# ruff: noqa: E402
"""
Scan every conv JSON under ``data/processed_data/<artifact>/convs/`` (or a user-supplied
path) and check — without running the tokenizer — that each sample's files
exist and that ``ensure_next_obs`` lands on real next-frame files.

Per sample we bucket into:
    ok_existing_next_obs  –  conv already had a populated next_obs that
                              points at files on disk.
    ok_derived_next_obs   –  next_obs was empty; derivation succeeded and
                              every derived path exists.
    eot_empty_next_obs    –  next_obs was empty and derivation also came
                              back empty — treated as end-of-trajectory,
                              expected near the last 10 frames of each rollout.
    broken_inputs         –  at least one image/action/state path referenced
                              by the sample itself is missing.  Real failure.
    broken_next_obs       –  next_obs claims files that do not exist.  Real
                              failure.

Run:
    python -m dreamer_vla.preprocess.validate_convs \
        --convs-dir data/processed_data/libero_goal/convs

Reports per split + an aggregate.  Exits non-zero if any broken_* count > 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dreamer_vla.utils.paths import processed_data_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dreamer_vla.preprocess.pre_tokenize_action_state_local import (
    derive_next_obs_from_paths,
)


def _all_exist(paths: list[str]) -> tuple[bool, list[str]]:
    missing = [p for p in paths if not Path(p).is_file()]
    return (not missing), missing


def _categorise(raw_item: dict, image_views_per_frame: int) -> tuple[str, list[str]]:
    """Return (bucket_name, missing_paths)."""
    images = list(raw_item.get("image", []) or [])
    actions = list(raw_item.get("action", []) or [])
    states = list(raw_item.get("state", []) or [])

    for paths, label in ((images, "image"), (actions, "action"), (states, "state")):
        ok, missing = _all_exist(paths)
        if not ok:
            return "broken_inputs", [f"{label}: {p}" for p in missing]

    original = raw_item.get("next_obs")
    original_has_content = isinstance(original, dict) and (
        list((original or {}).get("image", []) or [])
        or list((original or {}).get("state", []) or [])
    )

    if original_has_content:
        nimg_ok, nimg_missing = _all_exist(list(original.get("image", []) or []))
        nst_ok, nst_missing = _all_exist(list(original.get("state", []) or []))
        if nimg_ok and nst_ok:
            return "ok_existing_next_obs", []
        return "broken_next_obs", [f"next_obs.image: {p}" for p in nimg_missing] + [
            f"next_obs.state: {p}" for p in nst_missing
        ]

    derived = derive_next_obs_from_paths(
        raw_item, image_views_per_frame=image_views_per_frame
    )
    derived_has_content = bool(derived.get("image") or derived.get("state"))
    if not derived_has_content:
        return "eot_empty_next_obs", []

    nimg_ok, nimg_missing = _all_exist(list(derived.get("image", []) or []))
    nst_ok, nst_missing = _all_exist(list(derived.get("state", []) or []))
    if nimg_ok and nst_ok:
        return "ok_derived_next_obs", []
    return "broken_next_obs", [f"derived next_obs.image: {p}" for p in nimg_missing] + [
        f"derived next_obs.state: {p}" for p in nst_missing
    ]


def validate_split(
    path: Path,
    image_views_per_frame: int,
    sample_every: int,
    limit: int | None,
) -> tuple[Counter, dict[str, int], list[tuple[int, str, list[str]]]]:
    with path.open() as f:
        data = json.load(f)

    counts: Counter = Counter()
    per_task: dict[str, Counter] = defaultdict(Counter)
    failures: list[tuple[int, str, list[str]]] = []

    n = len(data)
    indices = range(0, n, max(sample_every, 1))
    for checked_count, idx in enumerate(indices):
        if limit is not None and checked_count >= limit:
            break
        item = data[idx]
        bucket, missing = _categorise(item, image_views_per_frame)
        counts[bucket] += 1
        per_task[str(item.get("task_name", "?"))][bucket] += 1
        if bucket.startswith("broken_"):
            failures.append((idx, bucket, missing))

    totals = {
        "num_total": n,
        "num_checked": sum(counts.values()),
    }
    return counts, totals, failures


def _format_percent(part: int, whole: int) -> str:
    if whole == 0:
        return "   —"
    return f"{100.0 * part / whole:5.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--convs-dir",
        default=str(processed_data_path("convs")),
    )
    parser.add_argument(
        "--pattern",
        default="libero_*.json",
        help="Glob pattern; defaults to every libero_*.json in --convs-dir.",
    )
    parser.add_argument("--image-views-per-frame", type=int, default=2)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Check every Nth sample (1 = check them all).  Use to spot-check.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many samples per split (debug).",
    )
    parser.add_argument(
        "--max-failures-shown",
        type=int,
        default=5,
        help="Number of broken items to echo per split.",
    )
    args = parser.parse_args()

    convs_dir = Path(args.convs_dir).expanduser().resolve()
    json_paths = sorted(convs_dir.glob(args.pattern))
    if not json_paths:
        print(f"no json matched {convs_dir}/{args.pattern}")
        sys.exit(2)

    aggregate = Counter()
    aggregate_totals = {"num_total": 0, "num_checked": 0}
    any_broken = 0

    for path in json_paths:
        counts, totals, failures = validate_split(
            path,
            image_views_per_frame=args.image_views_per_frame,
            sample_every=args.sample_every,
            limit=args.limit,
        )
        aggregate.update(counts)
        aggregate_totals["num_total"] += totals["num_total"]
        aggregate_totals["num_checked"] += totals["num_checked"]

        checked = totals["num_checked"]
        broken = counts["broken_inputs"] + counts["broken_next_obs"]
        any_broken += broken

        print(f"── {path.name} (total={totals['num_total']}, checked={checked}) ──")
        for bucket in (
            "ok_existing_next_obs",
            "ok_derived_next_obs",
            "eot_empty_next_obs",
            "broken_inputs",
            "broken_next_obs",
        ):
            n = counts[bucket]
            print(f"  {bucket:<24s}  {n:>7d}  ({_format_percent(n, checked)})")

        if failures:
            print(f"  first {min(args.max_failures_shown, len(failures))} failures:")
            for idx, bucket, missing in failures[: args.max_failures_shown]:
                miss_preview = missing[0] if missing else "(no detail)"
                print(f"    idx={idx:<6d}  {bucket}  ->  {miss_preview}")
        print()

    print("── aggregate over all splits ──")
    checked = aggregate_totals["num_checked"]
    for bucket in (
        "ok_existing_next_obs",
        "ok_derived_next_obs",
        "eot_empty_next_obs",
        "broken_inputs",
        "broken_next_obs",
    ):
        n = aggregate[bucket]
        print(f"  {bucket:<24s}  {n:>8d}  ({_format_percent(n, checked)})")
    print(f"  total samples across splits: {aggregate_totals['num_total']}")
    print(f"  samples actually checked:    {checked}")

    sys.exit(1 if any_broken else 0)


if __name__ == "__main__":
    main()
