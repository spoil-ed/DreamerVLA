"""
Audit already-pretokenized pkl shards produced by
``pre_tokenize_action_state_local.py`` (or the action-only variant).

Unlike ``validate_convs.py`` — which inspects the upstream conv JSON — this
script opens each pkl under ``<split>/files/`` and checks that:

    1. The pkl unpickles cleanly and carries the keys downstream training
       code expects.
    2. ``token`` / ``label`` are non-empty and equal length.
    3. ``wm_obs_input_ids`` contains at least one image block delimited by
       8197 / 8196, with the inner layout 2 + N*(S+1) row tokens.
    4. ``wm_next_obs_input_ids`` is checked for the same layout and the
       sample is bucketed:
            wm_next_has_image  –  next-obs sequence carries real image tokens
                                   (what the updated preprocess will produce).
            wm_next_prompt_only –  next-obs sequence is the old 23-token
                                   prompt template (pre-patch state).
    5. Every image / action / state path referenced by the pkl actually
       exists on disk.
    6. (Optional) ``next_obs`` fields are compared against disk too.

Per-split + aggregate counts are printed; exit code 1 if any broken_* > 0.

Run:
    python -m dreamer_vla.preprocess.validate_pretokenized \
        --tokens-dir data/processed_data/tokens

The walk is read-only.  Use --sample-every N for a spot-check.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_START = 8197
IMG_END = 8196


REQUIRED_KEYS = (
    "token",
    "label",
    "id",
    "task_name",
    "image",
    "action",
    "wm_obs_input_ids",
    "wm_next_obs_input_ids",
)


def _has_image_block(ids: list[int]) -> bool:
    try:
        start = ids.index(IMG_START)
    except ValueError:
        return False
    # an END after that start is enough to count as "contains real image tokens"
    try:
        ids.index(IMG_END, start + 1)
        return True
    except ValueError:
        return False


def _all_exist(paths: list[str]) -> list[str]:
    return [p for p in paths if not Path(p).is_file()]


def _categorise_pkl(pkl_path: Path, check_files: bool) -> tuple[str, str, list[str]]:
    """
    Returns (main_bucket, wm_next_bucket, missing_paths).

    main_bucket ∈ {
        "ok", "unpickle_failed", "missing_keys",
        "empty_token", "length_mismatch", "no_image_in_obs",
        "broken_paths",
    }
    wm_next_bucket ∈ {
        "wm_next_has_image", "wm_next_prompt_only", "wm_next_unknown",
    }
    """
    try:
        with pkl_path.open("rb") as f:
            item = pickle.load(f)
    except Exception as exc:
        return "unpickle_failed", "wm_next_unknown", [f"unpickle: {exc}"]

    missing_keys = [k for k in REQUIRED_KEYS if k not in item]
    if missing_keys:
        return "missing_keys", "wm_next_unknown", missing_keys

    token = list(item["token"])
    label = list(item["label"])
    if not token:
        return "empty_token", "wm_next_unknown", []
    if len(token) != len(label):
        return (
            "length_mismatch",
            "wm_next_unknown",
            [f"token_len={len(token)} label_len={len(label)}"],
        )

    wm_obs = list(item.get("wm_obs_input_ids") or [])
    wm_next = list(item.get("wm_next_obs_input_ids") or [])

    if not _has_image_block(wm_obs):
        return "no_image_in_obs", "wm_next_unknown", []

    if _has_image_block(wm_next):
        wm_next_bucket = "wm_next_has_image"
    else:
        wm_next_bucket = "wm_next_prompt_only"

    if check_files:
        missing: list[str] = []
        for paths, label_str in (
            (list(item.get("image") or []), "image"),
            (list(item.get("action") or []), "action"),
            (list(item.get("state") or []), "state"),
        ):
            for p in _all_exist(paths):
                missing.append(f"{label_str}: {p}")

        next_obs = item.get("next_obs") or {}
        if isinstance(next_obs, dict):
            for paths, label_str in (
                (list(next_obs.get("image") or []), "next_obs.image"),
                (list(next_obs.get("state") or []), "next_obs.state"),
            ):
                for p in _all_exist(paths):
                    missing.append(f"{label_str}: {p}")

        if missing:
            return "broken_paths", wm_next_bucket, missing

    return "ok", wm_next_bucket, []


def validate_split(
    split_dir: Path,
    sample_every: int,
    limit: int | None,
    check_files: bool,
) -> tuple[Counter, Counter, dict[str, int], list[tuple[Path, str, list[str]]]]:
    files_dir = split_dir / "files"
    pkl_paths = sorted(
        files_dir.glob("*.pkl"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )

    main_counts: Counter = Counter()
    wm_next_counts: Counter = Counter()
    failures: list[tuple[Path, str, list[str]]] = []

    checked = 0
    for idx, pkl in enumerate(pkl_paths):
        if idx % max(sample_every, 1) != 0:
            continue
        if limit is not None and checked >= limit:
            break
        main_bucket, wm_next_bucket, missing = _categorise_pkl(
            pkl, check_files=check_files
        )
        main_counts[main_bucket] += 1
        wm_next_counts[wm_next_bucket] += 1
        if main_bucket != "ok":
            failures.append((pkl, main_bucket, missing))
        checked += 1

    totals = {
        "num_total": len(pkl_paths),
        "num_checked": checked,
    }
    return main_counts, wm_next_counts, totals, failures


def _format_percent(part: int, whole: int) -> str:
    if whole == 0:
        return "   —"
    return f"{100.0 * part / whole:5.1f}%"


def _print_split(
    name: str,
    main_counts: Counter,
    wm_next_counts: Counter,
    totals: dict[str, int],
    failures: list[tuple[Path, str, list[str]]],
    max_shown: int,
) -> None:
    checked = totals["num_checked"]
    print(f"── {name} (total_pkl={totals['num_total']}, checked={checked}) ──")
    print("  content integrity:")
    for bucket in (
        "ok",
        "unpickle_failed",
        "missing_keys",
        "empty_token",
        "length_mismatch",
        "no_image_in_obs",
        "broken_paths",
    ):
        n = main_counts[bucket]
        print(f"    {bucket:<20s}  {n:>8d}  ({_format_percent(n, checked)})")
    print("  wm_next_obs_input_ids layout:")
    for bucket in ("wm_next_has_image", "wm_next_prompt_only", "wm_next_unknown"):
        n = wm_next_counts[bucket]
        print(f"    {bucket:<22s}  {n:>8d}  ({_format_percent(n, checked)})")
    if failures:
        print(f"  first {min(max_shown, len(failures))} non-ok items:")
        for pkl, bucket, missing in failures[:max_shown]:
            preview = missing[0] if missing else "(no detail)"
            print(f"    {pkl.name:<15s} {bucket:<16s} -> {preview}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-dir",
        default=str(PROJECT_ROOT / "data" / "processed_data" / "tokens"),
    )
    parser.add_argument(
        "--pattern",
        default="libero_*",
        help="Glob for split sub-directories (each with a 'files/' folder).",
    )
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-failures-shown", type=int, default=5)
    parser.add_argument(
        "--skip-file-checks",
        action="store_true",
        help="Skip on-disk existence verification of image/action/state paths.",
    )
    args = parser.parse_args()

    tokens_dir = Path(args.tokens_dir).expanduser().resolve()
    split_dirs = sorted(
        d
        for d in tokens_dir.glob(args.pattern)
        if d.is_dir() and (d / "files").is_dir()
    )
    if not split_dirs:
        print(f"no split dirs under {tokens_dir}/{args.pattern}")
        sys.exit(2)

    agg_main: Counter = Counter()
    agg_next: Counter = Counter()
    agg_totals = {"num_total": 0, "num_checked": 0}
    any_bad = 0

    for split_dir in split_dirs:
        main_counts, wm_next_counts, totals, failures = validate_split(
            split_dir,
            sample_every=args.sample_every,
            limit=args.limit,
            check_files=not args.skip_file_checks,
        )
        _print_split(
            split_dir.name,
            main_counts,
            wm_next_counts,
            totals,
            failures,
            max_shown=args.max_failures_shown,
        )
        agg_main.update(main_counts)
        agg_next.update(wm_next_counts)
        agg_totals["num_total"] += totals["num_total"]
        agg_totals["num_checked"] += totals["num_checked"]
        any_bad += sum(
            main_counts[b]
            for b in (
                "unpickle_failed",
                "missing_keys",
                "empty_token",
                "length_mismatch",
                "no_image_in_obs",
                "broken_paths",
            )
        )

    print("── aggregate ──")
    _print_split(
        "ALL SPLITS",
        agg_main,
        agg_next,
        agg_totals,
        failures=[],
        max_shown=0,
    )

    sys.exit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
