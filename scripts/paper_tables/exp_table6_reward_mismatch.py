#!/usr/bin/env python3
"""Table 6 interface: reward timing mismatch."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from exp_common import (
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_TABLES_DIR,
    as_float,
    format_metric,
    mean,
    parse_metric_triples,
    read_json,
    success_rate_from_json,
    write_json,
    write_latex_table,
)


VARIANTS = {
    "sparse": "Sparse reward head",
    "success_to_go": "Success-to-go",
    "combined": "Combined",
}


def _first_fire(curve: list[Any], threshold: float) -> int:
    for idx, value in enumerate(curve):
        numeric = as_float(value)
        if numeric is not None and numeric > threshold:
            return idx
    return -1


def _timing_metrics(payload: Mapping[str, Any], threshold: float) -> dict[str, float | None]:
    pred_steps, real_steps, timing_errors, early = [], [], [], []
    records = payload.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for mode_name in ("closed_loop", "closed_loop_sft_actions", "true_sft_actions", "env_raw_actions"):
                mode = record.get(mode_name)
                if not isinstance(mode, Mapping):
                    continue
                pred = as_float(mode.get("reward_fire_abs", mode.get("fire_pred")))
                real = as_float(record.get("ideal_fire_abs", mode.get("fire_gt")))
                if pred is None or real is None or pred < 0 or real < 0:
                    continue
                pred_steps.append(pred)
                real_steps.append(real)
                timing_errors.append(abs(pred - real))
                early.append(1.0 if pred < real else 0.0)
    curve = payload.get("reward_curve") or payload.get("post_reward_curve")
    sparse = payload.get("demo_sparse_rewards")
    if isinstance(curve, list):
        pred = _first_fire(curve, threshold)
        real = _first_fire(sparse, 0.5) if isinstance(sparse, list) else -1
        if pred >= 0 and real >= 0:
            pred_steps.append(float(pred))
            real_steps.append(float(real))
            timing_errors.append(float(abs(pred - real)))
            early.append(1.0 if pred < real else 0.0)
    return {
        "predicted_reward_fire_timestep": mean(pred_steps),
        "real_terminal_timestep": mean(real_steps),
        "early_fire_rate": mean(early),
        "timing_error": mean(timing_errors),
    }


def cmd_collect(args: argparse.Namespace) -> int:
    table: dict[str, dict[str, float | None]] = {}
    labels = dict(VARIANTS)
    for variant, kind, path in parse_metric_triples(args.result):
        payload = read_json(path)
        row = table.setdefault(variant, {})
        labels.setdefault(variant, variant)
        if kind == "timing":
            row.update(_timing_metrics(payload, threshold=args.threshold))
        elif kind == "real":
            row["real_success"] = success_rate_from_json(payload)
        else:
            raise SystemExit("Table 6 --result KIND must be 'timing' or 'real'")

    out_json = write_json(args.out_json, {labels.get(k, k): v for k, v in table.items()})
    order = [key for key in VARIANTS if key in table] + [key for key in table if key not in VARIANTS]
    rows = []
    for key in order:
        row = table[key]
        rows.append([
            labels.get(key, key),
            format_metric(row.get("early_fire_rate"), pct=True),
            format_metric(row.get("timing_error"), precision=1),
            format_metric(row.get("real_success"), pct=True),
        ])
    out_tex = write_latex_table(
        args.out_tex,
        caption="Reward timing mismatch diagnostics.",
        label="tab:reward_mismatch",
        columns=["Variant", "Early Fire", "Timing Error", "Real Success"],
        rows=rows,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_tex}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--result", nargs=3, action="append", metavar=("VARIANT", "KIND", "JSON"))
    collect.add_argument("--threshold", type=float, default=0.5)
    collect.add_argument("--out-json", default=str(DEFAULT_OUTPUTS_DIR / "reward_mismatch.json"))
    collect.add_argument("--out-tex", default=str(DEFAULT_TABLES_DIR / "reward_mismatch.tex"))
    collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

