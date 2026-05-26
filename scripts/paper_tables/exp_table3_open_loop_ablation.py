#!/usr/bin/env python3
"""Table 3 interface: open-loop world-model ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from exp_common import (
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_TABLES_DIR,
    as_float,
    collect_record_values,
    format_metric,
    mean,
    parse_metric_triples,
    read_json,
    run_or_print,
    success_rate_from_json,
    write_json,
    write_latex_table,
)


VARIANTS = {
    "teacher_forcing": {
        "label": "Teacher forcing only",
        "env": {"DINO_WM_ROLLOUT_LOSS_SCALE": "0.0", "DINO_WM_ROLLOUT_HORIZON": "0"},
    },
    "recursive_rollout": {
        "label": "Recursive rollout loss",
        "env": {"DINO_WM_ROLLOUT_LOSS_SCALE": "1.0", "DINO_WM_ROLLOUT_HORIZON": "8"},
    },
    "reward_head": {
        "label": "+ reward head",
        "env": {
            "DINO_WM_ROLLOUT_LOSS_SCALE": "1.0",
            "DINO_WM_ROLLOUT_HORIZON": "8",
            "DINO_WM_REWARD_HEAD_TYPE": "binary",
            "DINO_WM_REWARD_LOSS_SCALE": "1.0",
        },
    },
    "success_to_go": {
        "label": "+ success-to-go",
        "extra_overrides": ["world_model.success_return_head_type=binary", "world_model.success_return_loss_scale=1.0"],
    },
}


def _open_loop_metrics(payload: Mapping[str, Any]) -> dict[str, float | None]:
    records = payload.get("records")
    if isinstance(records, list):
        return {
            "one_step_mse": collect_record_values(records, "env", "mse_mean")
            or collect_record_values(records, "env_raw_actions", "mse_mean"),
            "rollout_mse": collect_record_values(records, "closed_loop", "mse_mean")
            or collect_record_values(records, "closed_loop_sft_actions", "mse_mean"),
        }
    starts = payload.get("starts")
    if isinstance(starts, list):
        return {"one_step_mse": None, "rollout_mse": mean(mean(item.get("feat_mse", [])) for item in starts if isinstance(item, Mapping))}
    return {
        "one_step_mse": as_float(payload.get("one_step_mse", payload.get("next_latent_mse"))),
        "rollout_mse": as_float(payload.get("rollout_mse")),
    }


def cmd_plan(args: argparse.Namespace) -> int:
    for key, spec in VARIANTS.items():
        env = {"WM_KIND": "rynn_dino", "CONFIG_NAME": "rynn_dino_wm_action_hidden_libero_goal"}
        env.update(spec.get("env", {}))
        env["OUT_DIR"] = str(Path(args.out_dir) / key)
        command = ["bash", "scripts/train_wm.sh", *spec.get("extra_overrides", [])]
        code = run_or_print(command, env=env, execute=args.execute)
        if code:
            return code
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    table: dict[str, dict[str, float | None]] = {}
    labels = {key: str(spec["label"]) for key, spec in VARIANTS.items()}
    for variant, kind, path in parse_metric_triples(args.result):
        payload = read_json(path)
        row = table.setdefault(variant, {})
        labels.setdefault(variant, variant)
        if kind == "wm":
            row.update(_open_loop_metrics(payload))
        elif kind == "real":
            row["real_success"] = success_rate_from_json(payload)
        else:
            raise SystemExit("Table 3 --result KIND must be 'wm' or 'real'")

    out_json = write_json(args.out_json, {labels.get(k, k): v for k, v in table.items()})
    order = [key for key in VARIANTS if key in table] + [key for key in table if key not in VARIANTS]
    rows = []
    for key in order:
        row = table[key]
        rows.append([
            labels.get(key, key),
            format_metric(row.get("one_step_mse"), precision=4),
            format_metric(row.get("rollout_mse"), precision=4),
            format_metric(row.get("real_success"), pct=True),
        ])
    out_tex = write_latex_table(
        args.out_tex,
        caption="Open-loop world-model ablation.",
        label="tab:open_loop_ablation",
        columns=["Variant", "One-step MSE", "Rollout MSE", "Real Success"],
        rows=rows,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_tex}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--out-dir", default="data/outputs/worldmodel/table3_open_loop_ablation")
    plan.add_argument("--execute", action="store_true")
    plan.set_defaults(func=cmd_plan)
    collect = sub.add_parser("collect")
    collect.add_argument("--result", nargs=3, action="append", metavar=("VARIANT", "KIND", "JSON"))
    collect.add_argument("--out-json", default=str(DEFAULT_OUTPUTS_DIR / "open_loop_ablation.json"))
    collect.add_argument("--out-tex", default=str(DEFAULT_TABLES_DIR / "open_loop_ablation.tex"))
    collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

