#!/usr/bin/env python3
"""Table 2 interface: hidden-state ablation."""

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
    "dino_visual": {
        "label": "DINO visual embedding",
        "env": {"WM_KIND": "dreamerv3_pixel", "CONFIG_NAME": "dreamerv3_pixel_libero_goal"},
    },
    "pi0_query": {
        "label": "pi0 query hidden",
        "env": {"WM_KIND": "oft_dino", "CONFIG_NAME": "oft_dino_wm_action_hidden_libero_goal_C"},
    },
    "full_action": {
        "label": "Full action hidden",
        "env": {"WM_KIND": "rynn_dino", "CONFIG_NAME": "rynn_dino_wm_action_hidden_libero_goal"},
    },
}


def _wm_metrics(payload: Mapping[str, Any]) -> dict[str, float | None]:
    records = payload.get("records")
    if isinstance(records, list):
        return {
            "rollout_mse": collect_record_values(records, "closed_loop", "mse_mean")
            or collect_record_values(records, "closed_loop_sft_actions", "mse_mean"),
            "rollout_cosine": collect_record_values(records, "closed_loop", "cos_mean")
            or collect_record_values(records, "closed_loop_sft_actions", "cos_mean"),
        }
    starts = payload.get("starts")
    if isinstance(starts, list):
        mses, cosines = [], []
        for item in starts:
            if not isinstance(item, Mapping):
                continue
            mses.append(mean(item.get("feat_mse", [])))
            cosines.append(mean(item.get("cosine", [])))
        return {"rollout_mse": mean(mses), "rollout_cosine": mean(cosines)}
    return {
        "rollout_mse": as_float(payload.get("rollout_mse", payload.get("mse"))),
        "rollout_cosine": as_float(payload.get("rollout_cosine", payload.get("cosine"))),
    }


def cmd_plan(args: argparse.Namespace) -> int:
    for key, spec in VARIANTS.items():
        env = dict(spec["env"])
        env.update(
            {
                "DINO_WM_ROLLOUT_HORIZON": str(args.rollout_horizon),
                "DINO_WM_ROLLOUT_LOSS_SCALE": str(args.rollout_loss_scale),
                "OUT_DIR": str(Path(args.out_dir) / key),
            }
        )
        code = run_or_print(["bash", "scripts/train_wm.sh"], env=env, execute=args.execute)
        if code:
            return code
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    rows_by_variant: dict[str, dict[str, float | None]] = {}
    labels: dict[str, str] = {key: str(spec["label"]) for key, spec in VARIANTS.items()}
    for variant, kind, path in parse_metric_triples(args.result):
        payload = read_json(path)
        row = rows_by_variant.setdefault(variant, {})
        labels.setdefault(variant, variant)
        if kind == "wm":
            row.update(_wm_metrics(payload))
        elif kind == "long":
            row["long_success"] = success_rate_from_json(payload)
        else:
            raise SystemExit("Table 2 --result KIND must be 'wm' or 'long'")

    out_json = write_json(args.out_json, {labels.get(k, k): v for k, v in rows_by_variant.items()})
    order = [key for key in VARIANTS if key in rows_by_variant] + [
        key for key in rows_by_variant if key not in VARIANTS
    ]
    tex_rows = []
    for key in order:
        row = rows_by_variant[key]
        tex_rows.append([
            labels.get(key, key),
            format_metric(row.get("rollout_mse"), precision=4),
            format_metric(row.get("rollout_cosine"), precision=3),
            format_metric(row.get("long_success"), pct=True),
        ])
    out_tex = write_latex_table(
        args.out_tex,
        caption="Hidden-state ablation with matched world-model training settings.",
        label="tab:hidden_ablation",
        columns=["Latent", "Rollout MSE", "Rollout Cos.", "Long"],
        rows=tex_rows,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_tex}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--rollout-horizon", type=int, default=8)
    plan.add_argument("--rollout-loss-scale", type=float, default=1.0)
    plan.add_argument("--out-dir", default="data/outputs/worldmodel/table2_hidden_ablation")
    plan.add_argument("--execute", action="store_true")
    plan.set_defaults(func=cmd_plan)

    collect = sub.add_parser("collect")
    collect.add_argument("--result", nargs=3, action="append", metavar=("VARIANT", "KIND", "JSON"))
    collect.add_argument("--out-json", default=str(DEFAULT_OUTPUTS_DIR / "hidden_ablation.json"))
    collect.add_argument("--out-tex", default=str(DEFAULT_TABLES_DIR / "hidden_ablation.tex"))
    collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

