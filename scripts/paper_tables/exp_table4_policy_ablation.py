#!/usr/bin/env python3
"""Table 4 interface: policy optimization ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from exp_common import (
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_TABLES_DIR,
    as_float,
    format_metric,
    parse_metric_triples,
    read_json,
    run_or_print,
    success_rate_from_json,
    write_json,
    write_latex_table,
)


VARIANTS = {
    "bc_only": {
        "label": "BC finetuning",
        "env": {"RUN_TAG": "table4_bc_only"},
        "overrides": [
            "training.run_actor_critic_phase=false",
            "algorithm.actor_bc_to_ref_scale=1.0",
        ],
    },
    "imagined_ppo": {
        "label": "Imagined PPO",
        "env": {"RUN_TAG": "table4_imagined_ppo"},
        "overrides": [
            "algorithm.update_type=wmpo_ppo",
            "algorithm.kl_coef=0.0",
            "algorithm.actor_bc_to_ref_scale=0.0",
        ],
    },
    "kl_regularized": {
        "label": "+ KL regularization",
        "env": {"RUN_TAG": "table4_kl_regularized"},
        "overrides": [
            "algorithm.update_type=wmpo_ppo",
            "algorithm.kl_coef=0.03",
            "algorithm.actor_bc_to_ref_scale=0.0",
        ],
    },
    "bc_anchor": {
        "label": "+ BC anchor",
        "env": {"RUN_TAG": "table4_bc_anchor"},
        "overrides": [
            "algorithm.update_type=wmpo_ppo",
            "algorithm.kl_coef=0.03",
            "algorithm.actor_bc_to_ref_scale=0.03",
        ],
    },
}


def _imagined_return(payload: dict) -> float | None:
    for key in ("returns_mean", "raw_returns_mean", "imagined_return", "epoch_returns"):
        value = as_float(payload.get(key))
        if value is not None:
            return value
    plateau = payload.get("plateau")
    if isinstance(plateau, dict):
        return as_float(plateau.get("returns_mean"))
    return None


def cmd_plan(args: argparse.Namespace) -> int:
    for key, spec in VARIANTS.items():
        env = {
            "CONFIG_NAME": args.config,
            "OUT_DIR": str(Path(args.out_dir) / key),
            **spec["env"],
        }
        code = run_or_print(
            ["bash", "scripts/train_dreamer_vla.sh", *spec["overrides"]],
            env=env,
            execute=args.execute,
        )
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
        if kind == "train":
            row["imagined_return"] = _imagined_return(payload)
        elif kind == "real":
            row["real_success"] = success_rate_from_json(payload)
        else:
            raise SystemExit("Table 4 --result KIND must be 'train' or 'real'")

    out_json = write_json(
        args.out_json, {labels.get(k, k): v for k, v in table.items()}
    )
    order = [key for key in VARIANTS if key in table] + [
        key for key in table if key not in VARIANTS
    ]
    rows = []
    for key in order:
        row = table[key]
        rows.append(
            [
                labels.get(key, key),
                format_metric(row.get("imagined_return"), precision=3),
                format_metric(row.get("real_success"), pct=True),
            ]
        )
    out_tex = write_latex_table(
        args.out_tex,
        caption="Policy optimization ablation.",
        label="tab:policy_ablation",
        columns=["Variant", "Imagined Return", "Real Success"],
        rows=rows,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_tex}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", default="dreamervla_rynn_dino_wm_actor_critic")
    plan.add_argument(
        "--out-dir", default="data/outputs/dreamervla/table4_policy_ablation"
    )
    plan.add_argument("--execute", action="store_true")
    plan.set_defaults(func=cmd_plan)
    collect = sub.add_parser("collect")
    collect.add_argument(
        "--result", nargs=3, action="append", metavar=("VARIANT", "KIND", "JSON")
    )
    collect.add_argument(
        "--out-json", default=str(DEFAULT_OUTPUTS_DIR / "policy_ablation.json")
    )
    collect.add_argument(
        "--out-tex", default=str(DEFAULT_TABLES_DIR / "policy_ablation.tex")
    )
    collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
