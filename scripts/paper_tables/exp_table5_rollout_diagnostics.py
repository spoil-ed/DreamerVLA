#!/usr/bin/env python3
"""Table 5 interface: world-model rollout diagnostics."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from exp_common import (
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_TABLES_DIR,
    as_float,
    format_metric,
    mean,
    read_json,
    run_or_print,
    write_json,
    write_latex_table,
)


STEPS = [1, 5, 10, 20]


def _values_at_step(payload: Mapping[str, Any], step: int) -> dict[str, float | None]:
    mses, cosines, norms = [], [], []
    starts = payload.get("starts")
    if isinstance(starts, list):
        for item in starts:
            if not isinstance(item, Mapping):
                continue
            idx = min(step, len(item.get("feat_mse", [])) - 1)
            if idx >= 0:
                mses.append(as_float(item.get("feat_mse", [])[idx]))
            idx = min(step, len(item.get("cosine", [])) - 1)
            if idx >= 0:
                cosines.append(as_float(item.get("cosine", [])[idx]))
            hidden_norm = item.get("hidden_norm")
            if isinstance(hidden_norm, list):
                idx = min(step, len(hidden_norm) - 1)
                if idx >= 0:
                    norms.append(as_float(hidden_norm[idx]))
    records = payload.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            mode = record.get("closed_loop") or record.get("closed_loop_sft_actions")
            if isinstance(mode, Mapping):
                mses.append(as_float(mode.get("mse_mean")))
                cosines.append(as_float(mode.get("cos_mean")))
                norms.append(as_float(mode.get("hidden_norm")))
    return {"mse": mean(mses), "cosine": mean(cosines), "hidden_norm": mean(norms)}


def cmd_plan(args: argparse.Namespace) -> int:
    command = [
        "python",
        "scripts/measure_wm_closed_loop.py",
        "--ckpt",
        args.ckpt,
        "--hidden-hdf5-dir",
        args.hidden_hdf5_dir,
        "--reward-hdf5-dir",
        args.reward_hdf5_dir,
        "--actor-cfg",
        args.actor_cfg,
        "--start-steps",
        *[str(step) for step in STEPS],
        "--out-json",
        args.out_json,
    ]
    return run_or_print(command, env=None, execute=args.execute)


def cmd_collect(args: argparse.Namespace) -> int:
    payload = read_json(args.input_json)
    out = {f"step_{step}": _values_at_step(payload, step) for step in STEPS}
    out_json = write_json(args.out_json, out)
    rows = [
        [
            str(step),
            format_metric(out[f"step_{step}"].get("mse"), precision=4),
            format_metric(out[f"step_{step}"].get("cosine"), precision=3),
            format_metric(out[f"step_{step}"].get("hidden_norm"), precision=3),
        ]
        for step in STEPS
    ]
    out_tex = write_latex_table(
        args.out_tex,
        caption="Closed-loop world-model rollout diagnostics.",
        label="tab:wm_rollout_diagnostics",
        columns=["Step", "MSE", "Cosine", "Hidden Norm"],
        rows=rows,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_tex}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--ckpt", required=True)
    plan.add_argument("--hidden-hdf5-dir", required=True)
    plan.add_argument("--reward-hdf5-dir", required=True)
    plan.add_argument(
        "--actor-cfg", default="configs/dreamervla_rynn_dino_wm_actor_critic.yaml"
    )
    plan.add_argument("--out-json", default="outputs/wm_rollout_diagnostics_raw.json")
    plan.add_argument("--execute", action="store_true")
    plan.set_defaults(func=cmd_plan)
    collect = sub.add_parser("collect")
    collect.add_argument("--input-json", required=True)
    collect.add_argument(
        "--out-json", default=str(DEFAULT_OUTPUTS_DIR / "wm_rollout_diagnostics.json")
    )
    collect.add_argument(
        "--out-tex", default=str(DEFAULT_TABLES_DIR / "wm_rollout_diagnostics.tex")
    )
    collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
