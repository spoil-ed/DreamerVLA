#!/usr/bin/env python3
"""Table 1 interface: main LIBERO results.

Collects per-suite LIBERO eval JSON files into:
  outputs/libero_main_results.json
  paper_tables/libero_main.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exp_common import (
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_TABLES_DIR,
    format_metric,
    mean,
    parse_metric_triples,
    read_json,
    run_or_print,
    success_rate_from_json,
    write_json,
    write_latex_table,
)


SUITES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}
DEFAULT_METHODS = [
    "RynnVLA SFT",
    "RynnVLA + BC finetuning",
    "RynnVLA + real-env PPO",
    "DreamerVLA",
]


def cmd_plan(args: argparse.Namespace) -> int:
    methods = args.method or {}
    for method, ckpt in methods.items():
        for suite_name in SUITES.values():
            env = {
                "TASK_SUITE": suite_name,
                "CKPT_PATH": ckpt,
                "NUM_EPISODES": str(args.num_episodes),
                "ACTION_STEPS": str(
                    args.action_steps
                    or (5 if suite_name in {"libero_goal", "libero_object"} else 10)
                ),
                "OUT_DIR": str(
                    Path(args.out_dir)
                    / method.replace(" ", "_").replace("+", "plus")
                    / suite_name
                ),
            }
            code = run_or_print(
                ["bash", "scripts/evals_libero/_eval_runner.sh"],
                env=env,
                execute=args.execute,
            )
            if code:
                return code
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    triples = parse_metric_triples(args.result)
    table: dict[str, dict[str, float | None]] = {}
    for method, suite, path in triples:
        if suite not in SUITES:
            raise SystemExit(
                f"Unknown suite key {suite!r}; expected one of {sorted(SUITES)}"
            )
        payload = read_json(path)
        table.setdefault(method, {})[suite] = success_rate_from_json(payload)

    method_order = args.method_order or DEFAULT_METHODS
    for method in list(table):
        if method not in method_order:
            method_order.append(method)
    for method in method_order:
        row = table.setdefault(method, {})
        values = [row.get(key) for key in SUITES]
        row["avg"] = mean(values)

    out_json = write_json(args.out_json, table)
    rows = []
    for method in method_order:
        row = table.get(method, {})
        rows.append(
            [
                method,
                format_metric(row.get("spatial"), pct=True),
                format_metric(row.get("object"), pct=True),
                format_metric(row.get("goal"), pct=True),
                format_metric(row.get("long"), pct=True),
                format_metric(row.get("avg"), pct=True),
            ]
        )
    out_tex = write_latex_table(
        args.out_tex,
        caption="Main LIBERO success rates.",
        label="tab:libero_main",
        columns=["Method", "Spatial", "Object", "Goal", "Long", "Avg."],
        rows=rows,
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_tex}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser(
        "plan", help="print or execute standardized LIBERO rollout commands"
    )
    plan.add_argument(
        "--method", nargs="+", action="append", default=[], metavar="NAME=CKPT"
    )
    plan.add_argument("--num-episodes", type=int, default=10)
    plan.add_argument("--action-steps", type=int, default=None)
    plan.add_argument("--out-dir", default="data/outputs/eval/table1_libero_main")
    plan.add_argument("--execute", action="store_true")
    plan.set_defaults(func=cmd_plan)

    collect = sub.add_parser("collect", help="aggregate eval JSONs and export Table 1")
    collect.add_argument(
        "--result", nargs=3, action="append", metavar=("METHOD", "SUITE", "JSON")
    )
    collect.add_argument("--method-order", nargs="*")
    collect.add_argument(
        "--out-json", default=str(DEFAULT_OUTPUTS_DIR / "libero_main_results.json")
    )
    collect.add_argument(
        "--out-tex", default=str(DEFAULT_TABLES_DIR / "libero_main.tex")
    )
    collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "plan":
        merged = {}
        for group in args.method:
            for item in group:
                if "=" not in item:
                    parser.error("--method expects NAME=CKPT")
                name, ckpt = item.split("=", 1)
                merged[name] = ckpt
        args.method = merged
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
