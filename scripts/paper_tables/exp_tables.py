#!/usr/bin/env python3
"""Master entrypoint for DreamerVLA paper table experiment interfaces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"

TABLE_SCRIPTS = {
    "1": "exp_table1_libero_main.py",
    "libero_main": "exp_table1_libero_main.py",
    "2": "exp_table2_hidden_ablation.py",
    "hidden_ablation": "exp_table2_hidden_ablation.py",
    "3": "exp_table3_open_loop_ablation.py",
    "open_loop_ablation": "exp_table3_open_loop_ablation.py",
    "4": "exp_table4_policy_ablation.py",
    "policy_ablation": "exp_table4_policy_ablation.py",
    "5": "exp_table5_rollout_diagnostics.py",
    "wm_rollout_diagnostics": "exp_table5_rollout_diagnostics.py",
    "6": "exp_table6_reward_mismatch.py",
    "reward_mismatch": "exp_table6_reward_mismatch.py",
}


def cmd_list() -> int:
    print("DreamerVLA experiment interfaces:")
    print("  1 / libero_main              -> outputs/libero_main_results.json")
    print("  2 / hidden_ablation          -> outputs/hidden_ablation.json")
    print("  3 / open_loop_ablation       -> outputs/open_loop_ablation.json")
    print("  4 / policy_ablation          -> outputs/policy_ablation.json")
    print("  5 / wm_rollout_diagnostics   -> outputs/wm_rollout_diagnostics.json")
    print("  6 / reward_mismatch          -> outputs/reward_mismatch.json")
    print("")
    print("Examples:")
    print("  python scripts/exp_tables.py 1 collect --result DreamerVLA goal path/to/eval_libero_metrics.json")
    print("  python scripts/exp_tables.py 5 plan --ckpt path/to/wm.ckpt --hidden-hdf5-dir ... --reward-hdf5-dir ...")
    return 0


def cmd_forward(table: str, remainder: list[str]) -> int:
    script = TABLE_SCRIPTS[table]
    command = [sys.executable, str(SCRIPT_DIR / script), *remainder]
    return subprocess.call(command, cwd=str(ROOT))


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "list"}:
        return cmd_list()
    table = argv[0]
    if table not in TABLE_SCRIPTS:
        choices = ", ".join(sorted(TABLE_SCRIPTS))
        print(f"Unknown table {table!r}. Choices: {choices}", file=sys.stderr)
        return 2
    return cmd_forward(table, argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
