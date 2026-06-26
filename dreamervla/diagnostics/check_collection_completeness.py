"""Inspect collected rollout completeness for episode-level resume.

Example:
  python -m dreamervla.diagnostics.check_collection_completeness \
    --reward-dir data/collected_rollouts/libero_goal/reward \
    --hidden-dir data/collected_rollouts/libero_goal/hidden \
    --target-episodes 500 --num-tasks 10 --json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from dreamervla.dataset.collection_manifest import complete_episode_ids_per_task


def build_collection_completeness_report(
    reward_dir: str | Path,
    hidden_dir: str | Path,
    *,
    target_episodes: int | None,
    num_tasks: int,
    task_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return complete and missing episode ids for collected rollout data."""
    complete_ids = complete_episode_ids_per_task(reward_dir, hidden_dir)
    selected_tasks = (
        [int(task_id) for task_id in task_ids]
        if task_ids is not None
        else list(range(int(num_tasks)))
    )
    target_per_task = (
        math.ceil(int(target_episodes) / int(num_tasks))
        if target_episodes is not None and int(num_tasks) > 0
        else None
    )
    missing: dict[int, list[int]] = {}
    if target_per_task is not None:
        for task_id in selected_tasks:
            present = complete_ids.get(int(task_id), set())
            gaps = [
                episode_id
                for episode_id in range(int(target_per_task))
                if episode_id not in present
            ]
            if gaps:
                missing[int(task_id)] = gaps
    total = sum(len(ids) for ids in complete_ids.values())
    complete = (
        target_episodes is not None
        and total >= int(target_episodes)
        and not missing
    )
    return {
        "reward_dir": str(Path(reward_dir).expanduser()),
        "hidden_dir": str(Path(hidden_dir).expanduser()),
        "target_episodes": None if target_episodes is None else int(target_episodes),
        "num_tasks": int(num_tasks),
        "target_per_task": target_per_task,
        "total_complete_episodes": int(total),
        "complete": bool(complete),
        "complete_episode_ids": {
            str(task_id): sorted(int(ep) for ep in ids)
            for task_id, ids in sorted(complete_ids.items())
        },
        "missing_episode_ids": {
            str(task_id): list(episodes)
            for task_id, episodes in sorted(missing.items())
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-dir", required=True)
    parser.add_argument("--hidden-dir", required=True)
    parser.add_argument("--target-episodes", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _parse_task_ids(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    if value.strip().lower() == "all":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_collection_completeness_report(
        args.reward_dir,
        args.hidden_dir,
        target_episodes=args.target_episodes,
        num_tasks=args.num_tasks,
        task_ids=_parse_task_ids(args.task_ids),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"reward_dir: {report['reward_dir']}")
        print(f"hidden_dir: {report['hidden_dir']}")
        print(
            f"complete episodes: {report['total_complete_episodes']} / "
            f"{report['target_episodes']}"
        )
        if report["missing_episode_ids"]:
            print("missing episode ids:")
            for task_id, episodes in report["missing_episode_ids"].items():
                joined = ",".join(str(ep) for ep in episodes)
                print(f"  task{task_id}: {joined}")
        else:
            print("missing episode ids: none")
    return 0 if report["target_episodes"] is None or report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
