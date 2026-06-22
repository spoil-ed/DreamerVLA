"""Manifest + episode-level resume helpers for the unified collected_rollouts space.

Cold-start collection writes to a stable ``data/collected_rollouts/<task>/`` space.
A ``collection_manifest.json`` next to the shards records metadata (task, target,
collected count, success, shards, config snapshot) and doubles as the resume state:
on relaunch we count what is already on disk and top up to the target by appending
new shards instead of overwriting.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

MANIFEST_NAME = "collection_manifest.json"


def count_collected_episodes(reward_dir: str | Path) -> int:
    """Total episodes already on disk, summed across reward shards.

    A shard that cannot be opened (partial/corrupt, e.g. an interrupted run) is
    skipped with a warning rather than crashing — it simply does not count, so the
    resume top-up will re-collect those episodes.
    """
    import h5py

    directory = Path(reward_dir).expanduser()
    if not directory.is_dir():
        return 0
    total = 0
    for shard in sorted(directory.glob("*.hdf5")):
        try:
            with h5py.File(str(shard), "r") as f:
                data = f.get("data")
                if data is None:
                    continue
                num = data.attrs.get("num_demos")
                total += int(num) if num is not None else len(list(data.keys()))
        except (OSError, KeyError) as exc:
            warnings.warn(f"skipping unreadable shard {shard}: {exc}", stacklevel=2)
    return total


def count_episodes_per_task(reward_dir: str | Path) -> dict[int, int]:
    """Episodes already on disk bucketed by their ``task_id`` demo attr."""
    import h5py

    directory = Path(reward_dir).expanduser()
    counts: dict[int, int] = {}
    if not directory.is_dir():
        return counts
    for shard in sorted(directory.glob("*.hdf5")):
        try:
            with h5py.File(str(shard), "r") as f:
                data = f.get("data")
                if data is None:
                    continue
                for key in data.keys():
                    tid = int(data[key].attrs.get("task_id", -1))
                    counts[tid] = counts.get(tid, 0) + 1
        except (OSError, KeyError) as exc:
            warnings.warn(f"skipping unreadable shard {shard}: {exc}", stacklevel=2)
    return counts


def summarize_collection(
    reward_dir: str | Path, *, target_total: int | None, num_tasks: int
) -> dict[str, Any]:
    """Inspect existing collected data and report progress toward the target."""
    per_task = count_episodes_per_task(reward_dir)
    total = sum(per_task.values())
    remaining: int | None = None
    target_per_task: int | None = None
    complete = False
    if target_total is not None:
        target_total = int(target_total)
        remaining = max(0, target_total - total)
        complete = remaining == 0
        target_per_task = math.ceil(target_total / num_tasks) if num_tasks > 0 else None
    return {
        "per_task": dict(sorted(per_task.items())),
        "total": total,
        "target_total": target_total,
        "target_per_task": target_per_task,
        "num_tasks": int(num_tasks),
        "remaining": remaining,
        "complete": complete,
    }


def format_collection_report(summary: dict[str, Any], *, root: str | Path) -> str:
    """Human-readable pre-collection report (counts, tasks, what is still needed)."""
    lines = [f"[collect] inspecting {root}"]
    total = summary["total"]
    target = summary["target_total"]
    if target is None:
        lines.append(f"  collected: {total} episodes (no target set)")
    elif summary["complete"]:
        lines.append(f"  collected: {total} / {target} target  (complete)")
    else:
        lines.append(
            f"  collected: {total} / {target} target  (need {summary['remaining']} more)"
        )
    per_task = summary["per_task"]
    if per_task:
        parts = " ".join(f"task{tid}={n}" for tid, n in per_task.items())
        tpt = summary["target_per_task"]
        suffix = f"  (target {tpt}/task)" if tpt is not None else ""
        lines.append(f"  per task:  {parts}{suffix}")
    else:
        lines.append("  per task:  (none collected yet)")
    return "\n".join(lines)


def next_shard_index(directory: str | Path, *, prefix: str) -> int:
    """Lowest unused ``{prefix}_{NNN}.hdf5`` index in ``directory`` (0 if none)."""
    path = Path(directory).expanduser()
    if not path.is_dir():
        return 0
    highest = -1
    for shard in path.glob(f"{prefix}_*.hdf5"):
        suffix = shard.name[len(prefix) + 1 : -len(".hdf5")]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def resume_plan(*, target_total: int, num_tasks: int, collected: int) -> dict[str, Any]:
    """Plan the next collection pass to reach ``target_total`` episodes.

    ``episodes_per_task`` is the per-task count for THIS pass (the remaining total
    spread uniformly, rounded up), which the collector consumes.
    """
    remaining = max(0, int(target_total) - int(collected))
    complete = remaining <= 0
    episodes_per_task = (
        math.ceil(remaining / num_tasks) if num_tasks > 0 and remaining > 0 else 0
    )
    return {
        "target": int(target_total),
        "collected": int(collected),
        "remaining": remaining,
        "episodes_per_task": episodes_per_task,
        "complete": complete,
    }


def read_manifest(root: str | Path) -> dict[str, Any] | None:
    path = Path(root).expanduser() / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(root: str | Path, data: dict[str, Any]) -> Path:
    directory = Path(root).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path
