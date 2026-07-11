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
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

_REWARD_DATASETS = (
    "actions",
    "dones",
    "rewards",
    "sparse_rewards",
    "robot_states",
    "states",
)
_OBS_DATASETS = (
    "agentview_rgb",
    "eye_in_hand_rgb",
    "ee_pos",
    "ee_ori",
    "ee_states",
    "gripper_states",
    "joint_states",
)

MANIFEST_NAME = "collection_manifest.json"
ONLINE_ROLLOUT_MANIFEST_NAME = "manifest.jsonl"
ONLINE_ROLLOUT_EPISODES_DIR = "episodes"
EPISODE_INDEX_NAME = "episode_index.jsonl"


def count_collected_episodes(
    reward_dir: str | Path, hidden_dir: str | Path | None = None
) -> int:
    """Total episodes already on disk, summed across reward shards.

    A shard that cannot be opened (partial/corrupt, e.g. an interrupted run) is
    skipped with a warning rather than crashing — it simply does not count, so the
    resume top-up will re-collect those episodes.
    """
    if hidden_dir is not None:
        return sum(len(ids) for ids in complete_episode_ids_per_task(reward_dir, hidden_dir).values())

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


def count_episodes_per_task(
    reward_dir: str | Path, hidden_dir: str | Path | None = None
) -> dict[int, int]:
    """Episodes already on disk bucketed by their ``task_id`` demo attr."""
    if hidden_dir is not None:
        return {
            task_id: len(ids)
            for task_id, ids in complete_episode_ids_per_task(reward_dir, hidden_dir).items()
        }

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


def complete_episode_ids_per_task(
    reward_dir: str | Path, hidden_dir: str | Path
) -> dict[int, set[int]]:
    """Complete ``episode_id`` values bucketed by ``task_id``.

    A complete collected episode must have a readable reward demo and a matching
    hidden sidecar demo in the same shard. ``complete=False`` on either side,
    missing required reward fields, missing ``obs_embedding``, or a hidden length
    shorter than the reward length makes the episode incomplete and therefore
    eligible for re-collection.
    """
    import h5py

    reward = Path(reward_dir).expanduser()
    hidden = Path(hidden_dir).expanduser()
    complete: dict[int, set[int]] = {}
    if not reward.is_dir() or not hidden.is_dir():
        return complete
    fallback_index: dict[int, int] = {}
    for shard in sorted(reward.glob("*.hdf5")):
        hidden_shard = hidden / shard.name
        if not hidden_shard.is_file():
            continue
        try:
            with h5py.File(str(shard), "r") as rf, h5py.File(str(hidden_shard), "r") as hf:
                reward_data = rf.get("data")
                hidden_data = hf.get("data")
                if reward_data is None or hidden_data is None:
                    continue
                for demo_key in sorted(reward_data.keys()):
                    reward_demo = reward_data[demo_key]
                    hidden_demo = hidden_data.get(demo_key)
                    if hidden_demo is None:
                        continue
                    length = _complete_reward_length(reward_demo)
                    if length is None:
                        continue
                    if not _complete_hidden_demo(hidden_demo, length):
                        continue
                    task_id = int(reward_demo.attrs.get("task_id", -1))
                    if task_id < 0:
                        continue
                    if "episode_id" in reward_demo.attrs:
                        episode_id = int(reward_demo.attrs["episode_id"])
                    else:
                        episode_id = fallback_index.get(task_id, 0)
                        fallback_index[task_id] = episode_id + 1
                    complete.setdefault(task_id, set()).add(episode_id)
        except (OSError, KeyError) as exc:
            warnings.warn(f"skipping unreadable shard {shard}: {exc}", stacklevel=2)
    return complete


def _attr_is_complete(group: Any) -> bool:
    return bool(group.attrs.get("complete", True))


def _complete_reward_length(demo: Any) -> int | None:
    if not _attr_is_complete(demo):
        return None
    for key in _REWARD_DATASETS:
        if key not in demo:
            return None
    obs = demo.get("obs")
    if obs is None:
        return None
    for key in _OBS_DATASETS:
        if key not in obs:
            return None
    try:
        length = int(demo.attrs.get("num_samples", demo["actions"].shape[0]))
    except (TypeError, ValueError):
        return None
    if length <= 0:
        return None
    for key in _REWARD_DATASETS:
        if int(demo[key].shape[0]) < length:
            return None
    for key in _OBS_DATASETS:
        if int(obs[key].shape[0]) < length:
            return None
    return length


def _complete_hidden_demo(demo: Any, length: int) -> bool:
    if not _attr_is_complete(demo):
        return False
    if "obs_embedding" not in demo:
        return False
    return int(demo["obs_embedding"].shape[0]) >= int(length)


def quarantine_corrupt_shards(
    reward_dir: str | Path, hidden_dir: str | Path
) -> list[str]:
    """Phase-1 integrity check: move truncated/unreadable shards out of the way.

    A crashed collect can leave a half-written shard (e.g. a 96-byte truncated HDF5).
    Readers that merely skip it stay alive, but the bad file lingers and re-breaks every
    later read. Here we fail-fast at the source: open each reward shard, and on ``OSError``
    move it AND its same-named hidden sidecar into a ``.corrupt/`` subdir (recoverable,
    not deleted) so collection-append and warmup see only valid shards. ``.corrupt/`` is
    not matched by the ``*.hdf5`` glob, so quarantined shards are not re-scanned. Returns
    the quarantined shard names.
    """
    import h5py

    reward = Path(reward_dir).expanduser()
    hidden = Path(hidden_dir).expanduser()
    quarantined: list[str] = []
    if not reward.is_dir():
        return quarantined
    for shard in sorted(reward.glob("*.hdf5")):
        try:
            with h5py.File(str(shard), "r"):
                pass
            continue
        except OSError as exc:
            reason = exc
        for directory in (reward, hidden):
            src = directory / shard.name
            if src.exists():
                dest_dir = directory / ".corrupt"
                dest_dir.mkdir(exist_ok=True)
                src.rename(dest_dir / src.name)
        warnings.warn(
            f"quarantined corrupt shard {shard.name} -> .corrupt/: {reason}",
            stacklevel=2,
        )
        quarantined.append(shard.name)
    return quarantined


def quarantine_incomplete_shards(
    reward_dir: str | Path, hidden_dir: str | Path
) -> list[str]:
    """Move unreadable or unpaired reward shards to ``.incomplete/``.

    This handles shard-level incompleteness such as a missing hidden sidecar.
    Per-demo gaps are intentionally left in place and ignored by
    ``complete_episode_ids_per_task`` so complete demos in the same shard remain
    reusable.
    """
    import h5py

    reward = Path(reward_dir).expanduser()
    hidden = Path(hidden_dir).expanduser()
    moved: list[str] = []
    if not reward.is_dir():
        return moved
    for shard in sorted(reward.glob("*.hdf5")):
        hidden_shard = hidden / shard.name
        reason: str | None = None
        if not hidden_shard.is_file():
            reason = "missing hidden sidecar"
        else:
            try:
                with h5py.File(str(shard), "r") as rf, h5py.File(str(hidden_shard), "r") as hf:
                    if rf.get("data") is None or hf.get("data") is None:
                        reason = "missing data group"
            except OSError as exc:
                reason = str(exc)
        if reason is None:
            continue
        for directory in (reward, hidden):
            src = directory / shard.name
            if src.exists():
                dest = directory / ".incomplete"
                dest.mkdir(exist_ok=True)
                src.rename(dest / src.name)
        warnings.warn(
            f"quarantined incomplete shard {shard.name} -> .incomplete/: {reason}",
            stacklevel=2,
        )
        moved.append(shard.name)
    return moved


def summarize_collection(
    reward_dir: str | Path,
    hidden_dir: str | Path | None = None,
    *,
    target_total: int | None,
    num_tasks: int,
) -> dict[str, Any]:
    """Inspect existing collected data and report progress toward the target."""
    per_task = count_episodes_per_task(reward_dir, hidden_dir)
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
    """One past the highest shard index in ``directory`` for ``prefix``.

    Supports numeric ``{prefix}_{NNN}.hdf5`` names and metadata-rich names such
    as ``{prefix}_gs000040_success_{NNN}.hdf5``.
    """
    path = Path(directory).expanduser()
    if not path.is_dir():
        return 0
    highest = -1
    for shard in path.glob(f"{prefix}_*.hdf5"):
        suffix = shard.name[len(prefix) + 1 : -len(".hdf5")]
        match = re.search(r"(?:^|_)(\d+)$", suffix)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def record_online_rollout_episode(
    root: str | Path,
    *,
    reward_path: str | Path,
    hidden_path: str | Path,
    task_id: int,
    episode_id: int,
    init_state_index: int | None,
    success: bool,
    complete: bool,
    global_step: int,
    env_step: int,
    keep_last_global_steps: int,
) -> dict[str, Any]:
    """Record one cotrain online rollout in the manifest and prune old steps.

    ``root`` is the existing online-cotrain dump root (the parent of ``reward`` and
    ``hidden`` by default). The original collect-compatible reward/hidden pair
    remains authoritative for dataset readers; ``episodes/.../ep_*.h5`` is a hard
    link to the reward file so the global-step layout is inspectable without a
    second full copy.
    """

    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    reward = Path(reward_path).expanduser()
    hidden = Path(hidden_path).expanduser()
    episode_path = _online_rollout_episode_path(
        root_path,
        task_id=int(task_id),
        global_step=int(global_step),
        success=bool(success),
        episode_id=int(episode_id),
    )
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    if episode_path.exists():
        episode_path.unlink()
    try:
        os.link(reward, episode_path)
    except OSError:
        shutil.copy2(reward, episode_path)

    entry: dict[str, Any] = {
        "global_step": int(global_step),
        "env_step": int(env_step),
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "init_state_index": int(init_state_index)
        if init_state_index is not None
        else int(episode_id),
        "success": bool(success),
        "complete": bool(complete),
        "episode_path": _manifest_path_value(root_path, episode_path),
        "reward_path": _manifest_path_value(root_path, reward),
        "hidden_path": _manifest_path_value(root_path, hidden),
    }
    entries = read_online_rollout_manifest(root_path)
    entries.append(entry)
    _write_online_rollout_manifest(root_path, entries)
    prune_online_rollout_history(
        root_path,
        keep_last_global_steps=int(keep_last_global_steps),
    )
    return entry


def read_online_rollout_manifest(root: str | Path) -> list[dict[str, Any]]:
    """Read the cotrain online rollout JSONL manifest."""
    path = Path(root).expanduser() / ONLINE_ROLLOUT_MANIFEST_NAME
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if isinstance(item, dict):
            entries.append(item)
    return entries


def prune_online_rollout_history(
    root: str | Path,
    *,
    keep_last_global_steps: int,
) -> list[dict[str, Any]]:
    """Keep only entries/files belonging to the latest K unique global steps."""
    root_path = Path(root).expanduser()
    entries = read_online_rollout_manifest(root_path)
    if int(keep_last_global_steps) <= 0 or not entries:
        return entries
    unique_steps = sorted({int(item["global_step"]) for item in entries})
    keep_steps = set(unique_steps[-int(keep_last_global_steps) :])
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in entries:
        if int(item.get("global_step", -1)) in keep_steps:
            kept.append(item)
        else:
            dropped.append(item)
    for item in dropped:
        for key in ("episode_path", "reward_path", "hidden_path"):
            _unlink_manifest_path(root_path, item.get(key))
    _write_online_rollout_manifest(root_path, kept)
    return kept


def online_rollout_episode_counts(
    root: str | Path,
    *,
    task_ids: tuple[int, ...],
) -> dict[int, int]:
    """Next episode id per task derived from the retained online manifest."""
    counts = {int(task_id): 0 for task_id in task_ids}
    for item in read_online_rollout_manifest(root):
        if not bool(item.get("complete", True)):
            continue
        task_id = int(item.get("task_id", -1))
        if task_id not in counts:
            continue
        counts[task_id] = max(counts[task_id], int(item.get("episode_id", -1)) + 1)
    return counts


def load_online_rollout_episodes(
    root: str | Path,
) -> list[list[dict[str, Any]]]:
    """Load retained manifest entries into ``OnlineReplay`` step dictionaries."""
    import h5py
    import numpy as np

    root_path = Path(root).expanduser()
    episodes: list[list[dict[str, Any]]] = []
    for item in read_online_rollout_manifest(root_path):
        if not bool(item.get("complete", True)):
            continue
        reward_path = _resolve_manifest_path(root_path, item.get("reward_path"))
        hidden_path = _resolve_manifest_path(root_path, item.get("hidden_path"))
        if reward_path is None or hidden_path is None:
            continue
        if not reward_path.is_file() or not hidden_path.is_file():
            continue
        try:
            with h5py.File(str(reward_path), "r") as rf, h5py.File(str(hidden_path), "r") as hf:
                rdata = rf.get("data")
                hdata = hf.get("data")
                if rdata is None or hdata is None:
                    continue
                demo_key = sorted(rdata.keys())[0]
                rdemo = rdata[demo_key]
                hdemo = hdata.get(demo_key)
                if hdemo is None or "obs_embedding" not in hdemo:
                    continue
                length = int(rdemo.attrs.get("num_samples", rdemo["actions"].shape[0]))
                length = min(length, int(hdemo["obs_embedding"].shape[0]))
                episode: list[dict[str, Any]] = []
                for index in range(length):
                    sparse_reward = (
                        float(rdemo["sparse_rewards"][index])
                        if "sparse_rewards" in rdemo
                        else 0.0
                    )
                    dense_reward = (
                        float(rdemo["rewards"][index])
                        if "rewards" in rdemo
                        else sparse_reward
                    )
                    done = bool(rdemo["dones"][index]) if "dones" in rdemo else index == length - 1
                    action = np.asarray(rdemo["actions"][index], dtype=np.float32)
                    step: dict[str, Any] = {
                        "obs_embedding": np.asarray(
                            hdemo["obs_embedding"][index], dtype=np.float32
                        ),
                        "actions": action,
                        "wm_action": action,
                        "reward": np.float32(dense_reward),
                        "rewards": np.float32(dense_reward),
                        "sparse_rewards": np.uint8(1 if sparse_reward > 0.5 else 0),
                        "done": np.float32(done),
                        "dones": np.uint8(1 if done else 0),
                        "is_terminal": np.float32(sparse_reward > 0.5),
                        "is_last": np.float32(done),
                        "task_id": int(item["task_id"]),
                        "episode_id": int(item["episode_id"]),
                        "init_state_index": int(item["init_state_index"]),
                        "success": bool(item["success"]) and index == length - 1,
                    }
                    obs = rdemo.get("obs")
                    if obs is not None and all(
                        key in obs for key in ("ee_pos", "ee_ori", "gripper_states")
                    ):
                        step["proprio"] = np.concatenate(
                            [
                                np.asarray(obs["ee_pos"][index], dtype=np.float32).reshape(-1),
                                np.asarray(obs["ee_ori"][index], dtype=np.float32).reshape(-1),
                                np.asarray(
                                    obs["gripper_states"][index],
                                    dtype=np.float32,
                                ).reshape(-1),
                            ],
                            axis=0,
                        ).astype(np.float32, copy=False)
                    elif "robot_states" in rdemo:
                        robot_state = np.asarray(
                            rdemo["robot_states"][index], dtype=np.float32
                        ).reshape(-1)
                        if robot_state.shape[0] == 8:
                            step["proprio"] = robot_state
                    if obs is not None and "agentview_rgb" in obs:
                        step["image"] = np.asarray(obs["agentview_rgb"][index], dtype=np.uint8)
                    if "lang_emb" in hdemo:
                        step["lang_emb"] = np.asarray(hdemo["lang_emb"][...], dtype=np.float32)
                    episode.append(step)
                if episode:
                    episodes.append(episode)
        except (OSError, KeyError, ValueError) as exc:
            warnings.warn(
                f"skipping unreadable online rollout episode {reward_path}: {exc}",
                stacklevel=2,
            )
    return episodes


def _write_online_rollout_manifest(
    root: str | Path,
    entries: list[dict[str, Any]],
) -> Path:
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / ONLINE_ROLLOUT_MANIFEST_NAME
    text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in entries)
    path.write_text(text, encoding="utf-8")
    return path


def _online_rollout_episode_path(
    root: Path,
    *,
    task_id: int,
    global_step: int,
    success: bool,
    episode_id: int,
) -> Path:
    return (
        root
        / ONLINE_ROLLOUT_EPISODES_DIR
        / f"task_{int(task_id):02d}"
        / f"global_step{int(global_step):06d}_success_{bool(success)}"
        / f"ep_{int(episode_id):06d}.h5"
    )


def _manifest_path_value(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.expanduser())


def _resolve_manifest_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _unlink_manifest_path(root: Path, value: Any) -> None:
    path = _resolve_manifest_path(root, value)
    if path is None:
        return
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        return
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        return
    _remove_empty_parents(path.parent, root)


def _remove_empty_parents(start: Path, stop: Path) -> None:
    current = start
    stop_resolved = stop.resolve()
    while True:
        try:
            if current.resolve() == stop_resolved:
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent


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


def append_episode_index_record(reward_dir: str | Path, record: dict[str, Any]) -> Path:
    """Append one per-episode JSON line to ``reward_dir/episode_index.jsonl``.

    Audit log tying each per-trajectory shard file to its identity metadata
    (task_id, episode_id, success, ...). Append-only: a re-collected episode
    adds a new line; readers should treat the LAST line per ``file`` as current.

    Parallel-safe: multiple collector ranks append to the SAME file, so each
    append takes an exclusive ``flock`` and lands as a single write — plain
    ``O_APPEND`` line atomicity is filesystem-dependent and not relied upon.
    """
    import fcntl

    directory = Path(reward_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EPISODE_INDEX_NAME
    line = json.dumps(record, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return path
