"""Load previously-collected cold-start trajectory HDF5 (RolloutDumpWriter
schema) into an OnlineReplay buffer, so WM/classifier warmup and the online
cotrain loop share one buffer + one set of step functions (no semantic drift).

Each demo (data/demo_<i>) in every reward shard is paired with its
obs_embedding sidecar and converted to the per-step transition dicts that
``OnlineReplay.add_episode`` consumes.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dreamervla.runners.online_replay import OnlineReplay


def _demo_to_transitions(
    demo: h5py.Group, emb: np.ndarray, task_id: int
) -> list[dict[str, Any]]:
    actions = np.asarray(demo["actions"][...], dtype=np.float32)        # (T, 7)
    sparse = np.asarray(demo["sparse_rewards"][...], dtype=np.float32)  # (T,)
    dones = np.asarray(demo["dones"][...], dtype=np.float32)            # (T,)
    images = np.asarray(demo["obs"]["agentview_rgb"][...], dtype=np.uint8)
    success_hits = np.flatnonzero(sparse > 0.5)
    attr_success = bool(demo.attrs.get("episode_success", bool(success_hits.size)))
    success = bool(attr_success or success_hits.size)
    if success_hits.size:
        success_step = int(success_hits[0])
    elif success:
        done_hits = np.flatnonzero(dones > 0.5)
        success_step = int(done_hits[0]) if done_hits.size else int(actions.shape[0]) - 1
    else:
        success_step = -1
    T = int(actions.shape[0])
    transitions: list[dict[str, Any]] = []
    for t in range(T):
        step_success = bool(t == success_step)
        reward = float(sparse[t])
        if step_success and reward <= 0.0:
            reward = 1.0
        transitions.append({
            "image": images[t],
            "obs_embedding": np.asarray(emb[t]),
            "reward": reward,                      # sparse reward = collector signal
            "done": float(dones[t]),
            "is_last": float(dones[t]),
            "is_terminal": float(step_success),    # terminal-success marker
            "wm_action": actions[t],               # collector stores env-scale wm_action
            "task_id": int(task_id),
            "success": step_success,
        })
    return transitions


def seed_replay_from_offline(
    replay: OnlineReplay,
    *,
    data_dir: str | Path,
    hidden_dir: str | Path,
    default_task_id: int | None = None,
    max_episodes_per_task: int | None = None,
) -> int:
    """Add demos from data_dir's reward shards to ``replay``. Returns the number of
    episodes actually added (demos shorter than sequence_length are skipped by
    add_episode).

    ``max_episodes_per_task`` caps how many episodes are added per task_id. The full-warmup
    seeding passes None (add everything); the online-replay seed passes a small cap so the
    bounded online buffer gets just enough per-task coverage to be training-ready (every
    task present) without evicting the room reserved for fresh online experience.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    hidden_dir = Path(hidden_dir).expanduser().resolve()
    shards = sorted(p.name for p in data_dir.glob("*.hdf5"))
    if not shards:
        raise FileNotFoundError(f"no reward HDF5 shards under {data_dir}")
    cap = None if max_episodes_per_task is None else int(max_episodes_per_task)
    per_task: dict[int, int] = {}
    n_added = 0
    for shard in shards:
        # Skip truncated/corrupt shards (e.g. a half-written file left by a crashed
        # collect) so one bad shard does not abort warmup — mirrors the tolerant
        # inspection in collection_manifest. A missing task_id is a real config error,
        # so the ValueError below is NOT swallowed (it is not OSError/KeyError).
        try:
            with h5py.File(data_dir / shard, "r") as rf, h5py.File(hidden_dir / shard, "r") as hf:
                for demo_key in rf["data"]:
                    demo = rf["data"][demo_key]
                    if "task_id" in demo.attrs:
                        task_id = int(demo.attrs["task_id"])
                    elif default_task_id is not None:
                        task_id = int(default_task_id)
                    else:
                        raise ValueError(
                            f"{shard}/{demo_key} has no task_id attr and no default_task_id "
                            "was provided; re-collect with the identity-aware collector or "
                            "set offline_warmup.task_id for single-task data."
                        )
                    if cap is not None and per_task.get(task_id, 0) >= cap:
                        continue
                    emb = np.asarray(hf["data"][demo_key]["obs_embedding"][...])
                    if (
                        replay.add_episode(
                            _demo_to_transitions(demo, emb, task_id),
                            source="coldstart",
                        )
                        is not None
                    ):
                        n_added += 1
                        per_task[task_id] = per_task.get(task_id, 0) + 1
        except (OSError, KeyError) as exc:
            warnings.warn(f"skipping unreadable shard {shard}: {exc}", stacklevel=2)
            continue
    return n_added
