"""Load previously-collected cold-start trajectory HDF5 (RolloutDumpWriter
schema) into an OnlineReplay buffer, so WM/classifier warmup and the online
cotrain loop share one buffer + one set of step functions (no semantic drift).

Each demo (data/demo_<i>) in every reward shard is paired with its
obs_embedding sidecar and converted to the per-step transition dicts that
``OnlineReplay.add_episode`` consumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dreamervla.preprocess.sidecar_schema import validate_hidden_token_sidecar_dir
from dreamervla.runtime.online_replay import OnlineReplay

_LIBERO_GOAL_TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
)


def _resolve_task_id(
    *,
    shard: str,
    demo_key: str,
    demo: h5py.Group,
    default_task_id: int | None,
    infer_task_id_from_shard: bool,
) -> int:
    if "task_id" in demo.attrs:
        return int(demo.attrs["task_id"])
    if default_task_id is not None:
        return int(default_task_id)
    if infer_task_id_from_shard:
        task_name = shard.removesuffix("_demo.hdf5")
        try:
            return _LIBERO_GOAL_TASKS.index(task_name)
        except ValueError as exc:
            raise ValueError(
                f"cannot infer task_id from shard {shard}; "
                "set offline_warmup.task_id or add a task-name mapping"
            ) from exc
    raise ValueError(
        f"{shard}/{demo_key} has no task_id attr and no default_task_id "
        "was provided; re-collect with the identity-aware collector or "
        "set offline_warmup.task_id for single-task data."
    )


def _preflight_task_ids(
    data_dir: Path,
    shards: list[str],
    *,
    default_task_id: int | None,
    infer_task_id_from_shard: bool,
) -> dict[tuple[str, str], int]:
    """Resolve every demo identity before replay mutation begins."""

    resolved: dict[tuple[str, str], int] = {}
    for shard in shards:
        with h5py.File(data_dir / shard, "r") as handle:
            for demo_key, demo in handle["data"].items():
                resolved[(shard, str(demo_key))] = _resolve_task_id(
                    shard=shard,
                    demo_key=str(demo_key),
                    demo=demo,
                    default_task_id=default_task_id,
                    infer_task_id_from_shard=infer_task_id_from_shard,
                )
    return resolved


def _demo_proprio_at(demo: h5py.Group, t: int) -> np.ndarray:
    obs = demo["obs"]
    return np.concatenate(
        [
            np.asarray(obs["ee_pos"][t], dtype=np.float32).reshape(-1),
            np.asarray(obs["ee_ori"][t], dtype=np.float32).reshape(-1),
            np.asarray(obs["gripper_states"][t], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _demo_to_transitions(
    demo: h5py.Group,
    emb: np.ndarray,
    task_id: int,
    *,
    lang_emb: np.ndarray | None = None,
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
        step = {
            "image": images[t],
            "obs_embedding": np.asarray(emb[t]),
            "proprio": _demo_proprio_at(demo, t),
            "reward": reward,                      # sparse reward = collector signal
            "done": float(dones[t]),
            "is_last": float(dones[t]),
            "is_terminal": float(step_success),    # terminal-success marker
            "wm_action": actions[t],               # collector stores env-scale wm_action
            "task_id": int(task_id),
            "success": step_success,
        }
        if lang_emb is not None:
            step["lang_emb"] = np.asarray(lang_emb, dtype=np.float32).reshape(-1)
        transitions.append(step)
    return transitions


def seed_replay_from_offline(
    replay: OnlineReplay,
    *,
    data_dir: str | Path,
    hidden_dir: str | Path,
    default_task_id: int | None = None,
    infer_task_id_from_shard: bool = False,
    max_episodes_per_task: int | None = None,
    require_reference_complete: bool = True,
) -> int:
    """Add demos from data_dir's reward shards to ``replay``. Returns the number of
    episodes actually added (demos shorter than sequence_length are skipped by
    add_episode).

    ``max_episodes_per_task`` caps how many episodes are added per task_id. The full-warmup
    seeding passes None (add everything); the online-replay seed passes a small cap so the
    bounded online buffer gets just enough per-task coverage to be training-ready (every
    task present) without evicting the room reserved for fresh online experience.

    ``require_reference_complete`` distinguishes collector-written reward shards,
    which carry per-demo completion markers, from structurally validated official
    LIBERO shards, which predate those markers.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    hidden_dir = Path(hidden_dir).expanduser().resolve()
    shards = sorted(p.name for p in data_dir.glob("*.hdf5"))
    if not shards:
        raise FileNotFoundError(f"no reward HDF5 shards under {data_dir}")
    # Warmup is a public training boundary, so validate every paired shard and
    # demo before adding even one transition. This prevents a valid first shard
    # from masking a later 56-token/flat legacy sidecar.
    validate_hidden_token_sidecar_dir(
        hidden_dir,
        expected_filenames=shards,
        reference_dir=data_dir,
        require_reference_complete=bool(require_reference_complete),
        require_sparse_rewards=True,
    )
    task_ids = _preflight_task_ids(
        data_dir,
        shards,
        default_task_id=default_task_id,
        infer_task_id_from_shard=infer_task_id_from_shard,
    )
    cap = None if max_episodes_per_task is None else int(max_episodes_per_task)
    per_task: dict[int, int] = {}
    n_added = 0
    for shard in shards:
        with h5py.File(data_dir / shard, "r") as rf, h5py.File(
            hidden_dir / shard, "r"
        ) as hf:
            for demo_key in rf["data"]:
                demo = rf["data"][demo_key]
                task_id = task_ids[(shard, str(demo_key))]
                if cap is not None and per_task.get(task_id, 0) >= cap:
                    continue
                hidden_demo = hf["data"][demo_key]
                emb = np.asarray(hidden_demo["obs_embedding"][...])
                lang_emb = (
                    np.asarray(hidden_demo["lang_emb"][...], dtype=np.float32)
                    if "lang_emb" in hidden_demo
                    else None
                )
                if (
                    replay.add_episode(
                        _demo_to_transitions(
                            demo,
                            emb,
                            task_id,
                            lang_emb=lang_emb,
                        ),
                        source="coldstart",
                    )
                    is not None
                ):
                    n_added += 1
                    per_task[task_id] = per_task.get(task_id, 0) + 1
    return n_added
