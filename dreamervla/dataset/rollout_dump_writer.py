"""Rollout dump writer: serializes one HDF5 demo (reward-dir schema) + obs_embedding
sidecar + preprocess_config.json into a format consumable by BalancedTerminalDataset.

Schema (reward HDF5, per demo at data/demo_<i>/):
    actions           (T, 7)         float64
    dones             (T,)           uint8
    rewards           (T,)           float32   — collector writes zeros
    sparse_rewards    (T,)           uint8     — 1 at terminal success frame, else 0
    robot_states      (T, 9)         float64
    states            (T, S)         float64   — S from data (not hardcoded)
    obs/agentview_rgb      (T,256,256,3) uint8
    obs/eye_in_hand_rgb    (T,256,256,3) uint8
    obs/ee_pos             (T, 3)    float64
    obs/ee_ori             (T, 3)    float64
    obs/ee_states          (T, 6)    float64
    obs/gripper_states     (T, 2)    float64
    obs/joint_states       (T, 7)    float64
    demo.attrs: init_state (ndarray), num_samples (str(T))
    data group attrs: env meta

    Sidecar (same filename, separate dir):
        data/demo_<i>/obs_embedding  (T, N, D) hidden_token float16
        data/demo_<i>/lang_emb       (D_lang,) optional demo-level float16

preprocess_config.json is written once to hidden_dir/preprocess_config.json.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dreamervla.preprocess.sidecar_schema import (
    validate_hidden_token_array_shape,
    validate_hidden_token_preprocess_config,
)

_CANONICAL_EPISODE_METADATA_KEYS = frozenset(("global_step", "env_step"))


class RolloutDumpWriter:
    """Writes one HDF5 reward file + sidecar HDF5 + preprocess_config.json.

    Parameters
    ----------
    reward_dir : directory for reward-schema HDF5 files
    hidden_dir : directory for sidecar HDF5 files (same filenames)
    shard_name : filename for both HDF5 files (e.g. "shard_000.hdf5")
    """

    def __init__(
        self,
        reward_dir: str | Path,
        hidden_dir: str | Path,
        shard_name: str,
    ) -> None:
        self.reward_dir = Path(reward_dir).expanduser().resolve()
        self.hidden_dir = Path(hidden_dir).expanduser().resolve()
        self.reward_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        self.shard_name = str(shard_name)

        self._reward_path = self.reward_dir / self.shard_name
        self._hidden_path = self.hidden_dir / self.shard_name

        self._reward_f: h5py.File = h5py.File(str(self._reward_path), "w")
        self._hidden_f: h5py.File = h5py.File(str(self._hidden_path), "w")

        self._reward_data: h5py.Group = self._reward_f.create_group("data")
        self._hidden_data: h5py.Group = self._hidden_f.create_group("data")

        self._num_demos: int = 0
        self._preprocess_config_written: bool = False
        self._data_attrs_written: bool = False
        self._closed: bool = False

    def write_demo(
        self,
        index: int,
        steps: list[dict[str, Any]],
        preprocess_config: dict[str, Any] | None = None,
        data_attrs: dict[str, Any] | None = None,
        task_id: int | None = None,
        episode_id: int | None = None,
        init_state_index: int | None = None,
        task_description: str | None = None,
        episode_success: bool | None = None,
        episode_horizon: int | None = None,
        episode_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one demo (list of per-step dicts) to both HDF5 files.

        Each step dict must contain:
            actions         array-like (7,)       float64
            rewards         scalar                float32
            sparse_rewards  scalar                uint8
            dones           scalar                uint8
            robot_states    array-like (9,)       float64
            states          array-like (S,)       float64
            obs: dict with:
                agentview_rgb    (256,256,3)       uint8
                eye_in_hand_rgb  (256,256,3)       uint8
                ee_pos           (3,)              float64
                ee_ori           (3,)              float64
                ee_states        (6,)              float64
                gripper_states   (2,)              float64
                joint_states     (7,)              float64
            obs_embedding   array-like (256,4096) hidden_token float16
            lang_emb         optional array-like (D_lang,) demo-level language embedding

        ``preprocess_config`` is written to hidden_dir/preprocess_config.json
        on the first call that provides a non-None value.

        ``data_attrs`` (env meta: bddl_file_name, env_name, tag, ...) is written
        to the reward HDF5 data-group attrs on the first call that provides it.
        """
        if self._closed:
            raise RuntimeError("RolloutDumpWriter has been closed")
        if not steps:
            return
        if not self._preprocess_config_written and preprocess_config is None:
            raise ValueError(
                "the first rollout demo must provide the canonical hidden-token "
                "preprocess_config"
            )
        if preprocess_config is not None:
            validate_hidden_token_preprocess_config(
                preprocess_config,
                context="RolloutDumpWriter preprocess_config",
            )

        T = len(steps)
        demo_key = f"demo_{index}"
        resolved_init_state_index = _resolve_init_state_index(
            init_state_index, steps
        )

        # Stack per-step arrays
        actions = np.stack(
            [np.asarray(s["actions"], dtype=np.float64) for s in steps], axis=0
        )  # (T, 7)
        rewards = np.array(
            [float(s["rewards"]) for s in steps], dtype=np.float32
        )  # (T,)
        sparse_rewards = np.array(
            [int(s["sparse_rewards"]) for s in steps], dtype=np.uint8
        )  # (T,)
        dones = np.array(
            [int(s["dones"]) for s in steps], dtype=np.uint8
        )  # (T,)
        robot_states = np.stack(
            [np.asarray(s["robot_states"], dtype=np.float64) for s in steps], axis=0
        )  # (T, 9)
        states = np.stack(
            [np.asarray(s["states"], dtype=np.float64) for s in steps], axis=0
        )  # (T, S)
        obs_embedding = np.stack(
            [np.asarray(s["obs_embedding"], dtype=np.float16) for s in steps], axis=0
        )  # (T, token_count, token_dim)
        if obs_embedding.ndim != 3:
            raise ValueError(
                "rollout obs_embedding must be tokenized [T,N,D], "
                f"got {obs_embedding.shape}"
            )
        validate_hidden_token_array_shape(
            obs_embedding.shape,
            context="rollout obs_embedding",
            token_count=(
                int(preprocess_config["token_count"])
                if preprocess_config is not None
                else None
            ),
            token_dim=(
                int(preprocess_config["token_dim"])
                if preprocess_config is not None
                else None
            ),
        )
        lang_emb = None
        if steps[0].get("lang_emb") is not None:
            lang_emb = np.asarray(steps[0]["lang_emb"], dtype=np.float16).reshape(-1)

        # obs sub-fields
        obs_keys_dtypes = {
            "agentview_rgb": np.uint8,
            "eye_in_hand_rgb": np.uint8,
            "ee_pos": np.float64,
            "ee_ori": np.float64,
            "ee_states": np.float64,
            "gripper_states": np.float64,
            "joint_states": np.float64,
        }
        obs_arrays: dict[str, np.ndarray] = {}
        for key, dtype in obs_keys_dtypes.items():
            obs_arrays[key] = np.stack(
                [np.asarray(s["obs"][key], dtype=dtype) for s in steps], axis=0
            )

        # Write reward HDF5
        demo_grp = self._reward_data.create_group(demo_key)
        demo_grp.create_dataset("actions", data=actions)
        demo_grp.create_dataset("dones", data=dones)
        demo_grp.create_dataset("rewards", data=rewards)
        demo_grp.create_dataset("sparse_rewards", data=sparse_rewards)
        demo_grp.create_dataset("robot_states", data=robot_states)
        demo_grp.create_dataset("states", data=states)

        obs_grp = demo_grp.create_group("obs")
        for key, arr in obs_arrays.items():
            obs_grp.create_dataset(key, data=arr)

        # Demo attrs: init_state from step 0's states, num_samples
        init_state = np.asarray(steps[0]["states"], dtype=np.float64)
        demo_grp.attrs["init_state"] = init_state
        demo_grp.attrs["num_samples"] = str(T)

        # Per-demo identity metadata (aligns collector output with canonical
        # LIBERO data, which encodes task identity via one-file-per-task; the
        # rank-sharded collector interleaves tasks so identity must be per-demo).
        if task_id is not None:
            demo_grp.attrs["task_id"] = int(task_id)
        if episode_id is not None:
            demo_grp.attrs["episode_id"] = int(episode_id)
        if resolved_init_state_index is not None:
            demo_grp.attrs["init_state_index"] = int(resolved_init_state_index)
        if task_description is not None:
            demo_grp.attrs["task_description"] = str(task_description)
        demo_grp.attrs["complete"] = True
        for key, value in _episode_attrs(
            preprocess_config=preprocess_config,
            data_attrs=data_attrs,
            task_description=task_description,
            episode_success=episode_success,
            episode_horizon=episode_horizon,
            episode_metadata=episode_metadata,
        ).items():
            demo_grp.attrs[key] = value

        # Write sidecar HDF5
        hidden_demo_grp = self._hidden_data.create_group(demo_key)
        hidden_demo_grp.create_dataset("obs_embedding", data=obs_embedding)
        if lang_emb is not None:
            hidden_demo_grp.create_dataset("lang_emb", data=lang_emb)
        hidden_demo_grp.attrs["num_samples"] = str(T)
        if task_id is not None:
            hidden_demo_grp.attrs["task_id"] = int(task_id)
        if episode_id is not None:
            hidden_demo_grp.attrs["episode_id"] = int(episode_id)
        if resolved_init_state_index is not None:
            hidden_demo_grp.attrs["init_state_index"] = int(resolved_init_state_index)
        if task_description is not None:
            hidden_demo_grp.attrs["task_description"] = str(task_description)
        hidden_demo_grp.attrs["complete"] = True
        for key, value in _episode_attrs(
            preprocess_config=preprocess_config,
            data_attrs=data_attrs,
            task_description=task_description,
            episode_success=episode_success,
            episode_horizon=episode_horizon,
            episode_metadata=episode_metadata,
        ).items():
            hidden_demo_grp.attrs[key] = value

        self._num_demos += 1

        # Write data-group env-meta attrs on first call (if provided)
        if data_attrs is not None and not self._data_attrs_written:
            for attr_key, attr_val in data_attrs.items():
                self._reward_data.attrs[attr_key] = attr_val
                self._hidden_data.attrs[attr_key] = attr_val
            self._data_attrs_written = True

        # Write preprocess_config.json on first call (if provided). Atomic
        # tmp+rename: per-trajectory collects rewrite this once per episode,
        # so a reader must never observe a torn write.
        if preprocess_config is not None and not self._preprocess_config_written:
            config_path = self.hidden_dir / "preprocess_config.json"
            tmp_path = config_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(preprocess_config, indent=2), encoding="utf-8")
            tmp_path.replace(config_path)
            self._preprocess_config_written = True

    def close(self) -> None:
        """Flush and close both HDF5 files."""
        if self._closed:
            return
        self._closed = True
        self._reward_data.attrs["num_demos"] = str(self._num_demos)
        self._hidden_data.attrs["num_demos"] = str(self._num_demos)
        self._reward_f.close()
        self._hidden_f.close()

    def __enter__(self) -> RolloutDumpWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RotatingRolloutDumpWriter:
    """Drop-in ``RolloutDumpWriter`` that rolls a new shard every N demos.

    Same ``write_demo`` / ``close`` / context-manager interface as
    ``RolloutDumpWriter``, so callers swap one for the other without further
    changes.  Shards are named ``{shard_prefix}_{NNN}.hdf5`` starting at
    ``start_index``; the demo group restarts at ``demo_0`` in each shard (the
    reader globs ``*.hdf5`` and iterates per-shard keys, so per-shard numbering
    is what it expects).  ``preprocess_config`` / ``data_attrs`` are captured from
    the first write that provides them and re-emitted on every shard's first demo
    so each shard is independently readable — mirroring ``RolloutDumpWorker``'s
    Ray-side rotation.  The caller's ``index`` argument is ignored (the wrapper
    owns shard-local numbering).
    """

    def __init__(
        self,
        reward_dir: str | Path,
        hidden_dir: str | Path,
        *,
        shard_prefix: str,
        demos_per_shard: int,
        start_index: int = 0,
    ) -> None:
        if int(demos_per_shard) <= 0:
            raise ValueError("RotatingRolloutDumpWriter requires demos_per_shard > 0")
        self.reward_dir = Path(reward_dir)
        self.hidden_dir = Path(hidden_dir)
        self.shard_prefix = str(shard_prefix)
        self.demos_per_shard = int(demos_per_shard)
        self._shard_idx = int(start_index)
        self._shard_demos = 0
        self._saved_config: dict[str, Any] | None = None
        self._saved_attrs: dict[str, Any] | None = None
        self._closed = False
        self._writer = RolloutDumpWriter(
            self.reward_dir, self.hidden_dir, self._shard_name(self._shard_idx)
        )

    def _shard_name(self, idx: int) -> str:
        return f"{self.shard_prefix}_{idx:03d}.hdf5"

    def write_demo(
        self,
        index: int,
        steps: list[dict[str, Any]],
        preprocess_config: dict[str, Any] | None = None,
        data_attrs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not steps:
            return
        if preprocess_config is not None and self._saved_config is None:
            self._saved_config = preprocess_config
        if data_attrs is not None and self._saved_attrs is None:
            self._saved_attrs = data_attrs
        if self._shard_demos >= self.demos_per_shard:
            self._writer.close()
            self._shard_idx += 1
            self._shard_demos = 0
            self._writer = RolloutDumpWriter(
                self.reward_dir, self.hidden_dir, self._shard_name(self._shard_idx)
            )
        first_in_shard = self._shard_demos == 0
        self._writer.write_demo(
            index=self._shard_demos,
            steps=steps,
            preprocess_config=self._saved_config if first_in_shard else None,
            data_attrs=self._saved_attrs if first_in_shard else None,
            **kwargs,
        )
        self._shard_demos += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close()

    def __enter__(self) -> RotatingRolloutDumpWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def per_trajectory_shard_name(prefix: str, task_id: int, episode_id: int) -> str:
    """Identity-based shard filename shared by all per-trajectory writers."""
    return f"{prefix}_t{int(task_id):02d}_ep{int(episode_id):06d}.hdf5"


class PerTrajectoryDumpWriter:
    """Drop-in ``RolloutDumpWriter`` that lands one file pair per trajectory.

    Same ``write_demo`` / ``close`` / context-manager surface as
    ``RotatingRolloutDumpWriter``.  Each demo is written to its own
    ``{file_prefix}_t{task:02d}_ep{episode:06d}.hdf5`` pair (reward + hidden)
    and closed immediately, so every finished trajectory is durable on disk.
    Identity comes from the globally-built work list (``task_id``/``episode_id``
    are REQUIRED), which makes the numbering unified across ranks without any
    cross-rank coordination; re-collecting an identity overwrites its files.
    ``preprocess_config`` / ``data_attrs`` are captured from the first write
    that provides them and re-emitted into every file so each file is
    independently readable.  Every write appends a line to
    ``reward_dir/episode_index.jsonl`` mapping the file to its metadata.
    The caller's ``index`` argument is ignored (each file holds ``demo_0``).
    """

    def __init__(
        self,
        reward_dir: str | Path,
        hidden_dir: str | Path,
        *,
        file_prefix: str = "traj",
    ) -> None:
        self.reward_dir = Path(reward_dir)
        self.hidden_dir = Path(hidden_dir)
        self.file_prefix = str(file_prefix)
        self._saved_config: dict[str, Any] | None = None
        self._saved_attrs: dict[str, Any] | None = None
        self._closed = False

    def write_demo(
        self,
        index: int,
        steps: list[dict[str, Any]],
        preprocess_config: dict[str, Any] | None = None,
        data_attrs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not steps:
            return
        task_id = kwargs.get("task_id")
        episode_id = kwargs.get("episode_id")
        if task_id is None or episode_id is None:
            raise ValueError(
                "PerTrajectoryDumpWriter requires task_id and episode_id kwargs"
            )
        if preprocess_config is not None and self._saved_config is None:
            self._saved_config = preprocess_config
        if data_attrs is not None and self._saved_attrs is None:
            self._saved_attrs = data_attrs
        shard_name = per_trajectory_shard_name(
            self.file_prefix, int(task_id), int(episode_id)
        )
        # Write under a .tmp name and atomically rename on success so a crash
        # mid-write never leaves a partial file at the canonical identity name
        # (readers/resume glob *.hdf5 and cannot see the .hdf5.tmp files).
        tmp_name = shard_name + ".tmp"
        try:
            with RolloutDumpWriter(
                self.reward_dir, self.hidden_dir, tmp_name
            ) as writer:
                writer.write_demo(
                    index=0,
                    steps=steps,
                    preprocess_config=self._saved_config,
                    data_attrs=self._saved_attrs,
                    **kwargs,
                )
        except BaseException:
            for directory in (self.reward_dir, self.hidden_dir):
                tmp_path = Path(directory) / tmp_name
                if tmp_path.exists():
                    tmp_path.unlink()
            raise
        for directory in (self.reward_dir, self.hidden_dir):
            (Path(directory) / tmp_name).replace(Path(directory) / shard_name)
        from dreamervla.dataset.collection_manifest import append_episode_index_record

        record: dict[str, Any] = {
            "file": shard_name,
            "task_id": int(task_id),
            "episode_id": int(episode_id),
            "horizon": len(steps),
        }
        init_state_index = _resolve_init_state_index(
            kwargs.get("init_state_index"), steps
        )
        if init_state_index is not None:
            record["init_state_index"] = int(init_state_index)
        episode_success = kwargs.get("episode_success")
        if episode_success is not None:
            record["success"] = bool(episode_success)
        task_description = kwargs.get("task_description")
        if task_description is not None:
            record["task_description"] = str(task_description)
        append_episode_index_record(self.reward_dir, record)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> PerTrajectoryDumpWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = [
    "PerTrajectoryDumpWriter",
    "RolloutDumpWriter",
    "RotatingRolloutDumpWriter",
    "per_trajectory_shard_name",
]


def _episode_attrs(
    *,
    preprocess_config: Mapping[str, Any] | None,
    data_attrs: Mapping[str, Any] | None,
    task_description: str | None,
    episode_success: bool | None,
    episode_horizon: int | None,
    episode_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if data_attrs is not None:
        suite = data_attrs.get("suite", data_attrs.get("task_suite_name"))
        if suite is not None:
            attrs["suite"] = suite
    if task_description is not None:
        attrs["task_name"] = str(task_description)
    if episode_success is not None:
        attrs["success"] = bool(episode_success)
    if preprocess_config is not None:
        for key in ("chunk_size", "hidden_key", "hidden_dim", "token_count", "token_dim"):
            if key in preprocess_config:
                attrs[key] = preprocess_config[key]
        if "hidden_dim" not in attrs:
            token_count = preprocess_config.get("token_count")
            token_dim = preprocess_config.get("token_dim")
            if token_count is not None and token_dim is not None:
                attrs["hidden_dim"] = int(token_count) * int(token_dim)
    if episode_metadata is not None:
        for key, value in dict(episode_metadata).items():
            if str(key) in _CANONICAL_EPISODE_METADATA_KEYS:
                attrs[str(key)] = value
    return {
        str(key): value
        for key, value in attrs.items()
        if _is_hdf5_attr_scalar(value)
    }


def _resolve_init_state_index(
    init_state_index: int | None,
    steps: list[dict[str, Any]],
) -> int | None:
    if init_state_index is not None:
        return int(init_state_index)
    if not steps:
        return None
    value = steps[0].get("init_state_index")
    if value is None:
        return None
    return int(value)


def _is_hdf5_attr_scalar(value: Any) -> bool:
    if value is None:
        return False
    return isinstance(
        value,
        (
            str,
            bytes,
            bool,
            int,
            float,
            np.bool_,
            np.integer,
            np.floating,
        ),
    )
