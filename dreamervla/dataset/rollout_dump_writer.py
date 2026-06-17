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
    data/demo_<i>/obs_embedding  (T, D) float16

preprocess_config.json is written once to hidden_dir/preprocess_config.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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
        task_description: str | None = None,
        episode_success: bool | None = None,
        episode_horizon: int | None = None,
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
            obs_embedding   array-like (D,)       float16

        ``preprocess_config`` is written to hidden_dir/preprocess_config.json
        on the first call that provides a non-None value.

        ``data_attrs`` (env meta: bddl_file_name, env_name, tag, ...) is written
        to the reward HDF5 data-group attrs on the first call that provides it.
        """
        if self._closed:
            raise RuntimeError("RolloutDumpWriter has been closed")
        if not steps:
            return

        T = len(steps)
        demo_key = f"demo_{index}"

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
        )  # (T, D)

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
        if task_description is not None:
            demo_grp.attrs["task_description"] = str(task_description)
        if episode_success is not None:
            demo_grp.attrs["episode_success"] = bool(episode_success)
        if episode_horizon is not None:
            demo_grp.attrs["episode_horizon"] = int(episode_horizon)

        # Write sidecar HDF5
        hidden_demo_grp = self._hidden_data.create_group(demo_key)
        hidden_demo_grp.create_dataset("obs_embedding", data=obs_embedding)

        self._num_demos += 1

        # Write data-group env-meta attrs on first call (if provided)
        if data_attrs is not None and not self._data_attrs_written:
            for attr_key, attr_val in data_attrs.items():
                self._reward_data.attrs[attr_key] = attr_val
            self._data_attrs_written = True

        # Write preprocess_config.json on first call (if provided)
        if preprocess_config is not None and not self._preprocess_config_written:
            config_path = self.hidden_dir / "preprocess_config.json"
            config_path.write_text(json.dumps(preprocess_config, indent=2), encoding="utf-8")
            self._preprocess_config_written = True

    def close(self) -> None:
        """Flush and close both HDF5 files."""
        if self._closed:
            return
        self._closed = True
        self._reward_data.attrs["num_demos"] = str(self._num_demos)
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


__all__ = ["RolloutDumpWriter"]
