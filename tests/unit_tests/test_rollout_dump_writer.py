"""Round-trip test: RolloutDumpWriter → BalancedTerminalDataset.

Writes two fake demos (one success, one failure) using RolloutDumpWriter,
then loads them through the REAL BalancedTerminalDataset and verifies that
the dataset returns a complete batch containing obs_embedding, images,
actions, rewards, and is_positive_window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── test parameters ────────────────────────────────────────────────────────────
T = 10          # episode length; must satisfy T >= sequence_length + 1 for negatives
SEQ_LEN = 4    # sequence_length; positive window at start=(T-SEQ_LEN), negative at 0..(T-SEQ_LEN-1)
IMAGE_H = 256
IMAGE_W = 256
ACTION_DIM = 7
HIDDEN_DIM = 229376   # 229376 = 56 * 4096
STATE_DIM = 79        # libero_goal S

PREPROCESS_CONFIG = {
    "action_dim": 7,
    "action_head_type": "oft_l1_regression",
    "center_crop": True,
    "chunk_size": 4,
    "hidden_key": "obs_embedding",
    "history": 2,
    "image_keys": ["agentview_rgb", "eye_in_hand_rgb"],
    "include_state": True,
    "model_path": "/fake/model/path",
    "num_images_in_input": 4,
    "obs_hidden_source": "action_query",
    "output_dtype": "float16",
    "prompt_style": "vla_policy",
    "resolution": 256,
    "rotate_images_180": True,
    "time_horizon": 8,
    "token_dim": 4096,
}


def _make_step(t: int, is_terminal: bool, episode_seed: int = 0) -> dict:
    """Build one fake timestep dict matching RolloutDumpWriter.write_demo input."""
    rng = np.random.default_rng(t + episode_seed)
    return {
        "actions": rng.standard_normal(ACTION_DIM),
        "rewards": np.float32(0.0),
        "sparse_rewards": np.uint8(1 if is_terminal else 0),
        "dones": np.uint8(1 if is_terminal else 0),
        "robot_states": rng.standard_normal(9),
        "states": rng.standard_normal(STATE_DIM),
        "obs": {
            "agentview_rgb": rng.integers(0, 255, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
            "eye_in_hand_rgb": rng.integers(0, 255, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
            "ee_pos": rng.standard_normal(3),
            "ee_ori": rng.standard_normal(3),
            "ee_states": rng.standard_normal(6),
            "gripper_states": rng.standard_normal(2),
            "joint_states": rng.standard_normal(7),
        },
        "obs_embedding": rng.standard_normal(HIDDEN_DIM).astype(np.float16),
    }


def _make_episode(success: bool) -> list[dict]:
    """Build a list of T steps; last step is terminal for success episodes."""
    steps = []
    episode_seed = 1000 if success else 2000
    for t in range(T):
        is_terminal = success and (t == T - 1)
        steps.append(_make_step(t, is_terminal, episode_seed))
    return steps


def test_round_trip_balanced_terminal_dataset(tmp_path: Path) -> None:
    """Writer → BalancedTerminalDataset round-trip."""
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    shard_name = "shard_000.hdf5"

    writer = RolloutDumpWriter(
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        shard_name=shard_name,
    )

    # demo_0: success — terminal at last step → positive window at start=(T-SEQ_LEN)
    writer.write_demo(index=0, steps=_make_episode(success=True), preprocess_config=PREPROCESS_CONFIG)
    # demo_1: failure — all sparse_rewards=0 → only negative windows
    writer.write_demo(index=1, steps=_make_episode(success=False), preprocess_config=PREPROCESS_CONFIG)
    writer.close()

    # Verify files exist
    assert (reward_dir / shard_name).is_file(), "reward HDF5 not written"
    assert (hidden_dir / shard_name).is_file(), "sidecar HDF5 not written"
    assert (hidden_dir / "preprocess_config.json").is_file(), "preprocess_config.json not written"

    # Verify preprocess_config.json was written correctly
    cfg = json.loads((hidden_dir / "preprocess_config.json").read_text())
    assert cfg["hidden_key"] == "obs_embedding"
    assert cfg["time_horizon"] == 8
    assert cfg["history"] == 2
    assert cfg["action_head_type"] == "oft_l1_regression"
    assert cfg["obs_hidden_source"] == "action_query"
    assert cfg["prompt_style"] == "vla_policy"
    assert cfg["include_state"] is True
    assert cfg["rotate_images_180"] is True

    # ── Round-trip: load via BalancedTerminalDataset ───────────────────────────
    from dreamervla.dataset.balanced_terminal_dataset import BalancedTerminalDataset

    dataset = BalancedTerminalDataset(
        hdf5_dir=str(reward_dir),
        hidden_dir=str(hidden_dir),
        sequence_length=SEQ_LEN,
        image_size=64,  # resize to small for speed
        expected_model_path="/fake/model/path",
        expected_time_horizon=8,
        expected_action_head_type="oft_l1_regression",
        expected_obs_hidden_source="action_query",
        expected_prompt_style="vla_policy",
        expected_history=2,
        expected_include_state=True,
        expected_rotate_images_180=True,
        reward_mode="sparse",
    )

    assert len(dataset) > 0, "dataset is empty"
    assert len(dataset.positive_indices) > 0, "no positive windows"
    assert len(dataset.negative_indices) > 0, "no negative windows"

    # Fetch one positive and one negative item
    pos_item = dataset[dataset.positive_indices[0]]
    neg_item = dataset[dataset.negative_indices[0]]

    for item, label in [(pos_item, "positive"), (neg_item, "negative")]:
        assert "obs_embedding" in item, f"{label} item missing obs_embedding"
        assert "images" in item, f"{label} item missing images"
        assert "actions" in item, f"{label} item missing actions"
        assert "rewards" in item, f"{label} item missing rewards"
        assert item["obs_embedding"].shape == (SEQ_LEN, HIDDEN_DIM), (
            f"{label} obs_embedding shape mismatch: {item['obs_embedding'].shape}"
        )
        assert item["images"].shape == (SEQ_LEN, 6, 64, 64), (
            f"{label} images shape mismatch: {item['images'].shape}"
        )
        assert item["actions"].shape == (SEQ_LEN, ACTION_DIM), (
            f"{label} actions shape mismatch: {item['actions'].shape}"
        )

    assert pos_item["is_positive_window"] is True
    assert neg_item["is_positive_window"] is False


def test_writer_creates_directories(tmp_path: Path) -> None:
    """Writer must create reward_dir and hidden_dir if they don't exist."""
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "deep" / "reward"
    hidden_dir = tmp_path / "deep" / "hidden"

    writer = RolloutDumpWriter(
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        shard_name="shard_000.hdf5",
    )
    writer.close()

    assert reward_dir.is_dir()
    assert hidden_dir.is_dir()


def test_writer_dtypes(tmp_path: Path) -> None:
    """Verify HDF5 datasets have the exact dtypes from the data contract."""
    import h5py

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    shard_name = "shard_000.hdf5"

    writer = RolloutDumpWriter(reward_dir=reward_dir, hidden_dir=hidden_dir, shard_name=shard_name)
    writer.write_demo(index=0, steps=_make_episode(success=True), preprocess_config=PREPROCESS_CONFIG)
    writer.close()

    with h5py.File(reward_dir / shard_name, "r") as f:
        demo = f["data"]["demo_0"]
        assert demo["actions"].dtype == np.float64,   f"actions dtype: {demo['actions'].dtype}"
        assert demo["dones"].dtype == np.uint8,       f"dones dtype: {demo['dones'].dtype}"
        assert demo["rewards"].dtype == np.float32,   f"rewards dtype: {demo['rewards'].dtype}"
        assert demo["sparse_rewards"].dtype == np.uint8, f"sparse_rewards dtype: {demo['sparse_rewards'].dtype}"
        assert demo["robot_states"].dtype == np.float64, f"robot_states dtype: {demo['robot_states'].dtype}"
        assert demo["states"].dtype == np.float64,    f"states dtype: {demo['states'].dtype}"
        assert demo["obs"]["agentview_rgb"].dtype == np.uint8
        assert demo["obs"]["eye_in_hand_rgb"].dtype == np.uint8

    with h5py.File(hidden_dir / shard_name, "r") as f:
        demo = f["data"]["demo_0"]
        assert demo["obs_embedding"].dtype == np.float16, f"obs_embedding dtype: {demo['obs_embedding'].dtype}"


def test_writer_shapes(tmp_path: Path) -> None:
    """Verify HDF5 dataset shapes match the data contract."""
    import h5py

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    shard_name = "shard_000.hdf5"

    writer = RolloutDumpWriter(reward_dir=reward_dir, hidden_dir=hidden_dir, shard_name=shard_name)
    writer.write_demo(index=0, steps=_make_episode(success=True), preprocess_config=PREPROCESS_CONFIG)
    writer.close()

    with h5py.File(reward_dir / shard_name, "r") as f:
        demo = f["data"]["demo_0"]
        assert demo["actions"].shape == (T, ACTION_DIM)
        assert demo["dones"].shape == (T,)
        assert demo["rewards"].shape == (T,)
        assert demo["sparse_rewards"].shape == (T,)
        assert demo["robot_states"].shape == (T, 9)
        assert demo["states"].shape == (T, STATE_DIM)
        assert demo["obs"]["agentview_rgb"].shape == (T, IMAGE_H, IMAGE_W, 3)
        assert demo["obs"]["eye_in_hand_rgb"].shape == (T, IMAGE_H, IMAGE_W, 3)
        assert demo["obs"]["ee_pos"].shape == (T, 3)
        assert demo["obs"]["ee_ori"].shape == (T, 3)
        assert demo["obs"]["ee_states"].shape == (T, 6)
        assert demo["obs"]["gripper_states"].shape == (T, 2)
        assert demo["obs"]["joint_states"].shape == (T, 7)
        # demo attrs
        assert "num_samples" in demo.attrs
        assert demo.attrs["num_samples"] == str(T)

    with h5py.File(hidden_dir / shard_name, "r") as f:
        demo = f["data"]["demo_0"]
        assert demo["obs_embedding"].shape == (T, HIDDEN_DIM)


def test_writer_data_attrs(tmp_path: Path) -> None:
    """data_attrs (env meta) are written once to the reward HDF5 data-group attrs."""
    import h5py

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    shard_name = "shard_000.hdf5"
    data_attrs = {"bddl_file_name": "x.bddl", "env_name": "Libero_Goal", "tag": "libero-v1"}

    writer = RolloutDumpWriter(reward_dir=reward_dir, hidden_dir=hidden_dir, shard_name=shard_name)
    writer.write_demo(
        index=0,
        steps=_make_episode(success=True),
        preprocess_config=PREPROCESS_CONFIG,
        data_attrs=data_attrs,
    )
    writer.close()

    with h5py.File(reward_dir / shard_name, "r") as f:
        attrs = f["data"].attrs
        assert attrs["bddl_file_name"] == "x.bddl"
        assert attrs["env_name"] == "Libero_Goal"
        assert attrs["tag"] == "libero-v1"


def test_writer_episode_metadata_attrs(tmp_path: Path) -> None:
    """episode_metadata scalar/string values are stored as per-demo attrs."""
    import h5py

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    writer = RolloutDumpWriter(
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        shard_name="shard_000.hdf5",
    )
    writer.write_demo(
        index=0,
        steps=_make_episode(success=True),
        preprocess_config=PREPROCESS_CONFIG,
        task_id=2,
        episode_id=7,
        episode_success=True,
        episode_horizon=300,
        episode_metadata={
            "suite": "libero_goal",
            "task_name": "open drawer",
            "global_episode_index": 123,
            "policy_name": "openvla_oft_default",
            "policy_ckpt": "/ckpts/policy",
            "policy_version": 5,
            "success_step": 9,
            "timeout": False,
            "chunk_size": 8,
            "action_scale": "raw",
            "seed": 17,
            "render_backend": "egl",
            "hidden_key": "obs_embedding",
            "hidden_dim": HIDDEN_DIM,
            "token_count": 56,
            "token_dim": 4096,
            "ignored_none": None,
            "ignored_dict": {"not": "an attr scalar"},
        },
    )
    writer.close()

    with h5py.File(reward_dir / "shard_000.hdf5", "r") as f:
        attrs = f["data"]["demo_0"].attrs
        assert attrs["task_id"] == 2
        assert attrs["episode_id"] == 7
        assert attrs["suite"] == "libero_goal"
        assert attrs["task_name"] == "open drawer"
        assert attrs["chunk_size"] == 8
        assert attrs["action_scale"] == "raw"
        assert attrs["timeout"] == np.False_
        assert attrs["hidden_dim"] == HIDDEN_DIM
        assert attrs["token_count"] == 56
        assert attrs["token_dim"] == 4096
        assert "ignored_none" not in attrs
        assert "ignored_dict" not in attrs
