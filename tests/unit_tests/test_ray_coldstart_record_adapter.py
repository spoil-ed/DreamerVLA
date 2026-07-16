from __future__ import annotations

import numpy as np

from dreamervla.workers.rollout.record_adapter import build_dump_step

HIDDEN_TOKEN_SHAPE = (256, 4096)


def _full_record() -> dict:
    return {
        "agentview_rgb": np.zeros((256, 256, 3), np.uint8),
        "eye_in_hand_rgb": np.zeros((256, 256, 3), np.uint8),
        "ee_pos": np.zeros(3, np.float64),
        "ee_ori": np.zeros(3, np.float64),
        "ee_states": np.zeros(6, np.float64),
        "gripper_states": np.zeros(2, np.float64),
        "joint_states": np.zeros(7, np.float64),
        "robot_states": np.zeros(9, np.float64),
        "states": np.zeros(45, np.float64),
    }


def test_build_dump_step_matches_writer_schema() -> None:
    full_record = _full_record()
    full_record["ee_pos"] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    full_record["ee_ori"] = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    full_record["gripper_states"] = np.array([7.0, 8.0], dtype=np.float64)
    step = build_dump_step(
        full_record=full_record,
        obs_embedding=np.zeros(HIDDEN_TOKEN_SHAPE, np.float16),
        lang_emb=np.arange(6, dtype=np.float32),
        action=np.ones(7, np.float32),
        reward=0.0,
        sparse_reward=1,
        done=True,
    )
    assert step["actions"].shape == (7,)
    assert step["obs_embedding"].shape == HIDDEN_TOKEN_SHAPE
    assert step["obs_embedding"].dtype == np.float16
    assert np.array_equal(step["lang_emb"], np.arange(6, dtype=np.float32))
    assert int(step["dones"]) == 1 and int(step["sparse_rewards"]) == 1
    assert step["robot_states"].shape == (9,)
    np.testing.assert_array_equal(step["proprio"], np.arange(1.0, 9.0, dtype=np.float32))
    for key in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "joint_states"):
        assert key in step["obs"]


def test_step_round_trips_through_rollout_dump_writer(
    tmp_path, hidden_token_preprocess_config
) -> None:
    import h5py

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    steps = [
        build_dump_step(
            full_record=_full_record(),
            obs_embedding=np.zeros(HIDDEN_TOKEN_SHAPE, np.float16),
            lang_emb=np.arange(6, dtype=np.float32),
            action=np.ones(7, np.float32),
            reward=0.0,
            sparse_reward=(1 if t == 2 else 0),
            done=(t == 2),
        )
        for t in range(3)
    ]
    writer = RolloutDumpWriter(tmp_path / "reward", tmp_path / "hidden", "shard.hdf5")
    writer.write_demo(
        index=0,
        steps=steps,
        preprocess_config=hidden_token_preprocess_config,
        task_id=0,
        episode_horizon=3,
        episode_success=True,
    )
    writer.close()
    with h5py.File(tmp_path / "hidden" / "shard.hdf5", "r") as handle:
        assert handle["data"]["demo_0"]["obs_embedding"].shape == (
            3,
            *HIDDEN_TOKEN_SHAPE,
        )
        assert handle["data"]["demo_0"]["lang_emb"].shape == (6,)
    with h5py.File(tmp_path / "reward" / "shard.hdf5", "r") as handle:
        assert handle["data"]["demo_0"]["obs"]["agentview_rgb"].shape == (3, 256, 256, 3)
        assert int(handle["data"]["demo_0"]["sparse_rewards"][-1]) == 1


def test_build_dump_step_preserves_hidden_token_shape(
    tmp_path, hidden_token_preprocess_config
) -> None:
    import h5py

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    tokenized = np.zeros(HIDDEN_TOKEN_SHAPE, dtype=np.float16)
    steps = [
        build_dump_step(
            full_record=_full_record(),
            obs_embedding=tokenized + t,
            lang_emb=np.arange(6, dtype=np.float32),
            action=np.ones(7, np.float32),
            reward=0.0,
            sparse_reward=(1 if t == 1 else 0),
            done=(t == 1),
        )
        for t in range(2)
    ]
    writer = RolloutDumpWriter(tmp_path / "reward", tmp_path / "hidden", "shard.hdf5")
    writer.write_demo(
        index=0,
        steps=steps,
        preprocess_config=hidden_token_preprocess_config,
        task_id=0,
        episode_horizon=2,
        episode_success=True,
    )
    writer.close()

    with h5py.File(tmp_path / "hidden" / "shard.hdf5", "r") as handle:
        assert handle["data"]["demo_0"]["obs_embedding"].shape == (
            2,
            *HIDDEN_TOKEN_SHAPE,
        )
