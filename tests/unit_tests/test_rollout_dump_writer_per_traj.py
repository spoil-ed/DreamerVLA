"""PerTrajectoryDumpWriter: one identity-named HDF5 pair per trajectory."""

import json

import h5py
import numpy as np
import pytest

from dreamervla.dataset.collection_manifest import (
    EPISODE_INDEX_NAME,
    complete_episode_ids_per_task,
    count_collected_episodes,
)
from dreamervla.dataset.rollout_dump_writer import (
    PerTrajectoryDumpWriter,
    per_trajectory_shard_name,
)


def _make_steps(n, success):
    steps = []
    for t in range(n):
        steps.append(
            {
                "actions": np.zeros(7),
                "rewards": 0.0,
                "sparse_rewards": 1 if (success and t == n - 1) else 0,
                "dones": 1 if t == n - 1 else 0,
                "robot_states": np.zeros(9),
                "states": np.arange(4, dtype=np.float64),
                "obs": {
                    "agentview_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    "eye_in_hand_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    "ee_pos": np.zeros(3),
                    "ee_ori": np.zeros(3),
                    "ee_states": np.zeros(6),
                    "gripper_states": np.zeros(2),
                    "joint_states": np.zeros(7),
                },
                "obs_embedding": np.zeros((256, 4096), dtype=np.float16),
                "success": success,
            }
        )
    return steps


def test_per_trajectory_shard_name():
    assert per_trajectory_shard_name("traj", 3, 41) == "traj_t03_ep000041.hdf5"


def test_writes_one_identity_named_pair_per_trajectory(
    tmp_path, input_token_preprocess_config
):
    reward_dir, hidden_dir = tmp_path / "reward", tmp_path / "hidden"
    with PerTrajectoryDumpWriter(reward_dir, hidden_dir) as writer:
        writer.write_demo(
            index=0,
            steps=_make_steps(3, success=True),
            preprocess_config={
                **input_token_preprocess_config,
                "chunk_size": 8,
            },
            data_attrs={"task_suite_name": "libero_goal"},
            task_id=0,
            episode_id=0,
            task_description="task zero",
            episode_success=True,
            episode_horizon=3,
        )
        writer.write_demo(
            index=1,
            steps=_make_steps(2, success=False),
            task_id=1,
            episode_id=3,
            task_description="task one",
            episode_success=False,
            episode_horizon=2,
        )

    names = sorted(p.name for p in reward_dir.glob("*.hdf5"))
    assert names == ["traj_t00_ep000000.hdf5", "traj_t01_ep000003.hdf5"]
    assert sorted(p.name for p in hidden_dir.glob("*.hdf5")) == names

    with h5py.File(reward_dir / "traj_t01_ep000003.hdf5", "r") as f:
        demo = f["data"]["demo_0"]
        assert int(demo.attrs["task_id"]) == 1
        assert int(demo.attrs["episode_id"]) == 3
        assert bool(demo.attrs["success"]) is False
        # captured config/attrs re-emitted so every file is independently readable
        assert f["data"].attrs["task_suite_name"] == "libero_goal"
        assert int(demo.attrs["chunk_size"]) == 8

    # resume helpers keep working over per-traj files
    assert count_collected_episodes(reward_dir) == 2
    assert complete_episode_ids_per_task(reward_dir, hidden_dir) == {0: {0}, 1: {3}}

    # episode index records correspondence
    lines = [
        json.loads(line)
        for line in (reward_dir / EPISODE_INDEX_NAME).read_text().splitlines()
    ]
    assert lines[0]["file"] == "traj_t00_ep000000.hdf5"
    assert lines[0]["task_id"] == 0 and lines[0]["episode_id"] == 0
    assert lines[0]["success"] is True and lines[0]["horizon"] == 3
    assert lines[1]["file"] == "traj_t01_ep000003.hdf5"


def test_rewrite_same_identity_overwrites(tmp_path, input_token_preprocess_config):
    reward_dir, hidden_dir = tmp_path / "reward", tmp_path / "hidden"
    with PerTrajectoryDumpWriter(reward_dir, hidden_dir) as writer:
        writer.write_demo(
            index=0,
            steps=_make_steps(2, False),
            preprocess_config=input_token_preprocess_config,
            task_id=0,
            episode_id=0,
            episode_success=False,
            episode_horizon=2,
        )
        writer.write_demo(
            index=1,
            steps=_make_steps(4, True),
            task_id=0,
            episode_id=0,
            episode_success=True,
            episode_horizon=4,
        )
    assert [p.name for p in reward_dir.glob("*.hdf5")] == ["traj_t00_ep000000.hdf5"]
    assert count_collected_episodes(reward_dir) == 1
    with h5py.File(reward_dir / "traj_t00_ep000000.hdf5", "r") as f:
        assert f["data"]["demo_0"]["actions"].shape[0] == 4


def test_missing_identity_raises(tmp_path):
    with PerTrajectoryDumpWriter(tmp_path / "r", tmp_path / "h") as writer:
        with pytest.raises(ValueError, match="task_id and episode_id"):
            writer.write_demo(index=0, steps=_make_steps(1, False))


def test_make_dump_writer_routes_one_to_per_traj(tmp_path):
    from dreamervla.dataset.rollout_dump_writer import (
        RolloutDumpWriter,
        RotatingRolloutDumpWriter,
    )
    from dreamervla.runners.collect_parallel_rollouts import _make_dump_writer

    kwargs = dict(shard_name="s_000.hdf5", shard_prefix="s", start_index=0)
    per_traj = _make_dump_writer(
        tmp_path / "r1", tmp_path / "h1", demos_per_shard=1, **kwargs
    )
    assert isinstance(per_traj, PerTrajectoryDumpWriter)
    rotating = _make_dump_writer(
        tmp_path / "r2", tmp_path / "h2", demos_per_shard=2, **kwargs
    )
    assert isinstance(rotating, RotatingRolloutDumpWriter)
    rotating.close()
    single = _make_dump_writer(
        tmp_path / "r3", tmp_path / "h3", demos_per_shard=0, **kwargs
    )
    assert isinstance(single, RolloutDumpWriter)
    single.close()


def test_no_canonical_file_left_when_write_crashes(tmp_path, monkeypatch):
    from dreamervla.dataset import rollout_dump_writer as rdw

    def _boom(self, *a, **kw):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(rdw.RolloutDumpWriter, "write_demo", _boom)
    reward_dir, hidden_dir = tmp_path / "reward", tmp_path / "hidden"
    writer = PerTrajectoryDumpWriter(reward_dir, hidden_dir)
    with pytest.raises(RuntimeError, match="simulated crash"):
        writer.write_demo(index=0, steps=_make_steps(2, False), task_id=0, episode_id=0)
    # the canonical identity name is never occupied by a partial file
    assert list(reward_dir.glob("*.hdf5")) == []
    assert list(hidden_dir.glob("*.hdf5")) == []


def test_no_tmp_files_left_after_successful_write(
    tmp_path, input_token_preprocess_config
):
    reward_dir, hidden_dir = tmp_path / "reward", tmp_path / "hidden"
    with PerTrajectoryDumpWriter(reward_dir, hidden_dir) as writer:
        writer.write_demo(
            index=0,
            steps=_make_steps(2, True),
            preprocess_config=input_token_preprocess_config,
            task_id=0,
            episode_id=0,
            episode_success=True,
            episode_horizon=2,
        )
    assert [p.name for p in reward_dir.glob("*.hdf5")] == ["traj_t00_ep000000.hdf5"]
    assert list(reward_dir.glob("*.tmp")) == []
    assert list(hidden_dir.glob("*.tmp")) == []
