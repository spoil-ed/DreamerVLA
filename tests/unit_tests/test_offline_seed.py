import h5py
import numpy as np
from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
from dreamervla.runners.online_replay import OnlineReplay
from dreamervla.runners.offline_seed import seed_replay_from_offline


def _demo_steps(T, success, emb_dim=16):
    steps = []
    for t in range(T):
        steps.append({
            "actions": np.full(7, t, np.float64),
            "rewards": np.float32(0.0),
            "sparse_rewards": np.uint8(1 if (success and t == T - 1) else 0),
            "dones": np.uint8(1 if t == T - 1 else 0),
            "robot_states": np.zeros(9, np.float64),
            "states": np.zeros(5, np.float64),
            "obs": {
                "agentview_rgb": np.zeros((256, 256, 3), np.uint8),
                "eye_in_hand_rgb": np.zeros((256, 256, 3), np.uint8),
                "ee_pos": np.zeros(3, np.float64), "ee_ori": np.zeros(3, np.float64),
                "ee_states": np.zeros(6, np.float64), "gripper_states": np.zeros(2, np.float64),
                "joint_states": np.zeros(7, np.float64),
            },
            "obs_embedding": np.full(emb_dim, t, np.float16),
        })
    return steps


def _write_fixture(reward_dir, hidden_dir):
    with RolloutDumpWriter(reward_dir, hidden_dir, "r0_shard.hdf5") as w:
        w.write_demo(index=0, steps=_demo_steps(6, success=True), task_id=2, episode_id=0)
        w.write_demo(index=1, steps=_demo_steps(6, success=False), task_id=5, episode_id=0)


def test_seed_replay_reads_all_demos_with_task_id(tmp_path):
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    _write_fixture(rdir, hdir)
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(2, 5), rank=0)
    n = seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)
    assert n == 2                                    # two episodes added
    assert replay.num_transitions == 12             # 6 + 6
    batch = replay.sample(2)
    assert set(int(t) for t in batch["task_ids"].tolist()) <= {2, 5}
    assert batch["obs_embedding"].shape[-1] == 16


def test_seed_replay_task_id_fallback(tmp_path):
    # Demo without task_id attr -> use provided default.
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        w.write_demo(index=0, steps=_demo_steps(6, success=True))   # no task_id
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(0,), rank=0)
    n = seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir, default_task_id=0)
    assert n == 1
    batch = replay.sample(1)
    assert int(batch["task_ids"][0]) == 0


def test_seed_replay_missing_task_id_raises(tmp_path):
    import pytest
    rdir, hdir = tmp_path / "reward", tmp_path / "hidden"
    with RolloutDumpWriter(rdir, hdir, "r0_shard.hdf5") as w:
        w.write_demo(index=0, steps=_demo_steps(6, success=True))   # no task_id
    replay = OnlineReplay(capacity=10_000, sequence_length=4, task_ids=(0,), rank=0)
    with pytest.raises(ValueError, match="task_id"):
        seed_replay_from_offline(replay, data_dir=rdir, hidden_dir=hdir)  # no default
