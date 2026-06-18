import h5py
import numpy as np

from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter


def _one_step():
    return {
        "actions": np.zeros(7, np.float64),
        "rewards": np.float32(0.0),
        "sparse_rewards": np.uint8(1),
        "dones": np.uint8(1),
        "robot_states": np.zeros(9, np.float64),
        "states": np.zeros(5, np.float64),
        "obs": {
            "agentview_rgb": np.zeros((256, 256, 3), np.uint8),
            "eye_in_hand_rgb": np.zeros((256, 256, 3), np.uint8),
            "ee_pos": np.zeros(3, np.float64),
            "ee_ori": np.zeros(3, np.float64),
            "ee_states": np.zeros(6, np.float64),
            "gripper_states": np.zeros(2, np.float64),
            "joint_states": np.zeros(7, np.float64),
        },
        "obs_embedding": np.zeros(16, np.float16),
    }


def test_write_demo_persists_identity_attrs(tmp_path):
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "shard_000.hdf5") as w:
        w.write_demo(
            index=0,
            steps=[_one_step(), _one_step()],
            task_id=7,
            episode_id=3,
            task_description="put the bowl on the plate",
            episode_success=True,
            episode_horizon=200,
        )
    with h5py.File(tmp_path / "r" / "shard_000.hdf5", "r") as f:
        demo = f["data"]["demo_0"]
        assert int(demo.attrs["task_id"]) == 7
        assert int(demo.attrs["episode_id"]) == 3
        assert str(demo.attrs["task_description"]) == "put the bowl on the plate"
        assert bool(demo.attrs["episode_success"]) is True
        assert int(demo.attrs["episode_horizon"]) == 200


def test_write_demo_identity_optional(tmp_path):
    # Backward compatible: omitting identity must not error and not write attrs.
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "shard_000.hdf5") as w:
        w.write_demo(index=0, steps=[_one_step()])
    with h5py.File(tmp_path / "r" / "shard_000.hdf5", "r") as f:
        assert "task_id" not in f["data"]["demo_0"].attrs
