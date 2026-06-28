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
            init_state_index=3,
            task_description="put the bowl on the plate",
            episode_success=True,
            episode_horizon=200,
            episode_metadata={
                "global_step": 120,
                "env_step": 4567,
                "update_step": 120,
                "learner_updates": 120,
                "policy_version": 120,
                "wm_version": 3,
                "classifier_version": 4,
            },
        )
    with h5py.File(tmp_path / "r" / "shard_000.hdf5", "r") as f:
        demo = f["data"]["demo_0"]
        assert int(demo.attrs["task_id"]) == 7
        assert int(demo.attrs["episode_id"]) == 3
        assert int(demo.attrs["init_state_index"]) == 3
        assert str(demo.attrs["task_description"]) == "put the bowl on the plate"
        assert bool(demo.attrs["success"]) is True
        assert bool(demo.attrs["complete"]) is True
        assert int(demo.attrs["global_step"]) == 120
        assert int(demo.attrs["env_step"]) == 4567
        for forbidden in (
            "episode_success",
            "episode_horizon",
            "update_step",
            "learner_updates",
            "policy_version",
            "wm_version",
            "classifier_version",
        ):
            assert forbidden not in demo.attrs
    with h5py.File(tmp_path / "h" / "shard_000.hdf5", "r") as f:
        hidden_demo = f["data"]["demo_0"]
        assert int(hidden_demo.attrs["init_state_index"]) == 3
        assert bool(hidden_demo.attrs["success"]) is True
        assert int(hidden_demo.attrs["global_step"]) == 120
        assert int(hidden_demo.attrs["env_step"]) == 4567


def test_write_demo_persists_data_attrs_on_hidden_sidecar(tmp_path):
    data_attrs = {
        "task_suite_name": "libero_goal",
        "env_name": "libero",
        "source": "online_cotrain_ray",
    }
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "shard_000.hdf5") as w:
        w.write_demo(
            index=0,
            steps=[_one_step(), _one_step()],
            data_attrs=data_attrs,
        )

    with h5py.File(tmp_path / "r" / "shard_000.hdf5", "r") as f:
        assert int(f["data"].attrs["num_demos"]) == 1
        assert f["data"].attrs["source"] == "online_cotrain_ray"
    with h5py.File(tmp_path / "h" / "shard_000.hdf5", "r") as f:
        assert int(f["data"].attrs["num_demos"]) == 1
        assert f["data"].attrs["task_suite_name"] == "libero_goal"
        assert f["data"].attrs["env_name"] == "libero"
        assert f["data"].attrs["source"] == "online_cotrain_ray"


def test_write_demo_identity_optional(tmp_path):
    # Backward compatible: omitting identity must not error and not write attrs.
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "shard_000.hdf5") as w:
        w.write_demo(index=0, steps=[_one_step()])
    with h5py.File(tmp_path / "r" / "shard_000.hdf5", "r") as f:
        assert "task_id" not in f["data"]["demo_0"].attrs


def test_write_demo_persists_demo_language_embedding(tmp_path):
    first = _one_step()
    second = _one_step()
    first["lang_emb"] = np.arange(8, dtype=np.float32)
    second["lang_emb"] = np.arange(8, dtype=np.float32) + 100.0

    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "shard_000.hdf5") as w:
        w.write_demo(index=0, steps=[first, second])

    with h5py.File(tmp_path / "h" / "shard_000.hdf5", "r") as f:
        lang_emb = f["data"]["demo_0"]["lang_emb"]
        assert lang_emb.shape == (8,)
        assert lang_emb.dtype == np.dtype("float16")
        assert np.allclose(lang_emb[...], np.arange(8, dtype=np.float16))
