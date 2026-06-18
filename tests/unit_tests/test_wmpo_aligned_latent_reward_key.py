import numpy as np

from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
from dreamervla.dataset.wmpo_aligned_latent_dataset import _load_demo


def _steps(T, success, emb_dim=16):
    out = []
    for t in range(T):
        out.append({
            "actions": np.zeros(7, np.float64),
            "rewards": np.float32(0.0),                                   # collector: always 0
            "sparse_rewards": np.uint8(1 if (success and t == T - 1) else 0),
            "dones": np.uint8(1 if t == T - 1 else 0),
            "robot_states": np.zeros(9, np.float64),
            "states": np.zeros(5, np.float64),
            "obs": {"agentview_rgb": np.zeros((256, 256, 3), np.uint8),
                    "eye_in_hand_rgb": np.zeros((256, 256, 3), np.uint8),
                    "ee_pos": np.zeros(3, np.float64), "ee_ori": np.zeros(3, np.float64),
                    "ee_states": np.zeros(6, np.float64), "gripper_states": np.zeros(2, np.float64),
                    "joint_states": np.zeros(7, np.float64)},
            "obs_embedding": np.zeros(emb_dim, np.float16),
        })
    return out


def test_load_demo_uses_sparse_rewards_for_complete(tmp_path):
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "s.hdf5") as w:
        w.write_demo(index=0, steps=_steps(6, success=True))
        w.write_demo(index=1, steps=_steps(6, success=False))
    raw, hid = tmp_path / "r" / "s.hdf5", tmp_path / "h" / "s.hdf5"
    # demo_key is the full HDF5 path ("data/demo_i"); _load_demo does fr[demo_key].
    assert _load_demo(raw, hid, "data/demo_0").complete is True   # sparse_rewards terminal 1
    assert _load_demo(raw, hid, "data/demo_1").complete is False  # no sparse reward
