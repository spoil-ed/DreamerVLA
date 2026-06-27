import h5py
import numpy as np
import torch

from dreamervla.dataset.lumos_aligned_latent_dataset import (
    LumosAlignedLatentTrainDataset,
    _load_demo,
)
from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter


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


def test_lumos_aligned_dataset_returns_proprio_and_language_sidecar(tmp_path):
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "s.hdf5") as writer:
        writer.write_demo(index=0, steps=_steps(6, success=True))
    raw_dir, hid_dir = tmp_path / "r", tmp_path / "h"
    with h5py.File(hid_dir / "s.hdf5", "a") as handle:
        handle["data/demo_0"].create_dataset(
            "lang_emb", data=np.arange(5, dtype=np.float32)
        )

    dataset = LumosAlignedLatentTrainDataset(
        raw_dir,
        hid_dir,
        None,
        None,
        window=2,
        stride=2,
        verbose=False,
        chunk_subsample=2,
        chunk_pool="last",
        proprio_keys=("ee_pos", "ee_ori", "gripper_states"),
        lang_emb_dir=hid_dir,
    )
    x, y, extra = next(iter(dataset))

    assert x.shape == (2, 16)
    assert y == 1
    assert extra["proprio"].shape == (2, 8)
    assert torch.allclose(extra["lang_emb"], torch.arange(5, dtype=torch.float32))
