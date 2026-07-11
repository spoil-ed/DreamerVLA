import h5py
import numpy as np
import pytest
import torch

from dreamervla.dataset.lumos_aligned_latent_dataset import (
    LumosAlignedLatentTrainDataset,
    _load_demo,
)
from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
from dreamervla.dataset.wm_replay_classifier_dataset import _find_demo_pairs


def _steps(T, success):
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
            "obs_embedding": np.zeros((256, 4096), np.float16),
        })
    return out


def test_load_demo_uses_sparse_rewards_for_complete(
    tmp_path, input_token_preprocess_config
):
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "s.hdf5") as w:
        w.write_demo(
            index=0,
            steps=_steps(6, success=True),
            preprocess_config=input_token_preprocess_config,
        )
        w.write_demo(
            index=1,
            steps=_steps(6, success=False),
            preprocess_config=input_token_preprocess_config,
        )
    raw, hid = tmp_path / "r" / "s.hdf5", tmp_path / "h" / "s.hdf5"
    # demo_key is the full HDF5 path ("data/demo_i"); _load_demo does fr[demo_key].
    assert _load_demo(raw, hid, "data/demo_0").complete is True   # sparse_rewards terminal 1
    assert _load_demo(raw, hid, "data/demo_1").complete is False  # no sparse reward


def test_find_demo_pairs_rejects_raw_hidden_demo_set_mismatch(tmp_path):
    raw_dir, hid_dir = tmp_path / "raw", tmp_path / "hidden"
    raw_dir.mkdir()
    hid_dir.mkdir()
    with h5py.File(raw_dir / "shard.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
        demo.create_dataset("rewards", data=np.zeros((4,), dtype=np.float32))
        demo.create_dataset("dones", data=np.zeros((4,), dtype=np.uint8))
    with h5py.File(hid_dir / "shard.hdf5", "w") as handle:
        data = handle.create_group("data")
        data.create_group("demo_0").create_dataset(
            "obs_embedding", data=np.zeros((4, 16), dtype=np.float16)
        )
        data.create_group("demo_1").create_dataset(
            "obs_embedding", data=np.zeros((4, 16), dtype=np.float16)
        )

    with pytest.raises(ValueError, match="demo set mismatch"):
        _find_demo_pairs(raw_dir, hid_dir)


def test_load_demo_rejects_raw_hidden_length_mismatch(tmp_path):
    raw_dir, hid_dir = tmp_path / "raw", tmp_path / "hidden"
    raw_dir.mkdir()
    hid_dir.mkdir()
    with h5py.File(raw_dir / "shard.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
        demo.create_dataset("rewards", data=np.zeros((3,), dtype=np.float32))
        demo.create_dataset("dones", data=np.zeros((3,), dtype=np.uint8))
    with h5py.File(hid_dir / "shard.hdf5", "w") as handle:
        handle.create_group("data/demo_0").create_dataset(
            "obs_embedding", data=np.zeros((4, 16), dtype=np.float16)
        )

    with pytest.raises(ValueError, match="raw/hidden length mismatch"):
        _load_demo(raw_dir / "shard.hdf5", hid_dir / "shard.hdf5", "data/demo_0")


def test_lumos_aligned_dataset_returns_proprio_and_language_sidecar(
    tmp_path, input_token_preprocess_config
):
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "s.hdf5") as writer:
        writer.write_demo(
            index=0,
            steps=_steps(6, success=True),
            preprocess_config=input_token_preprocess_config,
        )
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

    assert x.shape == (2, 256, 4096)
    assert y == 1
    assert extra["proprio"].shape == (2, 8)
    assert torch.allclose(extra["lang_emb"], torch.arange(5, dtype=torch.float32))


def test_lumos_aligned_dataset_can_read_language_from_source_hidden(
    tmp_path, input_token_preprocess_config
):
    with RolloutDumpWriter(tmp_path / "r", tmp_path / "h", "s.hdf5") as writer:
        writer.write_demo(
            index=0,
            steps=_steps(6, success=True),
            preprocess_config=input_token_preprocess_config,
        )
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
        lang_emb_dir="__source_hidden__",
    )
    _, _, extra = next(iter(dataset))

    assert torch.allclose(extra["lang_emb"], torch.arange(5, dtype=torch.float32))
