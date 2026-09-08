"""Opt-in CPU checks against the mounted native LIBERO v3 dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DVLA_TEST_LEROBOT_V3") != "1",
    reason="set DVLA_TEST_LEROBOT_V3=1 with LIBERO data and π0.5 assets mounted",
)


def _factory(*, num_workers: int = 0):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    cfg.task.pi05.base_ckpt_path = os.environ.get(
        "PI05_BASE_CKPT", "/jfs/oss-import/simate_pretrain_checkpoints/lerobot/pi05_base"
    )
    cfg.task.pi05.assets_path = os.environ.get(
        "PI05_ASSETS_CKPT",
        "/jfs/oss-import/simate_pretrain_checkpoints/RLinf/RLinf-Pi05-LIBERO-SFT",
    )
    cfg.data.loader.batch_size = 1
    cfg.data.loader.shuffle = False
    cfg.data.loader.num_workers = num_workers
    return instantiate(cfg.data.loader)


def test_native_video_and_actions_match_physical_source() -> None:
    import av
    import pyarrow.parquet as pq

    dataset = _factory().dataset
    assert dataset.num_episodes == 1693
    assert len(dataset) == 273465
    offset = int(dataset.episodes.iloc[0].length)
    sample = dataset[offset]
    episode = dataset.episodes.iloc[1]
    assert sample["episode_index"] == int(episode.episode_index)
    assert sample["frame_index"] == 0
    path = dataset.root / dataset.info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"])
    )
    rows = pq.read_table(path).slice(int(episode["_row_offset"]), dataset.sequence_length)
    np.testing.assert_array_equal(
        sample["actions"], np.array(rows["action"].to_pylist(), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        sample["state"], np.array(rows["observation.state"][0].as_py(), dtype=np.float32)
    )
    for field, camera in (("image", dataset.image_key), ("wrist_image", dataset.wrist_image_key)):
        prefix = f"videos/{camera}"
        path = dataset.root / dataset.info["video_path"].format(
            video_key=camera,
            chunk_index=int(episode[f"{prefix}/chunk_index"]),
            file_index=int(episode[f"{prefix}/file_index"]),
        )
        target = round(float(episode[f"{prefix}/from_timestamp"]) * dataset.fps)
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index == target:
                    np.testing.assert_array_equal(sample[field], frame.to_ndarray(format="rgb24"))
                    break
            else:
                pytest.fail(f"Missing reference video frame {target}")
    final = dataset[-1]
    assert final["actions"].shape == (dataset.sequence_length, 7)
    assert final["state"].shape == (8,)
    assert final["action_mask"].sum() == 1


def test_model_batch_and_eight_rank_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import torch

    from dreamervla.models.embodiment.pi05.sft_data import openpi_torch_loader

    factory = _factory(num_workers=1)
    bundle = factory.build(world_size=8, rank=0)
    loader = openpi_torch_loader(bundle.data_loader)
    sampler = loader.sampler
    assert sampler.num_replicas == 8
    first_indices = list(iter(sampler))[:4]
    assert first_indices == [0, 8, 16, 24]
    other = factory.build(world_size=8, rank=1)
    assert list(iter(openpi_torch_loader(other.data_loader).sampler))[:4] == [1, 9, 17, 25]
    observation, actions = next(iter(bundle.data_loader))
    assert actions.shape == (1, factory.action_horizon, 32)
    assert observation.state.shape == (1, 32)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(observation.state).all()
    assert observation.images["base_0_rgb"].shape == (1, 3, 224, 224)
    assert observation.images["left_wrist_0_rgb"].shape == (1, 3, 224, 224)
    assert bundle.source == str(factory.dataset.root)
    assert bundle.data_config.asset_id == "physical-intelligence/libero"
    assert bundle.data_config.repo_id == "lerobot/libero"
