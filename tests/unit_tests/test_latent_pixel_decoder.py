from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.models.embodiment.world_model.latent_pixel_decoder import (
    LatentTokenPixelDecoder,
    latent_pixel_reconstruction_loss,
)
from dreamervla.runners.latent_pixel_decoder_training_runner import (
    LatentPixelDecoderTrainingRunner,
)


def _decoder() -> LatentTokenPixelDecoder:
    return LatentTokenPixelDecoder(
        token_dim=32,
        token_count=48,
        tokens_per_view=16,
        view_indices=(0, 1),
        image_size=32,
        base_channels=16,
        channel_schedule=(16, 8),
        upsample_sizes=(16, 32),
    )


def test_decoder_accepts_wm_tokens_with_trailing_condition_width() -> None:
    decoder = _decoder()
    # WM prediction can append proprio conditioning after the visual token width.
    output = decoder(torch.randn(2, 3, 48, 37))
    assert output.shape == (2, 3, 2, 3, 32, 32)
    assert bool(((output >= 0.0) & (output <= 1.0)).all())


def test_decoder_layer_norm_matches_wm_input_invariance() -> None:
    decoder = _decoder().eval()
    tokens = torch.randn(2, 48, 32)
    with torch.no_grad():
        reference = decoder(tokens)
        shifted_scaled = decoder(tokens * 3.0 + 4.0)
    torch.testing.assert_close(reference, shifted_scaled, atol=2.0e-5, rtol=2.0e-5)


def test_spatial_decoder_mixes_tokens_and_uses_separate_view_heads() -> None:
    decoder = LatentTokenPixelDecoder(
        token_dim=32,
        token_count=48,
        tokens_per_view=16,
        view_indices=(0, 1),
        image_size=32,
        base_channels=32,
        channel_schedule=(16, 8),
        upsample_sizes=(16, 32),
        spatial_mixer_depth=2,
        spatial_mixer_heads=4,
        spatial_mixer_mlp_ratio=2.0,
        separate_view_heads=True,
    )

    output = decoder(torch.randn(2, 48, 32))

    assert output.shape == (2, 2, 3, 32, 32)
    assert isinstance(decoder.output, torch.nn.ModuleList)
    assert decoder.spatial_position is not None


def test_reconstruction_loss_backpropagates_and_reports_pixel_metrics() -> None:
    decoder = _decoder()
    source = torch.randn(2, 48, 32, requires_grad=True)
    prediction = decoder(source)
    losses = latent_pixel_reconstruction_loss(prediction, torch.rand_like(prediction))
    assert set(losses) == {"loss", "l1", "ssim", "psnr"}
    assert losses["loss"].requires_grad
    losses["loss"].backward()
    assert decoder.token_proj.weight.grad is not None
    assert source.grad is not None


def test_frozen_prefix_boundary_updates_only_decoder() -> None:
    class Producer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))

        def encode_observation_prefix(self, observation: torch.Tensor) -> torch.Tensor:
            return observation @ self.weight

    runner = LatentPixelDecoderTrainingRunner.__new__(LatentPixelDecoderTrainingRunner)
    runner.policy = Producer()
    for parameter in runner.policy.parameters():
        parameter.requires_grad_(False)
    decoder = torch.nn.Linear(8, 3)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)
    producer_before = runner.policy.weight.detach().clone()

    observation = torch.randn(2, 4, requires_grad=True)
    prefix = runner._encode_frozen_prefix(observation)
    assert not prefix.requires_grad
    assert not prefix.is_inference()
    loss = decoder(prefix).square().mean()
    loss.backward()
    optimizer.step()

    assert decoder.weight.grad is not None
    assert runner.policy.weight.grad is None
    assert observation.grad is None
    torch.testing.assert_close(runner.policy.weight, producer_before)


def test_prepare_observation_moves_tensor_leaves_to_runner_device() -> None:
    @dataclass
    class Observation:
        images: dict[str, torch.Tensor]
        state: torch.Tensor

    runner = LatentPixelDecoderTrainingRunner.__new__(LatentPixelDecoderTrainingRunner)
    runner.device = torch.device("cpu")
    runner.cfg = OmegaConf.create(
        {"decoder_training": {"target_image_keys": ["base_0_rgb", "left_wrist_0_rgb"]}}
    )
    observation = Observation(
        images={
            "base_0_rgb": torch.full((2, 3, 4, 4), -1.0),
            "left_wrist_0_rgb": torch.full((2, 3, 4, 4), 1.0),
        },
        state=torch.zeros(2, 8),
    )

    prepared, target = runner._prepare_observation((observation, torch.zeros(2, 10, 7)))

    assert prepared.state.device == runner.device
    assert prepared.images["base_0_rgb"].is_contiguous()
    assert target.shape == (2, 2, 3, 4, 4)
    torch.testing.assert_close(target[:, 0], torch.zeros(2, 3, 4, 4))
    torch.testing.assert_close(target[:, 1], torch.ones(2, 3, 4, 4))


def test_prepare_collected_replay_batch_resizes_pixels_and_removes_time_axis() -> None:
    runner = LatentPixelDecoderTrainingRunner.__new__(LatentPixelDecoderTrainingRunner)
    runner.device = torch.device("cpu")
    runner.cfg = OmegaConf.create({"pixel_decoder": {"image_size": 4}})
    prefix = torch.randn(2, 1, 48, 32, dtype=torch.float16, requires_grad=True)
    images = torch.full((2, 1, 2, 8, 8, 3), 255, dtype=torch.uint8)

    prepared_prefix, target = runner._prepare_replay_batch(
        {"obs_embedding": prefix, "images": images}
    )

    assert prepared_prefix.shape == (2, 48, 32)
    assert prepared_prefix.dtype == torch.float16
    assert not prepared_prefix.requires_grad
    assert target.shape == (2, 2, 3, 4, 4)
    torch.testing.assert_close(target, torch.ones_like(target))


def test_decoder_resume_tolerates_more_ranks_than_checkpoint_rng_states() -> None:
    runner = LatentPixelDecoderTrainingRunner.__new__(LatentPixelDecoderTrainingRunner)
    runner.distributed = SimpleNamespace(world_size=8, rank=7)
    runner._checkpoint_world_size = None
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dicts": {},
        "pickles": {},
        "rng_by_rank": [{}, {}],
    }

    runner.load_payload(payload, restore_rng=True)

    assert runner._checkpoint_world_size == 2


def test_decoder_resume_normalizes_data_cursor_after_ddp_resize() -> None:
    runner = LatentPixelDecoderTrainingRunner.__new__(LatentPixelDecoderTrainingRunner)
    runner.distributed = SimpleNamespace(world_size=8)
    runner.cfg = OmegaConf.create(
        {"decoder_training": {"micro_batch_size": 16, "global_batch_size": 128}}
    )
    runner.global_step = 6000
    runner.epoch = 2
    runner._data_epoch = 2
    runner._data_iter_offset = 4000
    runner._data_generator_state = torch.Generator().get_state()
    runner._checkpoint_world_size = 2

    runner._normalize_resume_cursor(num_batches=2137)

    assert runner.gradient_accumulation == 1
    assert runner.epoch == 2
    assert runner._data_epoch == 2
    assert runner._data_iter_offset == 1726
    assert runner._data_generator_state is None


def test_decoder_rejects_non_square_view_tokens() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        LatentTokenPixelDecoder(token_dim=8, token_count=20, tokens_per_view=10)


def test_pi05_pixel_decoder_config_validates_for_eight_gpu_ddp() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=["experiment=pi05_pixel_decoder"])
    validate_cfg(cfg, world_size=8)
    assert cfg.data.loader._target_ == "dreamervla.dataset.libero.LeRobotV3LIBERODataLoaderFactory"
    assert cfg.data.repo_id == "lerobot/libero"
    assert cfg.data.loader.normalization_asset_id == "physical-intelligence/libero"
    assert cfg.pixel_decoder.token_count == 768
    assert cfg.pixel_decoder.token_dim == 2048
    assert list(cfg.pixel_decoder.view_indices) == [0, 1]
    assert list(cfg.decoder_training.target_image_keys) == [
        "base_0_rgb",
        "left_wrist_0_rgb",
    ]


def test_collected_pixel_decoder_configs_cover_all_2000_object_trajectories() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        baseline = compose(
            config_name="train",
            overrides=["experiment=pi05_pixel_decoder_collected"],
        )
        spatial = compose(
            config_name="train",
            overrides=[
                "experiment=pi05_pixel_decoder_collected",
                "pixel_decoder=pi05-prefix-spatial",
            ],
        )

    validate_cfg(baseline, world_size=8)
    validate_cfg(spatial, world_size=8)
    assert baseline.decoder_training.expected_episodes == 2000
    assert baseline.data.replay.max_episodes_per_task == 200
    assert list(baseline.data.replay.task_ids) == list(range(10))
    assert OmegaConf.select(baseline, "pixel_decoder.spatial_mixer_depth", default=0) == 0
    spatial_decoder = instantiate(spatial.pixel_decoder)
    parameters = sum(parameter.numel() for parameter in spatial_decoder.parameters())
    assert 20_000_000 <= parameters <= 30_000_000
