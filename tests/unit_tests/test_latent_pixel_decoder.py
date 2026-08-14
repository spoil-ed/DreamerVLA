from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from dreamervla.config import validate_cfg
from dreamervla.models.embodiment.world_model.latent_pixel_decoder import (
    LatentTokenPixelDecoder,
    latent_pixel_reconstruction_loss,
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


def test_reconstruction_loss_backpropagates_and_reports_pixel_metrics() -> None:
    decoder = _decoder()
    prediction = decoder(torch.randn(2, 48, 32))
    losses = latent_pixel_reconstruction_loss(prediction, torch.rand_like(prediction))
    assert set(losses) == {"loss", "l1", "ssim", "psnr"}
    assert losses["loss"].requires_grad
    losses["loss"].backward()
    assert decoder.token_proj.weight.grad is not None


def test_decoder_rejects_non_square_view_tokens() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        LatentTokenPixelDecoder(token_dim=8, token_count=20, tokens_per_view=10)


def test_pi05_pixel_decoder_config_validates_for_two_gpu_ddp() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=["experiment=pi05_pixel_decoder"])
    validate_cfg(cfg, world_size=2)
    assert cfg.pixel_decoder.token_count == 768
    assert cfg.pixel_decoder.token_dim == 2048
    assert list(cfg.pixel_decoder.view_indices) == [0, 1]
    assert list(cfg.decoder_training.target_image_keys) == [
        "base_0_rgb",
        "left_wrist_0_rgb",
    ]
