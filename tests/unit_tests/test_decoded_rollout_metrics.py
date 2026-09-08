"""Evaluation must detect wrong motion without any image-gradient path."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dreamervla.diagnostics.evaluation.decoded_rollout_metrics import DecodedRolloutMetrics


class _Readout(nn.Module):
    tokens_per_view = 1
    view_indices = (0, 1)

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(tokens[..., :3] * self.scale)[..., None, None]


def _loss(**kwargs) -> DecodedRolloutMetrics:
    return DecodedRolloutMetrics(decoder=_Readout(), frame_stride=1, **kwargs)


def test_frozen_decoder_persistence_wrong_direction_and_exact_predictions() -> None:
    loss = _loss()
    anchor = torch.zeros(2, 1, 2, 3, requires_grad=True)
    target = torch.arange(1.0, 5.0)[None, :, None, None].expand(2, 4, 2, 3).requires_grad_()
    prediction = nn.Parameter(torch.zeros_like(target))
    result = loss(prediction, target, anchor)
    assert result["decoded_reconstruction_ratio"].item() == pytest.approx(1.0)
    assert result["decoded_temporal_error_ratio"].item() == pytest.approx(1.0)
    assert "_loss" not in result
    assert all(not value.requires_grad for value in result.values())
    assert prediction.grad is None
    assert target.grad is None and anchor.grad is None
    assert all(p.grad is None and not p.requires_grad for p in loss.parameters())
    assert loss(target, target, anchor)["decoded_reconstruction_ratio"].item() == pytest.approx(0.0)
    assert (
        loss(-target, target, anchor)["decoded_reconstruction_ratio"].item()
        > result["decoded_reconstruction_ratio"].item()
    )
    loss.train()
    assert not loss.training and not loss.decoder.training and not loss.decoder.dropout.training


def test_static_targets_do_not_reward_hallucination_and_padding_has_no_gradient() -> None:
    loss = _loss()
    anchor = torch.zeros(1, 1, 2, 3)
    target = torch.zeros(1, 3, 2, 3)
    prediction = nn.Parameter(torch.ones_like(target))
    mask = torch.tensor([[[True, False]] * 3])
    result = loss(prediction, target, anchor, token_mask=mask, anchor_mask=mask[:, :1])
    assert result["decoded_reconstruction_ratio"].item() > 0 and torch.isfinite(
        result["decoded_reconstruction_ratio"]
    )
    assert prediction.grad is None
    prediction.data[:, :, 1] = 1000
    masked = loss(prediction, target, anchor, token_mask=mask, anchor_mask=mask[:, :1])
    torch.testing.assert_close(
        masked["decoded_reconstruction_ratio"], result["decoded_reconstruction_ratio"]
    )
    assert loss(target, target, anchor)["decoded_reconstruction_ratio"].item() == 0


def test_ongoing_motion_rejects_initial_jump_and_wrong_direction() -> None:
    loss = _loss(temporal_include_anchor=False)
    anchor = torch.zeros(1, 1, 2, 3)
    target = torch.arange(10.0, 14.0)[None, :, None, None].expand(1, 4, 2, 3)
    jump_then_static = target[:, :1].expand_as(target).clone().requires_grad_()
    result = loss(jump_then_static, target, anchor)
    assert result["decoded_motion_ratio"] > 0.98
    assert result["decoded_ongoing_motion_ratio"] == 0
    assert result["decoded_late_motion_ratio"] == 0
    assert result["decoded_temporal_error_ratio"] == pytest.approx(1.0)
    assert jump_then_static.grad is None
    assert all(not value.requires_grad for value in result.values())
    exact = loss(target, target, anchor)
    assert exact["decoded_ongoing_motion_ratio"] == pytest.approx(1.0)
    assert exact["decoded_ongoing_motion_cosine"] == pytest.approx(1.0)
    wrong = loss(-target, target, anchor)
    assert wrong["decoded_ongoing_motion_ratio"] == pytest.approx(1.0)
    assert wrong["decoded_ongoing_motion_cosine"] == pytest.approx(-1.0)
    assert wrong["decoded_ongoing_temporal_error_ratio"] == pytest.approx(4.0)


def test_ongoing_metrics_handle_empty_pairs_and_masks() -> None:
    anchor = torch.zeros(1, 1, 2, 3)
    target = torch.ones_like(anchor)
    result = _loss()(target, target, anchor)
    assert result["decoded_ongoing_valid_pairs"] == 0
    assert all(torch.isfinite(value) for value in result.values())
    with pytest.raises(ValueError, match="at least two"):
        _loss(temporal_include_anchor=False)(target, target, anchor)
    target = target.expand(1, 3, 2, 3)
    mask = torch.zeros(1, 3, 2, dtype=torch.bool)
    result = _loss(temporal_include_anchor=False)(
        target, target, anchor, token_mask=mask, anchor_mask=mask[:, :1]
    )
    assert all(torch.isfinite(value) for value in result.values())
    assert result["decoded_ongoing_valid_pairs"] == result["decoded_reconstruction_ratio"] == 0


def test_evaluation_preserves_no_grad_boundary_and_roundtrip(tmp_path) -> None:
    first = _loss()
    second = _loss()
    p1 = torch.randn(2, 4, 2, 3, requires_grad=True)
    p2 = p1.detach().clone().requires_grad_()
    target, anchor = torch.randn_like(p1), torch.randn(2, 1, 2, 3)
    assert not first(p1, target, anchor)["decoded_reconstruction_ratio"].requires_grad
    assert not second(p2, target, anchor)["decoded_reconstruction_ratio"].requires_grad
    assert p1.grad is p2.grad is None
    path = tmp_path / "readout.pt"
    torch.save(first.state_dict(), path)
    restored = copy.deepcopy(second)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(
        restored(p1, target, anchor)["decoded_reconstruction_ratio"],
        first(p1, target, anchor)["decoded_reconstruction_ratio"],
    )


@pytest.mark.parametrize("stride", [2, 5, 10])
def test_final_frame_always_measured(stride: int) -> None:
    loss = DecodedRolloutMetrics(decoder=_Readout(), frame_stride=stride)
    prediction = torch.zeros(1, 3, 2, 3, requires_grad=True)
    target = torch.ones_like(prediction)
    result = loss(prediction, target, torch.zeros(1, 1, 2, 3))
    assert result["decoded_reconstruction_ratio"] > 0
    assert prediction.grad is None


def test_checkpoint_loader_is_strict_and_restores_frozen_parameters(tmp_path) -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "pixel_decoder": {
                "_target_": "dreamervla.models.embodiment.world_model.latent_pixel_decoder.LatentTokenPixelDecoder",
                "token_dim": 4,
                "token_count": 2,
                "tokens_per_view": 1,
                "view_indices": [0, 1],
                "image_size": 2,
                "base_channels": 32,
                "channel_schedule": [32],
                "upsample_sizes": [2],
            }
        }
    )
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(config, tmp_path / ".hydra" / "config.yaml")
    decoder = instantiate(config.pixel_decoder)
    path = tmp_path / "decoder.ckpt"
    torch.save({"state_dicts": {"pixel_decoder": decoder.state_dict()}}, path)
    loss = DecodedRolloutMetrics(checkpoint_path=str(path))
    for key, value in decoder.state_dict().items():
        torch.testing.assert_close(loss.decoder.state_dict()[key], value, rtol=0, atol=0)
    assert all(not p.requires_grad for p in loss.parameters())
    state = decoder.state_dict()
    state.pop("token_proj.weight")
    torch.save({"state_dicts": {"pixel_decoder": state}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        DecodedRolloutMetrics(checkpoint_path=str(path))
