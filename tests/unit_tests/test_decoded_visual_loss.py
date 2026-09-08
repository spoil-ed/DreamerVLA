"""Frozen readout supervision must penalize wrong motion without target leakage."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dreamervla.models.embodiment.world_model.decoded_visual_loss import FrozenDecodedVisualLoss


class _Readout(nn.Module):
    tokens_per_view = 1
    view_indices = (0, 1)

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(tokens[..., :3] * self.scale)[..., None, None]


def _loss(**kwargs) -> FrozenDecodedVisualLoss:
    return FrozenDecodedVisualLoss(decoder=_Readout(), frame_stride=1, **kwargs)


def test_frozen_decoder_persistence_wrong_direction_and_exact_predictions() -> None:
    loss = _loss()
    anchor = torch.zeros(2, 1, 2, 3, requires_grad=True)
    target = torch.arange(1.0, 5.0)[None, :, None, None].expand(2, 4, 2, 3).requires_grad_()
    prediction = nn.Parameter(torch.zeros_like(target))
    result = loss(prediction, target, anchor)
    assert result["decoded_reconstruction_ratio"].item() == pytest.approx(1.0)
    assert result["decoded_temporal_error_ratio"].item() == pytest.approx(1.0)
    result["_loss"].backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.norm() > 0
    assert target.grad is None and anchor.grad is None
    assert all(p.grad is None and not p.requires_grad for p in loss.parameters())
    assert loss(target, target, anchor)["_loss"].item() == pytest.approx(0.0)
    assert loss(-target, target, anchor)["_loss"].item() > result["_loss"].item()
    loss.train()
    assert not loss.training and not loss.decoder.training and not loss.decoder.dropout.training


def test_static_targets_do_not_reward_hallucination_and_padding_has_no_gradient() -> None:
    loss = _loss()
    anchor = torch.zeros(1, 1, 2, 3)
    target = torch.zeros(1, 3, 2, 3)
    prediction = nn.Parameter(torch.ones_like(target))
    mask = torch.tensor([[[True, False]] * 3])
    result = loss(prediction, target, anchor, token_mask=mask, anchor_mask=mask[:, :1])
    assert result["_loss"].item() > 0 and torch.isfinite(result["_loss"])
    result["_loss"].backward()
    assert prediction.grad[:, :, 0].norm() > 0
    assert prediction.grad[:, :, 1].count_nonzero() == 0
    assert loss(target, target, anchor)["_loss"].item() == 0


def test_checkpointed_decode_preserves_gradients_and_roundtrip(tmp_path) -> None:
    first = _loss(gradient_checkpointing=True)
    second = _loss(gradient_checkpointing=False)
    p1 = torch.randn(2, 4, 2, 3, requires_grad=True)
    p2 = p1.detach().clone().requires_grad_()
    target, anchor = torch.randn_like(p1), torch.randn(2, 1, 2, 3)
    first(p1, target, anchor)["_loss"].backward()
    second(p2, target, anchor)["_loss"].backward()
    torch.testing.assert_close(p1.grad, p2.grad)
    path = tmp_path / "readout.pt"
    torch.save(first.state_dict(), path)
    restored = copy.deepcopy(second)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(
        restored(p1, target, anchor)["_loss"], first(p1, target, anchor)["_loss"]
    )


@pytest.mark.parametrize("stride", [2, 5, 10])
def test_final_frame_always_supervised(stride: int) -> None:
    loss = FrozenDecodedVisualLoss(decoder=_Readout(), frame_stride=stride)
    prediction = torch.zeros(1, 3, 2, 3, requires_grad=True)
    target = torch.ones_like(prediction)
    result = loss(prediction, target, torch.zeros(1, 1, 2, 3))
    result["_loss"].backward()
    assert prediction.grad[:, -1].norm() > 0


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
    loss = FrozenDecodedVisualLoss(checkpoint_path=str(path))
    for key, value in decoder.state_dict().items():
        torch.testing.assert_close(loss.decoder.state_dict()[key], value, rtol=0, atol=0)
    assert all(not p.requires_grad for p in loss.parameters())
    state = decoder.state_dict()
    state.pop("token_proj.weight")
    torch.save({"state_dicts": {"pixel_decoder": state}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        FrozenDecodedVisualLoss(checkpoint_path=str(path))
