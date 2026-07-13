"""DIAG-01: opt-in world-model state-dict loader."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dreamervla.runtime.online_utils import load_world_model_state_from_dict


class _TinyWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Canonical nested reward-head layout.
        self.reward_head = nn.Module()
        self.reward_head.net = nn.Module()
        self.reward_head.net.net = nn.Linear(4, 1)
        self.backbone = nn.Linear(3, 3)


def _state(*, mismatch: bool, old_reward_key: bool = False) -> dict[str, torch.Tensor]:
    reward_prefix = "reward_head.net" if old_reward_key else "reward_head.net.net"
    state = {
        f"module.{reward_prefix}.weight": torch.zeros(1, 4),
        f"module.{reward_prefix}.bias": torch.zeros(1),
        "module.backbone.bias": torch.zeros(3),
    }
    state["module.backbone.weight"] = (
        torch.zeros(5, 5) if mismatch else torch.zeros(3, 3)
    )
    return state


def test_exact_loader_strips_distributed_prefix() -> None:
    model = _TinyWM()
    missing, unexpected = load_world_model_state_from_dict(model, _state(mismatch=False))
    assert missing == []
    assert unexpected == []


def test_old_reward_key_is_rejected() -> None:
    model = _TinyWM()
    with pytest.raises(RuntimeError):
        load_world_model_state_from_dict(
            model,
            _state(mismatch=False, old_reward_key=True),
        )


def test_shape_mismatch_is_rejected() -> None:
    model = _TinyWM()
    with pytest.raises(RuntimeError):
        load_world_model_state_from_dict(
            model, _state(mismatch=True)
        )


def test_reset_reward_head_drops_reward_tensors() -> None:
    model = _TinyWM()
    missing, _ = load_world_model_state_from_dict(
        model, _state(mismatch=False), reset_reward_head=True
    )
    assert "reward_head.net.net.weight" in missing
    assert "backbone.bias" not in missing
