"""DIAG-01: opt-in world-model state-dict loader."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dreamervla.runners.online_utils import load_world_model_state_from_dict


class _TinyWM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Current layout: reward_head.net.net.* (legacy ckpts use reward_head.net.*).
        self.reward_head = nn.Module()
        self.reward_head.net = nn.Module()
        self.reward_head.net.net = nn.Linear(4, 1)
        self.backbone = nn.Linear(3, 3)


def _legacy_state(*, mismatch: bool) -> dict[str, torch.Tensor]:
    state = {
        "module.reward_head.net.weight": torch.zeros(1, 4),
        "module.reward_head.net.bias": torch.zeros(1),
        "module.backbone.bias": torch.zeros(3),
    }
    state["module.backbone.weight"] = (
        torch.zeros(5, 5) if mismatch else torch.zeros(3, 3)
    )
    return state


def test_remap_and_skip_on_default() -> None:
    model = _TinyWM()
    missing, _ = load_world_model_state_from_dict(model, _legacy_state(mismatch=True))
    # module. stripped + reward head remapped -> loaded (not missing).
    assert "reward_head.net.net.weight" not in missing
    # shape-mismatched backbone.weight skipped -> reported missing.
    assert "backbone.weight" in missing
    assert "backbone.bias" not in missing


def test_remap_off_leaves_legacy_key_unmapped() -> None:
    model = _TinyWM()
    missing, unexpected = load_world_model_state_from_dict(
        model, _legacy_state(mismatch=False), remap_reward_head=False
    )
    # Without remap the legacy key stays reward_head.net.weight -> unexpected;
    # the model's reward_head.net.net.weight goes unfilled -> missing.
    assert "reward_head.net.weight" in unexpected
    assert "reward_head.net.net.weight" in missing


def test_skip_off_raises_on_shape_mismatch() -> None:
    model = _TinyWM()
    with pytest.raises(RuntimeError):
        load_world_model_state_from_dict(
            model, _legacy_state(mismatch=True), skip_shape_mismatch=False
        )


def test_reset_reward_head_drops_reward_tensors() -> None:
    model = _TinyWM()
    missing, _ = load_world_model_state_from_dict(
        model, _legacy_state(mismatch=False), reset_reward_head=True
    )
    assert "reward_head.net.net.weight" in missing
    assert "backbone.bias" not in missing
