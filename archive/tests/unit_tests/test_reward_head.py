from __future__ import annotations

import torch

from dreamervla.models.world_model.dreamerv3_torch import BinaryRewardHead


def test_binary_reward_head_applies_positive_weight() -> None:
    head = BinaryRewardHead(in_dim=3, units=4, pos_weight=7.0)
    logits = torch.zeros(2, 1)
    target = torch.tensor([0.0, 1.0])

    loss = head.loss(logits, target)

    assert torch.allclose(loss[1], loss[0] * 7.0)
