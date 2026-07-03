"""Unit tests for shared GRPO helpers."""

from __future__ import annotations

import torch

from dreamervla.algorithms.ppo.grpo import group_variance_mask


def test_group_variance_mask_zeros_degenerate_groups() -> None:
    # group 0 = all-fail (no variance), group 1 = mixed (has variance)
    returns = torch.tensor([1.0, 1.0, 0.0, 1.0])
    mask = group_variance_mask(returns, group_size=2, eps=1e-6)
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_group_variance_mask_all_kept_when_group_size_one() -> None:
    returns = torch.tensor([1.0, 0.0, 1.0])
    mask = group_variance_mask(returns, group_size=1, eps=1e-6)
    assert mask.tolist() == [1.0, 1.0, 1.0]


def test_group_variance_mask_rejects_non_multiple() -> None:
    returns = torch.tensor([1.0, 0.0, 1.0])
    try:
        group_variance_mask(returns, group_size=2, eps=1e-6)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-multiple batch")
