"""Regression tests for the GRPO group-advantage helper.

Pins the contract that ``_group_advantage`` raises on misaligned batches
(rather than silently falling back to global normalization) and that the
documented escape hatch ``group_size=1`` still produces a global z-score.
"""
from __future__ import annotations

import pytest
import torch

from src.algorithms.ppo.grpo import _group_advantage


def test_group_advantage_raises_on_misaligned_batch():
    score = torch.zeros(5)
    with pytest.raises(ValueError, match=r"not a positive multiple"):
        _group_advantage(score, group_size=4, eps=1e-6)


def test_group_advantage_raises_on_undersized_batch():
    score = torch.zeros(2)
    with pytest.raises(ValueError, match=r"not a positive multiple"):
        _group_advantage(score, group_size=4, eps=1e-6)


def test_group_advantage_group_size_one_global_normalization():
    score = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = _group_advantage(score, group_size=1, eps=1e-6)
    assert torch.isclose(out.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(out.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_group_advantage_aligned_batch_per_group_zscore():
    score = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = _group_advantage(score, group_size=2, eps=1e-6)
    # Group A (1, 2) → (-1, 1); Group B (3, 4) → (-1, 1)
    assert torch.allclose(out, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-6)
