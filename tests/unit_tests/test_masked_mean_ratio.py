"""outcome-norm flip: RLinf masked_mean_ratio per-rollout normalization."""

from __future__ import annotations

import torch

from dreamervla.algorithms.ppo.grpo import masked_mean_ratio_chunk_term


def _global_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The OLD outcome normalization: global per-(chunk, rollout) masked mean."""
    return (values * mask).sum() / mask.sum()


def test_masked_mean_ratio_weights_rollouts_equally() -> None:
    # rollout 0: 3 valid chunks (value 2.0 each); rollout 1: 1 valid chunk (8.0).
    values = torch.tensor([[2.0, 8.0], [2.0, 0.0], [2.0, 0.0]])  # [num_chunks, B_eff]
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    num_chunks, b_eff = values.shape
    per_rollout_count = mask.sum(dim=0).clamp(min=1.0)

    total = sum(
        masked_mean_ratio_chunk_term(values[c], mask[c], per_rollout_count, b_eff)
        for c in range(num_chunks)
    )

    # per-rollout means: rollout0 = 2.0, rollout1 = 8.0 -> equal-weight mean = 5.0.
    assert torch.allclose(total, torch.tensor(5.0))
    # The OLD global masked mean over-weights the long rollout: (2+2+2+8)/4 = 3.5.
    assert torch.allclose(_global_masked_mean(values, mask), torch.tensor(3.5))


def test_masked_mean_ratio_matches_rlinf_formula() -> None:
    # RLinf: (values / loss_mask_ratio * mask).mean(), ratio = valid_count/num_chunks.
    values = torch.tensor([[1.0, -3.0, 2.0], [4.0, 0.0, 0.0]]).T  # [num_chunks=3, B=2]
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]).T
    num_chunks, b_eff = values.shape
    per_rollout_count = mask.sum(dim=0).clamp(min=1.0)
    loss_mask_ratio = per_rollout_count / num_chunks

    rlinf = (values / loss_mask_ratio * mask).mean()
    ours = sum(
        masked_mean_ratio_chunk_term(values[c], mask[c], per_rollout_count, b_eff)
        for c in range(num_chunks)
    )
    assert torch.allclose(ours, rlinf)


def test_masked_mean_ratio_empty_rollout_contributes_zero() -> None:
    # A fully masked rollout (clamped count) must contribute 0, not NaN.
    values = torch.tensor([[5.0, 9.0]])  # [num_chunks=1, B=2]
    mask = torch.tensor([[1.0, 0.0]])
    per_rollout_count = mask.sum(dim=0).clamp(min=1.0)
    term = masked_mean_ratio_chunk_term(values[0], mask[0], per_rollout_count, 2)
    # rollout0 mean 5.0 weighted 1/B_eff=1/2 -> 2.5; rollout1 masked -> 0.
    assert torch.allclose(term, torch.tensor(2.5))
    assert torch.isfinite(term)
