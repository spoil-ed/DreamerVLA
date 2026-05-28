"""Group-relative advantage and GRPO helpers shared by step + chunk PPO loops."""

from __future__ import annotations

from typing import Any

import torch


def _repeat_latent(value: Any, repeats: int) -> Any:
    """Replicate a latent (Tensor or nested dict of Tensors) along the batch dim.

    GRPO samples ``ppo_rollouts_per_start`` independent rollouts from each real
    starting frame; this duplicates the latent so that each rollout sees the
    same initial state.
    """
    if int(repeats) <= 1:
        return value
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(int(repeats), dim=0)
    if isinstance(value, dict):
        return {key: _repeat_latent(item, repeats) for key, item in value.items()}
    raise TypeError(f"Unsupported latent type for repeat: {type(value).__name__}")


def _group_advantage(score: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    """Group-relative z-score normalization of per-rollout returns (GRPO).

    Each group of ``group_size`` consecutive rollouts shares a common prompt;
    the advantage is ``(score - group_mean) / group_std``.

    Raises:
        ValueError: If ``score.numel()`` is not a positive multiple of
            ``group_size``.  GRPO requires the batch to be partitionable into
            equal-sized groups; silently falling back to global normalization
            (former behavior) hides DDP shard / filter / config bugs and makes
            the advantage scale incomparable across batches.  Set
            ``group_size=1`` if a global normalization is genuinely intended.
    """
    g = int(group_size)
    n = int(score.numel())
    if g <= 1:
        return (score - score.mean()) / score.std(unbiased=False).clamp_min(float(eps))
    if n < g or n % g != 0:
        raise ValueError(
            f"_group_advantage: score.numel()={n} is not a positive multiple of "
            f"group_size={g}. GRPO requires `B_eff = B * group_size`. Check "
            f"dataloader.batch_size, ppo_rollouts_per_start, and per-rank DDP "
            f"shard sizes."
        )
    groups = score.reshape(-1, g)
    mean = groups.mean(dim=1, keepdim=True)
    std = groups.std(dim=1, unbiased=False, keepdim=True).clamp_min(float(eps))
    return ((groups - mean) / std).reshape_as(score)


__all__ = ["_group_advantage", "_repeat_latent"]
