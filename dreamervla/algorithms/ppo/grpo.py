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


def _slice_latent(value: Any, lo: int, hi: int) -> Any:
    """Slice a latent (Tensor or nested dict of Tensors) along the batch dim [lo:hi].

    Batch-dim companion to ``_repeat_latent``: lets the LUMOS step process
    the effective batch in group-aligned micro-batches without materializing the
    whole imagination on GPU. Slice boundaries must be multiples of group_size so
    each slice holds whole GRPO groups (the advantage is group-relative).
    """
    if isinstance(value, torch.Tensor):
        return value[int(lo) : int(hi)]
    if isinstance(value, dict):
        return {key: _slice_latent(item, lo, hi) for key, item in value.items()}
    raise TypeError(f"Unsupported latent type for slice: {type(value).__name__}")


def _entropy_coef(algorithm_cfg: Any) -> float:
    """The single PPO entropy coefficient, honored identically by every route.

    Reads ``actent`` first, then ``entropy_coef`` (compatibility alias), default 0.
    """
    return float(algorithm_cfg.get("actent", algorithm_cfg.get("entropy_coef", 0.0)))


def masked_mean_ratio_chunk_term(
    value_vec: torch.Tensor,  # [B_eff] this chunk's per-rollout values
    mask_c: torch.Tensor,  # [B_eff] this chunk's 0/1 validity
    per_rollout_count: torch.Tensor,  # [B_eff] each rollout's total valid-chunk count (>=1)
    b_eff: int,
) -> torch.Tensor:
    """One chunk's contribution to RLinf ``masked_mean_ratio`` over the
    ``[num_chunks, B_eff]`` outcome layout.

    Summed across all chunks this equals ``mean_over_rollouts(
    mean_over_valid_chunks(value))`` — every rollout weighted equally regardless
    of episode length, matching RLinf's ``(value / loss_mask_ratio * mask).mean()``
    (``loss_mask_ratio = valid_count / num_chunks``). Replaces the prior global
    per-(chunk, rollout) masked mean, which over-weighted long/failed rollouts.

    Computed per chunk so the caller can keep backpropagating chunk-by-chunk
    (the outcome route holds one chunk's graph at a time to bound memory).
    """
    return ((value_vec * mask_c) / per_rollout_count).sum() / float(b_eff)


def _ppo_ratio(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    *,
    clip_log_ratio: float | None = None,
) -> torch.Tensor:
    """Importance ratio ``exp(log_prob - old_log_prob)``.

    ``clip_log_ratio`` (default off) clamps the log-ratio to ``[-c, c]`` before
    ``exp`` for numerical stability (matches RLinf's log-ratio clamp); it is
    especially relevant here because callers sum log-probs over a trajectory.
    """
    log_ratio = log_prob - old_log_prob
    if clip_log_ratio is not None:
        log_ratio = log_ratio.clamp(-float(clip_log_ratio), float(clip_log_ratio))
    return torch.exp(log_ratio)


def _ppo_clip_term(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_low: float,
    clip_high: float,
    *,
    clip_ratio_c: float | None = None,
) -> torch.Tensor:
    """Per-element PPO clipped surrogate loss (negated, ready to minimize).

    Returns ``max(-adv * ratio, -adv * ratio_clipped)`` elementwise. When
    ``clip_ratio_c`` (>1, default off) is set, the loss is additionally capped at
    ``clip_ratio_c * |adv|`` (PPO dual-clip), bounding the negative-advantage /
    exploded-ratio case. The caller applies its own reduction.
    """
    ratio_clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    term = torch.maximum(-advantage * ratio, -advantage * ratio_clipped)
    if clip_ratio_c is not None:
        term = torch.minimum(term, float(clip_ratio_c) * advantage.abs())
    return term


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


def group_variance_mask(score: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    """Return a ``[N]`` float mask zeroing rollouts whose GRPO group has no
    return variance (all-success or all-fail). Their group-relative advantage is
    0 anyway, so masking them out of the loss is a pure compute/stability win.

    Mirrors the keep/skip decision in ``outcome.py::_adaptive_group_advantage_and_mask``
    for the fixed-width (non-adaptive) actor batch; ``group_size <= 1`` keeps
    everything.
    """
    g = int(group_size)
    n = int(score.numel())
    if g <= 1:
        return torch.ones_like(score)
    if n < g or n % g != 0:
        raise ValueError(
            f"group_variance_mask: numel={n} is not a positive multiple of group_size={g}."
        )
    groups = score.reshape(-1, g)
    has_var = groups.std(dim=1, unbiased=False) > float(eps)  # [n_groups]
    return has_var.to(score.dtype).repeat_interleave(g).reshape_as(score)


__all__ = [
    "_entropy_coef",
    "_group_advantage",
    "_ppo_clip_term",
    "_ppo_ratio",
    "_repeat_latent",
    "group_variance_mask",
]
