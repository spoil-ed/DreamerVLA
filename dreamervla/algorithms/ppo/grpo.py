"""Group-relative advantage and GRPO helpers shared by step + chunk PPO loops."""

from __future__ import annotations

import math
from typing import Any

import torch


def _repeat_latent(value: Any, repeats: int) -> Any:
    """Replicate a latent (Tensor or nested dict of Tensors) along the batch dim.

    GRPO samples ``ppo_rollouts_per_start`` independent rollouts from each real
    starting frame; this duplicates the latent so that each rollout sees the
    same initial state.
    """
    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError(f"repeats must be > 0, got {repeats!r}")
    if repeats == 1:
        return value
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
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
    lo = int(lo)
    hi = int(hi)
    if isinstance(value, torch.Tensor):
        batch = int(value.shape[0]) if value.ndim > 0 else 0
        if lo < 0 or hi <= lo or hi > batch:
            raise ValueError(
                f"slice bounds must satisfy 0 <= lo < hi <= batch ({batch}), got [{lo}:{hi}]"
            )
        return value[lo:hi]
    if isinstance(value, dict):
        return {key: _slice_latent(item, lo, hi) for key, item in value.items()}
    raise TypeError(f"Unsupported latent type for slice: {type(value).__name__}")


def _entropy_coef(algorithm_cfg: Any) -> float:
    """The single PPO entropy coefficient, honored identically by every route.

    Reads ``actent`` first, then ``entropy_coef`` (compatibility alias), default 0.
    """
    value = float(algorithm_cfg.get("actent", algorithm_cfg.get("entropy_coef", 0.0)))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"entropy coefficient must be finite and >= 0, got {value!r}")
    return value


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
    if int(b_eff) <= 0:
        raise ValueError(f"b_eff must be > 0, got {b_eff!r}")
    if (
        value_vec.ndim != 1
        or mask_c.shape != value_vec.shape
        or per_rollout_count.shape != value_vec.shape
    ):
        raise ValueError(
            "value_vec, mask_c, and per_rollout_count must have matching 1D shapes, got "
            f"{tuple(value_vec.shape)}, {tuple(mask_c.shape)}, {tuple(per_rollout_count.shape)}"
        )
    if int(value_vec.shape[0]) > int(b_eff):
        raise ValueError(
            f"vector length {int(value_vec.shape[0])} cannot exceed global b_eff={int(b_eff)}"
        )
    if not bool(torch.isfinite(value_vec).all()) or not bool(torch.isfinite(mask_c).all()):
        raise ValueError("value_vec and mask_c must contain only finite values")
    if not bool(torch.isfinite(per_rollout_count).all()) or bool((per_rollout_count <= 0).any()):
        raise ValueError("per_rollout_count must contain only finite positive values")
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
    if log_prob.shape != old_log_prob.shape:
        raise ValueError(
            "log_prob and old_log_prob must have matching shapes, got "
            f"{tuple(log_prob.shape)} and {tuple(old_log_prob.shape)}"
        )
    if not bool(torch.isfinite(log_prob).all()) or not bool(torch.isfinite(old_log_prob).all()):
        raise ValueError("log_prob and old_log_prob must contain only finite values")
    if clip_log_ratio is not None and (
        not math.isfinite(float(clip_log_ratio)) or float(clip_log_ratio) <= 0.0
    ):
        raise ValueError(
            f"clip_log_ratio must be finite and > 0 when configured, got {clip_log_ratio!r}"
        )
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
    broadcastable_advantage = (
        ratio.ndim == advantage.ndim
        and ratio.ndim > 0
        and int(ratio.shape[0]) == int(advantage.shape[0])
        and all(
            int(advantage_dim) in {1, int(ratio_dim)}
            for ratio_dim, advantage_dim in zip(ratio.shape[1:], advantage.shape[1:], strict=True)
        )
    )
    if not broadcastable_advantage:
        raise ValueError(
            "ratio and advantage must have matching shapes or explicit singleton "
            "advantage axes, got "
            f"{tuple(ratio.shape)} and {tuple(advantage.shape)}"
        )
    if not bool(torch.isfinite(ratio).all()) or not bool(torch.isfinite(advantage).all()):
        raise ValueError("ratio and advantage must contain only finite values")
    if not math.isfinite(float(clip_low)) or not 0.0 <= float(clip_low) < 1.0:
        raise ValueError(f"clip_low must be finite and in [0, 1), got {clip_low!r}")
    if not math.isfinite(float(clip_high)) or float(clip_high) < 0.0:
        raise ValueError(f"clip_high must be finite and >= 0, got {clip_high!r}")
    if clip_ratio_c is not None and (
        not math.isfinite(float(clip_ratio_c)) or float(clip_ratio_c) <= 1.0
    ):
        raise ValueError(f"clip_ratio_c must be finite and > 1, got {clip_ratio_c!r}")
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
    if n == 0:
        raise ValueError("_group_advantage: score must be non-empty")
    if not bool(torch.isfinite(score).all()):
        raise ValueError("_group_advantage: score must contain only finite values")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError(f"_group_advantage: eps must be finite and > 0, got {eps!r}")
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
    if n == 0:
        raise ValueError("group_variance_mask: score must be non-empty")
    if not bool(torch.isfinite(score).all()):
        raise ValueError("group_variance_mask: score must contain only finite values")
    if not math.isfinite(float(eps)) or float(eps) < 0.0:
        raise ValueError(f"group_variance_mask: eps must be finite and >= 0, got {eps!r}")
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
