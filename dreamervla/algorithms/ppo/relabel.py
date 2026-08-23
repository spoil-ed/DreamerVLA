"""Real-rollout relabel PPO loss — replays cached (hidden, action, old_lp, advantage)
tuples from real env interactions through the current policy so the actor sees
gradient signal from real trajectories alongside imagined ones.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from dreamervla.algorithms.ppo.grpo import _ppo_clip_term, _ppo_ratio

_ZERO_METRICS = {
    "real_relabel_applied": 0.0,
    "real_relabel_loss": 0.0,
    "real_relabel_ratio_mean": 1.0,
    "real_relabel_clipfrac": 0.0,
    "real_relabel_advantage_mean": 0.0,
    "real_relabel_weight_mean": 0.0,
}


def _real_relabel_anchor_loss(
    policy: nn.Module,
    real_relabel_batch: Mapping[str, torch.Tensor] | None,
    clip_low: float,
    clip_high: float,
    *,
    clip_log_ratio: float | None = None,
    clip_ratio_c: float | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if real_relabel_batch is None:
        return None, dict(_ZERO_METRICS)
    fields: dict[str, torch.Tensor] = {}
    for key in ("hidden", "action", "old_log_prob", "advantage", "weight"):
        if key not in real_relabel_batch:
            raise KeyError(f"real_relabel_batch is missing required field {key!r}")
        value = real_relabel_batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"real_relabel_batch.{key} must be a Tensor, got {type(value).__name__}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"real_relabel_batch.{key} must contain only finite values")
        fields[key] = value

    hidden = fields["hidden"]
    action = fields["action"]
    if hidden.ndim < 2 or hidden.shape[0] <= 0:
        raise ValueError(
            f"real_relabel_batch.hidden must be non-empty [B,...], got {tuple(hidden.shape)}"
        )
    batch = int(hidden.shape[0])
    if action.ndim < 2 or int(action.shape[0]) != batch:
        raise ValueError(
            f"real_relabel_batch.action must have batch={batch}, got {tuple(action.shape)}"
        )
    scalars: dict[str, torch.Tensor] = {}
    for key in ("old_log_prob", "advantage", "weight"):
        value = fields[key]
        if value.numel() != batch:
            raise ValueError(
                f"real_relabel_batch.{key} must contain batch={batch} values, "
                f"got shape {tuple(value.shape)}"
            )
        scalars[key] = value.reshape(batch)
    old_log_prob = scalars["old_log_prob"]
    advantage = scalars["advantage"]
    weight = scalars["weight"]
    if bool((weight < 0).any()):
        raise ValueError("real_relabel_batch.weight must be non-negative")
    if not bool((weight > 0).any()):
        return None, dict(_ZERO_METRICS)

    log_prob, _entropy, _extra = policy(
        {
            "mode": "evaluate",
            "hidden": hidden.float(),
            "action": action.float(),
        }
    )
    if not isinstance(log_prob, torch.Tensor) or log_prob.numel() != batch:
        shape = tuple(log_prob.shape) if isinstance(log_prob, torch.Tensor) else None
        raise ValueError(f"policy evaluate log_prob must contain batch={batch} values, got {shape}")
    log_prob = log_prob.reshape(batch)
    if not bool(torch.isfinite(log_prob).all()):
        raise ValueError("policy evaluate log_prob must contain only finite values")
    advantage = advantage.to(device=log_prob.device, dtype=log_prob.dtype)
    old_log_prob = old_log_prob.to(device=log_prob.device, dtype=log_prob.dtype)
    weight = weight.to(device=log_prob.device, dtype=log_prob.dtype)
    ratio = _ppo_ratio(log_prob, old_log_prob, clip_log_ratio=clip_log_ratio)
    per_item = _ppo_clip_term(ratio, advantage, clip_low, clip_high, clip_ratio_c=clip_ratio_c)
    denom = weight.sum()
    loss = (per_item * weight).sum() / denom
    clipfrac = (
        ((ratio.detach() < 1.0 - clip_low) | (ratio.detach() > 1.0 + clip_high)).float().mean()
    )
    metrics = {
        "real_relabel_applied": 1.0,
        "real_relabel_loss": float(loss.detach().cpu()),
        "real_relabel_ratio_mean": float(ratio.detach().mean().cpu()),
        "real_relabel_ratio_min": float(ratio.detach().min().cpu()),
        "real_relabel_ratio_max": float(ratio.detach().max().cpu()),
        "real_relabel_clipfrac": float(clipfrac.cpu()),
        "real_relabel_advantage_mean": float(advantage.detach().mean().cpu()),
        "real_relabel_advantage_abs_mean": float(advantage.detach().abs().mean().cpu()),
        "real_relabel_weight_mean": float(weight.detach().mean().cpu()),
    }
    return loss, metrics


__all__ = ["_real_relabel_anchor_loss"]
