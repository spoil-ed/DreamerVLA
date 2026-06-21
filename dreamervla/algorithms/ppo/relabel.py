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
    if not real_relabel_batch:
        return None, dict(_ZERO_METRICS)
    hidden = real_relabel_batch.get("hidden")
    action = real_relabel_batch.get("action")
    old_log_prob = real_relabel_batch.get("old_log_prob")
    advantage = real_relabel_batch.get("advantage")
    weight = real_relabel_batch.get("weight")
    if not all(
        isinstance(x, torch.Tensor)
        for x in (hidden, action, old_log_prob, advantage, weight)
    ):
        return None, dict(_ZERO_METRICS)
    if hidden.numel() == 0:
        return None, dict(_ZERO_METRICS)

    log_prob, _entropy, _extra = policy(
        {
            "mode": "evaluate",
            "hidden": hidden.float(),
            "action": action.float(),
        }
    )
    advantage = advantage.to(device=log_prob.device, dtype=log_prob.dtype)
    old_log_prob = old_log_prob.to(device=log_prob.device, dtype=log_prob.dtype)
    weight = weight.to(device=log_prob.device, dtype=log_prob.dtype).clamp_min(0.0)
    ratio = _ppo_ratio(log_prob, old_log_prob, clip_log_ratio=clip_log_ratio)
    per_item = _ppo_clip_term(
        ratio, advantage, clip_low, clip_high, clip_ratio_c=clip_ratio_c
    )
    denom = weight.sum().clamp_min(1.0)
    loss = (per_item * weight).sum() / denom
    clipfrac = (
        ((ratio.detach() < 1.0 - clip_low) | (ratio.detach() > 1.0 + clip_high))
        .float()
        .mean()
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
