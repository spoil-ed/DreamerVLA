from __future__ import annotations

import torch
from torch import Tensor


def compute_group_relative_advantages(
    scores: Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> Tensor:
    if group_size <= 0:
        raise ValueError("`group_size` must be positive.")

    # Group view
    original_shape = scores.shape
    if scores.ndim == 1:
        if scores.numel() % group_size != 0:
            raise ValueError("Flattened scores must be divisible by group_size.")
        grouped_scores = scores.view(-1, group_size)
    elif scores.ndim == 2 and scores.shape[1] == group_size:
        grouped_scores = scores
    else:
        raise ValueError("`scores` must have shape [batch * group_size] or [batch, group_size].")

    # Group stats
    group_mean = grouped_scores.mean(dim=1, keepdim=True)
    group_std = grouped_scores.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped_scores - group_mean) / (group_std + eps)
    return advantages.reshape(-1) if len(original_shape) == 1 else advantages


def compute_ppo_actor_loss(
    log_prob_new: Tensor,
    log_prob_old: Tensor,
    advantages: Tensor,
    clip_ratio: float,
    entropy: Tensor,
    entropy_coef: float,
    log_prob_ref: Tensor | None = None,
    kl_coef: float = 0.0,
) -> dict[str, Tensor]:
    # PPO ratio
    ratio = torch.exp(log_prob_new - log_prob_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Surrogate loss
    surrogate_unclipped = ratio * advantages
    surrogate_clipped = clipped_ratio * advantages
    surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)

    # Loss terms
    policy_loss = -surrogate.mean()
    entropy_bonus = entropy.mean()
    approx_kl_old = (log_prob_old - log_prob_new).mean()
    clip_fraction = ((ratio > 1.0 + clip_ratio) | (ratio < 1.0 - clip_ratio)).float().mean()

    if log_prob_ref is None:
        approx_kl_ref = policy_loss.new_zeros(())
    else:
        approx_kl_ref = (log_prob_new - log_prob_ref).mean()

    # Total loss
    total_loss = policy_loss - entropy_coef * entropy_bonus + kl_coef * approx_kl_ref
    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "entropy_bonus": entropy_bonus,
        "approx_kl_old": approx_kl_old,
        "approx_kl_ref": approx_kl_ref,
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio.mean(),
        "advantage_mean": advantages.mean(),
        "advantage_std": advantages.std(unbiased=False),
    }
