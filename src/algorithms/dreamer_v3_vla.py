"""DreamerV3-style actor-critic imagination step for DreamerVLA.

Differences vs `dreamer_vla.imagine_actor_critic_step` (V1/V2 style):

  • Critic is a *twohot* categorical over `symlog(value)` bins; the critic loss
    is −log_prob of the twohot target of `stop_grad(λ-returns)`.
  • A slow-updated *target critic* provides bootstrap values for λ-returns.
    Updated every step by Polyak averaging (τ ≈ 0.02).
  • Actor advantages are normalised by a running percentile scale
    S = max(1, EMA(P95) − EMA(P5)) so the actor loss remains well-conditioned
    across reward magnitudes. (Hafner et al., 2023, §B.3.)
  • Actor loss = −E[ discount · (λ-return / S) ] + η · H[π]   (dynamics
    back-prop through the WM reward head, same as V1/V2 for continuous
    reparameterised actors).

The WM pretrain step is unchanged — re-use `dreamer_vla.world_model_pretrain_step`.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Normal

from src.algorithms.dreamer_vla import compute_lambda_returns, embed_observation
from src.models.critic.twohot_critic import ReturnPercentileTracker, soft_update
from src.utils.torch_utils import move_mapping_to_device


def imagine_actor_critic_step_v3(
    policy: nn.Module,
    world_model: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    return_tracker: ReturnPercentileTracker,
    obs: Mapping[str, Any],
    device: torch.device,
    algorithm_cfg: DictConfig,
    optim_cfg: DictConfig,
) -> dict[str, float]:
    """Single DreamerV3 actor-critic update over WM imagination.

    Expects `critic` and `target_critic` to be `TwohotCritic` instances exposing
    `forward(feat) -> expected_value` and `log_prob_of(feat, target_values)`.
    """
    horizon = int(algorithm_cfg.imagination_horizon)
    gamma = float(algorithm_cfg.gamma)
    lam = float(algorithm_cfg.lam)
    entropy_coef = float(algorithm_cfg.get("entropy_coef", 3.0e-4))
    target_tau = float(algorithm_cfg.get("target_critic_tau", 0.02))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    zero_grad = bool(optim_cfg.get("zero_grad_set_to_none", True))

    world_model.eval()
    target_critic.eval()
    policy.train()
    critic.train()

    # ── 1. Initial latent (no grad back to encoder / posterior) ────────────
    with torch.no_grad():
        hidden = embed_observation(policy, move_mapping_to_device(obs, device))
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach()
        initial_latent = world_model.encode_latent(hidden)
    current_feat = initial_latent.feature().detach()

    # ── 2. H-step imagination (grad flows through action → reward head) ────
    feats: list[torch.Tensor] = [current_feat]
    rewards: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for _ in range(horizon):
        action, _, extra = policy.sample_action_from_embedding(current_feat, deterministic=False)
        if extra.get("std") is not None:
            entropies.append(Normal(extra["mean"], extra["std"]).entropy().sum(dim=-1))

        with torch.no_grad():
            current_latent = world_model.encode_latent(current_feat)
            next_latent = world_model.predict_next(current_latent, action.detach())
            next_feat = next_latent.feature().detach()

        reward = world_model.reward(current_latent, action, next_latent)
        rewards.append(reward)
        feats.append(next_feat)
        current_feat = next_feat

    # ── 3. Target-critic bootstrap values (stop-grad) ──────────────────────
    with torch.no_grad():
        feat_stack_all = torch.stack(feats, dim=0)                # [H+1, B, D]
        Hp1, B, D = feat_stack_all.shape
        values_flat = target_critic(feat_stack_all.view(Hp1 * B, D)).view(Hp1, B)
        values = [values_flat[t] for t in range(Hp1)]

    # ── 4. λ-returns (grad through rewards → action → actor) ───────────────
    returns = compute_lambda_returns(rewards, values, gamma, lam)  # [H, B]

    # ── 5. Percentile return normalisation ─────────────────────────────────
    return_tracker.update(returns)
    scale = return_tracker.scale()
    scale_tensor = torch.as_tensor(scale, device=device, dtype=returns.dtype).clamp_min(1.0)

    # ── 6. Actor loss ──────────────────────────────────────────────────────
    discount = torch.tensor(
        [gamma ** t for t in range(horizon)], device=device, dtype=returns.dtype,
    ).unsqueeze(-1)                                                # [H, 1]
    actor_loss = -(discount * (returns / scale_tensor)).mean()
    if entropies:
        actor_loss = actor_loss - entropy_coef * torch.stack(entropies).mean()

    actor_optimizer.zero_grad(set_to_none=zero_grad)
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
    actor_optimizer.step()

    # ── 7. Critic twohot loss against stop_grad(λ-returns) ─────────────────
    feat_stack_h = feat_stack_all[:-1]                             # [H, B, D]
    H, B2, D2 = feat_stack_h.shape
    log_probs = critic.log_prob_of(feat_stack_h.view(H * B2, D2), returns.detach().view(H * B2))
    critic_loss = -log_probs.mean()

    critic_optimizer.zero_grad(set_to_none=zero_grad)
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip)
    critic_optimizer.step()

    # ── 8. Polyak-average target critic ────────────────────────────────────
    soft_update(target_critic, critic, tau=target_tau)

    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "returns_mean": float(returns.detach().mean().cpu()),
        "returns_std": float(returns.detach().std().cpu()),
        "return_scale": float(scale),
        "reward_mean": float(torch.stack(rewards).detach().mean().cpu()),
        "value_mean": float(values_flat[:-1].mean().cpu()),
        "actor_grad_norm": float(torch.as_tensor(actor_grad_norm).detach().cpu()),
        "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).detach().cpu()),
    }


__all__ = ["imagine_actor_critic_step_v3"]
