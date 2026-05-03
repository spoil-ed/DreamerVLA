"""DreamerV3-style actor-critic imagination + WM pretrain step for DreamerVLA.

Phase-1 (`world_model_pretrain_step`) trains the WM on (obs, action, reward,
next_obs) tuples by routing through `world_model(batch).compute_loss_dict`.

Phase-2 (`imagine_actor_critic_step`) runs an H-step imagination rollout in
the WM latent space (Hafner et al. 2023, *DreamerV3*):

  • Critic is a *twohot* categorical over `symlog(value)` bins; the critic
    loss is −log_prob of the twohot target of `stop_grad(λ-returns)`.
  • A slow-updated *target critic* provides bootstrap values for λ-returns,
    refreshed every step by Polyak averaging (τ ≈ 0.02).
  • Actor advantages are normalised by a running percentile scale
    S = max(1, EMA(P95) − EMA(P5)) so the actor loss is well-conditioned
    across reward magnitudes (DreamerV3 §B.3).
  • Actor loss = −E[ discount · (λ-return / S) ] + η · H[π]   (dynamics
    back-prop through the WM reward head only — transition backbone is
    detached).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Normal

from src.models.critic.twohot_critic import ReturnPercentileTracker, soft_update
from src.utils.torch_utils import move_mapping_to_device


@contextmanager
def _temporarily_freeze(module: nn.Module):
    params = list(module.parameters())
    requires_grad = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, requires_grad):
            p.requires_grad_(flag)


def _named_grad_norm(module: nn.Module, name_fragment: str) -> float:
    total = torch.zeros((), device=next(module.parameters()).device)
    for name, param in module.named_parameters():
        if name_fragment not in name or param.grad is None:
            continue
        total = total + param.grad.detach().float().pow(2).sum()
    return float(total.sqrt().cpu())


def world_model_pretrain_step(
    policy: nn.Module,
    world_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    device: torch.device,
    optim_cfg: DictConfig,
) -> dict[str, float]:
    """Phase-1 WM update: dispatch through ``world_model(batch)`` (forward).

    Both ``TSSMWorldModel`` and ``TSSMWorldModelTransDreamer`` accept the same
    batch dict ``{obs_embedding, next_obs_embedding, action, reward}`` via
    their ``compute_loss_dict`` entry point, so the workspace can pick whichever
    WM class it wants. ``policy`` is unused here but kept in the signature for
    backwards-compat with callers.
    """
    del policy  # batch is already encoded by the workspace; nothing to do here.
    flat_batch: dict[str, Any] = {}
    for key in ("obs_embedding", "next_obs_embedding", "action", "action_mask",
                "reward", "next_obs_image_hiddens", "next_obs_image_token_ids"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            flat_batch[key] = value.to(device)
        elif value is not None:
            flat_batch[key] = value

    world_model.train()
    losses = world_model(flat_batch)

    optimizer.zero_grad(set_to_none=bool(optim_cfg.get("zero_grad_set_to_none", True)))
    losses["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        world_model.parameters(),
        max_norm=float(optim_cfg.get("grad_clip_norm", 1.0)),
    )
    optimizer.step()

    def _f(key: str, default: float = 0.0) -> float:
        v = losses.get(key)
        return float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(default)

    return {
        "loss": _f("loss"),
        "kl_loss": _f("kl_loss"),
        "dyn_kl": _f("dyn_kl"),
        "rep_kl": _f("rep_kl"),
        "transition_loss": _f("transition_loss"),
        "reward_loss": _f("reward_loss"),
        "delta_latent_loss": _f("delta_latent_loss"),
        "action_margin_loss": _f("action_margin_loss"),
        "image_recon_ce_loss": _f("image_recon_ce_loss"),
        "image_static_ce_loss": _f("image_static_ce_loss"),
        "image_dynamic_ce_loss": _f("image_dynamic_ce_loss"),
        "image_recon_mse_loss": _f("image_recon_mse_loss"),
        "image_decoder_loss": _f("image_decoder_loss"),
        "image_recon_accuracy": _f("image_recon_accuracy"),
        "image_static_accuracy": _f("image_static_accuracy"),
        "image_dynamic_accuracy": _f("image_dynamic_accuracy"),
        "image_dynamic_fraction": _f("image_dynamic_fraction"),
        "pred_entropy": _f("pred_entropy"),
        "pred_unique_tokens": _f("pred_unique_tokens"),
        "gt_unique_tokens": _f("gt_unique_tokens"),
        "predicted_reward_mean": _f("predicted_reward_mean"),
        "latent_norm": _f("latent_norm"),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
    }


def compute_lambda_returns(
    rewards: list[torch.Tensor],  # H tensors, each [B]
    values: list[torch.Tensor],   # H+1 tensors (includes bootstrap at H), each [B]
    gamma: float,
    lam: float,
) -> torch.Tensor:                # [H, B]
    """λ-return: G_t = r_t + γ[(1-λ)V(s_{t+1}) + λ G_{t+1}].

    Setting λ=1 recovers a pure discounted Monte-Carlo return bootstrapped by
    the last value estimate; setting λ=0 gives a 1-step TD return.
    """
    H = len(rewards)
    ret = values[H]           # bootstrap from the last critic estimate [B]
    returns: list[torch.Tensor] = []
    for t in reversed(range(H)):
        ret = rewards[t] + gamma * ((1.0 - lam) * values[t + 1] + lam * ret)
        returns.append(ret)
    returns.reverse()
    return torch.stack(returns, dim=0)  # [H, B]


def imagine_actor_critic_step(
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
    actor_loss_type = str(algorithm_cfg.get("actor_loss_type", "pathwise"))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    zero_grad = bool(optim_cfg.get("zero_grad_set_to_none", True))
    use_pg_actor_loss = actor_loss_type in {"dreamerv3_pg", "pg", "policy_gradient"}
    if actor_loss_type not in {"pathwise", "dreamerv3_pg", "pg", "policy_gradient"}:
        raise ValueError(f"Unknown actor_loss_type: {actor_loss_type!r}")

    world_model.eval()
    target_critic.eval()
    policy.train()
    critic.train()

    # ── 1. Initial latent (no grad back to encoder / posterior) ────────────
    obs = move_mapping_to_device(obs, device)
    hidden = obs["obs_embedding"]
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(
            f"imagine_actor_critic_step expects obs['obs_embedding'] to be a Tensor, "
            f"got {type(hidden).__name__}"
        )

    # FSDP only triggers all-gather on `__call__` / `forward`; route every
    # custom op through `module({'mode': ..., ...})` so sharded params get
    # gathered before the matmul and grads flow back through FSDP's backward
    # hook (reduce-scatter to the local shard).
    with torch.no_grad():
        initial_latent = world_model({"mode": "encode_latent", "hidden": hidden.detach()})
    # Float32 boundary: WM runs in bf16, policy/critic stay in fp32.
    current_feat = initial_latent.feature().detach().float()

    # ── 2. H-step imagination (grad flows through action → reward head) ────
    feats: list[torch.Tensor] = [current_feat]
    rewards: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []

    # Keep WM frozen until the actor backward has finished. FSDP registers
    # different backward hooks depending on `requires_grad` at forward time, so
    # restoring WM params before `actor_loss.backward()` can trip FSDP's hook
    # state assertion.
    with _temporarily_freeze(world_model):
        for _ in range(horizon):
            action, _, extra = policy({
                "mode": "sample", "hidden": current_feat, "deterministic": False,
            })
            if extra.get("std") is not None:
                dist = Normal(extra["mean"], extra["std"])
                if use_pg_actor_loss:
                    log_probs.append(dist.log_prob(action.detach()).sum(dim=-1))
                entropies.append(dist.entropy().sum(dim=-1))

            action_for_world = action.detach() if use_pg_actor_loss else action
            if use_pg_actor_loss:
                # DreamerV3-style actor update treats imagined returns as
                # stop-gradient advantages. The world model rollout only
                # produces scores, so avoid retaining its graph here.
                with torch.no_grad():
                    current_latent = world_model({"mode": "encode_latent", "hidden": current_feat})
                    next_latent = world_model({
                        "mode": "predict_next", "latent": current_latent, "actions": action_for_world,
                    })
                    next_feat = next_latent.feature().detach().float()
                    reward = world_model({
                        "mode": "reward", "latent": current_latent, "actions": action_for_world,
                        "next_latent": next_latent,
                    }).float()
            else:
                with torch.no_grad():
                    current_latent = world_model({"mode": "encode_latent", "hidden": current_feat})
                next_latent = world_model({
                    "mode": "predict_next", "latent": current_latent, "actions": action_for_world,
                })
                next_feat = next_latent.feature().detach().float()

                reward = world_model({
                    "mode": "reward", "latent": current_latent, "actions": action_for_world,
                    "next_latent": next_latent,
                }).float()
            rewards.append(reward)
            feats.append(next_feat)
            current_feat = next_feat

        # ── 3. Target-critic bootstrap values (stop-grad) ──────────────────
        with torch.no_grad():
            feat_stack_all = torch.stack(feats, dim=0)             # [H+1, B, D]
            Hp1, B, D = feat_stack_all.shape
            values_flat = target_critic(feat_stack_all.view(Hp1 * B, D)).view(Hp1, B)
            values = [values_flat[t] for t in range(Hp1)]

        # ── 4. λ-returns (grad through rewards → action → actor) ───────────
        returns = compute_lambda_returns(rewards, values, gamma, lam)  # [H, B]

        # ── 5. Percentile return normalisation ─────────────────────────────
        return_tracker.update(returns)
        scale = return_tracker.scale()
        scale_tensor = torch.as_tensor(scale, device=device, dtype=returns.dtype).clamp_min(1.0)

        # ── 6. Actor loss ──────────────────────────────────────────────────
        discount = torch.tensor(
            [gamma ** t for t in range(horizon)], device=device, dtype=returns.dtype,
        ).unsqueeze(-1)                                             # [H, 1]
        if use_pg_actor_loss:
            if not log_probs:
                raise RuntimeError("DreamerV3 PG actor loss requires stochastic policy log_probs.")
            log_prob_stack = torch.stack(log_probs, dim=0)           # [H, B]
            baseline = values_flat[:-1].detach()                    # [H, B]
            advantages = (returns.detach() - baseline) / scale_tensor
            actor_loss = -(discount * log_prob_stack * advantages.detach()).mean()
        else:
            actor_loss = -(discount * (returns / scale_tensor)).mean()
        if entropies:
            actor_loss = actor_loss - entropy_coef * torch.stack(entropies).mean()

        actor_optimizer.zero_grad(set_to_none=zero_grad)
        actor_loss.backward()
        actor_adapter_grad_norm = _named_grad_norm(policy, "adapter")
        actor_action_head_grad_norm = _named_grad_norm(policy, "action")
        actor_log_std_grad_norm = _named_grad_norm(policy, "log_std")
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
        actor_optimizer.step()

    # ── 7. Critic twohot loss against stop_grad(λ-returns) ─────────────────
    feat_stack_h = feat_stack_all[:-1]                             # [H, B, D]
    H, B2, D2 = feat_stack_h.shape
    log_probs_critic = critic({
        "mode": "log_prob",
        "hidden": feat_stack_h.view(H * B2, D2),
        "values": returns.detach().view(H * B2),
    })
    critic_loss = -log_probs_critic.mean()

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
        "actor_grad_norm_adapter": actor_adapter_grad_norm,
        "actor_grad_norm_action_head": actor_action_head_grad_norm,
        "actor_grad_norm_log_std": actor_log_std_grad_norm,
        "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).detach().cpu()),
    }


__all__ = [
    "compute_lambda_returns",
    "imagine_actor_critic_step",
    "world_model_pretrain_step",
]
