"""DreamerV3-style actor-critic imagination + WM pretrain step for DreamerVLA.

Phase-1 (`world_model_pretrain_step`) trains the WM on (obs, action, reward,
next_obs) tuples by routing through `world_model(batch).compute_loss_dict`.

Phase-2 (`imagine_actor_critic_step`) follows the public DreamerV3 training
loss: posterior starts are taken from the replay sequence, then an H-step
imagination rollout trains the actor and value head:

  • Critic is a *twohot* categorical over `symlog(value)` bins; the critic
    loss is −log_prob of the twohot target of `stop_grad(λ-returns)`.
  • A slow-updated *target critic* provides bootstrap values for λ-returns,
    refreshed every step by Polyak averaging (τ ≈ 0.02).
  • Returns are continuation-weighted λ-returns from predicted reward and
    predicted continuation, matching `dreamerv3/agent.py::imag_loss`.
  • Actor loss is `-logpi(stop_grad(action)) * stop_grad(adv)` plus the
    DreamerV3 action entropy term; imagined states are stop-gradient by
    default (`ac_grads: False` in the official config).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
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
    for key in (
        "obs_embedding", "next_obs_embedding", "action", "action_mask", "reward",
        "done",
        "next_obs_image_hiddens", "next_obs_image_token_ids",
        # DreamerV3 sequence WM batches.
        "images", "tokens", "actions", "rewards", "dones", "is_first",
        "is_terminal", "is_last",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            flat_batch[key] = value.to(device)
        elif value is not None:
            flat_batch[key] = value

    world_model.train()
    losses = world_model(flat_batch)
    loss_tensor = losses.get("_loss", losses.get("loss"))
    if not isinstance(loss_tensor, torch.Tensor):
        raise KeyError("world_model output must contain Tensor key 'loss' or '_loss'")

    optimizer.zero_grad(set_to_none=bool(optim_cfg.get("zero_grad_set_to_none", True)))
    loss_tensor.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        world_model.parameters(),
        max_norm=float(optim_cfg.get("grad_clip_norm", 1.0)),
    )
    optimizer.step()

    def _f(key: str, default: float = 0.0) -> float:
        v = losses.get(key)
        return float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(default)

    return {
        "loss": float(loss_tensor.detach().cpu()),
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
        "hidden_rec_loss": _f("hidden_rec_loss"),
        "hidden_rec_scaled_loss": _f("hidden_rec_scaled_loss"),
        "hidden_cosine_loss": _f("hidden_cosine_loss"),
        "full_hidden_rec_loss": _f("full_hidden_rec_loss"),
        "full_hidden_rec_scaled_loss": _f("full_hidden_rec_scaled_loss"),
        "full_hidden_cosine_loss": _f("full_hidden_cosine_loss"),
        "hidden_pred_norm": _f("hidden_pred_norm"),
        "hidden_target_norm": _f("hidden_target_norm"),
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


def _detach_latent(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if is_dataclass(value):
        return replace(
            value,
            **{field.name: _detach_latent(getattr(value, field.name)) for field in fields(value)}
        )
    if isinstance(value, dict):
        return {key: _detach_latent(item) for key, item in value.items()}
    return value


def _world_model_actor_input(world_model: nn.Module, latent: Any) -> torch.Tensor:
    try:
        return world_model({"mode": "actor_input", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "actor_input" not in message and "Unknown" not in message:
            raise
    return latent.feature()


def _world_model_actor_input_sequence(world_model: nn.Module, latent: Any) -> torch.Tensor:
    try:
        return world_model({"mode": "actor_input_sequence", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "actor_input_sequence" not in message and "Unknown" not in message:
            raise
    raise RuntimeError("Configured actor_input_mode=sequence, but world_model has no actor_input_sequence.")


def _world_model_critic_input(world_model: nn.Module, latent: Any) -> torch.Tensor:
    try:
        return world_model({"mode": "critic_input", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "critic_input" not in message and "Unknown" not in message:
            raise
    return _world_model_actor_input(world_model, latent)


def _latent_time_dim(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim < 2:
            raise ValueError(f"Expected latent sequence tensor with [B,T,...], got {tuple(value.shape)}")
        return int(value.shape[1])
    if is_dataclass(value):
        for field in fields(value):
            return _latent_time_dim(getattr(value, field.name))
    if isinstance(value, dict):
        for item in value.values():
            return _latent_time_dim(item)
    raise TypeError(f"Cannot infer latent sequence length from {type(value).__name__}")


def _flatten_last_steps(value: Any, steps: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim < 2:
            raise ValueError(f"Expected [B,T,...] tensor, got {tuple(value.shape)}")
        bsz = int(value.shape[0])
        sliced = value[:, -steps:]
        return sliced.reshape(bsz * steps, *value.shape[2:])
    if is_dataclass(value):
        return replace(
            value,
            **{field.name: _flatten_last_steps(getattr(value, field.name), steps) for field in fields(value)}
        )
    if isinstance(value, dict):
        return {key: _flatten_last_steps(item, steps) for key, item in value.items()}
    return value


def _world_model_observe_starts(world_model: nn.Module, obs: Mapping[str, Any], imag_last: int) -> Any:
    observed = world_model({"mode": "observe_sequence", **obs})
    if not isinstance(observed, Mapping) or "latent" not in observed:
        raise TypeError("world_model observe_sequence must return a mapping with key 'latent'")
    latent_seq = observed["latent"]
    seq_len = _latent_time_dim(latent_seq)
    starts = min(int(imag_last) if int(imag_last) > 0 else seq_len, seq_len)
    return _flatten_last_steps(latent_seq, starts)


def _world_model_state_reward(world_model: nn.Module, latent: Any) -> torch.Tensor:
    return world_model({"mode": "reward", "latent": latent})


def _world_model_continue(world_model: nn.Module, latent: Any, like: torch.Tensor) -> torch.Tensor:
    try:
        return world_model({"mode": "continue", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "continue" not in message and "Unknown" not in message:
            raise
    return torch.ones_like(like)


def compute_lambda_returns(
    rewards: torch.Tensor,  # [B,H+1]
    continues: torch.Tensor,  # [B,H+1]
    values: torch.Tensor,  # [B,H+1]
    boot: torch.Tensor,  # [B,H+1]
    disc: float,
    lam: float,
) -> torch.Tensor:  # [B,H]
    """DreamerV3 lambda return, matching dreamerv3/agent.py::lambda_return.

    Reward/continue/value are defined for the start state plus H imagined
    states. The return for action t uses reward/continue/value at t+1.
    """
    if not (rewards.shape == continues.shape == values.shape == boot.shape):
        raise ValueError(
            "lambda_return expects equal [B,H+1] shapes, got "
            f"{tuple(rewards.shape)}, {tuple(continues.shape)}, "
            f"{tuple(values.shape)}, {tuple(boot.shape)}"
        )
    live = continues[:, 1:] * float(disc)
    cont = torch.full_like(live, float(lam))
    interm = rewards[:, 1:] + (1.0 - cont) * live * boot[:, 1:]
    ret = boot[:, -1]
    returns: list[torch.Tensor] = []
    for idx in reversed(range(live.shape[1])):
        ret = interm[:, idx] + live[:, idx] * cont[:, idx] * ret
        returns.append(ret)
    returns.reverse()
    return torch.stack(returns, dim=1)


@dataclass
class ReturnNormalizationOutput:
    returns: torch.Tensor
    values: torch.Tensor
    low: float
    high: float
    scale: float
    enabled: bool


def normalize_returns_for_actor_critic(
    returns: torch.Tensor,
    values: torch.Tensor,
    algorithm_cfg: DictConfig,
) -> ReturnNormalizationOutput:
    norm_cfg = algorithm_cfg.get("return_normalization", None)
    mode = "none" if norm_cfg is None else str(norm_cfg.get("mode", "none")).lower()
    if mode in {"none", "identity", "off", "false", "0"}:
        return ReturnNormalizationOutput(
            returns=returns,
            values=values,
            low=float("nan"),
            high=float("nan"),
            scale=1.0,
            enabled=False,
        )
    if mode not in {"minmax01", "percentile01"}:
        raise ValueError("algorithm.return_normalization.mode must be one of: none, minmax01")

    low_q = float(norm_cfg.get("low", 0.05))
    high_q = float(norm_cfg.get("high", 0.95))
    eps = float(norm_cfg.get("eps", 1.0e-6))
    flat = returns.detach().float().flatten()
    low = torch.quantile(flat, low_q).to(device=returns.device, dtype=returns.dtype)
    high = torch.quantile(flat, high_q).to(device=returns.device, dtype=returns.dtype)
    scale = (high - low).clamp_min(eps)

    def _map(x: torch.Tensor) -> torch.Tensor:
        return ((x - low) / scale).clamp(0.0, 1.0)

    return ReturnNormalizationOutput(
        returns=_map(returns),
        values=_map(values),
        low=float(low.detach().cpu()),
        high=float(high.detach().cpu()),
        scale=float(scale.detach().cpu()),
        enabled=True,
    )


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
    actor_input_mode = str(algorithm_cfg.get("actor_input_mode", "pooled")).lower()
    if actor_input_mode not in {"pooled", "sequence"}:
        raise ValueError("algorithm.actor_input_mode must be one of: pooled, sequence")
    lam = float(algorithm_cfg.lam)
    entropy_coef = float(algorithm_cfg.get("actent", algorithm_cfg.get("entropy_coef", 3.0e-4)))
    target_tau = float(algorithm_cfg.get("target_critic_tau", 0.02))
    imag_last = int(algorithm_cfg.get("imag_last", 0))
    env_horizon = int(algorithm_cfg.get("horizon", 333))
    contdisc = bool(algorithm_cfg.get("contdisc", True))
    slowtar = bool(algorithm_cfg.get("slowtar", False))
    slowreg = float(algorithm_cfg.get("slowreg", 1.0))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    zero_grad = bool(optim_cfg.get("zero_grad_set_to_none", True))
    disc = 1.0 if contdisc else 1.0 - 1.0 / float(env_horizon)

    world_model.eval()
    target_critic.eval()
    policy.train()
    critic.train()

    # ── 1. Posterior starts from a replay sequence, as in DreamerV3 ─────────
    obs = move_mapping_to_device(obs, device)
    with torch.no_grad():
        current_latent = _detach_latent(_world_model_observe_starts(world_model, obs, imag_last))
        actor_input_ids = None
        actor_attention_mask = None
        if actor_input_mode == "sequence":
            if "actor_input_ids" not in obs or "actor_attention_mask" not in obs:
                raise RuntimeError(
                    "algorithm.actor_input_mode=sequence requires obs.actor_input_ids "
                    "and obs.actor_attention_mask from the dataset."
                )
            actor_seq_len = _latent_time_dim(obs["actor_input_ids"])
            actor_starts = min(int(imag_last) if int(imag_last) > 0 else actor_seq_len, actor_seq_len)
            actor_input_ids = _flatten_last_steps(obs["actor_input_ids"], actor_starts)
            actor_attention_mask = _flatten_last_steps(
                obs["actor_attention_mask"],
                actor_starts,
            )

    # ── 2. H-step imagination. DreamerV3 stops gradients through imagined
    # states by default (`ac_grads: False` in the official config) and trains
    # the actor with log-probability advantages.
    latents: list[Any] = [current_latent]
    entropies: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []

    with _temporarily_freeze(world_model):
        for _ in range(horizon):
            if actor_input_mode == "sequence":
                current_actor_seq = _world_model_actor_input_sequence(world_model, current_latent).detach().float()
                action, _, extra = policy({
                    "mode": "sample",
                    "hidden_states": current_actor_seq,
                    "input_ids": actor_input_ids,
                    "attention_mask": actor_attention_mask,
                    "deterministic": False,
                })
            else:
                current_actor_feat = _world_model_actor_input(world_model, current_latent).detach().float()
                action, _, extra = policy({"mode": "sample", "hidden": current_actor_feat, "deterministic": False})
            if extra.get("std") is not None:
                dist = Normal(extra["mean"], extra["std"])
                log_probs.append(dist.log_prob(action.detach()).sum(dim=-1))
                entropies.append(dist.entropy().sum(dim=-1))

            with torch.no_grad():
                next_latent = world_model({
                    "mode": "predict_next", "latent": current_latent, "actions": action.detach(),
                })
                current_latent = _detach_latent(next_latent)
                latents.append(current_latent)

        if not log_probs:
            raise RuntimeError("DreamerV3 actor loss requires stochastic policy log_probs.")

        with torch.no_grad():
            critic_feat_stack = torch.stack(
                [_world_model_critic_input(world_model, latent).detach().float() for latent in latents],
                dim=1,
            )  # [B*K,H+1,D]
            reward_stack = torch.stack(
                [_world_model_state_reward(world_model, latent).detach().float() for latent in latents],
                dim=1,
            )
            continue_stack = torch.stack(
                [
                    _world_model_continue(world_model, latent, reward_stack[:, idx]).detach().float()
                    for idx, latent in enumerate(latents)
                ],
                dim=1,
            ).clamp(0.0, 1.0)

            BKHp1, Hp1, D = critic_feat_stack.shape
            flat_feats = critic_feat_stack.reshape(BKHp1 * Hp1, D)
            critic_values = critic({"mode": "value", "hidden": flat_feats}).view(BKHp1, Hp1)
            slow_values = target_critic({"mode": "value", "hidden": flat_feats}).view(BKHp1, Hp1)
            target_values = slow_values if slowtar else critic_values
            returns = compute_lambda_returns(
                reward_stack,
                continue_stack,
                target_values,
                target_values,
                disc=disc,
                lam=lam,
            )

        raw_returns = returns
        raw_baseline_values = target_values[:, :-1]
        normalized = normalize_returns_for_actor_critic(raw_returns, raw_baseline_values, algorithm_cfg)
        returns = normalized.returns
        baseline_values = normalized.values

        return_tracker.update(returns)
        scale = 1.0 if normalized.enabled else return_tracker.scale()
        scale_tensor = torch.as_tensor(scale, device=device, dtype=returns.dtype).clamp_min(1.0)
        weights = torch.cumprod(float(disc) * continue_stack, dim=1) / float(disc)
        weights = weights[:, :-1]
        advantages = (returns - baseline_values) / scale_tensor

        log_prob_stack = torch.stack(log_probs, dim=1)              # [B*K,H]
        entropy_stack = torch.stack(entropies, dim=1) if entropies else torch.zeros_like(log_prob_stack)
        actor_loss = (weights.detach() * -(
            log_prob_stack * advantages.detach() + entropy_coef * entropy_stack
        )).mean()

        actor_optimizer.zero_grad(set_to_none=zero_grad)
        actor_loss.backward()
        actor_adapter_grad_norm = _named_grad_norm(policy, "adapter")
        actor_action_head_grad_norm = _named_grad_norm(policy, "action")
        actor_policy_head_grad_norm = _named_grad_norm(policy, "policy_head")
        actor_log_std_grad_norm = _named_grad_norm(policy, "log_std")
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
        actor_optimizer.step()

    # ── 3. Value loss, matching DreamerV3 value target from imagined returns.
    feat_stack_h = critic_feat_stack[:, :-1].detach()                # [B*K,H,D]
    B2, H, D2 = feat_stack_h.shape
    log_probs_critic = critic({
        "mode": "log_prob",
        "hidden": feat_stack_h.reshape(B2 * H, D2),
        "values": returns.detach().reshape(B2 * H),
    })
    value_loss = -log_probs_critic.view(B2, H)
    if slowreg > 0:
        with torch.no_grad():
            slow_targets = target_critic({
                "mode": "value",
                "hidden": feat_stack_h.reshape(B2 * H, D2),
            }).view(B2, H)
        slow_log_probs = critic({
            "mode": "log_prob",
            "hidden": feat_stack_h.reshape(B2 * H, D2),
            "values": slow_targets.detach().reshape(B2 * H),
        })
        value_loss = value_loss + slowreg * (-slow_log_probs.view(B2, H))
    critic_loss = (weights.detach() * value_loss).mean()

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
        "raw_returns_mean": float(raw_returns.detach().mean().cpu()),
        "raw_returns_std": float(raw_returns.detach().std().cpu()),
        "return_norm_enabled": float(normalized.enabled),
        "return_norm_low": normalized.low,
        "return_norm_high": normalized.high,
        "return_norm_scale": normalized.scale,
        "return_scale": float(scale),
        "reward_mean": float(reward_stack[:, 1:].detach().mean().cpu()),
        "continue_mean": float(continue_stack[:, 1:].detach().mean().cpu()),
        "value_mean": float(critic_values[:, :-1].detach().mean().cpu()),
        "imagine_weight_mean": float(weights.detach().mean().cpu()),
        "actor_grad_norm": float(torch.as_tensor(actor_grad_norm).detach().cpu()),
        "actor_grad_norm_adapter": actor_adapter_grad_norm,
        "actor_grad_norm_action_head": actor_action_head_grad_norm,
        "actor_grad_norm_policy_head": actor_policy_head_grad_norm,
        "actor_grad_norm_log_std": actor_log_std_grad_norm,
        "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).detach().cpu()),
    }


__all__ = [
    "compute_lambda_returns",
    "normalize_returns_for_actor_critic",
    "imagine_actor_critic_step",
    "world_model_pretrain_step",
]
