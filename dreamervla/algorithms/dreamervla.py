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
  • Returns are raw continuation-weighted λ-returns from predicted reward and
    predicted continuation, matching `dreamerv3/agent.py::imag_loss`; the
    return normalizer scales the actor advantage, not the critic target.
  • Actor loss is `-logpi(stop_grad(action)) * stop_grad(adv)` plus the
    DreamerV3 action entropy term; imagined states are stop-gradient by
    default (`ac_grads: False` in the official config).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Normal

from dreamervla.algorithms.critic.twohot_critic import ReturnPercentileTracker
from dreamervla.utils.polyak import soft_update
from dreamervla.utils.torch_utils import autocast_context, move_mapping_to_device
from dreamervla.utils.update_timing import GradientUpdateTimer


@contextmanager
def _temporarily_freeze(module: nn.Module):
    params = list(module.parameters())
    requires_grad = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, requires_grad, strict=True):
            p.requires_grad_(flag)


def _manual_autocast_context(
    optim_cfg: Mapping[str, Any],
    device: torch.device,
):
    precision = str(optim_cfg.get("precision", optim_cfg.get("dtype", "fp32")))
    try:
        return autocast_context(device, precision)
    except ValueError as exc:
        raise ValueError(
            f"optim.precision must be one of fp32, bf16, or fp16; got {precision!r}"
        ) from exc


def _named_grad_norm(module: nn.Module, name_fragment: str) -> float:
    total = torch.zeros((), device=next(module.parameters()).device)
    for name, param in module.named_parameters():
        if name_fragment not in name or param.grad is None:
            continue
        total = total + param.grad.detach().float().pow(2).sum()
    return float(total.sqrt().cpu())


_WM_LOG_METRIC_KEYS = (
    "loss",
    "hidden_rec_loss",
    "hidden_mse",
    "next_latent_mse",
    "hidden_cosine_loss",
    "one_step_cosine_similarity",
    "persistence_cosine_similarity",
    "chunk_cosine_similarity",
    "rollout_cosine_similarity",
    "full_hidden_rec_loss",
    "full_hidden_cosine_loss",
    "proprio_reconstruction_loss",
    "proprio_pred_norm",
    "proprio_target_norm",
    "reward_loss",
    "reward_pred_mean",
    "reward_target_mean",
    "reward_binary_acc",
    "rollout_loss",
    "rollout_mse",
    "rollout_cosine_loss",
    "rollout_chunks",
    "hidden_pred_norm",
    "hidden_target_norm",
    "grad_norm",
)


def namespaced_world_model_metrics(raw: Mapping[str, Any]) -> dict[str, float]:
    """Return the public ``wm/*`` metric subset emitted by learner loops."""
    metrics: dict[str, float] = {}

    def _as_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    for key in _WM_LOG_METRIC_KEYS:
        if key not in raw:
            continue
        metrics[f"wm/{key}"] = _as_float(raw[key])
    if "wm/hidden_rec_loss" not in metrics:
        for alias in ("hidden_mse", "next_latent_mse"):
            if alias in raw:
                metrics["wm/hidden_rec_loss"] = _as_float(raw[alias])
                break
    return metrics


def world_model_pretrain_step(
    policy: nn.Module,
    world_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    device: torch.device,
    optim_cfg: DictConfig,
    profile_timings: dict[str, float] | None = None,
    metrics_mode: str = "full",
) -> dict[str, Any]:
    """Phase-1 WM update: dispatch through ``world_model(batch)`` (forward).

    Retained WM implementations accept either embedding batches or DreamerV3
    sequence batches through their forward entry point. ``policy`` is unused
    here but kept in the signature for callers shared with the actor phase.
    """
    del policy  # batch is already encoded by the workspace; nothing to do here.

    resolved_metrics_mode = str(metrics_mode).lower()
    if resolved_metrics_mode not in {"full", "loss_only", "loss_tensor"}:
        raise ValueError("metrics_mode must be one of: full, loss_only, loss_tensor")
    timer = GradientUpdateTimer(device, enabled=profile_timings is not None)
    flat_batch: dict[str, Any] = {}
    with timer.device_stage("h2d"):
        for key in (
            "obs_embedding",
            "next_obs_embedding",
            "action",
            "action_mask",
            "reward",
            "done",
            "next_obs_image_hiddens",
            "next_obs_image_token_ids",
            # DreamerV3 sequence WM batches.
            "images",
            "tokens",
            "actions",
            "current_actions",
            "proprio",
            "lang_emb",
            "rewards",
            "dones",
            "is_first",
            "is_terminal",
            "is_last",
            "success_to_go",
            "return_to_go",
            "return_targets",
            "task_ids",
        ):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                flat_batch[key] = value.to(device, non_blocking=True)
            elif value is not None:
                flat_batch[key] = value

    world_model.train()
    optimizer.zero_grad(set_to_none=bool(optim_cfg.get("zero_grad_set_to_none", True)))
    with timer.device_stage("forward"):
        with _manual_autocast_context(optim_cfg, device):
            losses = world_model(flat_batch)
    loss_tensor = losses.get("_loss", losses.get("loss"))
    if not isinstance(loss_tensor, torch.Tensor):
        raise KeyError("world_model output must contain Tensor key 'loss' or '_loss'")

    with timer.device_stage("backward"):
        loss_tensor.backward()
    with timer.device_stage("grad_clip"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            world_model.parameters(),
            max_norm=float(optim_cfg.get("grad_clip_norm", 1.0)),
        )
    with timer.device_stage("optimizer"):
        optimizer.step()

    # CUDA events let all kernels run asynchronously. One synchronization here
    # resolves every profiled device stage without forcing a sync per boundary.
    timer.synchronize_device()

    def _f(key: str, default: float = 0.0) -> float:
        v = losses.get(key)
        return float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(default)

    with timer.wall_stage("metrics"):
        if resolved_metrics_mode == "loss_tensor":
            metrics = {
                "loss": loss_tensor.detach(),
                "grad_norm": torch.as_tensor(grad_norm).detach(),
            }
            for key in (
                "next_latent_mse",
                "hidden_cosine_loss",
                "hidden_pred_norm",
                "hidden_target_norm",
                "rollout_loss",
                "rollout_mse",
                "rollout_cosine_loss",
                "one_step_cosine_similarity",
                "persistence_cosine_similarity",
                "chunk_cosine_similarity",
                "rollout_cosine_similarity",
                "proprio_reconstruction_loss",
                "reward_loss",
            ):
                value = losses.get(key)
                if isinstance(value, torch.Tensor):
                    metrics[key] = value.detach()
        else:
            loss_value = float(loss_tensor.detach().cpu())
            grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu())
        if resolved_metrics_mode == "loss_only":
            metrics = {"loss": loss_value, "grad_norm": grad_norm_value}
        elif resolved_metrics_mode == "full":
            hidden_mse = _f("hidden_mse", _f("next_latent_mse"))
            next_latent_mse = _f("next_latent_mse", hidden_mse)
            hidden_rec_loss = _f("hidden_rec_loss", hidden_mse)
            metrics = {
                "loss": loss_value,
                "kl_loss": _f("kl_loss"),
                "dyn_kl": _f("dyn_kl"),
                "rep_kl": _f("rep_kl"),
                "transition_loss": _f("transition_loss"),
                "reward_loss": _f("reward_loss"),
                "reward_pred_mean": _f("reward_pred_mean"),
                "reward_target_mean": _f("reward_target_mean"),
                "reward_binary_acc": _f("reward_binary_acc"),
                "success_return_loss": _f("success_return_loss"),
                "success_return_pred_mean": _f("success_return_pred_mean"),
                "success_return_target_mean": _f("success_return_target_mean"),
                "success_return_mse": _f("success_return_mse"),
                "delta_latent_loss": _f("delta_latent_loss"),
                "action_margin_loss": _f("action_margin_loss"),
                "image_recon_ce_loss": _f("image_recon_ce_loss"),
                "image_static_ce_loss": _f("image_static_ce_loss"),
                "image_dynamic_ce_loss": _f("image_dynamic_ce_loss"),
                "image_recon_mse_loss": _f("image_recon_mse_loss"),
                "image_decoder_loss": _f("image_decoder_loss"),
                "image_recon_accuracy": _f("image_recon_accuracy"),
                "hidden_rec_loss": hidden_rec_loss,
                "hidden_mse": hidden_mse,
                "next_latent_mse": next_latent_mse,
                "hidden_rec_scaled_loss": _f("hidden_rec_scaled_loss"),
                "hidden_cosine_loss": _f("hidden_cosine_loss"),
                "one_step_cosine_similarity": _f("one_step_cosine_similarity"),
                "persistence_cosine_similarity": _f("persistence_cosine_similarity"),
                "chunk_cosine_similarity": _f("chunk_cosine_similarity"),
                "rollout_cosine_similarity": _f("rollout_cosine_similarity"),
                "full_hidden_rec_loss": _f("full_hidden_rec_loss"),
                "full_hidden_rec_scaled_loss": _f("full_hidden_rec_scaled_loss"),
                "full_hidden_cosine_loss": _f("full_hidden_cosine_loss"),
                "hidden_pred_norm": _f("hidden_pred_norm"),
                "hidden_target_norm": _f("hidden_target_norm"),
                "proprio_reconstruction_loss": _f("proprio_reconstruction_loss"),
                "proprio_pred_norm": _f("proprio_pred_norm"),
                "proprio_target_norm": _f("proprio_target_norm"),
                "image_static_accuracy": _f("image_static_accuracy"),
                "image_dynamic_accuracy": _f("image_dynamic_accuracy"),
                "image_dynamic_fraction": _f("image_dynamic_fraction"),
                "pred_entropy": _f("pred_entropy"),
                "pred_unique_tokens": _f("pred_unique_tokens"),
                "gt_unique_tokens": _f("gt_unique_tokens"),
                "predicted_reward_mean": _f("predicted_reward_mean"),
                "latent_norm": _f("latent_norm"),
                "grad_norm": grad_norm_value,
            }
    if profile_timings is not None:
        profile_timings.update(timer.finish())
    return metrics


def _detach_latent(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if is_dataclass(value):
        return replace(
            value,
            **{field.name: _detach_latent(getattr(value, field.name)) for field in fields(value)},
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
    raise RuntimeError(
        "Configured actor_input_mode=sequence, but world_model has no actor_input_sequence."
    )


def _world_model_critic_input(world_model: nn.Module, latent: Any) -> torch.Tensor:
    try:
        return world_model({"mode": "critic_input", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "critic_input" not in message and "Unknown" not in message:
            raise
    return _world_model_actor_input(world_model, latent)


def _world_model_success_return(world_model: nn.Module, latent: Any) -> torch.Tensor:
    try:
        return world_model({"mode": "success_return", "latent": latent})
    except ValueError as exc:
        message = str(exc)
        if "success_return" not in message and "Unknown" not in message:
            raise
    return world_model({"mode": "return", "latent": latent})


def _latent_time_dim(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim < 2:
            raise ValueError(
                f"Expected latent sequence tensor with [B,T,...], got {tuple(value.shape)}"
            )
        return int(value.shape[1])
    if is_dataclass(value):
        for field in fields(value):
            return _latent_time_dim(getattr(value, field.name))
    if isinstance(value, dict):
        for item in value.values():
            return _latent_time_dim(item)
    raise TypeError(f"Cannot infer latent sequence length from {type(value).__name__}")


def _latent_batch_dim(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim < 2:
            raise ValueError(
                f"Expected latent sequence tensor with [B,T,...], got {tuple(value.shape)}"
            )
        return int(value.shape[0])
    if is_dataclass(value):
        for field in fields(value):
            return _latent_batch_dim(getattr(value, field.name))
    if isinstance(value, dict):
        for item in value.values():
            return _latent_batch_dim(item)
    raise TypeError(f"Cannot infer latent batch size from {type(value).__name__}")


def _slice_last_steps(value: Any, steps: int, *, _key: str | None = None) -> Any:
    if isinstance(value, torch.Tensor):
        if _key == "lang" and value.ndim == 2:
            return value
        if value.ndim < 2:
            raise ValueError(f"Expected [B,T,...] tensor, got {tuple(value.shape)}")
        return value[:, -steps:]
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _slice_last_steps(getattr(value, field.name), steps, _key=field.name)
                for field in fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: _slice_last_steps(item, steps, _key=key) for key, item in value.items()}
    return value


def _flatten_last_steps(value: Any, steps: int, *, _key: str | None = None) -> Any:
    if isinstance(value, torch.Tensor):
        if _key == "lang" and value.ndim == 2:
            return value.repeat_interleave(int(steps), dim=0)
        if value.ndim < 2:
            raise ValueError(f"Expected [B,T,...] tensor, got {tuple(value.shape)}")
        bsz = int(value.shape[0])
        sliced = value[:, -steps:]
        return sliced.reshape(bsz * steps, *value.shape[2:])
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _flatten_last_steps(getattr(value, field.name), steps, _key=field.name)
                for field in fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: _flatten_last_steps(item, steps, _key=key) for key, item in value.items()}
    return value


def _flatten_strided_steps(value: Any, num_starts: int, min_start: int = 0) -> Any:
    """Flatten ``num_starts`` imagination start positions into the batch dim.

    Unlike :func:`_flatten_last_steps` (which takes the last ``num_starts``
    *adjacent* frames — near-identical consecutive states), this spreads the
    starts EVENLY over the valid range ``[min_start, T-1]`` so the imagined
    rollouts begin from diverse phases of the real trajectory. ``min_start``
    should be ``num_hist - 1`` so every start carries a full (unpadded) history.
    Same start count (=> same effective batch / memory), better state coverage.
    """
    T = _latent_time_dim(value)
    lo = max(0, min(int(min_start), T - 1))
    avail = T - lo
    n = max(1, min(int(num_starts), avail))
    if n >= avail:
        idx = torch.arange(lo, T)
    else:
        idx = torch.unique(torch.linspace(lo, T - 1, steps=n).round().long())

    n_selected = int(idx.numel())

    def _apply(v: Any, key: str | None = None) -> Any:
        if isinstance(v, torch.Tensor):
            if key == "lang" and v.ndim == 2:
                return v.repeat_interleave(n_selected, dim=0)
            if v.ndim < 2:
                raise ValueError(f"Expected [B,T,...] tensor, got {tuple(v.shape)}")
            bsz = int(v.shape[0])
            sel = v.index_select(1, idx.to(v.device))
            return sel.reshape(bsz * sel.shape[1], *v.shape[2:])
        if is_dataclass(v):
            return replace(
                v,
                **{f.name: _apply(getattr(v, f.name), f.name) for f in fields(v)},
            )
        if isinstance(v, dict):
            return {item_key: _apply(item, item_key) for item_key, item in v.items()}
        return v

    return _apply(value)


def _world_model_observe_sequence(world_model: nn.Module, obs: Mapping[str, Any]) -> Any:
    observed = world_model({"mode": "observe_sequence", **obs})
    if not isinstance(observed, Mapping) or "latent" not in observed:
        raise TypeError("world_model observe_sequence must return a mapping with key 'latent'")
    return observed["latent"]


def _world_model_observe_starts(
    world_model: nn.Module, obs: Mapping[str, Any], imag_last: int
) -> Any:
    latent_seq = _world_model_observe_sequence(world_model, obs)
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


def _actor_action_to_env_scale(
    action: torch.Tensor,
    algorithm_cfg: DictConfig,
    *,
    clip: bool | None = None,
) -> torch.Tensor:
    low = torch.as_tensor(
        algorithm_cfg.get(
            "rssm_action_low",
            [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0],
        ),
        device=action.device,
        dtype=action.dtype,
    )
    high = torch.as_tensor(
        algorithm_cfg.get(
            "rssm_action_high",
            [0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0],
        ),
        device=action.device,
        dtype=action.dtype,
    )
    mapped = (action + 1.0) * 0.5 * (high - low) + low
    do_clip = bool(algorithm_cfg.get("rssm_action_clip", True)) if clip is None else bool(clip)
    if do_clip:
        mapped = torch.maximum(torch.minimum(mapped, high), low)
    return mapped


def _actor_action_for_world_model(action: torch.Tensor, algorithm_cfg: DictConfig) -> torch.Tensor:
    scale = str(algorithm_cfg.get("rssm_action_scale", "env")).lower()
    if scale in {"policy", "raw", "identity", ""}:
        return action
    if scale not in {"env", "libero_env", "libero"}:
        raise ValueError("algorithm.rssm_action_scale must be one of: policy, env")
    return _actor_action_to_env_scale(action, algorithm_cfg)


def _policy_reference_action_chunk(policy: nn.Module, hidden: torch.Tensor) -> torch.Tensor | None:
    module: Any = policy
    visited: set[int] = set()
    while module is not None and id(module) not in visited:
        visited.add(id(module))
        reference = getattr(module, "reference_action_chunk", None)
        if callable(reference):
            return reference(hidden)
        module = getattr(module, "module", None)
    return None


def _lambda_return_recurrence(
    live: torch.Tensor,  # [B,T-1]
    cont: torch.Tensor,  # [B,T-1]
    rewards: torch.Tensor,  # [B,T]
    boot: torch.Tensor,  # [B,T]
) -> torch.Tensor:  # [B,T-1]
    """Backward DreamerV3 lambda-return recurrence shared by the imagine/replay
    helpers.

    ``live`` (discounted continuation) and ``cont`` (lambda trace weight) are
    constructed by the caller; everything downstream is identical.
    """
    interm = rewards[:, 1:] + (1.0 - cont) * live * boot[:, 1:]
    ret = boot[:, -1]
    returns: list[torch.Tensor] = []
    for idx in reversed(range(live.shape[1])):
        ret = interm[:, idx] + live[:, idx] * cont[:, idx] * ret
        returns.append(ret)
    returns.reverse()
    return torch.stack(returns, dim=1)


def compute_lambda_returns(
    rewards: torch.Tensor,  # [B,H+1]
    continues: torch.Tensor,  # [B,H+1]
    boot: torch.Tensor,  # [B,H+1]
    disc: float,
    lam: float,
) -> torch.Tensor:  # [B,H]
    """DreamerV3 lambda return, matching dreamerv3/agent.py::lambda_return.

    Reward/continue/bootstrap are defined for the start state plus H imagined
    states. The return for action t uses reward/continue/bootstrap at t+1.
    """
    if not (rewards.shape == continues.shape == boot.shape):
        raise ValueError(
            "lambda_return expects equal [B,H+1] shapes, got "
            f"{tuple(rewards.shape)}, {tuple(continues.shape)}, {tuple(boot.shape)}"
        )
    live = continues[:, 1:] * float(disc)
    cont = torch.full_like(live, float(lam))
    return _lambda_return_recurrence(live, cont, rewards, boot)


def compute_replay_lambda_returns(
    last: torch.Tensor,  # [B,T]
    terminal: torch.Tensor,  # [B,T]
    rewards: torch.Tensor,  # [B,T]
    boot: torch.Tensor,  # [B,T]
    disc: float,
    lam: float,
) -> torch.Tensor:  # [B,T-1]
    """DreamerV3 replay lambda return, matching agent.py::lambda_return.

    `last` stops the lambda trace across episode boundaries. `terminal`
    removes environment continuation. `boot` is the imagined return seeded from
    each replay posterior state, exactly like DreamerV3's `repval_loss`.
    """
    if not (last.shape == terminal.shape == rewards.shape == boot.shape):
        raise ValueError(
            "replay lambda_return expects equal [B,T] shapes, got "
            f"{tuple(last.shape)}, {tuple(terminal.shape)}, "
            f"{tuple(rewards.shape)}, {tuple(boot.shape)}"
        )
    live = (1.0 - terminal.float())[:, 1:] * float(disc)
    cont = (1.0 - last.float())[:, 1:] * float(lam)
    return _lambda_return_recurrence(live, cont, rewards, boot)


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
    if mode not in {"dreamerv3", "perc", "percentile", "percentile_scale"}:
        raise ValueError("algorithm.return_normalization.mode must be one of: none, dreamerv3")

    low_q = float(norm_cfg.get("low", 0.05))
    high_q = float(norm_cfg.get("high", 0.95))
    eps = float(norm_cfg.get("eps", 1.0e-6))
    flat = returns.detach().float().flatten()
    low = torch.quantile(flat, low_q).to(device=returns.device, dtype=returns.dtype)
    high = torch.quantile(flat, high_q).to(device=returns.device, dtype=returns.dtype)
    scale = (high - low).clamp_min(eps)

    return ReturnNormalizationOutput(
        returns=returns,
        values=values,
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
    ref_policy: nn.Module | None = None,
    prev_policy: nn.Module | None = None,
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
    actor_bc_scale = float(
        algorithm_cfg.get("actor_bc_to_vla_scale", algorithm_cfg.get("actor_bc_scale", 0.0))
    )
    actor_bc_ref_scale = float(algorithm_cfg.get("actor_bc_to_ref_scale", 0.0))
    kl_coef = float(algorithm_cfg.get("kl_coef", 0.0))
    kl_penalty_kind = str(algorithm_cfg.get("kl_penalty_kind", "kl")).lower()
    success_return_shaping_scale = float(algorithm_cfg.get("success_return_shaping_scale", 0.0))
    success_return_shaping_discount = float(
        algorithm_cfg.get("success_return_shaping_discount", algorithm_cfg.get("ppo_gamma", 1.0))
    )
    # LUMOS-style second KL: π_new vs π_prev (snapshot before previous update step).
    prev_kl_coef = float(algorithm_cfg.get("prev_kl_coef", 0.0))
    use_ref_kl = (ref_policy is not None) and (kl_coef > 0.0)
    use_prev_kl = (prev_policy is not None) and (prev_kl_coef > 0.0)
    use_ref_bc = (ref_policy is not None) and (actor_bc_ref_scale > 0.0)
    target_tau = float(algorithm_cfg.get("target_critic_tau", 0.02))
    imag_last = int(algorithm_cfg.get("imag_last", 0))
    env_horizon = int(algorithm_cfg.get("horizon", 333))
    contdisc = bool(algorithm_cfg.get("contdisc", True))
    slowtar = bool(algorithm_cfg.get("slowtar", False))
    slowreg = float(algorithm_cfg.get("slowreg", 1.0))
    repl_cfg = algorithm_cfg.get("repl_loss", {})
    repl_lam = float(repl_cfg.get("lam", lam)) if hasattr(repl_cfg, "get") else lam
    repl_slowreg = float(repl_cfg.get("slowreg", slowreg)) if hasattr(repl_cfg, "get") else slowreg
    repl_slowtar = bool(repl_cfg.get("slowtar", slowtar)) if hasattr(repl_cfg, "get") else slowtar
    repval_enabled = bool(algorithm_cfg.get("repval_loss", True))
    repval_scale = float(algorithm_cfg.get("repval_scale", 0.3))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    zero_grad = bool(optim_cfg.get("zero_grad_set_to_none", True))
    # Per-step actor grad-norm / cosine instrumentation is diagnostic-only and
    # costs up to 4 extra retain_graph autograd.grad passes plus 5 full-parameter
    # traversals every step. Gate it OFF by default (PERF-H4).
    grad_diagnostics = bool(optim_cfg.get("grad_diagnostics", False))
    disc = 1.0 if contdisc else 1.0 - 1.0 / float(env_horizon)
    replay_disc = 1.0 - 1.0 / float(env_horizon)

    world_model.eval()
    target_critic.eval()
    policy.train()
    critic.train()

    # ── 1. Posterior starts from a replay sequence, as in DreamerV3 ─────────
    obs = move_mapping_to_device(obs, device)
    with torch.no_grad():
        latent_seq = _detach_latent(_world_model_observe_sequence(world_model, obs))
        seq_len = _latent_time_dim(latent_seq)
        starts = min(int(imag_last) if int(imag_last) > 0 else seq_len, seq_len)
        replay_latent = _slice_last_steps(latent_seq, starts)
        replay_batch_size = _latent_batch_dim(replay_latent)
        current_latent = _flatten_last_steps(latent_seq, starts)
        actor_input_ids = None
        actor_attention_mask = None
        if actor_input_mode == "sequence":
            if "actor_input_ids" not in obs or "actor_attention_mask" not in obs:
                raise RuntimeError(
                    "algorithm.actor_input_mode=sequence requires obs.actor_input_ids "
                    "and obs.actor_attention_mask from the dataset."
                )
            actor_seq_len = _latent_time_dim(obs["actor_input_ids"])
            actor_starts = min(starts, actor_seq_len)
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
    ref_kls: list[torch.Tensor] = []
    prev_kls: list[torch.Tensor] = []
    actor_bc_losses: list[torch.Tensor] = []
    actor_bc_ref_losses: list[torch.Tensor] = []
    actor_drift_raw_mses: list[torch.Tensor] = []
    actor_drift_env_mses: list[torch.Tensor] = []
    actor_drift_env_clip_mses: list[torch.Tensor] = []
    actor_drift_env_maes: list[torch.Tensor] = []

    with _temporarily_freeze(world_model):
        for _ in range(horizon):
            if actor_input_mode == "sequence":
                current_actor_seq = (
                    _world_model_actor_input_sequence(world_model, current_latent).detach().float()
                )
                action, _, extra = policy(
                    {
                        "mode": "sample",
                        "hidden_states": current_actor_seq,
                        "input_ids": actor_input_ids,
                        "attention_mask": actor_attention_mask,
                        "deterministic": False,
                    }
                )
            else:
                current_actor_feat = (
                    _world_model_actor_input(world_model, current_latent).detach().float()
                )
                action, _, extra = policy(
                    {
                        "mode": "sample",
                        "hidden": current_actor_feat,
                        "deterministic": False,
                    }
                )
                action_chunk = extra.get("action_chunk")
                reference_chunk = _policy_reference_action_chunk(policy, current_actor_feat)
                if actor_bc_scale > 0.0:
                    if reference_chunk is None:
                        raise RuntimeError(
                            "algorithm.actor_bc_to_vla_scale requires policy.reference_action_chunk(hidden)."
                        )
                    if not isinstance(action_chunk, torch.Tensor):
                        raise RuntimeError(
                            "policy sample output must include extra['action_chunk'] for BC anchoring."
                        )
                    actor_bc_losses.append(
                        (action_chunk.float() - reference_chunk.float()).square().mean()
                    )
                if use_ref_bc and isinstance(action_chunk, torch.Tensor):
                    with torch.no_grad():
                        _, _, ref_extra = ref_policy(
                            {
                                "mode": "sample",
                                "hidden": current_actor_feat,
                                "deterministic": True,
                                "return_chunk": True,
                            }
                        )
                    ref_action_chunk = ref_extra.get("action_chunk")
                    if not isinstance(ref_action_chunk, torch.Tensor):
                        raise RuntimeError(
                            "algorithm.actor_bc_to_ref_scale requires ref_policy sample extra to include 'action_chunk'."
                        )
                    actor_bc_ref_losses.append(
                        (action_chunk.float() - ref_action_chunk.detach().float()).square().mean()
                    )
                # Drift is monitoring-only (never in loss). Compute against the
                # startup-time frozen ref_policy snapshot — not policy.reference_action_chunk()
                # which is broken under adapter_type=identity (both branches use the
                # same trainable output_projection and yield drift==0 by construction).
                if ref_policy is not None and isinstance(action_chunk, torch.Tensor):
                    with torch.no_grad():
                        _, _, _drift_extra = ref_policy(
                            {
                                "mode": "sample",
                                "hidden": current_actor_feat,
                                "deterministic": True,
                                "return_chunk": True,
                            }
                        )
                    drift_ref_chunk = _drift_extra.get("action_chunk")
                    if isinstance(drift_ref_chunk, torch.Tensor):
                        action_chunk_f = action_chunk.detach().float()
                        drift_ref_chunk_f = drift_ref_chunk.detach().float()
                        actor_drift_raw_mses.append(
                            (action_chunk_f - drift_ref_chunk_f).square().mean()
                        )
                        action_env = _actor_action_to_env_scale(
                            action_chunk_f, algorithm_cfg, clip=False
                        )
                        ref_env = _actor_action_to_env_scale(
                            drift_ref_chunk_f, algorithm_cfg, clip=False
                        )
                        action_env_clip = _actor_action_to_env_scale(
                            action_chunk_f, algorithm_cfg, clip=True
                        )
                        ref_env_clip = _actor_action_to_env_scale(
                            drift_ref_chunk_f, algorithm_cfg, clip=True
                        )
                        actor_drift_env_mses.append((action_env - ref_env).square().mean())
                        actor_drift_env_clip_mses.append(
                            (action_env_clip - ref_env_clip).square().mean()
                        )
                        actor_drift_env_maes.append((action_env - ref_env).abs().mean())
            if extra.get("std") is not None:
                dist = Normal(extra["mean"], extra["std"])
                log_prob_t = dist.log_prob(action.detach()).sum(dim=-1)
                log_probs.append(log_prob_t)
                entropies.append(dist.entropy().sum(dim=-1))

                if use_ref_kl:
                    with torch.no_grad():
                        if actor_input_mode == "sequence":
                            ref_log_prob_t, _, _ = ref_policy(
                                {
                                    "mode": "evaluate",
                                    "hidden_states": current_actor_seq,
                                    "input_ids": actor_input_ids,
                                    "attention_mask": actor_attention_mask,
                                    "action": action.detach(),
                                }
                            )
                        else:
                            ref_log_prob_t, _, _ = ref_policy(
                                {
                                    "mode": "evaluate",
                                    "hidden": current_actor_feat,
                                    "action": action.detach(),
                                }
                            )
                    kl_raw = log_prob_t.detach() - ref_log_prob_t
                    if kl_penalty_kind == "abs":
                        kl_t = kl_raw.abs()
                    elif kl_penalty_kind == "mse":
                        kl_t = 0.5 * kl_raw.square()
                    else:
                        kl_t = kl_raw
                    ref_kls.append(kl_t.detach())

                if use_prev_kl:
                    with torch.no_grad():
                        if actor_input_mode == "sequence":
                            prev_log_prob_t, _, _ = prev_policy(
                                {
                                    "mode": "evaluate",
                                    "hidden_states": current_actor_seq,
                                    "input_ids": actor_input_ids,
                                    "attention_mask": actor_attention_mask,
                                    "action": action.detach(),
                                }
                            )
                        else:
                            prev_log_prob_t, _, _ = prev_policy(
                                {
                                    "mode": "evaluate",
                                    "hidden": current_actor_feat,
                                    "action": action.detach(),
                                }
                            )
                    prev_kl_raw = log_prob_t.detach() - prev_log_prob_t
                    if kl_penalty_kind == "abs":
                        prev_kl_t = prev_kl_raw.abs()
                    elif kl_penalty_kind == "mse":
                        prev_kl_t = 0.5 * prev_kl_raw.square()
                    else:
                        prev_kl_t = prev_kl_raw
                    prev_kls.append(prev_kl_t.detach())

            with torch.no_grad():
                rssm_action = _actor_action_for_world_model(action.detach(), algorithm_cfg)
                next_latent = world_model(
                    {
                        "mode": "predict_next",
                        "latent": current_latent,
                        "actions": rssm_action,
                    }
                )
                current_latent = _detach_latent(next_latent)
                latents.append(current_latent)

        if not log_probs:
            raise RuntimeError("DreamerV3 actor loss requires stochastic policy log_probs.")

        with torch.no_grad():
            critic_feat_stack = torch.stack(
                [
                    _world_model_critic_input(world_model, latent).detach().float()
                    for latent in latents
                ],
                dim=1,
            )  # [B*K,H+1,D]
            reward_stack = torch.stack(
                [
                    _world_model_state_reward(world_model, latent).detach().float()
                    for latent in latents
                ],
                dim=1,
            )
            reward_stack_raw = reward_stack.detach().clone()  # pre-KL WM reward
            success_return_stack: torch.Tensor | None = None
            success_return_delta: torch.Tensor | None = None
            if success_return_shaping_scale != 0.0:
                success_return_stack = torch.stack(
                    [
                        _world_model_success_return(world_model, latent).detach().float()
                        for latent in latents
                    ],
                    dim=1,
                )
                success_return_delta = (
                    float(success_return_shaping_discount) * success_return_stack[:, 1:]
                    - success_return_stack[:, :-1]
                )
                reward_stack = reward_stack.clone()
                reward_stack[:, 1:] = (
                    reward_stack[:, 1:] + float(success_return_shaping_scale) * success_return_delta
                )
            kl_stack_raw: torch.Tensor | None = None
            kl_penalty_mean = torch.zeros((), device=device, dtype=reward_stack.dtype)
            if use_ref_kl and ref_kls:
                kl_stack = torch.stack(ref_kls, dim=1).to(dtype=reward_stack.dtype)
                kl_stack_raw = kl_stack.detach().clone()
                # reward_stack is [B*K, H+1]; KL applies to H action steps → r_0..r_{H-1}
                reward_stack = reward_stack.clone()
                reward_stack[:, :-1] = reward_stack[:, :-1] - kl_coef * kl_stack
                kl_penalty_mean = kl_stack.mean().detach()
            prev_kl_penalty_mean = torch.zeros((), device=device, dtype=reward_stack.dtype)
            if use_prev_kl and prev_kls:
                prev_kl_stack = torch.stack(prev_kls, dim=1).to(dtype=reward_stack.dtype)
                reward_stack = reward_stack.clone()
                reward_stack[:, :-1] = reward_stack[:, :-1] - prev_kl_coef * prev_kl_stack
                prev_kl_penalty_mean = prev_kl_stack.mean().detach()
            continue_stack = torch.stack(
                [
                    _world_model_continue(world_model, latent, reward_stack[:, idx])
                    .detach()
                    .float()
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
                disc=disc,
                lam=lam,
            )

        raw_returns = returns
        raw_baseline_values = target_values[:, :-1]
        normalized = normalize_returns_for_actor_critic(
            raw_returns, raw_baseline_values, algorithm_cfg
        )
        returns = raw_returns
        baseline_values = raw_baseline_values

        if normalized.enabled:
            norm_low, norm_high = return_tracker.update(raw_returns)
            scale = return_tracker.scale()
            offset = return_tracker.offset()
        else:
            norm_low, norm_high = normalized.low, normalized.high
            scale = 1.0
            offset = 0.0
        scale_tensor = torch.as_tensor(scale, device=device, dtype=returns.dtype).clamp_min(1.0)
        offset_tensor = torch.as_tensor(offset, device=device, dtype=returns.dtype)
        ret_normed = (returns - offset_tensor) / scale_tensor
        weights = torch.cumprod(float(disc) * continue_stack, dim=1) / float(disc)
        weights = weights[:, :-1]
        advantages = (returns - baseline_values) / scale_tensor

        log_prob_stack = torch.stack(log_probs, dim=1)  # [B*K,H]
        entropy_stack = (
            torch.stack(entropies, dim=1) if entropies else torch.zeros_like(log_prob_stack)
        )
        # Split actor loss into named components so each gradient contribution can be measured.
        actor_pg_loss = (weights.detach() * -(log_prob_stack * advantages.detach())).mean()
        if entropy_coef != 0.0:
            actor_entropy_loss = (weights.detach() * -(entropy_coef * entropy_stack)).mean()
        else:
            actor_entropy_loss = torch.zeros((), device=device, dtype=actor_pg_loss.dtype)
        actor_bc_loss = (
            torch.stack(actor_bc_losses).mean()
            if actor_bc_losses
            else torch.zeros((), device=device, dtype=actor_pg_loss.dtype)
        )
        actor_bc_loss_scaled = (
            actor_bc_scale * actor_bc_loss
            if actor_bc_scale > 0.0
            else torch.zeros((), device=device, dtype=actor_pg_loss.dtype)
        )
        actor_bc_ref_loss = (
            torch.stack(actor_bc_ref_losses).mean()
            if actor_bc_ref_losses
            else torch.zeros((), device=device, dtype=actor_pg_loss.dtype)
        )
        actor_bc_ref_loss_scaled = (
            actor_bc_ref_scale * actor_bc_ref_loss
            if actor_bc_ref_scale > 0.0
            else torch.zeros((), device=device, dtype=actor_pg_loss.dtype)
        )
        actor_loss = (
            actor_pg_loss + actor_entropy_loss + actor_bc_loss_scaled + actor_bc_ref_loss_scaled
        )

        # ── Per-component actor-gradient instrumentation (diagnostic-only) ───
        # For each loss term, compute the gradient w.r.t. policy parameters and
        # record its L2 norm (and cosine with the PG term). This lets us see
        # whether the actor is being moved by the REINFORCE signal or by the
        # BC anchor pulling it back toward SFT. Each ``_flat_grad`` runs an extra
        # ``autograd.grad(..., retain_graph=True)`` (up to 4/step, pinning the
        # actor graph), so the whole block is gated behind ``grad_diagnostics``
        # (PERF-H4). Defaults below match the OFF / unavailable convention (0.0).
        actor_grad_pg_norm = 0.0
        actor_grad_bc_ref_norm = 0.0
        actor_grad_entropy_norm = 0.0
        actor_grad_bc_vla_norm = 0.0
        cos_pg_bcref = 0.0
        if grad_diagnostics:
            actor_params = [p for p in policy.parameters() if p.requires_grad]

            def _flat_grad(loss_t: torch.Tensor) -> torch.Tensor | None:
                # Returns a flat tensor of size sum(p.numel() for p in actor_params).
                # Params with no gradient w.r.t. loss_t (e.g. log_std for a
                # deterministic MSE loss) are filled with zeros so different
                # components can be compared element-wise (norms, cosines).
                if not loss_t.requires_grad or loss_t.grad_fn is None:
                    return None
                try:
                    grads = torch.autograd.grad(
                        loss_t,
                        actor_params,
                        retain_graph=True,
                        allow_unused=True,
                        create_graph=False,
                    )
                except RuntimeError:
                    return None
                pieces = []
                has_any = False
                for g, p in zip(grads, actor_params, strict=True):
                    if g is None:
                        pieces.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
                    else:
                        pieces.append(g.detach().reshape(-1))
                        has_any = True
                if not has_any:
                    return None
                return torch.cat(pieces)

            g_pg = _flat_grad(actor_pg_loss)
            g_bcref = _flat_grad(actor_bc_ref_loss_scaled) if actor_bc_ref_scale > 0.0 else None
            g_ent = _flat_grad(actor_entropy_loss) if entropy_coef != 0.0 else None
            g_bcvla = _flat_grad(actor_bc_loss_scaled) if actor_bc_scale > 0.0 else None

            def _norm(g: torch.Tensor | None) -> float:
                return float(g.norm().cpu()) if g is not None else 0.0

            actor_grad_pg_norm = _norm(g_pg)
            actor_grad_bc_ref_norm = _norm(g_bcref)
            actor_grad_entropy_norm = _norm(g_ent)
            actor_grad_bc_vla_norm = _norm(g_bcvla)

            if g_pg is not None and g_bcref is not None:
                denom = (g_pg.norm() * g_bcref.norm()).clamp_min(1e-12)
                cos_pg_bcref = float(((g_pg * g_bcref).sum() / denom).cpu())
        # log_prob and advantage diagnostics
        log_prob_mean = float(log_prob_stack.detach().mean().cpu())
        log_prob_std = float(log_prob_stack.detach().std().cpu())
        adv_pos_frac = float((advantages.detach() > 0).float().mean().cpu())

        actor_optimizer.zero_grad(set_to_none=zero_grad)
        actor_loss.backward()
        # Per-submodule grad norms are diagnostic-only (5 full-parameter
        # traversals); gate behind ``grad_diagnostics`` (PERF-H4).
        actor_adapter_grad_norm = 0.0
        actor_action_head_grad_norm = 0.0
        actor_output_projection_grad_norm = 0.0
        actor_policy_head_grad_norm = 0.0
        actor_log_std_grad_norm = 0.0
        if grad_diagnostics:
            actor_adapter_grad_norm = _named_grad_norm(policy, "adapter")
            actor_action_head_grad_norm = _named_grad_norm(policy, "action")
            actor_output_projection_grad_norm = _named_grad_norm(policy, "output_projection")
            actor_policy_head_grad_norm = _named_grad_norm(policy, "policy_head")
            actor_log_std_grad_norm = _named_grad_norm(policy, "log_std")
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
        actor_optimizer.step()

    # ── 3. Value loss, matching DreamerV3 value target from imagined returns.
    feat_stack_h = critic_feat_stack[:, :-1].detach()  # [B*K,H,D]
    B2, H, D2 = feat_stack_h.shape
    log_probs_critic = critic(
        {
            "mode": "log_prob",
            "hidden": feat_stack_h.reshape(B2 * H, D2),
            "values": returns.detach().reshape(B2 * H),
        }
    )
    value_loss = -log_probs_critic.view(B2, H)
    if slowreg > 0:
        with torch.no_grad():
            slow_targets = target_critic(
                {
                    "mode": "value",
                    "hidden": feat_stack_h.reshape(B2 * H, D2),
                }
            ).view(B2, H)
        slow_log_probs = critic(
            {
                "mode": "log_prob",
                "hidden": feat_stack_h.reshape(B2 * H, D2),
                "values": slow_targets.detach().reshape(B2 * H),
            }
        )
        value_loss = value_loss + slowreg * (-slow_log_probs.view(B2, H))
    critic_loss = (weights.detach() * value_loss).mean()

    repval_loss = torch.zeros((), device=device)
    repval_weight_mean = torch.zeros((), device=device)
    repval_applied = False
    if repval_enabled and repval_scale > 0.0 and starts > 1:

        def _sequence_field(*keys: str) -> torch.Tensor | None:
            for key in keys:
                value = obs.get(key)
                if isinstance(value, torch.Tensor):
                    tensor = value
                    if tensor.ndim == 3 and tensor.shape[-1] == 1:
                        tensor = tensor.squeeze(-1)
                    if tensor.ndim != 2:
                        raise ValueError(
                            f"obs.{key} must be [B,T] for replay value loss, got {tuple(value.shape)}"
                        )
                    return tensor[:, -starts:].to(device=device, dtype=returns.dtype)
            return None

        replay_rewards = _sequence_field("rewards", "reward")
        if replay_rewards is not None:
            replay_terminal = _sequence_field("is_terminal", "dones")
            replay_last = _sequence_field("is_last", "dones")
            if replay_terminal is None:
                replay_terminal = torch.zeros_like(replay_rewards)
            if replay_last is None:
                replay_last = replay_terminal
            with torch.no_grad():
                replay_critic_feat = (
                    _world_model_critic_input(world_model, replay_latent).detach().float()
                )
                if replay_critic_feat.ndim != 3:
                    raise ValueError(
                        "Replay value loss expects critic replay features [B,K,D], "
                        f"got {tuple(replay_critic_feat.shape)}"
                    )
                B_rep, K_rep, D_rep = replay_critic_feat.shape
                if B_rep != replay_batch_size or K_rep != starts:
                    raise ValueError(
                        "Replay latent shape changed unexpectedly: "
                        f"B={B_rep}, K={K_rep}, expected B={replay_batch_size}, K={starts}"
                    )
                # A4: bootstrap the replay lambda-return with the critic's
                # per-state value on the replay posterior states (standard
                # DreamerV3 repval), not the single imagined return scalar.
                # repl_loss.slowtar selects the target vs fast critic.
                replay_boot_feats = replay_critic_feat.reshape(B_rep * K_rep, D_rep)
                replay_fast_values = critic({"mode": "value", "hidden": replay_boot_feats}).view(
                    B_rep, K_rep
                )
                replay_slow_values = target_critic(
                    {"mode": "value", "hidden": replay_boot_feats}
                ).view(B_rep, K_rep)
                replay_boot = (replay_slow_values if repl_slowtar else replay_fast_values).to(
                    dtype=returns.dtype
                )
                replay_returns = compute_replay_lambda_returns(
                    last=replay_last,
                    terminal=replay_terminal,
                    rewards=replay_rewards,
                    boot=replay_boot,
                    disc=replay_disc,
                    lam=repl_lam,
                )

            replay_feat_h = replay_critic_feat[:, :-1].detach()
            B_rep, K_minus_1, D_rep = replay_feat_h.shape
            replay_log_probs = critic(
                {
                    "mode": "log_prob",
                    "hidden": replay_feat_h.reshape(B_rep * K_minus_1, D_rep),
                    "values": replay_returns.detach().reshape(B_rep * K_minus_1),
                }
            )
            replay_value_loss = -replay_log_probs.view(B_rep, K_minus_1)
            if repl_slowreg > 0:
                with torch.no_grad():
                    replay_slow_targets = target_critic(
                        {
                            "mode": "value",
                            "hidden": replay_feat_h.reshape(B_rep * K_minus_1, D_rep),
                        }
                    ).view(B_rep, K_minus_1)
                replay_slow_log_probs = critic(
                    {
                        "mode": "log_prob",
                        "hidden": replay_feat_h.reshape(B_rep * K_minus_1, D_rep),
                        "values": replay_slow_targets.detach().reshape(B_rep * K_minus_1),
                    }
                )
                replay_value_loss = replay_value_loss + repl_slowreg * (
                    -replay_slow_log_probs.view(B_rep, K_minus_1)
                )
            replay_weights = (1.0 - replay_last[:, :-1].float()).detach()
            repval_weight_mean = replay_weights.mean()
            repval_loss = (replay_weights * replay_value_loss).mean()
            critic_loss = critic_loss + repval_scale * repval_loss
            repval_applied = True

    critic_optimizer.zero_grad(set_to_none=zero_grad)
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip)
    critic_optimizer.step()

    # ── 8. Polyak-average target critic ────────────────────────────────────
    soft_update(target_critic, critic, tau=target_tau)

    _metrics = {
        "actor_loss": float(actor_loss.detach().cpu()),
        "actor_bc_loss": float(actor_bc_loss.detach().cpu()),
        "actor_bc_scale": float(actor_bc_scale),
        "actor_bc_ref_loss": float(actor_bc_ref_loss.detach().cpu()),
        "actor_bc_ref_scale": float(actor_bc_ref_scale),
        "ref_kl_mean": float(kl_penalty_mean.detach().cpu()),
        "prev_kl_mean": float(prev_kl_penalty_mean.detach().cpu()),
        "kl_coef": float(kl_coef),
        "prev_kl_coef": float(prev_kl_coef),
        "actor_vla_drift_raw_mse": float(
            torch.stack(actor_drift_raw_mses).mean().detach().cpu()
            if actor_drift_raw_mses
            else torch.zeros(())
        ),
        "actor_vla_drift_env_mse": float(
            torch.stack(actor_drift_env_mses).mean().detach().cpu()
            if actor_drift_env_mses
            else torch.zeros(())
        ),
        "actor_vla_drift_env_mse_clipped": float(
            torch.stack(actor_drift_env_clip_mses).mean().detach().cpu()
            if actor_drift_env_clip_mses
            else torch.zeros(())
        ),
        "actor_vla_drift_env_mae": float(
            torch.stack(actor_drift_env_maes).mean().detach().cpu()
            if actor_drift_env_maes
            else torch.zeros(())
        ),
        "critic_loss": float(critic_loss.detach().cpu()),
        "returns_mean": float(ret_normed.detach().mean().cpu()),
        "returns_std": float(ret_normed.detach().std().cpu()),
        "raw_returns_mean": float(raw_returns.detach().mean().cpu()),
        "raw_returns_std": float(raw_returns.detach().std().cpu()),
        "advantage_mean": float(advantages.detach().mean().cpu()),
        "advantage_std": float(advantages.detach().std().cpu()),
        "advantage_mag": float(advantages.detach().abs().mean().cpu()),
        "return_norm_enabled": float(normalized.enabled),
        "return_norm_low": float(norm_low),
        "return_norm_high": float(norm_high),
        "return_norm_offset": float(offset),
        "return_norm_scale": float(scale),
        "return_norm_batch_scale": normalized.scale,
        "return_scale": float(scale),
        "ret_normed_min": float(ret_normed.detach().min().cpu()),
        "ret_normed_max": float(ret_normed.detach().max().cpu()),
        "ret_normed_rate": float((ret_normed.detach().abs() >= 1.0).float().mean().cpu()),
        "reward_mean": float(reward_stack[:, 1:].detach().mean().cpu()),
        # Reward distribution diagnostics (over imagine horizon × batch).
        "reward_raw_mean": float(reward_stack_raw[:, :-1].mean().cpu()),
        "reward_raw_std": float(reward_stack_raw[:, :-1].std().cpu()),
        "reward_raw_min": float(reward_stack_raw[:, :-1].min().cpu()),
        "reward_raw_max": float(reward_stack_raw[:, :-1].max().cpu()),
        "reward_raw_p10": float(reward_stack_raw[:, :-1].quantile(0.1).cpu()),
        "reward_raw_p50": float(reward_stack_raw[:, :-1].quantile(0.5).cpu()),
        "reward_raw_p90": float(reward_stack_raw[:, :-1].quantile(0.9).cpu()),
        "reward_post_min": float(reward_stack[:, :-1].detach().min().cpu()),
        "reward_post_max": float(reward_stack[:, :-1].detach().max().cpu()),
        "reward_post_p10": float(reward_stack[:, :-1].detach().quantile(0.1).cpu()),
        "reward_post_p50": float(reward_stack[:, :-1].detach().quantile(0.5).cpu()),
        "reward_post_p90": float(reward_stack[:, :-1].detach().quantile(0.9).cpu()),
        "success_return_shaping_scale": float(success_return_shaping_scale),
        "success_return_shaping_discount": float(success_return_shaping_discount),
        "success_return_mean": (
            float(success_return_stack[:, 1:].detach().mean().cpu())
            if success_return_stack is not None
            else 0.0
        ),
        "success_return_delta_mean": (
            float(success_return_delta.detach().mean().cpu())
            if success_return_delta is not None
            else 0.0
        ),
        "success_return_delta_std": (
            float(success_return_delta.detach().std().cpu())
            if success_return_delta is not None
            else 0.0
        ),
        "kl_p10": float(kl_stack_raw.quantile(0.1).cpu()) if kl_stack_raw is not None else 0.0,
        "kl_p50": float(kl_stack_raw.quantile(0.5).cpu()) if kl_stack_raw is not None else 0.0,
        "kl_p90": float(kl_stack_raw.quantile(0.9).cpu()) if kl_stack_raw is not None else 0.0,
        "kl_max": float(kl_stack_raw.max().cpu()) if kl_stack_raw is not None else 0.0,
        "continue_mean": float(continue_stack[:, 1:].detach().mean().cpu()),
        "value_mean": float(critic_values[:, :-1].detach().mean().cpu()),
        "critic_target_mean": float(returns.detach().mean().cpu()),
        "imagine_weight_mean": float(weights.detach().mean().cpu()),
        "repval_loss": float(repval_loss.detach().cpu()),
        "repval_scale": float(repval_scale),
        "repval_applied": float(repval_applied),
        "repval_weight_mean": float(repval_weight_mean.detach().cpu()),
        "actor_grad_norm": float(torch.as_tensor(actor_grad_norm).detach().cpu()),
        "actor_grad_norm_adapter": actor_adapter_grad_norm,
        "actor_grad_norm_action_head": actor_action_head_grad_norm,
        "actor_grad_norm_output_projection": actor_output_projection_grad_norm,
        "actor_grad_norm_policy_head": actor_policy_head_grad_norm,
        "actor_grad_norm_log_std": actor_log_std_grad_norm,
        # Per-loss-component actor gradient norms (diagnostic).
        "actor_pg_loss": float(actor_pg_loss.detach().cpu()),
        "actor_entropy_loss": float(actor_entropy_loss.detach().cpu()),
        "actor_grad_norm_pg": actor_grad_pg_norm,
        "actor_grad_norm_bc_ref": actor_grad_bc_ref_norm,
        "actor_grad_norm_entropy": actor_grad_entropy_norm,
        "actor_grad_norm_bc_vla": actor_grad_bc_vla_norm,
        "actor_grad_cos_pg_bcref": cos_pg_bcref,
        "log_prob_mean": log_prob_mean,
        "log_prob_std": log_prob_std,
        "advantage_pos_frac": adv_pos_frac,
        "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).detach().cpu()),
    }

    # One-shot tensor snapshot for offline reward-distribution analysis.
    # Set env var REWARD_DUMP_DIR to enable. Dump happens on the first call;
    # marker file prevents subsequent overwrites.
    _dump_dir = os.environ.get("REWARD_DUMP_DIR")
    if _dump_dir:
        _dump_path = Path(_dump_dir)
        _dump_path.mkdir(parents=True, exist_ok=True)
        _marker = _dump_path / ".dumped"
        if not _marker.exists():
            torch.save(
                {
                    "reward_raw": reward_stack_raw.cpu(),
                    "reward_post_kl": reward_stack.detach().cpu(),
                    "success_return": (
                        success_return_stack.cpu() if success_return_stack is not None else None
                    ),
                    "success_return_delta": (
                        success_return_delta.cpu() if success_return_delta is not None else None
                    ),
                    "kl": (kl_stack_raw.cpu() if kl_stack_raw is not None else None),
                    "kl_coef": float(kl_coef),
                    "advantages": advantages.detach().cpu(),
                    "returns": returns.detach().cpu(),
                    "value": critic_values.detach().cpu(),
                    "log_prob": log_prob_stack.detach().cpu(),
                },
                _dump_path / "reward_snapshot.pt",
            )
            _marker.write_text("ok")
            print(
                f"[reward-dump] snapshot written to {_dump_path / 'reward_snapshot.pt'}",
                flush=True,
            )

    return _metrics


__all__ = [
    "_actor_action_for_world_model",
    "compute_lambda_returns",
    "compute_replay_lambda_returns",
    "imagine_actor_critic_step",
    "normalize_returns_for_actor_critic",
    "world_model_pretrain_step",
]
