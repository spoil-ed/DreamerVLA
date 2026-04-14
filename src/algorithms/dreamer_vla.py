from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from omegaconf import DictConfig
from torch import nn

from src.algorithms.ppo_grpo import (
    compute_group_relative_advantages,
    compute_ppo_actor_loss,
)
from src.utils.torch_utils import (
    freeze_module,
    move_mapping_to_device,
    repeat_tensor_mapping,
)


@dataclass
class PreparedPPOBatch:
    grouped_obs: Mapping[str, Any]
    embedded_obs: Any | None
    sampled_action: torch.Tensor
    scores: torch.Tensor
    advantages: torch.Tensor
    log_prob_old: torch.Tensor
    log_prob_ref: torch.Tensor


def _print_once(obj: object, attr: str, message: str) -> None:
    if getattr(obj, attr, False):
        return
    print(message, flush=True)
    setattr(obj, attr, True)


def sync_policy_snapshot(source: nn.Module, target: nn.Module) -> None:
    # Snapshot sync
    if hasattr(source, "snapshot_state_dict") and hasattr(target, "load_snapshot_state_dict"):
        target.load_snapshot_state_dict(source.snapshot_state_dict())
    else:
        target.load_state_dict(source.state_dict())
    freeze_module(target)


def embed_observation(policy: nn.Module, obs: Mapping[str, Any]) -> Any:
    shared_embedding = getattr(policy, "embedding", None)
    if shared_embedding is None:
        hidden = policy.encode(obs)
        _print_once(
            policy,
            "_trace_encode_to_world_model",
            f"[Trace] policy.encode -> hidden shape {tuple(hidden.shape)}",
        )
        return hidden
    embedded = shared_embedding.embed_observation(obs)
    _print_once(
        policy,
        "_trace_embedding_to_world_model",
        f"[Trace] shared embedding -> sequence shape {tuple(embedded.embeddings.shape)} "
        f"mask shape {tuple(embedded.attention_mask.shape)}",
    )
    return embedded


def score_candidate_actions(
    policy: nn.Module,
    world_model: nn.Module,
    obs: Mapping[str, Any],
    actions: torch.Tensor,
    score_source: str,
    embedded_obs: Any | None = None,
) -> torch.Tensor:
    # World score
    with torch.no_grad():
        hidden = embedded_obs if embedded_obs is not None else embed_observation(policy, obs)
        latent = world_model.encode_latent(hidden)
        next_latent = world_model.predict_next(latent, actions)
        attention_mask = getattr(hidden, "attention_mask", None)
        if attention_mask is None and isinstance(hidden, Mapping):
            attention_mask = hidden.get("attention_mask")
        if score_source == "reward_head":
            scores = world_model.reward(latent, actions, next_latent, attention_mask=attention_mask)
        elif score_source == "dummy_l2":
            scores = -(actions.pow(2).mean(dim=-1))
        else:
            raise ValueError(f"Unsupported score source: {score_source}")
    return scores


def world_model_pretrain_step(
    policy: nn.Module,
    world_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    device: torch.device,
    optim_cfg: DictConfig,
) -> dict[str, float]:
    # Batch tensors
    obs = move_mapping_to_device(batch["obs"], device)
    next_obs = move_mapping_to_device(batch["next_obs"], device)
    action = batch["action"].to(device)
    reward = batch.get("reward")
    if reward is not None:
        reward = reward.to(device)

    # Frozen encoder
    world_model.train()
    policy.eval()
    with torch.no_grad():
        hidden = embed_observation(policy, obs)
        next_hidden = embed_observation(policy, next_obs)
        _print_once(
            world_model,
            "_trace_pretrain_bridge",
            "[Trace] world_model_pretrain_step received encoder outputs; entering pretrain_loss.",
        )

    # Model loss
    losses = world_model.pretrain_loss(
        hidden=hidden,
        action=action,
        next_hidden=next_hidden,
        reward_target=reward,
    )

    # Optim step
    optimizer.zero_grad(set_to_none=bool(optim_cfg.get("zero_grad_set_to_none", True)))
    losses["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        world_model.parameters(),
        max_norm=float(optim_cfg.get("grad_clip_norm", 1.0)),
    )
    optimizer.step()

    return {
        "loss": float(losses["loss"].detach().cpu()),
        "transition_loss": float(losses["transition_loss"].detach().cpu()),
        "reward_loss": float(losses["reward_loss"].detach().cpu()),
        "predicted_reward_mean": float(losses["predicted_reward_mean"].detach().cpu()),
        "latent_norm": float(losses["latent_norm"].detach().cpu()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
    }


def actor_update_step(
    new_policy: nn.Module,
    old_policy: nn.Module,
    ref_policy: nn.Module,
    world_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    device: torch.device,
    algorithm_cfg: DictConfig,
    optim_cfg: DictConfig,
) -> dict[str, float]:
    prepared = prepare_ppo_batch(
        new_policy=new_policy,
        old_policy=old_policy,
        ref_policy=ref_policy,
        world_model=world_model,
        batch=batch,
        device=device,
        algorithm_cfg=algorithm_cfg,
    )
    return ppo_update_step(
        new_policy=new_policy,
        optimizer=optimizer,
        prepared=prepared,
        algorithm_cfg=algorithm_cfg,
        optim_cfg=optim_cfg,
    )


def prepare_ppo_batch(
    new_policy: nn.Module,
    old_policy: nn.Module,
    ref_policy: nn.Module,
    world_model: nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    algorithm_cfg: DictConfig,
) -> PreparedPPOBatch:
    # Grouped batch
    obs = move_mapping_to_device(batch["obs"], device)
    group_size = int(algorithm_cfg.group_size)
    grouped_obs = repeat_tensor_mapping(obs, group_size)

    # Snapshot the rollout policy once. This frozen copy defines old_log_prob for
    # all subsequent PPO updates on the same sampled batch.
    sync_policy_snapshot(new_policy, old_policy)
    if ref_policy is not old_policy:
        freeze_module(ref_policy)
    new_policy.train()
    world_model.eval()

    with torch.no_grad():
        embedded_grouped_obs = embed_observation(new_policy, grouped_obs)
        _print_once(
            world_model,
            "_trace_actor_bridge",
            "[Trace] actor_update_step received shared encoder outputs; querying world model reward head.",
        )
        if getattr(new_policy, "embedding", None) is None:
            sampled_action, _, _ = new_policy.sample_action(
                grouped_obs,
                deterministic=False,
            )
        else:
            sampled_action, _, _ = new_policy.sample_action_from_embedding(
                embedded_grouped_obs,
                deterministic=False,
            )

        scores = score_candidate_actions(
            policy=new_policy,
            world_model=world_model,
            obs=grouped_obs,
            actions=sampled_action,
            score_source=str(algorithm_cfg.score_source),
            embedded_obs=embedded_grouped_obs,
        )
        advantages = compute_group_relative_advantages(
            scores=scores,
            group_size=group_size,
            eps=float(algorithm_cfg.advantage_eps),
        )
        if getattr(old_policy, "embedding", None) is None:
            log_prob_old, _, _ = old_policy.evaluate_action(grouped_obs, sampled_action)
            log_prob_ref, _, _ = ref_policy.evaluate_action(grouped_obs, sampled_action)
        else:
            log_prob_old, _, _ = old_policy.evaluate_action_from_embedding(
                embedded_grouped_obs,
                sampled_action,
            )
            log_prob_ref, _, _ = ref_policy.evaluate_action_from_embedding(
                embedded_grouped_obs,
                sampled_action,
            )

    return PreparedPPOBatch(
        grouped_obs=grouped_obs,
        embedded_obs=embedded_grouped_obs,
        sampled_action=sampled_action,
        scores=scores.detach(),
        advantages=advantages.detach(),
        log_prob_old=log_prob_old.detach(),
        log_prob_ref=log_prob_ref.detach(),
    )


def ppo_update_step(
    new_policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    prepared: PreparedPPOBatch,
    algorithm_cfg: DictConfig,
    optim_cfg: DictConfig,
) -> dict[str, float]:
    # PPO loss
    if getattr(new_policy, "embedding", None) is None:
        log_prob_new, entropy, _ = new_policy.evaluate_action(prepared.grouped_obs, prepared.sampled_action)
    else:
        log_prob_new, entropy, _ = new_policy.evaluate_action_from_embedding(
            prepared.embedded_obs,
            prepared.sampled_action,
        )
    losses = compute_ppo_actor_loss(
        log_prob_new=log_prob_new,
        log_prob_old=prepared.log_prob_old,
        advantages=prepared.advantages,
        clip_ratio=float(algorithm_cfg.clip_ratio),
        entropy=entropy,
        entropy_coef=float(algorithm_cfg.entropy_coef),
        log_prob_ref=prepared.log_prob_ref,
        kl_coef=float(algorithm_cfg.kl_coef),
    )

    # Optim step
    optimizer.zero_grad(set_to_none=bool(optim_cfg.get("zero_grad_set_to_none", True)))
    losses["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        new_policy.parameters(),
        max_norm=float(optim_cfg.get("grad_clip_norm", 1.0)),
    )
    optimizer.step()

    return {
        "loss": float(losses["loss"].detach().cpu()),
        "policy_loss": float(losses["policy_loss"].detach().cpu()),
        "entropy_bonus": float(losses["entropy_bonus"].detach().cpu()),
        "approx_kl_old": float(losses["approx_kl_old"].detach().cpu()),
        "approx_kl_ref": float(losses["approx_kl_ref"].detach().cpu()),
        "clip_fraction": float(losses["clip_fraction"].detach().cpu()),
        "ratio_mean": float(losses["ratio_mean"].detach().cpu()),
        "advantage_mean": float(losses["advantage_mean"].detach().cpu()),
        "advantage_std": float(losses["advantage_std"].detach().cpu()),
        "score_mean": float(prepared.scores.mean().detach().cpu()),
        "log_prob_new_mean": float(log_prob_new.mean().detach().cpu()),
        "log_prob_old_mean": float(prepared.log_prob_old.mean().detach().cpu()),
        "log_prob_ref_mean": float(prepared.log_prob_ref.mean().detach().cpu()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
    }


def run_actor_ppo_updates(
    new_policy: nn.Module,
    old_policy: nn.Module,
    ref_policy: nn.Module,
    world_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    device: torch.device,
    algorithm_cfg: DictConfig,
    optim_cfg: DictConfig,
    num_updates: int,
) -> list[dict[str, float]]:
    prepared = prepare_ppo_batch(
        new_policy=new_policy,
        old_policy=old_policy,
        ref_policy=ref_policy,
        world_model=world_model,
        batch=batch,
        device=device,
        algorithm_cfg=algorithm_cfg,
    )
    metrics: list[dict[str, float]] = []
    for update_idx in range(int(num_updates)):
        step_metrics = ppo_update_step(
            new_policy=new_policy,
            optimizer=optimizer,
            prepared=prepared,
            algorithm_cfg=algorithm_cfg,
            optim_cfg=optim_cfg,
        )
        step_metrics["update_idx"] = float(update_idx)
        metrics.append(step_metrics)
    return metrics
