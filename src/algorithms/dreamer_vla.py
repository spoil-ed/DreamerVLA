from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Normal

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


# ─────────────────────────────────────────────────────────────────────────────
# Dreamer-style actor-critic with imagined rollouts
# ─────────────────────────────────────────────────────────────────────────────

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
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    obs: Mapping[str, Any],
    device: torch.device,
    algorithm_cfg: DictConfig,
    optim_cfg: DictConfig,
) -> dict[str, float]:
    """One Dreamer-style actor-critic update using WM imagination.

    Training flow
    ─────────────
    1. Encode real obs → WM latent state (detached; no grad back to encoder).
    2. Imagine H steps:
         policy samples action from latent feature (reparameterised → grad)
         WM dynamics predict next latent (detached; no grad through backbone)
         WM reward head scores (state, action, next_state) → grad through action
    3. Critic estimates V(s_t) for each imagined state → bootstrap λ-returns.
    4. Actor loss  = −E[γᵗ · G_t^λ] + entropy bonus.
    5. Critic loss =  MSE(V(s_t), stop_grad(G_t^λ)).

    Note: gradients flow through the WM *reward head* only (not the transition
    backbone). This keeps memory bounded and avoids back-propagating through
    the large causal transformer for H steps.
    """
    horizon = int(algorithm_cfg.imagination_horizon)
    gamma = float(algorithm_cfg.gamma)
    lam = float(algorithm_cfg.lam)
    entropy_coef = float(algorithm_cfg.get("entropy_coef", 0.0))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    zero_grad = bool(optim_cfg.get("zero_grad_set_to_none", True))

    # ── 1. Initial latent state (no gradient back to encoder / WM encoder) ─
    world_model.eval()
    policy.train()
    critic.train()

    with torch.no_grad():
        hidden = embed_observation(policy, move_mapping_to_device(obs, device))
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach()
        initial_latent = world_model.encode_latent(hidden)

    current_feat = initial_latent.feature().detach()  # [B, latent_dim+deter_dim]

    # ── 2. H-step imagination ─────────────────────────────────────────────
    feats: list[torch.Tensor] = [current_feat]
    rewards: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for _ in range(horizon):
        # Actor: sample action with reparameterisation (grad flows here)
        action, _, extra = policy.sample_action_from_embedding(current_feat, deterministic=False)

        if entropy_coef > 0.0 and extra.get("std") is not None:
            dist = Normal(extra["mean"], extra["std"])
            entropies.append(dist.entropy().sum(dim=-1))   # [B]

        # Dynamics: predict next state — detach action to avoid backprop
        # through the large transition backbone
        with torch.no_grad():
            current_latent = world_model.encode_latent(current_feat)
            next_latent = world_model.predict_next(current_latent, action.detach())
            next_feat = next_latent.feature().detach()     # [B, D]

        # Reward: gradient CAN flow action → reward_head → actor_loss
        # cat([current_feat (detached), action (grad), next_feat (detached)])
        reward = world_model.reward(current_latent, action, next_latent)  # [B]

        rewards.append(reward)
        feats.append(next_feat)
        current_feat = next_feat

    # ── 3. Critic values for bootstrap (stop_gradient from actor's perspective) ─
    with torch.no_grad():
        feat_stack_all = torch.stack(feats, dim=0)           # [H+1, B, D]
        Hp1, B, D = feat_stack_all.shape
        values_flat = critic(feat_stack_all.view(Hp1 * B, D)).view(Hp1, B)
        values = [values_flat[t] for t in range(Hp1)]        # list of H+1 tensors [B]

    # ── 4. λ-returns (gradient through rewards → action → policy) ─────────
    returns = compute_lambda_returns(rewards, values, gamma, lam)  # [H, B]

    # ── 5. Actor loss ─────────────────────────────────────────────────────
    discount = torch.tensor(
        [gamma ** t for t in range(horizon)],
        device=device, dtype=returns.dtype,
    ).unsqueeze(-1)                              # [H, 1]
    actor_loss = -(discount * returns).mean()
    if entropies and entropy_coef > 0.0:
        actor_loss = actor_loss - entropy_coef * torch.stack(entropies).mean()

    actor_optimizer.zero_grad(set_to_none=zero_grad)
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
    actor_optimizer.step()

    # ── 6. Critic loss: fit stop_grad(λ-returns) ──────────────────────────
    # Re-run the critic forward *with* gradient (fresh graph, no dependency on actor)
    feat_stack_h = feat_stack_all[:-1]          # [H, B, D]  (H states)
    H, B2, D2 = feat_stack_h.shape
    value_preds = critic(feat_stack_h.view(H * B2, D2)).view(H, B2)  # [H, B]
    critic_loss = F.mse_loss(value_preds, returns.detach())

    critic_optimizer.zero_grad(set_to_none=zero_grad)
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip)
    critic_optimizer.step()

    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "returns_mean": float(returns.detach().mean().cpu()),
        "returns_std": float(returns.detach().std().cpu()),
        "reward_mean": float(torch.stack(rewards).detach().mean().cpu()),
        "value_mean": float(values_flat[:-1].mean().cpu()),
        "actor_grad_norm": float(torch.as_tensor(actor_grad_norm).detach().cpu()),
        "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).detach().cpu()),
    }
