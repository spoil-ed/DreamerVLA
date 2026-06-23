"""Dense-reward WMPO / PPO / GRPO route.

**Reward form**: dense per-step state-reward. For each imagined env-step we
decode a scalar reward from the world-model hidden state and sum the
per-step rewards (γ-discounted) into the per-rollout return. Every step
contributes signal — the actor learns from a shaped trajectory rather than
a single terminal label.

For each real start frame we imagine ``imagination_horizon`` env-steps with
a frozen world model, sample one action per step from the current policy,
and run one (or more) PPO clip updates with GRPO group-relative advantages.
The current rollout loop drives the WM with the single-frame ``predict_next``
call (one latent in → one latent out).

Contrast with ``dino_wmpo_outcome_step`` (``ppo/outcome.py``), which produces
a single sparse outcome reward from a ``LatentSuccessClassifier`` scoring the
imagined latent video at success/finish, and which drives the WM in chunk
mode (``predict_next_chunk``) to align with the actor's K-step action chunk.

Optional add-ons:
  * ``ref_policy``: KL penalty against a fixed reference (subtracted from the
    pre-advantage return, matching the WMPO/verl convention).
  * ``actor_bc_to_ref_scale``: behavior-cloning anchor on the deterministic
    action chunk, drawn either against the ref policy or against
    ``policy.reference_action_chunk``.
  * ``real_rollout_relabel``: a side PPO loss on cached real-env samples.
  * ``tdmpc_ac``: terminal-value bootstrap and a TD-MPC critic side-update.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from dreamervla.algorithms.dreamervla import (
    _actor_action_for_world_model,
    _actor_action_to_env_scale,
    _detach_latent,
    _flatten_last_steps,
    _latent_time_dim,
    _named_grad_norm,
    _policy_reference_action_chunk,
    _temporarily_freeze,
    _world_model_actor_input,
    _world_model_observe_sequence,
    _world_model_state_reward,
)
from dreamervla.algorithms.ppo.grpo import (
    _entropy_coef,
    _group_advantage,
    _ppo_clip_term,
    _ppo_ratio,
    _repeat_latent,
)
from dreamervla.algorithms.ppo.relabel import _real_relabel_anchor_loss
from dreamervla.algorithms.ppo.tdmpc_critic import (
    _sequence_field,
    _tdmpc_action_dim,
    _tdmpc_critic_hidden,
    _tdmpc_value_mode,
)
from dreamervla.models.critic.twohot_critic import soft_update
from dreamervla.utils.torch_utils import move_mapping_to_device


def _dense_actor_backward_microbatched(
    *,
    policy: nn.Module,
    ref_policy: nn.Module | None,
    use_ref: bool,
    actor_feats: list[torch.Tensor],
    actions: list[torch.Tensor],
    action_token_ids: list[torch.Tensor | None],
    advantages: torch.Tensor,  # [B_eff] detached
    old_log_prob_traj: torch.Tensor,  # [B_eff] detached
    slice_bounds: list[tuple[int, int]],
    b_eff: int,
    algorithm_cfg: DictConfig,
    clip_low: float,
    clip_high: float,
    clip_ratio_c: float | None,
    clip_log_ratio: float | None,
    entropy_coef: float,
    actor_bc_ref_scale: float,
    real_relabel_term: torch.Tensor,
) -> dict[str, Any]:
    """Run the dense PPO actor backward over group-aligned B_eff micro-batches.

    The actor loss is ``mean_over_B_eff(ppo + entropy)`` plus a BC anchor that is
    ``mean_over_bc_steps(mean_over_(B_eff,K,A))`` and a batch-independent
    ``real_relabel_term``. Each slice backprops globally-normalized contributions
    (``sum / B_eff`` for PPO/entropy; ``sum / (b_eff*K*A*bc_steps)`` for BC) so the
    accumulated gradient equals the full-batch single backward bit-for-bit. The
    ``real_relabel_term`` is independent of the imagined batch, so it is added once
    (on the first slice). ``slice_bounds`` covering the whole batch in one slice
    reproduces the original behavior. Returns assembled metric tensors/values.

    The caller must have called ``actor_optimizer.zero_grad(...)`` before this.
    """
    ratio_records: list[torch.Tensor] = []
    log_prob_records: list[torch.Tensor] = []
    bc_value_sum = 0.0
    pg_value_sum = 0.0
    entropy_value_sum = 0.0
    drift_raw_mses: list[torch.Tensor] = []
    drift_env_mses: list[torch.Tensor] = []
    drift_env_clip_mses: list[torch.Tensor] = []
    drift_env_maes: list[torch.Tensor] = []
    total_loss = 0.0
    for slice_idx, (lo, hi) in enumerate(slice_bounds):
        new_log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        bc_terms: list[torch.Tensor] = []
        for actor_feat, action_detached, token_ids in zip(
            actor_feats, actions, action_token_ids, strict=True
        ):
            af = actor_feat[lo:hi]
            eval_batch = {
                "mode": "evaluate",
                "hidden": af,
                "action": action_detached[lo:hi],
            }
            if token_ids is not None:
                eval_batch["action_token_ids"] = token_ids[lo:hi]
            log_prob_t, entropy_t, _ = policy(eval_batch)
            new_log_probs.append(log_prob_t)
            entropies.append(entropy_t)

            _, _, extra = policy(
                {
                    "mode": "sample",
                    "hidden": af,
                    "deterministic": True,
                    "return_chunk": True,
                }
            )
            action_chunk = extra.get("action_chunk")
            if use_ref:
                if isinstance(action_chunk, torch.Tensor):
                    with torch.no_grad():
                        _, _, ref_extra = ref_policy(
                            {
                                "mode": "sample",
                                "hidden": af,
                                "deterministic": True,
                                "return_chunk": True,
                            }
                        )
                    ref_action_chunk = ref_extra.get("action_chunk")
                    if isinstance(ref_action_chunk, torch.Tensor):
                        action_chunk_f = action_chunk.float()
                        ref_chunk_f = ref_action_chunk.detach().float()
                        bc_terms.append((action_chunk_f - ref_chunk_f).square().sum())
                        drift_raw_mses.append(
                            (action_chunk_f.detach() - ref_chunk_f).square().mean()
                        )
                        action_env = _actor_action_to_env_scale(
                            action_chunk_f.detach(), algorithm_cfg, clip=False
                        )
                        ref_env = _actor_action_to_env_scale(
                            ref_chunk_f, algorithm_cfg, clip=False
                        )
                        action_env_clip = _actor_action_to_env_scale(
                            action_chunk_f.detach(), algorithm_cfg, clip=True
                        )
                        ref_env_clip = _actor_action_to_env_scale(
                            ref_chunk_f, algorithm_cfg, clip=True
                        )
                        drift_env_mses.append((action_env - ref_env).square().mean())
                        drift_env_clip_mses.append(
                            (action_env_clip - ref_env_clip).square().mean()
                        )
                        drift_env_maes.append((action_env - ref_env).abs().mean())
            else:
                reference_chunk = _policy_reference_action_chunk(policy, af)
                if isinstance(action_chunk, torch.Tensor) and isinstance(
                    reference_chunk, torch.Tensor
                ):
                    bc_terms.append(
                        (action_chunk.float() - reference_chunk.detach().float())
                        .square()
                        .sum()
                    )

        log_prob_stack = torch.stack(new_log_probs, dim=1)  # [mb, horizon]
        entropy_stack = torch.stack(entropies, dim=1)
        log_prob_traj = log_prob_stack.sum(dim=1)  # [mb]
        ratio = _ppo_ratio(
            log_prob_traj, old_log_prob_traj[lo:hi], clip_log_ratio=clip_log_ratio
        )
        # Sum / global B_eff (NOT slice size) so per-slice grads sum to the
        # full-batch `.mean()` gradient.
        actor_pg_loss = (
            _ppo_clip_term(
                ratio, advantages[lo:hi], clip_low, clip_high, clip_ratio_c=clip_ratio_c
            ).sum()
            / b_eff
        )
        actor_entropy_loss = (
            -(entropy_coef * entropy_stack.sum(dim=1)).sum() / b_eff
            if entropy_coef
            else actor_pg_loss.new_zeros(())
        )
        # BC anchor: original = mean over bc_steps of mean over (slice elements).
        # Each ``bc_terms`` element is the per-step SUM of squared diffs over this
        # slice's (mb, K, A); the global normalizer divides by the full count so
        # the per-slice sums accumulate to the full-batch mean exactly. ``bc_steps``
        # (the count of contributing horizon steps) is slice-invariant — validity
        # depends on the policy/ref returning a chunk tensor, not on the batch slice.
        if bc_terms:
            bc_steps = len(bc_terms)
            elems_per_step = action_chunk.float().reshape(action_chunk.shape[0], -1).shape[1]
            bc_norm = float(b_eff) * float(elems_per_step) * float(bc_steps)
            bc_loss = torch.stack(bc_terms).sum() / bc_norm
            # Σ_slices bc_loss == the original ``stack(bc_ref_losses).mean()``.
            bc_value_sum += float(bc_loss.detach().cpu())
        else:
            bc_loss = actor_pg_loss.new_zeros(())
        slice_loss = actor_pg_loss + actor_entropy_loss + actor_bc_ref_scale * bc_loss
        if slice_idx == 0:
            slice_loss = slice_loss + real_relabel_term
        slice_loss.backward()

        # Σ_slices of each globally-normalized term == the full-batch term value.
        pg_value_sum += float(actor_pg_loss.detach().cpu())
        entropy_value_sum += float(actor_entropy_loss.detach().cpu())
        total_loss += float(slice_loss.detach().cpu())
        ratio_records.append(ratio.detach())
        log_prob_records.append(log_prob_stack.detach())

    return {
        "actor_loss": total_loss,
        "actor_pg_loss": pg_value_sum,
        "actor_entropy_loss": entropy_value_sum,
        "ratio": torch.cat(ratio_records, dim=0),
        "log_prob_stack": torch.cat(log_prob_records, dim=0),
        "bc_ref_loss_value": bc_value_sum,
        "drift_raw_mses": drift_raw_mses,
        "drift_env_mses": drift_env_mses,
        "drift_env_clip_mses": drift_env_clip_mses,
        "drift_env_maes": drift_env_maes,
    }


def dino_wmpo_dense_step(
    policy: nn.Module,
    world_model: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    obs: Mapping[str, Any],
    device: torch.device,
    algorithm_cfg: DictConfig,
    optim_cfg: DictConfig,
    ref_policy: nn.Module | None = None,
    real_relabel_batch: Mapping[str, torch.Tensor] | None = None,
    critic: nn.Module | None = None,
    target_critic: nn.Module | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One PPO/GRPO-like update using DINO-WM imagined trajectory rewards.

    The world model is used as a frozen imagination environment for this RL
    update.  It may still be trained by the separate supervised WM phase.
    """
    horizon = int(algorithm_cfg.get("imagination_horizon", 5))
    imag_last = int(algorithm_cfg.get("imag_last", 4))
    group_size = int(algorithm_cfg.get("ppo_rollouts_per_start", 4))
    update_epochs = max(1, int(algorithm_cfg.get("ppo_update_epochs", 1)))
    clip_low = float(algorithm_cfg.get("clip_ratio_low", 0.2))
    clip_high = float(algorithm_cfg.get("clip_ratio_high", 0.28))
    clip_ratio_c = algorithm_cfg.get("clip_ratio_c", None)
    clip_log_ratio = algorithm_cfg.get("clip_log_ratio", None)
    entropy_coef = _entropy_coef(algorithm_cfg)
    kl_coef = float(algorithm_cfg.get("kl_coef", 0.0))
    actor_bc_ref_scale = float(algorithm_cfg.get("actor_bc_to_ref_scale", 0.0))
    real_relabel_cfg = algorithm_cfg.get("real_rollout_relabel", {}) or {}
    real_relabel_scale = float(real_relabel_cfg.get("loss_scale", 0.0))
    tdmpc_ac_cfg = algorithm_cfg.get("tdmpc_ac", {}) or {}
    tdmpc_ac_enabled = bool(tdmpc_ac_cfg.get("enabled", False))
    tdmpc_value_mode = _tdmpc_value_mode(tdmpc_ac_cfg)
    tdmpc_critic_action_dim = _tdmpc_action_dim(
        tdmpc_ac_cfg, int(algorithm_cfg.get("rssm_action_dim", 7))
    )
    tdmpc_ac_ready = (
        tdmpc_ac_enabled
        and critic is not None
        and target_critic is not None
        and critic_optimizer is not None
    )
    tdmpc_terminal_value_scale = float(tdmpc_ac_cfg.get("terminal_value_scale", 1.0))
    tdmpc_critic_loss_scale = float(tdmpc_ac_cfg.get("critic_loss_scale", 1.0))
    tdmpc_imagined_critic_loss_scale = float(
        tdmpc_ac_cfg.get("imagined_critic_loss_scale", tdmpc_critic_loss_scale)
    )
    tdmpc_replay_critic_loss_scale = float(
        tdmpc_ac_cfg.get("replay_critic_loss_scale", 1.0)
    )
    tdmpc_target_tau = float(
        tdmpc_ac_cfg.get(
            "target_critic_tau", algorithm_cfg.get("target_critic_tau", 0.02)
        )
    )
    gamma = float(algorithm_cfg.get("ppo_gamma", 1.0))
    adv_eps = float(algorithm_cfg.get("advantage_eps", 1.0e-6))
    zero_grad = bool(optim_cfg.get("zero_grad_set_to_none", True))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    use_ref = ref_policy is not None

    world_model.eval()
    policy.train()
    if critic is not None:
        critic.train()
    if target_critic is not None:
        target_critic.eval()
    if ref_policy is not None:
        ref_policy.eval()

    obs = move_mapping_to_device(obs, device)
    with torch.no_grad():
        latent_seq = _detach_latent(_world_model_observe_sequence(world_model, obs))
        seq_len = _latent_time_dim(latent_seq)
        starts = min(imag_last if imag_last > 0 else seq_len, seq_len)
        current_latent = _repeat_latent(
            _flatten_last_steps(latent_seq, starts), group_size
        )

    latents: list[Any] = [current_latent]
    actor_feats: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    action_token_ids: list[torch.Tensor | None] = []
    old_log_probs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    ref_kls: list[torch.Tensor] = []

    with _temporarily_freeze(world_model):
        for step in range(horizon):
            del step
            actor_feat = (
                _world_model_actor_input(world_model, current_latent).detach().float()
            )
            with torch.no_grad():
                action, old_log_prob_t, extra = policy(
                    {"mode": "sample", "hidden": actor_feat, "deterministic": False}
                )
            action_detached = action.detach()
            actor_feats.append(actor_feat)
            actions.append(action_detached)
            sampled_token_ids = extra.get("action_token_ids")
            action_token_ids.append(
                sampled_token_ids.detach()
                if isinstance(sampled_token_ids, torch.Tensor)
                else None
            )
            old_log_probs.append(old_log_prob_t.detach())

            if use_ref:
                with torch.no_grad():
                    ref_eval_batch = {
                        "mode": "evaluate",
                        "hidden": actor_feat,
                        "action": action_detached,
                    }
                    if action_token_ids[-1] is not None:
                        ref_eval_batch["action_token_ids"] = action_token_ids[-1]
                    ref_log_prob_t, _, ref_extra_eval = ref_policy(ref_eval_batch)
                ref_kls.append((old_log_prob_t.detach() - ref_log_prob_t).detach())
                del ref_extra_eval

            with torch.no_grad():
                wm_action = _actor_action_for_world_model(
                    action_detached, algorithm_cfg
                )
                current_latent = _detach_latent(
                    world_model(
                        {
                            "mode": "predict_next",
                            "latent": current_latent,
                            "actions": wm_action,
                        }
                    )
                )
                latents.append(current_latent)
                rewards.append(
                    _world_model_state_reward(world_model, current_latent)
                    .detach()
                    .float()
                )

    if not actor_feats:
        raise RuntimeError("DINO-WM PPO requires at least one imagined actor step.")

    # The advantage is built from the imagined rewards (+ optional TD-MPC terminal
    # value) ONLY — it never depends on the policy re-evaluation forwards — so we
    # compute it first, then run the actor backward over group-aligned B_eff
    # micro-batches (the forwards / loss / backward all live in
    # ``_dense_actor_backward_microbatched``).
    old_log_prob_traj = torch.stack(old_log_probs, dim=1).sum(dim=1).detach()  # [B_eff]
    reward_stack = torch.stack(rewards, dim=1)
    adjusted_reward = reward_stack
    kl_stack = None
    if ref_kls and kl_coef > 0.0:
        kl_stack = torch.stack(ref_kls, dim=1).to(dtype=reward_stack.dtype)
        adjusted_reward = reward_stack - kl_coef * kl_stack

    discounts = torch.pow(
        torch.full((horizon,), gamma, device=device, dtype=adjusted_reward.dtype),
        torch.arange(horizon, device=device, dtype=adjusted_reward.dtype),
    )
    traj_score = (adjusted_reward * discounts[None]).sum(dim=1)
    tdmpc_terminal_value = torch.zeros_like(traj_score)
    tdmpc_critic_loss = torch.zeros((), device=device, dtype=traj_score.dtype)
    tdmpc_imagined_critic_loss = torch.zeros((), device=device, dtype=traj_score.dtype)
    tdmpc_replay_critic_loss = torch.zeros((), device=device, dtype=traj_score.dtype)
    tdmpc_critic_grad_norm = torch.zeros((), device=device)
    tdmpc_ac_applied = False
    tdmpc_replay_value_applied = False
    tdmpc_replay_reward_mean = torch.zeros((), device=device, dtype=traj_score.dtype)
    tdmpc_replay_target_mean = torch.zeros((), device=device, dtype=traj_score.dtype)
    tdmpc_replay_value_mean = torch.zeros((), device=device, dtype=traj_score.dtype)
    if tdmpc_ac_ready:
        with torch.no_grad():
            terminal_action = None
            if tdmpc_value_mode == "state_action":
                terminal_actor_feat = (
                    _world_model_actor_input(world_model, latents[-1]).detach().float()
                )
                terminal_action, _, _ = policy(
                    {
                        "mode": "sample",
                        "hidden": terminal_actor_feat,
                        "deterministic": True,
                    }
                )
                terminal_action = _actor_action_for_world_model(
                    terminal_action.detach(), algorithm_cfg
                )
            terminal_feat = _tdmpc_critic_hidden(
                world_model,
                latents[-1],
                terminal_action,
                value_mode=tdmpc_value_mode,
                action_dim=tdmpc_critic_action_dim,
            )
            tdmpc_terminal_value = (
                target_critic({"mode": "value", "hidden": terminal_feat})
                .detach()
                .to(dtype=traj_score.dtype)
            )
            tdmpc_terminal_value = tdmpc_terminal_value.reshape_as(traj_score)
        if tdmpc_terminal_value_scale != 0.0:
            traj_score = (
                traj_score
                + (float(gamma) ** horizon)
                * tdmpc_terminal_value_scale
                * tdmpc_terminal_value
            )
    advantages = _group_advantage(traj_score.detach(), group_size, adv_eps)

    # ─── group-aligned micro-batch slices (PERF-W6) ───────────────────────
    # Bound the actor backward's peak memory to ONE group-aligned slice of B_eff
    # instead of all `horizon` policy forwards for the full batch at once. GRPO
    # groups are CONTIGUOUS `group_size` blocks (`_repeat_latent` =
    # repeat_interleave); the advantage is computed ONCE on the full batch then
    # detached, so slicing `advantages[lo:hi]` reproduces the exact per-rollout
    # weights. Slices are in START units (one start = `group_size` rollouts). The
    # PPO + entropy + BC loss are means over B_eff, so each slice backprops a
    # global-B_eff-normalized contribution and the accumulated gradient equals the
    # full-batch single backward bit-for-bit. `wmpo.update_micro_batch_starts`
    # <= 0 or >= n_starts ⇒ ONE full-batch slice = the original behavior.
    b_eff = int(advantages.shape[0])
    n_starts = b_eff // group_size
    mb_starts_cfg = int(
        (algorithm_cfg.get("wmpo", {}) or {}).get("update_micro_batch_starts", 0)
    )
    mb_starts = n_starts if mb_starts_cfg <= 0 else min(max(1, mb_starts_cfg), n_starts)
    slice_bounds = [
        (s * group_size, min(s + mb_starts, n_starts) * group_size)
        for s in range(0, n_starts, mb_starts)
    ]

    real_relabel_loss, real_relabel_metrics = _real_relabel_anchor_loss(
        policy=policy,
        real_relabel_batch=real_relabel_batch,
        clip_low=clip_low,
        clip_high=clip_high,
        clip_log_ratio=clip_log_ratio,
        clip_ratio_c=clip_ratio_c,
    )
    if real_relabel_loss is None or real_relabel_scale <= 0.0:
        real_relabel_term = traj_score.new_zeros(())
    else:
        real_relabel_term = float(real_relabel_scale) * real_relabel_loss

    actor_optimizer.zero_grad(set_to_none=zero_grad)
    update_out = _dense_actor_backward_microbatched(
        policy=policy,
        ref_policy=ref_policy,
        use_ref=use_ref,
        actor_feats=actor_feats,
        actions=actions,
        action_token_ids=action_token_ids,
        advantages=advantages,
        old_log_prob_traj=old_log_prob_traj,
        slice_bounds=slice_bounds,
        b_eff=b_eff,
        algorithm_cfg=algorithm_cfg,
        clip_low=clip_low,
        clip_high=clip_high,
        clip_ratio_c=clip_ratio_c,
        clip_log_ratio=clip_log_ratio,
        entropy_coef=entropy_coef,
        actor_bc_ref_scale=actor_bc_ref_scale,
        real_relabel_term=real_relabel_term,
    )
    ratio = update_out["ratio"]
    log_prob_stack = update_out["log_prob_stack"]
    bc_ref_loss_value = update_out["bc_ref_loss_value"]
    actor_loss_value = update_out["actor_loss"]
    actor_pg_loss_value = update_out["actor_pg_loss"]
    actor_entropy_loss_value = update_out["actor_entropy_loss"]
    drift_raw_mses = update_out["drift_raw_mses"]
    drift_env_mses = update_out["drift_env_mses"]
    drift_env_clip_mses = update_out["drift_env_clip_mses"]
    drift_env_maes = update_out["drift_env_maes"]
    actor_adapter_grad_norm = _named_grad_norm(policy, "adapter")
    actor_output_projection_grad_norm = _named_grad_norm(policy, "output_projection")
    actor_log_std_grad_norm = _named_grad_norm(policy, "log_std")
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), max_norm=grad_clip
    )
    actor_optimizer.step()

    if tdmpc_ac_ready and (
        tdmpc_imagined_critic_loss_scale > 0.0 or tdmpc_replay_critic_loss_scale > 0.0
    ):
        if tdmpc_imagined_critic_loss_scale > 0.0:
            with torch.no_grad():
                target_return = (
                    tdmpc_terminal_value_scale * tdmpc_terminal_value.detach()
                )
                returns_reversed: list[torch.Tensor] = []
                for step in reversed(range(horizon)):
                    target_return = (
                        adjusted_reward[:, step].detach() + float(gamma) * target_return
                    )
                    returns_reversed.append(target_return)
                returns_reversed.reverse()
                tdmpc_targets = torch.stack(returns_reversed, dim=1)
                critic_feat_stack = torch.stack(
                    [
                        _tdmpc_critic_hidden(
                            world_model,
                            latent,
                            _actor_action_for_world_model(action, algorithm_cfg)
                            if tdmpc_value_mode == "state_action"
                            else None,
                            value_mode=tdmpc_value_mode,
                            action_dim=tdmpc_critic_action_dim,
                        )
                        for latent, action in zip(latents[:-1], actions, strict=True)
                    ],
                    dim=1,
                )
            B, H, D = critic_feat_stack.shape
            tdmpc_log_probs = critic(
                {
                    "mode": "log_prob",
                    "hidden": critic_feat_stack.reshape(B * H, D),
                    "values": tdmpc_targets.reshape(B * H),
                }
            )
            tdmpc_imagined_critic_loss = (
                -tdmpc_log_probs.view(B, H).mean() * tdmpc_imagined_critic_loss_scale
            )

        replay_rewards = _sequence_field(
            obs,
            ("rewards", "reward"),
            device=device,
            dtype=traj_score.dtype,
        )
        if tdmpc_replay_critic_loss_scale > 0.0 and replay_rewards is not None:
            replay_terminal = _sequence_field(
                obs,
                ("is_terminal", "dones"),
                device=device,
                dtype=traj_score.dtype,
            )
            replay_last = _sequence_field(
                obs,
                ("is_last", "dones"),
                device=device,
                dtype=traj_score.dtype,
            )
            if replay_terminal is None:
                replay_terminal = torch.zeros_like(replay_rewards)
            if replay_last is None:
                replay_last = replay_terminal
            replay_actions = obs.get("actions")
            if tdmpc_value_mode == "state_action" and not isinstance(
                replay_actions, torch.Tensor
            ):
                raise KeyError(
                    "TD-MPC state-action replay critic requires obs['actions']."
                )
            replay_critic_feat = _tdmpc_critic_hidden(
                world_model,
                latent_seq,
                replay_actions if isinstance(replay_actions, torch.Tensor) else None,
                value_mode=tdmpc_value_mode,
                action_dim=tdmpc_critic_action_dim,
            )
            if replay_critic_feat.ndim != 3:
                raise ValueError(
                    "TD-MPC replay critic loss expects critic features [B,T,D], "
                    f"got {tuple(replay_critic_feat.shape)}"
                )
            replay_steps = min(
                int(replay_critic_feat.shape[1]),
                int(replay_rewards.shape[1]),
                int(replay_terminal.shape[1]),
                int(replay_last.shape[1]),
            )
            if replay_steps >= 2:
                replay_critic_feat = replay_critic_feat[:, -replay_steps:]
                replay_rewards = replay_rewards[:, -replay_steps:]
                replay_terminal = replay_terminal[:, -replay_steps:]
                replay_last = replay_last[:, -replay_steps:]
                B_rep, T_rep, D_rep = replay_critic_feat.shape
                replay_current_feat = replay_critic_feat[:, :-1]
                replay_next_feat = replay_critic_feat[:, 1:]
                with torch.no_grad():
                    if tdmpc_value_mode == "state_action":
                        next_latent = latent_seq
                        if isinstance(latent_seq, dict):
                            next_latent = {
                                key: value[:, -replay_steps:][:, 1:]
                                if isinstance(value, torch.Tensor) and value.ndim >= 3
                                else value
                                for key, value in latent_seq.items()
                            }
                        elif isinstance(latent_seq, torch.Tensor):
                            next_latent = latent_seq[:, -replay_steps:][:, 1:]
                        next_actor_feat = (
                            _world_model_actor_input(world_model, next_latent)
                            .detach()
                            .float()
                        )
                        next_action, _, _ = policy(
                            {
                                "mode": "sample",
                                "hidden": next_actor_feat.reshape(
                                    B_rep * (T_rep - 1), -1
                                ),
                                "deterministic": True,
                            }
                        )
                        next_action = _actor_action_for_world_model(
                            next_action.detach(), algorithm_cfg
                        )
                        next_action = next_action.reshape(B_rep, T_rep - 1, -1)
                        next_feat = _tdmpc_critic_hidden(
                            world_model,
                            next_latent,
                            next_action,
                            value_mode=tdmpc_value_mode,
                            action_dim=tdmpc_critic_action_dim,
                        )
                        replay_next_feat = next_feat
                    replay_next_value = target_critic(
                        {
                            "mode": "value",
                            "hidden": replay_next_feat.reshape(
                                B_rep * (T_rep - 1), D_rep
                            ),
                        }
                    ).view(B_rep, T_rep - 1)
                    replay_target = (
                        replay_rewards[:, 1:]
                        + float(gamma)
                        * (1.0 - replay_terminal[:, 1:].float())
                        * replay_next_value
                    )
                    replay_mask = (1.0 - replay_last[:, :-1].float()).clamp_min(0.0)
                    tdmpc_replay_reward_mean = replay_rewards[:, 1:].detach().mean()
                    tdmpc_replay_target_mean = replay_target.detach().mean()
                    tdmpc_replay_value_mean = replay_next_value.detach().mean()
                replay_log_probs = critic(
                    {
                        "mode": "log_prob",
                        "hidden": replay_current_feat.reshape(
                            B_rep * (T_rep - 1), D_rep
                        ),
                        "values": replay_target.detach().reshape(B_rep * (T_rep - 1)),
                    }
                )
                replay_loss_per_step = -replay_log_probs.view(B_rep, T_rep - 1)
                tdmpc_replay_critic_loss = (
                    (replay_loss_per_step * replay_mask).sum()
                    / replay_mask.sum().clamp_min(1.0)
                ) * tdmpc_replay_critic_loss_scale
                tdmpc_replay_value_applied = True

        tdmpc_critic_loss = tdmpc_imagined_critic_loss + tdmpc_replay_critic_loss
        if tdmpc_critic_loss.requires_grad:
            critic_optimizer.zero_grad(set_to_none=zero_grad)
            tdmpc_critic_loss.backward()
            tdmpc_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), max_norm=grad_clip
            )
            critic_optimizer.step()
            soft_update(target_critic, critic, tau=tdmpc_target_tau)
            tdmpc_ac_applied = True

    for _update_epoch in range(1, update_epochs):
        real_relabel_loss, real_relabel_metrics = _real_relabel_anchor_loss(
            policy=policy,
            real_relabel_batch=real_relabel_batch,
            clip_low=clip_low,
            clip_high=clip_high,
            clip_log_ratio=clip_log_ratio,
            clip_ratio_c=clip_ratio_c,
        )
        if real_relabel_loss is None or real_relabel_scale <= 0.0:
            real_relabel_term = traj_score.new_zeros(())
        else:
            real_relabel_term = float(real_relabel_scale) * real_relabel_loss

        actor_optimizer.zero_grad(set_to_none=zero_grad)
        update_out = _dense_actor_backward_microbatched(
            policy=policy,
            ref_policy=ref_policy,
            use_ref=use_ref,
            actor_feats=actor_feats,
            actions=actions,
            action_token_ids=action_token_ids,
            advantages=advantages,
            old_log_prob_traj=old_log_prob_traj,
            slice_bounds=slice_bounds,
            b_eff=b_eff,
            algorithm_cfg=algorithm_cfg,
            clip_low=clip_low,
            clip_high=clip_high,
            clip_ratio_c=clip_ratio_c,
            clip_log_ratio=clip_log_ratio,
            entropy_coef=entropy_coef,
            actor_bc_ref_scale=actor_bc_ref_scale,
            real_relabel_term=real_relabel_term,
        )
        ratio = update_out["ratio"]
        log_prob_stack = update_out["log_prob_stack"]
        bc_ref_loss_value = update_out["bc_ref_loss_value"]
        actor_loss_value = update_out["actor_loss"]
        actor_pg_loss_value = update_out["actor_pg_loss"]
        actor_entropy_loss_value = update_out["actor_entropy_loss"]
        drift_raw_mses = update_out["drift_raw_mses"]
        drift_env_mses = update_out["drift_env_mses"]
        drift_env_clip_mses = update_out["drift_env_clip_mses"]
        drift_env_maes = update_out["drift_env_maes"]
        actor_adapter_grad_norm = _named_grad_norm(policy, "adapter")
        actor_output_projection_grad_norm = _named_grad_norm(
            policy, "output_projection"
        )
        actor_log_std_grad_norm = _named_grad_norm(policy, "log_std")
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=grad_clip
        )
        actor_optimizer.step()

    def _mean_or_zero(items: list[torch.Tensor]) -> float:
        if not items:
            return 0.0
        return float(torch.stack(items).mean().detach().cpu())

    return {
        "actor_loss": float(actor_loss_value),
        "actor_pg_loss": float(actor_pg_loss_value),
        "actor_entropy_loss": float(actor_entropy_loss_value),
        "actor_bc_ref_loss": float(bc_ref_loss_value),
        "actor_bc_ref_scale": float(actor_bc_ref_scale),
        "real_relabel_scale": float(real_relabel_scale),
        "real_relabel_term": float(real_relabel_term.detach().cpu()),
        "actor_bc_loss": float(bc_ref_loss_value),
        "actor_bc_scale": float(actor_bc_ref_scale),
        "ppo_update_epochs": float(update_epochs),
        "critic_loss": float(tdmpc_critic_loss.detach().cpu()),
        "returns_mean": float(traj_score.detach().mean().cpu()),
        "returns_std": float(traj_score.detach().std().cpu()),
        "raw_returns_mean": float(traj_score.detach().mean().cpu()),
        "raw_returns_std": float(traj_score.detach().std().cpu()),
        "advantage_mean": float(advantages.detach().mean().cpu()),
        "advantage_std": float(advantages.detach().std().cpu()),
        "advantage_mag": float(advantages.detach().abs().mean().cpu()),
        "return_scale": 1.0,
        "reward_mean": float(reward_stack.detach().mean().cpu()),
        "reward_raw_mean": float(reward_stack.detach().mean().cpu()),
        "reward_raw_std": float(reward_stack.detach().std().cpu()),
        "ref_kl_mean": float(kl_stack.detach().mean().cpu())
        if kl_stack is not None
        else 0.0,
        "kl_coef": float(kl_coef),
        "continue_mean": 1.0,
        "value_mean": float(tdmpc_terminal_value.detach().mean().cpu())
        if tdmpc_ac_ready
        else 0.0,
        "critic_target_mean": float(traj_score.detach().mean().cpu()),
        "actor_grad_norm": float(torch.as_tensor(actor_grad_norm).detach().cpu()),
        "critic_grad_norm": float(
            torch.as_tensor(tdmpc_critic_grad_norm).detach().cpu()
        ),
        "tdmpc_ac_applied": float(tdmpc_ac_applied),
        "tdmpc_terminal_value_mean": float(tdmpc_terminal_value.detach().mean().cpu())
        if tdmpc_ac_ready
        else 0.0,
        "tdmpc_terminal_value_scale": float(tdmpc_terminal_value_scale),
        "tdmpc_critic_loss_scale": float(tdmpc_critic_loss_scale),
        "tdmpc_value_mode": tdmpc_value_mode,
        "tdmpc_critic_action_dim": float(tdmpc_critic_action_dim),
        "tdmpc_imagined_critic_loss": float(tdmpc_imagined_critic_loss.detach().cpu()),
        "tdmpc_imagined_critic_loss_scale": float(tdmpc_imagined_critic_loss_scale),
        "tdmpc_replay_critic_loss": float(tdmpc_replay_critic_loss.detach().cpu()),
        "tdmpc_replay_critic_loss_scale": float(tdmpc_replay_critic_loss_scale),
        "tdmpc_replay_value_applied": float(tdmpc_replay_value_applied),
        "tdmpc_replay_reward_mean": float(tdmpc_replay_reward_mean.detach().cpu()),
        "tdmpc_replay_target_mean": float(tdmpc_replay_target_mean.detach().cpu()),
        "tdmpc_replay_value_mean": float(tdmpc_replay_value_mean.detach().cpu()),
        "actor_grad_norm_adapter": actor_adapter_grad_norm,
        "actor_grad_norm_output_projection": actor_output_projection_grad_norm,
        "actor_grad_norm_log_std": actor_log_std_grad_norm,
        "actor_vla_drift_raw_mse": _mean_or_zero(drift_raw_mses),
        "actor_vla_drift_env_mse": _mean_or_zero(drift_env_mses),
        "actor_vla_drift_env_mse_clipped": _mean_or_zero(drift_env_clip_mses),
        "actor_vla_drift_env_mae": _mean_or_zero(drift_env_maes),
        "ppo_ratio_mean": float(ratio.detach().mean().cpu()),
        "ppo_ratio_min": float(ratio.detach().min().cpu()),
        "ppo_ratio_max": float(ratio.detach().max().cpu()),
        "ppo_clipfrac": float(
            ((ratio.detach() < 1.0 - clip_low) | (ratio.detach() > 1.0 + clip_high))
            .float()
            .mean()
            .cpu()
        ),
        "log_prob_mean": float(log_prob_stack.detach().mean().cpu()),
        "log_prob_std": float(log_prob_stack.detach().std().cpu()),
        **real_relabel_metrics,
    }


__all__ = ["dino_wmpo_dense_step"]
