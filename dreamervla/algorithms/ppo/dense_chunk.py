"""Dense-reward LUMOS PPO route driven by the chunk WM.

Same **reward form** as ``dino_lumos_dense_step`` (dense per-step state-reward
decoded from the WM hidden at every imagined env-step), but the rollout is
driven by ``ChunkAwareWorldModel.predict_next_chunk`` so each
actor decision produces a K-step action chunk that the WM consumes in one
call.

Semantics:
    K       = lumos.chunk_size (VLA actor time_horizon, default 5)
    horizon = imagination_horizon, redefined as **number of chunk decisions**
              (one actor call per chunk). Total imagined env-steps = horizon * K.
    group_size = ppo_rollouts_per_start
    B_eff   = B * group_size

Per chunk c we record:
    actor_feat[c]                   — VLA hidden, frozen actor input
    action_chunk[c], old_lp[c]      — sampled K-step action chunk + chunk-level log_prob
    rewards[c, :, k]                — dense state-reward at imagined frame k

PPO ratio is computed on the chunk-level log_prob (sum over all K actions in
the chunk), and the return is the γ-discounted sum of all horizon * K
per-frame rewards. GRPO group-relative advantage is broadcast across all
chunks of a rollout.

Contrast with ``dino_lumos_dense_step`` (``ppo/dense.py``) which calls the
single-frame WM per env-step and the actor per env-step (wasting the actor's
K-1 chunk outputs), and with ``dino_lumos_step`` (``ppo/outcome.py``)
which uses the same chunk WM but a sparse outcome reward from a classifier.

Optional ``ref_policy``: KL penalty against a fixed reference, subtracted
from the chunk-level return before GRPO (LUMOS/verl convention).

Not yet wired (raise if requested for explicit failure):
  * TD-MPC critic terminal bootstrap + side-update
  * real_rollout_relabel side loss
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
from dreamervla.algorithms.validation import validate_ppo_hyperparameters
from dreamervla.utils.torch_utils import move_mapping_to_device


def dino_lumos_dense_chunk_step(
    policy: nn.Module,
    chunk_world_model: nn.Module,
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
    """One PPO/GRPO update using chunk-WM imagined trajectories + dense state-reward."""
    validate_ppo_hyperparameters(algorithm_cfg, prefix="algorithm_cfg")
    if (
        real_relabel_batch is not None
        and float((algorithm_cfg.get("real_rollout_relabel", {}) or {}).get("loss_scale", 0.0))
        > 0.0
    ):
        raise NotImplementedError(
            "dino_lumos_dense_chunk_step: real_rollout_relabel not yet wired; "
            "use dino_lumos_dense_step or set real_rollout_relabel.loss_scale=0."
        )
    if (critic is not None or target_critic is not None or critic_optimizer is not None) and bool(
        (algorithm_cfg.get("tdmpc_ac", {}) or {}).get("enabled", False)
    ):
        raise NotImplementedError(
            "dino_lumos_dense_chunk_step: TD-MPC critic side-update not yet wired; "
            "use dino_lumos_dense_step or set tdmpc_ac.enabled=false."
        )

    lumos_cfg = algorithm_cfg.get("lumos", {})
    K = int(lumos_cfg.get("chunk_size", 5))
    horizon = int(algorithm_cfg.get("imagination_horizon", 5))  # NOW: number of chunks
    if horizon < 1 or K < 1:
        raise ValueError(f"horizon={horizon}, K={K}; both must be >= 1")
    imag_last = int(algorithm_cfg.get("imag_last", 4))
    group_size = int(algorithm_cfg.get("ppo_rollouts_per_start", 4))
    update_epochs = int(algorithm_cfg.get("ppo_update_epochs", 1))
    clip_low = float(algorithm_cfg.get("clip_ratio_low", 0.2))
    clip_high = float(algorithm_cfg.get("clip_ratio_high", 0.28))
    clip_ratio_c = algorithm_cfg.get("clip_ratio_c", None)
    clip_log_ratio = algorithm_cfg.get("clip_log_ratio", None)
    entropy_coef = _entropy_coef(algorithm_cfg)
    kl_coef = float(algorithm_cfg.get("kl_coef", 0.0))
    gamma = float(algorithm_cfg.get("ppo_gamma", 1.0))
    adv_eps = float(algorithm_cfg.get("advantage_eps", 1.0e-6))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    zero_grad = bool(optim_cfg.get("zero_grad_set_to_none", True))
    use_ref = ref_policy is not None

    chunk_world_model.eval()
    policy.train()
    if ref_policy is not None:
        ref_policy.eval()

    obs = move_mapping_to_device(dict(obs), device)
    with torch.no_grad():
        latent_seq = _detach_latent(_world_model_observe_sequence(chunk_world_model, obs))
        seq_len = _latent_time_dim(latent_seq)
        starts = min(imag_last if imag_last > 0 else seq_len, seq_len)
        current = _repeat_latent(_flatten_last_steps(latent_seq, starts), group_size)

    actor_feats: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []  # [B_eff, K, A] per chunk — basis for PPO ratio
    action_token_ids: list[torch.Tensor | None] = []
    old_log_probs: list[torch.Tensor] = []
    ref_kls: list[torch.Tensor] = []
    chunk_rewards: list[torch.Tensor] = []  # [B_eff, K] dense state-reward per chunk
    chunk_clip_fracs: list[float] = []  # per-chunk fraction of action elements clipped
    chunk_clip_max: list[float] = []  # per-chunk max |clipped - unclipped| in env units

    with _temporarily_freeze(chunk_world_model):
        for _ in range(horizon):
            actor_feat = _world_model_actor_input(chunk_world_model, current).detach().float()
            with torch.no_grad():
                # return_chunk=True: actor samples the full K-step chunk stochastically
                # and returns chunk-level log_prob (sum over K * action_dim). This is
                # required for PPO to credit all K actions executed by the WM, not just
                # the first one.
                action_chunk, old_lp, _sample_extra = policy(
                    {
                        "mode": "sample",
                        "hidden": actor_feat,
                        "deterministic": False,
                        "return_chunk": True,
                    }
                )
            if action_chunk.ndim != 3 or action_chunk.shape[1] != K:
                raise ValueError(
                    f"action_chunk shape mismatch: got {tuple(action_chunk.shape)}, expected [B,K={K},action_dim]"
                )

            actor_feats.append(actor_feat)
            action_chunks.append(action_chunk.detach())
            sampled_token_ids = _sample_extra.get("action_token_ids")
            action_token_ids.append(
                sampled_token_ids.detach() if isinstance(sampled_token_ids, torch.Tensor) else None
            )
            old_log_probs.append(old_lp.detach())

            if use_ref:
                with torch.no_grad():
                    ref_eval_batch = {
                        "mode": "evaluate",
                        "hidden": actor_feat,
                        "action": action_chunk.detach(),
                    }
                    if action_token_ids[-1] is not None:
                        ref_eval_batch["action_token_ids"] = action_token_ids[-1]
                    ref_lp, _, _ = ref_policy(ref_eval_batch)
                # k1 KL estimator (signed) — unbiased in expectation but can
                # go negative on individual samples; this is the verl/DAPO
                # convention used downstream (subtract from reward before
                # GRPO normalization, not as a direct loss). Dashboards
                # comparing to clipped-KL penalties will see negatives —
                # interpret as "policy assigned higher log-prob than ref".
                ref_kls.append((old_lp.detach() - ref_lp).detach())

            with torch.no_grad():
                # When ``rssm_action_clip=True`` (default), the env-scale map
                # clips out-of-bounds samples before the WM executes them.
                # The stored ``action_chunk`` (used for PPO ratio) is the
                # PRE-clip raw policy sample, so any clipped element is
                # off-policy by the same shape as the original F1+F2 bug
                # (credit ≠ execution). We don't fix it here (would need a
                # tanh-bounded policy or to clip-then-credit) but we expose
                # the per-step clip fraction via dense_chunk/action_clip_*
                # metrics so the bias is observable instead of silent.
                # Detection: compare clipped vs unclipped env-scale maps
                # (the affine shift is identical, so the diff is purely the
                # clipping effect).
                wm_action_chunk = _actor_action_for_world_model(
                    action_chunk.detach(), algorithm_cfg
                )
                env_scale = str(algorithm_cfg.get("rssm_action_scale", "env")).lower()
                if env_scale in {"env", "libero_env", "libero"}:
                    env_unclipped = _actor_action_to_env_scale(
                        action_chunk.detach(),
                        algorithm_cfg,
                        clip=False,
                    )
                    env_clipped = _actor_action_to_env_scale(
                        action_chunk.detach(),
                        algorithm_cfg,
                        clip=True,
                    )
                    clip_delta = (env_clipped - env_unclipped).abs()
                    chunk_clip_fracs.append(float((clip_delta > 0).float().mean().item()))
                    chunk_clip_max.append(float(clip_delta.max().item()))
                next_seq = chunk_world_model(
                    {
                        "mode": "predict_next_chunk",
                        "latent": current,
                        "actions": wm_action_chunk,
                    }
                )
                hidden_seq = next_seq["hidden_seq"]  # [B_eff, K, ...]
                current = _detach_latent(
                    {
                        "history": next_seq["history"],
                        "actions": next_seq["actions"],
                        "hidden": next_seq["hidden"],
                    }
                )
                # Dense per-frame state-reward — decode at each of the K emitted frames.
                # The WM's reward head consumes the latent dict shape used by predict_next
                # (which downstreams to the hidden tensor), so a single-frame dict suffices.
                per_frame = [
                    _world_model_state_reward(chunk_world_model, {"hidden": hidden_seq[:, k]})
                    .detach()
                    .float()
                    for k in range(K)
                ]
                chunk_rewards.append(torch.stack(per_frame, dim=1))  # [B_eff, K]

    # rewards: [B_eff, horizon, K]
    reward_stack = torch.stack(chunk_rewards, dim=1)

    # γ-discount across the full horizon*K env-step axis.
    total_steps = horizon * K
    discounts = torch.pow(
        torch.full((total_steps,), gamma, device=device, dtype=reward_stack.dtype),
        torch.arange(total_steps, device=device, dtype=reward_stack.dtype),
    ).view(horizon, K)
    discounted = (reward_stack * discounts.unsqueeze(0)).sum(dim=(1, 2))  # [B_eff]

    # KL-into-reward (LUMOS/verl): subtract chunk-level KL sum from the return.
    if use_ref and ref_kls and kl_coef > 0.0:
        kl_per_chunk = torch.stack(ref_kls, dim=1).to(dtype=reward_stack.dtype)  # [B_eff, horizon]
        kl_per_rollout = kl_per_chunk.sum(dim=1)
        traj_score = discounted - kl_coef * kl_per_rollout
    else:
        kl_per_rollout = torch.zeros_like(discounted)
        traj_score = discounted

    advantages = _group_advantage(traj_score.detach(), group_size, adv_eps)  # [B_eff]
    old_log_prob_traj = torch.stack(old_log_probs, dim=1).sum(dim=1).detach()  # [B_eff]

    # ─── group-aligned micro-batch slices (PERF-W6) ───────────────────────
    # Bound the actor backward's peak memory to ONE group-aligned slice of the
    # effective batch instead of all `horizon` policy forwards for the full
    # B_eff at once. GRPO groups are CONTIGUOUS `group_size` blocks in B_eff
    # (`_repeat_latent` = repeat_interleave); the advantage is computed ONCE on
    # the full batch then detached, so slicing `advantages[lo:hi]` reproduces the
    # exact per-rollout weights. We slice in START units (one start =
    # `group_size` rollouts). The PPO + entropy loss is a plain mean over B_eff,
    # so each slice backprops `term.sum() / B_eff` (global normalizer) and the
    # accumulated gradient equals the full-batch `.mean()` backward bit-for-bit.
    # `lumos.update_micro_batch_starts` <= 0 or >= n_starts ⇒ one full-batch slice
    # = the original single backward.
    b_eff = int(advantages.shape[0])
    n_starts = b_eff // group_size
    mb_starts_cfg = int(lumos_cfg.get("update_micro_batch_starts", 0))
    mb_starts = n_starts if mb_starts_cfg <= 0 else min(max(1, mb_starts_cfg), n_starts)
    slice_bounds = [
        (s * group_size, min(s + mb_starts, n_starts) * group_size)
        for s in range(0, n_starts, mb_starts)
    ]

    # PPO update — chunk-level log_prob, ratio aggregated over the horizon chunks.
    total_actor_loss = 0.0
    total_entropy = 0.0
    grad_norm = torch.zeros((), device=device)
    ratio_last = torch.zeros((), device=device)
    for _ in range(update_epochs):
        actor_optimizer.zero_grad(set_to_none=zero_grad)
        epoch_loss = 0.0
        epoch_entropy_sum = 0.0
        ratio_records: list[torch.Tensor] = []
        for lo, hi in slice_bounds:
            new_log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            for actor_feat, action_detached, token_ids in zip(
                actor_feats, action_chunks, action_token_ids, strict=True
            ):
                # action_detached is [B_eff, K, A]; actor's evaluate path with ndim==3
                # returns chunk-level log_prob and entropy summed over (K, action_dim),
                # matching the chunk-level old_lp recorded above.
                eval_batch = {
                    "mode": "evaluate",
                    "hidden": actor_feat[lo:hi],
                    "action": action_detached[lo:hi],
                }
                if token_ids is not None:
                    eval_batch["action_token_ids"] = token_ids[lo:hi]
                new_lp, entropy_t, _ = policy(eval_batch)
                new_log_probs.append(new_lp)
                entropies.append(entropy_t)

            log_prob_stack = torch.stack(new_log_probs, dim=1)  # [mb, horizon]
            entropy_stack = torch.stack(entropies, dim=1)

            log_prob_traj = log_prob_stack.sum(dim=1)  # [mb]
            ratio = _ppo_ratio(
                log_prob_traj, old_log_prob_traj[lo:hi], clip_log_ratio=clip_log_ratio
            )
            # Sum / global B_eff (NOT slice size) so per-slice grads sum to the
            # full-batch `.mean()` gradient.
            pg_loss = (
                _ppo_clip_term(
                    ratio, advantages[lo:hi], clip_low, clip_high, clip_ratio_c=clip_ratio_c
                ).sum()
                / b_eff
            )
            ent_loss = (
                -(entropy_coef * entropy_stack.sum(dim=1)).sum() / b_eff
                if entropy_coef
                else pg_loss.new_zeros(())
            )
            loss = pg_loss + ent_loss
            loss.backward()

            epoch_loss += float(loss.detach().cpu())
            epoch_entropy_sum += float(entropy_stack.detach().sum().cpu())
            ratio_records.append(ratio.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
        actor_optimizer.step()

        total_actor_loss += epoch_loss
        # entropy_stack metric is a mean over (B_eff, horizon); reassemble from slices.
        total_entropy += epoch_entropy_sum / max(1, b_eff * horizon)
        ratio_last = torch.cat(ratio_records, dim=0)

    actor_loss_val = total_actor_loss / max(1, update_epochs)
    returns_mean = float(traj_score.detach().mean().cpu())
    returns_std = float(traj_score.detach().std().cpu())
    reward_mean = float(reward_stack.detach().mean().cpu())
    advantage_mag = float(advantages.detach().abs().mean().cpu())
    return {
        # Flat keys — matched to dreamervla_runner metric extraction so
        # this route is drop-in compatible with the existing dispatch.
        "actor_loss": actor_loss_val,
        "actor_bc_loss": 0.0,
        "actor_bc_scale": 0.0,
        "actor_vla_drift_raw_mse": 0.0,
        "actor_vla_drift_env_mse": 0.0,
        "actor_vla_drift_env_mse_clipped": 0.0,
        "actor_vla_drift_env_mae": 0.0,
        "critic_loss": 0.0,
        "returns_mean": returns_mean,
        "returns_std": returns_std,
        "raw_returns_mean": returns_mean,
        "raw_returns_std": returns_std,
        "advantage_mean": float(advantages.detach().mean().cpu()),
        "advantage_std": float(advantages.detach().std().cpu()),
        "advantage_mag": advantage_mag,
        "return_scale": 1.0,
        "reward_mean": reward_mean,
        "value_mean": 0.0,  # no critic in dense_chunk yet
        "actor_grad_norm": float(grad_norm.detach().cpu()),
        "critic_grad_norm": 0.0,
        "ppo_update_epochs": float(update_epochs),
        "ppo_ratio_mean": float(ratio_last.mean().cpu()) if ratio_last.numel() else 1.0,
        "ppo_ratio_min": float(ratio_last.min().cpu()) if ratio_last.numel() else 1.0,
        "ppo_ratio_max": float(ratio_last.max().cpu()) if ratio_last.numel() else 1.0,
        "ppo_clipfrac": float(
            ((ratio_last < 1.0 - clip_low) | (ratio_last > 1.0 + clip_high)).float().mean().cpu()
        )
        if ratio_last.numel()
        else 0.0,
        "continue_mean": 1.0,
        # Namespaced detail — for run-specific dashboards.
        "dense_chunk/actor_loss": actor_loss_val,
        "dense_chunk/avg_entropy": total_entropy / max(1, update_epochs),
        "dense_chunk/grad_norm": float(grad_norm.detach().cpu()),
        "dense_chunk/returns_mean": returns_mean,
        "dense_chunk/returns_std": returns_std,
        "dense_chunk/advantage_mag": advantage_mag,
        "dense_chunk/reward_mean": reward_mean,
        "dense_chunk/ref_kl_mean": float(kl_per_rollout.detach().mean().cpu()),
        "dense_chunk/ratio_mean": float(ratio_last.mean().cpu()) if ratio_last.numel() else 0.0,
        "dense_chunk/horizon_chunks": float(horizon),
        "dense_chunk/chunk_size": float(K),
        "dense_chunk/total_env_steps": float(total_steps),
        "dense_chunk/actor_grad_norm_adapter": _named_grad_norm(policy, "adapter"),
        "dense_chunk/actor_grad_norm_output_projection": _named_grad_norm(
            policy, "output_projection"
        ),
        "dense_chunk/actor_grad_norm_log_std": _named_grad_norm(policy, "log_std"),
        # Action-clip observability: the WM consumes a (potentially) clipped
        # version of the sampled action, but PPO credits the raw sample's
        # density. If ``action_clip_frac`` is non-trivial (>1-2%), the
        # PPO ratio is biased on those samples; consider tanh-bounding the
        # actor or switching to ``rssm_action_clip=False``.
        "dense_chunk/action_clip_frac": sum(chunk_clip_fracs) / max(1, len(chunk_clip_fracs)),
        "dense_chunk/action_clip_max_env": max(chunk_clip_max) if chunk_clip_max else 0.0,
    }


__all__ = ["dino_lumos_dense_chunk_step"]
