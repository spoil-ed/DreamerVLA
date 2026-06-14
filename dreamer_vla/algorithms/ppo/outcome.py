# ruff: noqa: E402
"""Outcome-reward WMPO PPO route.

**Reward form**: sparse outcome reward. After imagining a full episode in
the world model, ``LatentSuccessClassifier.predict_success`` scores the
imagined latent video and emits ``(complete, finish_step)``. We place
``float(complete)`` at ``finish_step`` and zero elsewhere — one positive
signal per successful rollout, none otherwise.

This is the DreamerVLA-side reproduction of the WMPO/verl PPO loop. The
rollout drives the WM in chunk mode (``ChunkAwareRynnDinoWMWorldModel.
predict_next_chunk``) so one WM call advances ``action_chunks_len`` env
steps in lockstep with the RynnVLA actor's K-step action chunk.

Contrast with ``dino_wmpo_dense_step`` (``ppo/dense.py``), which decodes a
dense per-step state-reward from the WM hidden at every imagined env-step.

    real start frame
        → encode to WM latent
        → repeat for GRPO group
        → loop episode_max_steps // K chunks:
              RynnVLA actor (chunk-output)  → action_chunk[B, K, 7]
              chunk WM (chunk-input)     → next K latent frames
              accumulate K latents to a video buffer
        → LatentSuccessClassifier.predict_success on the video
            → (complete[B], finish_step[B])
        → reward[i, finish_step[i]] = float(complete[i])
        → GRPO group-relative advantage, broadcast across all chunks
        → PPO clip + KL-to-ref + entropy loss
        → actor update
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn

_logger = logging.getLogger(__name__)
_warned_missing_cfg: bool = False

from dreamer_vla.algorithms.dreamer_vla import (
    _detach_latent,
    _flatten_last_steps,
    _latent_time_dim,
    _policy_reference_action_chunk,
    _temporarily_freeze,
    _world_model_actor_input,
    _world_model_observe_sequence,
)
from dreamer_vla.algorithms.ppo.grpo import _group_advantage, _repeat_latent
from dreamer_vla.utils.torch_utils import move_mapping_to_device


def build_valid_chunk_count(
    finish_step: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
) -> torch.Tensor:
    """Number of valid actor chunks per rollout, aligned with WMPO's eos_mask.

    A chunk c spans env-steps ``[c*K, (c+1)*K)``. The chunk that contains
    ``finish_step`` is ``finish_step // K``. We INCLUDE that chunk (the actor's
    decision drove the env up to and including the success frame) and mask
    everything strictly after — so ``valid_chunks = (finish_step // K) + 1``.

    Args:
        finish_step: [B] env-step index of the success frame, or T_max-1 for
            failed episodes.
        chunk_size: K, env-steps per actor decision (e.g., 5 for RynnVLA).
        num_chunks: total chunks in the episode (=T_max // K).

    Returns:
        [B] long tensor, each value in [1, num_chunks].
    """
    K = int(chunk_size)
    counts = (finish_step // K) + 1
    return counts.long().clamp_(min=1, max=int(num_chunks))


def _build_reward_tensor(
    *,
    batch: int,
    max_steps: int,
    chunk_size: int,
    finish_step: torch.Tensor,
    complete: torch.Tensor,
) -> torch.Tensor:
    """Place a sparse outcome reward at finish_step for complete episodes.

    Args:
        batch: B_eff (B * group_size after repeat).
        max_steps: T_max (episode horizon in env-step units, not chunks).
        chunk_size: K. Currently unused for placement (env-step units), kept for
            API parity with WMPO's RobRewardManager which uses action_token_len.
        finish_step: [B] env-step indices.
        complete: [B] bool.

    Returns:
        [B, T_max] float32 tensor on CPU. Caller moves to device.
    """
    del chunk_size  # placement uses env-step index directly
    reward = torch.zeros((batch, max_steps), dtype=torch.float32)
    if max_steps <= 0:
        return reward
    finish = finish_step.detach().cpu().long().clamp(min=0, max=max_steps - 1)
    comp = complete.detach().cpu().bool()
    for i in range(batch):
        if comp[i].item():
            reward[i, finish[i].item()] = 1.0
    return reward


def _zip_lists(
    actor_feats: list[torch.Tensor],
    actions: list[torch.Tensor],
    action_token_ids: list[torch.Tensor | None],
    old_log_probs: list[torch.Tensor],
    ref_kls: list[torch.Tensor] | None,
):
    if ref_kls is None:
        for a, b, c, d in zip(
            actor_feats, actions, action_token_ids, old_log_probs, strict=True
        ):
            yield a, b, c, d, None
    else:
        for a, b, c, d, e in zip(
            actor_feats, actions, action_token_ids, old_log_probs, ref_kls, strict=True
        ):
            yield a, b, c, d, e


def dino_wmpo_outcome_step(
    policy: nn.Module,
    chunk_world_model: nn.Module,
    classifier: nn.Module,
    classifier_threshold: float,
    actor_optimizer: torch.optim.Optimizer,
    obs: Mapping[str, Any],
    device: torch.device,
    algorithm_cfg: DictConfig,
    optim_cfg: DictConfig,
    ref_policy: nn.Module | None = None,
) -> dict[str, float]:
    """One WMPO PPO step.

    Shape conventions:
        K       = algorithm_cfg.wmpo.chunk_size (RynnVLA actor time_horizon, default 5)
        T_max   = algorithm_cfg.wmpo.episode_max_steps (libero_goal: 300)
        num_chunks = T_max // K
        group_size = algorithm_cfg.ppo_rollouts_per_start
        B_eff   = B * group_size
    """
    wmpo_cfg = algorithm_cfg.get("wmpo", {})
    K = int(wmpo_cfg.get("chunk_size", 5))
    T_max = int(wmpo_cfg.get("episode_max_steps", 300))
    num_chunks = T_max // K
    if num_chunks < 1:
        raise ValueError(f"episode_max_steps={T_max} too small for chunk_size={K}")
    # min_steps for the classifier sliding-window sweep — windows ending before
    # this index are skipped.  Unit MUST match the classifier's native
    # granularity (read from ``classifier_module.cfg.granularity`` below):
    # action classifier → env-step, chunk classifier → chunk.  Default below
    # is in chunk units (~num_chunks/15: e.g. 60/15=4 for libero_goal at K=5,
    # 37/15=2 for K=8). For an action-granularity classifier the YAML MUST
    # set ``algorithm.wmpo.classifier_min_steps`` explicitly in env-step units.
    classifier_min_steps = int(
        wmpo_cfg.get("classifier_min_steps", max(1, num_chunks // 15))
    )
    # Drop GRPO groups with no variance in returns (all-success or all-fail in
    # the same prompt's rollouts). Their normalized advantage is 0 anyway, so
    # this is purely a compute optimization; matches WMPO ray_trainer filter().
    filter_zero_variance_groups = bool(
        wmpo_cfg.get("filter_zero_variance_groups", True)
    )

    group_size = int(algorithm_cfg.get("ppo_rollouts_per_start", 4))
    update_epochs = max(1, int(algorithm_cfg.get("ppo_update_epochs", 1)))
    clip_low = float(algorithm_cfg.get("clip_ratio_low", 0.2))
    clip_high = float(algorithm_cfg.get("clip_ratio_high", 0.28))
    kl_coef = float(algorithm_cfg.get("kl_coef", 0.0))
    actor_bc_ref_scale = float(algorithm_cfg.get("actor_bc_to_ref_scale", 0.0))
    entropy_coef = float(algorithm_cfg.get("entropy_coef", 0.0))
    adv_eps = float(algorithm_cfg.get("advantage_eps", 1.0e-6))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    zero_grad_set_to_none = bool(optim_cfg.get("zero_grad_set_to_none", True))
    use_ref = ref_policy is not None

    chunk_world_model.eval()
    classifier_module = (
        classifier.module if hasattr(classifier, "module") else classifier
    )
    classifier_module.eval()
    policy.train()
    if ref_policy is not None:
        ref_policy.eval()

    obs = move_mapping_to_device(dict(obs), device)

    with torch.no_grad():
        latent_seq = _detach_latent(
            _world_model_observe_sequence(chunk_world_model, obs)
        )
        T_hist = _latent_time_dim(latent_seq)
        current = _repeat_latent(_flatten_last_steps(latent_seq, T_hist), group_size)

    actor_feats: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    action_token_ids: list[torch.Tensor | None] = []
    old_log_probs: list[torch.Tensor] = []
    ref_kls: list[torch.Tensor] = []
    video_latents: list[torch.Tensor] = []

    with _temporarily_freeze(chunk_world_model):
        for _ in range(num_chunks):
            actor_feat = (
                _world_model_actor_input(chunk_world_model, current).detach().float()
            )
            with torch.no_grad():
                # Stochastic full action chunk. This is the PPO action unit for
                # RynnVLA/WMPO: one policy decision emits K env actions.
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
                    f"action_chunk shape mismatch: got {tuple(action_chunk.shape)}, "
                    f"expected [B,K={K},action_dim]"
                )
            actor_feats.append(actor_feat)
            actions.append(action_chunk.detach())
            sampled_token_ids = _sample_extra.get("action_token_ids")
            action_token_ids.append(
                sampled_token_ids.detach()
                if isinstance(sampled_token_ids, torch.Tensor)
                else None
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
                # k1 KL estimator (signed) — unbiased in expectation but
                # can go negative on individual samples; this is the
                # verl/DAPO convention used here (subtract from reward
                # before GRPO normalization, not as a direct loss).
                ref_kls.append((old_lp.detach() - ref_lp).detach())

            with torch.no_grad():
                next_seq = chunk_world_model(
                    {
                        "mode": "predict_next_chunk",
                        "latent": current,
                        "actions": action_chunk.detach(),
                    }
                )
                video_latents.append(next_seq["hidden_seq"])
                current = _detach_latent(
                    {
                        "history": next_seq["history"],
                        "actions": next_seq["actions"],
                        "hidden": next_seq["hidden"],
                    }
                )

    # [B_eff, num_chunks * K, latent_dim]
    video = torch.cat(video_latents, dim=1)
    B_eff = video.shape[0]
    with torch.no_grad():
        success_info = classifier_module.predict_success(
            video,
            threshold=float(classifier_threshold),
            stride=1,
            min_steps=classifier_min_steps,
        )
    complete = success_info["complete"]
    # predict_success returns finish_step in the classifier's NATIVE unit
    # (chunk for chunk classifier).  Map to env-step at the boundary so
    # _build_reward_tensor / build_valid_chunk_count stay in env-step units.
    # Defensive ``getattr(module, "cfg", None)`` keeps this path safe for
    # action-granularity classifiers / minimal mocks that don't expose ``cfg``.
    # When ``cfg`` is missing we fall back to "action" granularity, which
    # produces wrong (off-by-K) reward placement if the classifier is in
    # fact chunk-granular. The wmpo override ``classifier_granularity``
    # lets callers force the correct branch when ``cfg`` is unreachable;
    # we also log a one-shot warning so the silent miscoding becomes
    # observable instead of dashboard-invisible.
    classifier_cfg = getattr(classifier_module, "cfg", None)
    cfg_override = wmpo_cfg.get("classifier_granularity", None)
    if classifier_cfg is None and cfg_override is None:
        global _warned_missing_cfg
        if not _warned_missing_cfg:
            _warned_missing_cfg = True
            _logger.warning(
                "dino_wmpo_outcome_step: classifier_module has no `.cfg`; "
                "defaulting granularity='action'. If the classifier is "
                "chunk-granular, set `algorithm.wmpo.classifier_granularity: "
                "chunk` (and optionally `chunk_pool`) to avoid silent "
                "off-by-K reward placement."
            )
    if cfg_override is not None:
        cls_gran = str(cfg_override)
    else:
        cls_gran = str(getattr(classifier_cfg, "granularity", "action"))
    if cls_gran == "chunk":
        pool = str(
            wmpo_cfg.get("classifier_chunk_pool", None)
            or getattr(classifier_cfg, "chunk_pool", "last")
        )
        offset = K - 1 if pool == "last" else (0 if pool == "first" else K // 2)
        # Unfired entries already encode "T_scan - 1" in chunk units → maps to
        # the very last env-step within the last chunk, which equals T_max - 1.
        finish_step = success_info["finish_step"] * K + offset
    else:
        finish_step = success_info["finish_step"]

    reward_tensor = _build_reward_tensor(
        batch=B_eff,
        max_steps=T_max,
        chunk_size=K,
        finish_step=finish_step,
        complete=complete,
    ).to(device)
    returns = reward_tensor.sum(dim=-1)  # for sparse 0/1 this equals float(complete)

    # ─── eos_mask, aligned with WMPO ───────────────────────────────────────
    # WMPO masks PPO loss past finish_step. We do the chunk-level equivalent:
    # chunk c spans env-steps [c*K, (c+1)*K). Chunk containing success is
    # ``finish_step // K`` (included). For failed episodes (complete=0,
    # finish_step = T_max-1) every chunk is valid (uniform mask).
    valid_chunk_count = build_valid_chunk_count(finish_step, K, num_chunks).to(device)
    chunk_indices_t = torch.arange(num_chunks, device=device).unsqueeze(
        1
    )  # [num_chunks, 1]
    chunk_mask = (
        chunk_indices_t < valid_chunk_count.unsqueeze(0)
    ).float()  # [num_chunks, B_eff]

    # ─── KL subtracted from reward (WMPO style) ────────────────────────────
    # WMPO compute_rewards: token_score - kl * kl_ratio, BEFORE GRPO advantage.
    # We compute total masked KL per rollout and subtract from the scalar return.
    if use_ref and ref_kls:
        ref_kl_stack = torch.stack(ref_kls, dim=0)  # [num_chunks, B_eff]
        kl_per_rollout = (ref_kl_stack * chunk_mask).sum(dim=0)  # [B_eff]
        returns_adjusted = returns - kl_coef * kl_per_rollout
    else:
        kl_per_rollout = torch.zeros_like(returns)
        returns_adjusted = returns

    # ─── Group-relative advantage, then zero-variance filter ──────────────
    # WMPO's ray_trainer filters out groups where every rollout has the same
    # return (no within-group variance) — those produce zero advantage anyway
    # and waste compute on policy forwards. We mark them via a per-rollout
    # mask and multiply into chunk_mask so the entire group is skipped.
    advantages = _group_advantage(returns_adjusted, group_size=group_size, eps=adv_eps)
    # Preserve the finish-only mask BEFORE the variance filter — BC anchor is
    # a regularizer that should still fire on zero-variance groups (it does
    # not need a learning signal from the return), so it uses the finish mask
    # alone and is decoupled from ``filter_zero_variance_groups``.
    bc_chunk_mask = chunk_mask.clone()
    if filter_zero_variance_groups and B_eff >= group_size:
        groups = returns_adjusted.reshape(-1, group_size)
        group_has_variance = (groups.std(dim=-1, unbiased=False) > adv_eps).float()
        per_rollout_group_mask = group_has_variance.repeat_interleave(
            group_size
        )  # [B_eff]
        chunk_mask = chunk_mask * per_rollout_group_mask.unsqueeze(0)
    else:
        per_rollout_group_mask = torch.ones_like(returns_adjusted)

    total_actor_loss = 0.0
    total_bc_ref_loss = 0.0
    total_kl = 0.0
    total_entropy_sum = 0.0  # sum_{epoch, chunk, rollout} (entropy * mask)
    grad_norm = 0.0
    mask_sum_total = float(chunk_mask.sum().item())  # PPO loss normalizer
    bc_mask_sum_total = float(bc_chunk_mask.sum().item())  # BC loss normalizer
    # Optimizer step is skipped when no chunk contributes a real gradient
    # (no PPO mask AND no BC anchor signal). Without this guard, Adam decays
    # its momentum/velocity state on a zero-gradient step, which moves
    # parameters in the direction of stale momentum and silently drifts the
    # actor during cold-start (every group all-fail, BC disabled).
    #
    # ``bc_mask_sum_total`` is always ≥ B_eff because ``build_valid_chunk_count``
    # clamps the per-rollout count to ``min=1`` — so ``has_bc_signal`` reduces
    # to ``actor_bc_ref_scale > 0``, but we keep the explicit form for clarity
    # if that clamp is ever loosened.
    has_ppo_signal = mask_sum_total > 0.0
    has_bc_signal = actor_bc_ref_scale > 0.0 and bc_mask_sum_total > 0.0
    should_step = has_ppo_signal or has_bc_signal
    # DDP correctness: every rank must reach the SAME step() decision, else
    # some ranks call ``optimizer.step()`` while others don't and parameters
    # diverge across ranks (breaking the all-reduced-gradient invariant DDP
    # relies on).  Use logical-OR (MAX): if any rank has gradient signal,
    # all ranks step — the silent ranks already had their (zero) local grad
    # all-reduced into the averaged grad, so stepping is a no-op for them.
    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor(
            [1.0 if should_step else 0.0],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        should_step = bool(flag.item() > 0.0)

    # Last-epoch ratio statistics, restricted to chunks/rollouts that actually
    # contribute to the PPO loss (mask_c == 1). Reported to the workspace so
    # the dashboard reflects real ratio drift instead of the stub 1.0 / 0.0
    # the route used to emit. Aggregation is over all valid (chunk, rollout)
    # pairs of the *final* epoch — sufficient for diagnosing clip pressure.
    last_epoch_ratio_records: list[torch.Tensor] = []  # each [n_valid] flat

    for epoch_idx in range(update_epochs):
        actor_optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
        epoch_actor_loss = 0.0
        epoch_bc_ref_loss_sum = 0.0
        epoch_bc_ref_count = 0
        if epoch_idx == update_epochs - 1:
            last_epoch_ratio_records = []
        for c, (
            actor_feat,
            action_detached,
            token_ids,
            old_lp,
            _ref_kl_unused,
        ) in enumerate(
            _zip_lists(
                actor_feats,
                actions,
                action_token_ids,
                old_log_probs,
                ref_kls if use_ref else None,
            )
        ):
            eval_batch = {
                "mode": "evaluate",
                "hidden": actor_feat,
                "action": action_detached,
            }
            if token_ids is not None:
                eval_batch["action_token_ids"] = token_ids
            new_lp, entropy_t, _ = policy(eval_batch)
            mask_c = chunk_mask[c]  # [B_eff], 0/1 per rollout
            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantages
            ppo_loss = -(torch.min(unclipped, clipped) * mask_c).sum()
            ent_term = (entropy_t * mask_c).sum()
            # Backprop chunk-by-chunk instead of accumulating all chunk graphs.
            # Long-imagine PPO has many actor forwards (T_max / K); holding them
            # all until a single backward can exceed 80GB even for small batches.
            # Note: kl_coef is no longer applied as a separate loss term — it
            # has been folded into advantages via returns_adjusted above.
            loss_c = (ppo_loss - entropy_coef * ent_term) / max(1.0, mask_sum_total)
            total_entropy_sum += float((entropy_t.detach() * mask_c).sum().item())

            if epoch_idx == update_epochs - 1:
                valid = mask_c > 0
                if valid.any():
                    last_epoch_ratio_records.append(ratio.detach()[valid].reshape(-1))

            if actor_bc_ref_scale > 0.0:
                _, _, extra = policy(
                    {
                        "mode": "sample",
                        "hidden": actor_feat,
                        "deterministic": True,
                        "return_chunk": True,
                    }
                )
                action_chunk = extra.get("action_chunk")
                if isinstance(action_chunk, torch.Tensor):
                    if ref_policy is not None:
                        with torch.no_grad():
                            _, _, ref_extra = ref_policy(
                                {
                                    "mode": "sample",
                                    "hidden": actor_feat,
                                    "deterministic": True,
                                    "return_chunk": True,
                                }
                            )
                        ref_action_chunk = ref_extra.get("action_chunk")
                    else:
                        ref_action_chunk = _policy_reference_action_chunk(
                            policy, actor_feat
                        )
                    if isinstance(ref_action_chunk, torch.Tensor):
                        # BC anchor is masked by the FINISH-only mask
                        # (``bc_chunk_mask``), not the post-variance-filter
                        # PPO mask. This keeps BC active on zero-variance
                        # groups (where it's still a valid regularizer) while
                        # eliminating spurious BC signal on past-finish
                        # chunks. Normalizer is per-(chunk, rollout) like
                        # PPO, so ``actor_bc_to_ref_scale`` carries the
                        # literal relative weight against PPO inside the
                        # active region.
                        bc_mask_c = bc_chunk_mask[c]
                        bc_per_rollout = (
                            (action_chunk.float() - ref_action_chunk.detach().float())
                            .square()
                            .mean(dim=(-1, -2))
                        )  # [B_eff]
                        bc_loss_c = (bc_per_rollout * bc_mask_c).sum()
                        loss_c = loss_c + actor_bc_ref_scale * bc_loss_c / max(
                            1.0, bc_mask_sum_total
                        )
                        epoch_bc_ref_loss_sum += float(bc_loss_c.detach().item())
                        epoch_bc_ref_count += int(bc_mask_c.sum().item())
            loss_c.backward()
            epoch_actor_loss += float(loss_c.detach().item())
        if should_step:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=grad_clip
                ).item()
            )
            actor_optimizer.step()
        else:
            grad_norm = 0.0
        total_actor_loss += epoch_actor_loss
        total_bc_ref_loss += epoch_bc_ref_loss_sum / max(1, epoch_bc_ref_count)
        if use_ref:
            total_kl += float(kl_per_rollout.detach().mean().item())

    # Aggregate last-epoch ratio diagnostics. When mask_sum_total == 0 there
    # were no contributing PPO updates this step and we report the neutral
    # ratio=1.0 / clipfrac=0.0 sentinels.
    if last_epoch_ratio_records:
        ratio_flat = torch.cat(last_epoch_ratio_records, dim=0)
        ppo_ratio_mean = float(ratio_flat.mean().item())
        ppo_ratio_min = float(ratio_flat.min().item())
        ppo_ratio_max = float(ratio_flat.max().item())
        ppo_clipfrac = float(
            ((ratio_flat < 1.0 - clip_low) | (ratio_flat > 1.0 + clip_high))
            .float()
            .mean()
            .item()
        )
    else:
        ppo_ratio_mean = 1.0
        ppo_ratio_min = 1.0
        ppo_ratio_max = 1.0
        ppo_clipfrac = 0.0

    actor_loss_val = total_actor_loss / max(1, update_epochs)
    bc_ref_loss_val = total_bc_ref_loss / max(1, update_epochs)
    returns_mean = float(returns_adjusted.detach().mean().item())
    returns_std = float(returns_adjusted.detach().std(unbiased=False).item())
    reward_mean = float(
        returns.detach().mean().item()
    )  # imagined success rate per rollout

    # mean_finish_step averaged ONLY over completed rollouts — the prior
    # implementation averaged finish_step over all rollouts including
    # failures (where finish_step is pinned to T_max - 1), pinning the
    # metric near T_max whenever most episodes failed.
    #
    # Sentinel = -1.0 when no rollouts complete. Avoids float('nan'), which
    # propagates through the workspace's ``reduce_mean_dict`` (all_reduce
    # SUM) and contaminates the metric on every rank.  Readers should pair
    # this with ``wmpo/success_rate``: success_rate==0 ⇒ metric is the
    # sentinel and should be ignored.
    complete_bool = complete.detach().bool()
    if complete_bool.any():
        mean_finish_step_complete = float(
            finish_step.detach().float()[complete_bool].mean().item()
        )
    else:
        mean_finish_step_complete = -1.0
    # avg_entropy normalized by sum-of-mask across all epochs, matching the
    # PPO loss normalization granularity (per valid (chunk, rollout) pair).
    entropy_denom = float(update_epochs) * max(1.0, mask_sum_total)
    avg_entropy_val = total_entropy_sum / entropy_denom

    # Per-group success breakdown — each group is `group_size`
    # (= ppo_rollouts_per_start) rollouts from the same starting state.
    # Used by the JSONL ppo_groups log: timestamp + per-group success rate +
    # whether the group has variance (i.e. is actually useful for GRPO).
    if B_eff >= group_size and B_eff % group_size == 0:
        groups_returns = returns.detach().reshape(-1, group_size)  # [G, K]
        groups_complete = complete.detach().bool().reshape(-1, group_size)
        groups_finish_step = finish_step.detach().long().reshape(-1, group_size)
        group_success_rates: list[float] = groups_returns.mean(dim=-1).cpu().tolist()
        group_success_counts: list[int] = groups_complete.sum(dim=-1).cpu().tolist()
        group_rollout_successes: list[list[bool]] = groups_complete.cpu().tolist()
        group_finish_steps: list[list[int]] = groups_finish_step.cpu().tolist()
        group_has_variance_bool = (
            (groups_returns.std(dim=-1, unbiased=False) > adv_eps).cpu().tolist()
        )
        num_groups = int(groups_returns.shape[0])
        num_all_success = int((groups_returns.sum(dim=-1) == group_size).sum().item())
        num_all_fail = int((groups_returns.sum(dim=-1) == 0).sum().item())
        num_mixed = num_groups - num_all_success - num_all_fail
    else:
        group_success_rates = []
        group_success_counts = []
        group_rollout_successes = []
        group_finish_steps = []
        group_has_variance_bool = []
        num_groups = 0
        num_all_success = 0
        num_all_fail = 0
        num_mixed = 0

    return {
        # Flat keys — for compatibility with workspace/script metric extraction.
        "actor_loss": actor_loss_val,
        "actor_bc_loss": bc_ref_loss_val,
        "actor_bc_scale": actor_bc_ref_scale,
        "actor_bc_ref_loss": bc_ref_loss_val,
        "actor_bc_ref_scale": actor_bc_ref_scale,
        "actor_vla_drift_raw_mse": 0.0,
        "actor_vla_drift_env_mse": 0.0,
        "actor_vla_drift_env_mse_clipped": 0.0,
        "actor_vla_drift_env_mae": 0.0,
        "critic_loss": 0.0,
        "returns_mean": returns_mean,
        "returns_std": returns_std,
        "raw_returns_mean": returns_mean,
        "raw_returns_std": returns_std,
        "advantage_mean": float(advantages.detach().mean().item()),
        "advantage_std": float(advantages.detach().std(unbiased=False).item()),
        "advantage_mag": float(advantages.detach().abs().mean().item()),
        "return_scale": 1.0,
        "reward_mean": reward_mean,
        "value_mean": 0.0,
        "actor_grad_norm": grad_norm,
        "critic_grad_norm": 0.0,
        "ppo_update_epochs": float(update_epochs),
        # Real ratio diagnostics emitted now — used to be hard-coded 1.0 / 0.0,
        # which masked clip pressure whenever ``ppo_update_epochs > 1``.
        "ppo_ratio_mean": ppo_ratio_mean,
        "ppo_ratio_min": ppo_ratio_min,
        "ppo_ratio_max": ppo_ratio_max,
        "ppo_clipfrac": ppo_clipfrac,
        "ppo_step_applied": float(should_step),
        "continue_mean": 1.0,
        "ref_kl_mean": total_kl / max(1, update_epochs),
        "kl_coef": float(kl_coef),
        # Namespaced detail — WMPO-specific diagnostics.
        "wmpo/actor_loss": actor_loss_val,
        "wmpo/actor_bc_ref_loss": bc_ref_loss_val,
        "wmpo/actor_bc_ref_scale": actor_bc_ref_scale,
        "wmpo/avg_entropy": avg_entropy_val,
        "wmpo/avg_kl": total_kl / max(1, update_epochs),
        "wmpo/grad_norm": grad_norm,
        "wmpo/success_rate": float(complete.float().mean().item()),
        # Failure-aware: average finish step ONLY over completed rollouts.
        # The prior all-rollouts average pinned this metric near T_max when
        # most episodes failed (finish_step = T_max-1 for failures).
        "wmpo/mean_finish_step": mean_finish_step_complete,
        "wmpo/mean_finish_step_all": float(finish_step.float().mean().item()),
        "wmpo/num_chunks": float(num_chunks),
        "wmpo/T_max": float(T_max),
        "wmpo/start_points_per_window": float(T_hist),
        "wmpo/classifier_min_steps": float(classifier_min_steps),
        "wmpo/valid_chunk_frac": float(
            chunk_mask.sum().item() / max(1, num_chunks * B_eff)
        ),
        "wmpo/group_var_keep_frac": float(per_rollout_group_mask.mean().item()),
        # ── per-group breakdown for ppo_groups.jsonl log ─────────────────
        "wmpo/group_size": float(group_size),
        "wmpo/num_groups": float(num_groups),
        "wmpo/num_all_success_groups": float(num_all_success),
        "wmpo/num_all_fail_groups": float(num_all_fail),
        "wmpo/num_mixed_groups": float(num_mixed),
        "wmpo/group_success_rates": group_success_rates,
        "wmpo/group_success_counts": group_success_counts,
        "wmpo/group_rollout_successes": group_rollout_successes,
        "wmpo/group_finish_steps": group_finish_steps,
        "wmpo/group_has_variance": group_has_variance_bool,
    }


__all__ = ["dino_wmpo_outcome_step", "build_valid_chunk_count", "_build_reward_tensor"]
