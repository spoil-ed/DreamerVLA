# ruff: noqa: E402
"""Outcome-reward WMPO PPO route.

**Reward form**: sparse outcome reward. After imagining a full episode in
the world model, ``LatentSuccessClassifier.predict_success`` scores the
imagined latent video and emits ``(complete, finish_step)``. We place
``float(complete)`` at ``finish_step`` and zero elsewhere — one positive
signal per successful rollout, none otherwise.

This is the DreamerVLA-side reproduction of the WMPO/verl PPO loop. The
rollout drives the WM in chunk mode (``ChunkAwareDinoWMWorldModel.
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

from dreamervla.algorithms.dreamervla import (
    _detach_latent,
    _flatten_strided_steps,
    _latent_batch_dim,
    _latent_time_dim,
    _policy_reference_action_chunk,
    _temporarily_freeze,
    _world_model_actor_input,
    _world_model_observe_sequence,
)
from dreamervla.algorithms.ppo.grpo import (
    _entropy_coef,
    _group_advantage,
    _ppo_clip_term,
    _ppo_ratio,
    _repeat_latent,
    _slice_latent,
    masked_mean_ratio_chunk_term,
)
from dreamervla.utils.torch_utils import move_mapping_to_device


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
    # Vectorized sparse placement: write float(complete) at the finish column of
    # each row in a single scatter_. One index per row (no accumulation), and a
    # 0.0 write for incomplete rows is a no-op against the zero base — bit-for-bit
    # identical to the prior ``if comp[i]: reward[i, finish[i]] = 1.0`` loop.
    reward.scatter_(1, finish.unsqueeze(1), comp.float().unsqueeze(1))
    return reward


def _predict_next_chunk_mb(
    world_model: nn.Module, current: Any, action_chunk: torch.Tensor, micro_batch: int
) -> dict[str, torch.Tensor]:
    """Run the WM ``predict_next_chunk`` in micro-batches and concatenate.

    The imagination forward is per-rollout independent, so slicing the batch and
    concatenating the outputs is numerically identical to one full call — but the
    attention activations are bounded to ``micro_batch`` rollouts. ``micro_batch
    <= 0`` keeps the single full call. This is a SECONDARY bound inside a slice;
    it is also the only bound on the WM forward for the full-batch fallback.
    """
    b = int(action_chunk.shape[0])
    call = lambda cur, act: world_model(  # noqa: E731
        {"mode": "predict_next_chunk", "latent": cur, "actions": act}
    )
    if micro_batch <= 0 or micro_batch >= b:
        return call(current, action_chunk)
    parts: list[dict[str, torch.Tensor]] = []
    for lo in range(0, b, micro_batch):
        hi = min(lo + micro_batch, b)
        parts.append(call(_slice_latent(current, lo, hi), action_chunk[lo:hi]))
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}


def _imagine_and_score_slice(
    *,
    policy: nn.Module,
    chunk_world_model: nn.Module,
    classifier_module: nn.Module,
    classifier_threshold: float,
    current: Any,
    device: torch.device,
    chunk_size: int,
    num_chunks: int,
    chunk_granular: bool,
    chunk_pool: str | None,
    finish_offset: int,
    feat_dtype: torch.dtype,
    use_ref: bool,
    ref_policy: nn.Module | None,
    imag_mb: int,
    eval_micro_batch: int,
    classifier_min_steps: int,
) -> dict[str, Any]:
    """Imagine ONE group-aligned start slice and score it — MEM-RL-01.

    This is the transient "imagination buffer" for one slice. The imagination is
    pure DATA: the PPO gradient flows only through the later ``policy.evaluate``
    re-eval, never the WM dynamics (the rollout is ``no_grad``). So processing the
    effective batch one group-aligned slice at a time is numerically identical to
    the full-batch rollout — but the imagination forward + classifier sweep never
    sit on GPU at full ``B_eff``. Per-chunk policy inputs are offloaded to CPU so
    the multi-epoch update can stream them back one slice / one chunk at a time.

    Returns per-chunk host buffers (``actor_feats`` on CPU; ``actions``,
    ``action_token_ids``, ``old_log_probs``, ``ref_kls`` on device) plus this
    slice's ``complete`` / ``finish_step`` (already mapped to env-step units).
    """
    K = int(chunk_size)
    actor_feats: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    action_token_ids: list[torch.Tensor | None] = []
    old_log_probs: list[torch.Tensor] = []
    ref_kls: list[torch.Tensor] = []
    video_latents: list[torch.Tensor] = []

    for _ in range(num_chunks):
        actor_feat = (
            _world_model_actor_input(chunk_world_model, current)
            .detach()
            .to(feat_dtype)
        )
        with torch.no_grad():
            # Stochastic full action chunk — the PPO action unit for RynnVLA/WMPO:
            # one policy decision emits K env actions.
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
        # Offload per-chunk policy inputs to CPU: detached (no graph), only re-read
        # in the multi-epoch loss loop. Keeps GPU to ~1 chunk of this slice instead
        # of all num_chunks. Streamed back to device just-in-time at the eval.
        actor_feats.append(actor_feat.to("cpu"))
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
            # k1 KL estimator (signed) — verl/DAPO convention: subtracted from the
            # reward before GRPO normalization, not applied as a direct loss.
            ref_kls.append((old_lp.detach() - ref_lp).detach())

        with torch.no_grad():
            next_seq = _predict_next_chunk_mb(
                chunk_world_model, current, action_chunk.detach(), imag_mb
            )
            hidden_seq = next_seq["hidden_seq"]  # [b, K, ...] all K frames
            if chunk_granular:
                # Store ONE pooled frame per chunk (== classifier _chunk_aggregate
                # over this chunk's K frames). 1/K memory.
                if chunk_pool == "first":
                    pooled = hidden_seq[:, 0]
                elif chunk_pool == "mean":
                    pooled = hidden_seq.mean(dim=1)
                else:  # "last"
                    pooled = hidden_seq[:, -1]
                video_latents.append(pooled.unsqueeze(1).to("cpu"))  # [b, 1, ...]
            else:
                video_latents.append(hidden_seq.to("cpu"))
            current = _detach_latent(
                {
                    "history": next_seq["history"],
                    "actions": next_seq["actions"],
                    "hidden": next_seq["hidden"],
                }
            )

    # predict_success is a per-rollout temporal scan (no cross-rollout coupling),
    # so even within this slice we sweep in ``eval_micro_batch`` sub-batches and
    # never materialize the slice's full [b, num_chunks, latent_dim] video on GPU.
    # Tokenized WMs emit a 4-D hidden_seq; flatten the trailing token axes to the
    # flat [b, T, latent_dim] contract the classifier expects. ``pre_pooled`` only
    # matters on the chunk path (we pooled each chunk while generating).
    b = int(video_latents[0].shape[0])
    mb = max(1, min(int(eval_micro_batch) if eval_micro_batch else b, b))
    complete_parts: list[torch.Tensor] = []
    finish_parts: list[torch.Tensor] = []
    for s in range(0, b, mb):
        e = min(s + mb, b)
        video_s = torch.cat([v[s:e].to(device) for v in video_latents], dim=1)
        if video_s.ndim > 3:
            video_s = video_s.reshape(video_s.shape[0], video_s.shape[1], -1)
        with torch.no_grad():
            info_s = classifier_module.predict_success(
                video_s,
                threshold=float(classifier_threshold),
                stride=1,
                min_steps=classifier_min_steps,
                **({"pre_pooled": True} if chunk_granular else {}),
            )
        complete_parts.append(info_s["complete"])
        finish_parts.append(info_s["finish_step"])
        del video_s
    complete = torch.cat(complete_parts, dim=0)
    finish_native = torch.cat(finish_parts, dim=0)
    # Map finish_step from the classifier's NATIVE unit back to env-step units. For
    # a chunk classifier finish_step is in chunks; convert at the boundary. Unfired
    # entries already encode "T_scan - 1" → the last env-step of the last chunk.
    finish_step = finish_native * K + finish_offset if chunk_granular else finish_native

    return {
        "actor_feats": actor_feats,
        "actions": actions,
        "action_token_ids": action_token_ids,
        "old_log_probs": old_log_probs,
        "ref_kls": ref_kls if use_ref else None,
        "complete": complete,
        "finish_step": finish_step,
    }


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
    # Bound the imagination forward's activation memory to this many rollouts
    # (per-rollout independent → numerically identical). 0 = full effective batch.
    imag_mb = int(wmpo_cfg.get("imagine_micro_batch", 0))
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
    clip_ratio_c = algorithm_cfg.get("clip_ratio_c", None)
    clip_log_ratio = algorithm_cfg.get("clip_log_ratio", None)
    kl_coef = float(algorithm_cfg.get("kl_coef", 0.0))
    actor_bc_ref_scale = float(algorithm_cfg.get("actor_bc_to_ref_scale", 0.0))
    entropy_coef = _entropy_coef(algorithm_cfg)
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

    # Cap the imagination start points per window with ``imag_last`` and spread
    # them EVENLY over the window instead of taking the last-N adjacent frames.
    # Every frame in the observed window is a valid start (observe_sequence
    # builds a num_hist history for each), but the frames are consecutive
    # states from one real trajectory: taking the last N gives near-identical,
    # redundant starts, while using all 36 explodes the effective batch
    # B_eff = B * starts * group_size through the WM. The window length is sized
    # for WM chunk-rollout training (H + N*K + 1), not for RL. ``imag_last``
    # sets how many starts; strided selection over [num_hist-1, T-1] makes them
    # diverse (different trajectory phases) while keeping a full real history.
    imag_last = int(algorithm_cfg.get("imag_last", 4))
    wm_module = getattr(chunk_world_model, "module", chunk_world_model)
    num_hist = int(getattr(wm_module, "num_hist", 1))
    with torch.no_grad():
        latent_seq = _detach_latent(
            _world_model_observe_sequence(chunk_world_model, obs)
        )
        seq_len = _latent_time_dim(latent_seq)
        T_hist = min(imag_last if imag_last > 0 else seq_len, seq_len)
        current = _repeat_latent(
            _flatten_strided_steps(latent_seq, T_hist, min_start=num_hist - 1),
            group_size,
        )

    # ─── group-aligned micro-batch slices (MEM-RL-01) ─────────────────────
    # Bound the update's peak GPU memory to ONE group-aligned slice instead of
    # the full effective batch B_eff. GRPO groups are CONTIGUOUS blocks of
    # group_size in the B_eff dim (``_repeat_latent`` = repeat_interleave), so a
    # slice MUST cover whole groups for its group-relative advantage to match the
    # full-batch one. We slice in START units: each slice is ``mb_starts`` real
    # starts = mb_starts * group_size rollouts, the contiguous block
    # ``[lo*g : hi*g]``. ``update_micro_batch_starts`` <= 0 or >= n_starts ⇒ one
    # full-batch slice — bit-for-bit the original behavior.
    n_starts = _latent_batch_dim(current) // group_size
    B_eff = n_starts * group_size
    mb_starts_cfg = int(wmpo_cfg.get("update_micro_batch_starts", 0))
    mb_starts = n_starts if mb_starts_cfg <= 0 else min(max(1, mb_starts_cfg), n_starts)
    slice_bounds = [
        (s, min(s + mb_starts, n_starts)) for s in range(0, n_starts, mb_starts)
    ]
    # Inner micro-batch for the classifier sweep (``eval_micro_batch``) — composes
    # with the slice and is the only bound for the full-batch fallback.
    eval_micro_batch = int(wmpo_cfg.get("eval_micro_batch", 64) or B_eff)

    # Classifier granularity drives BOTH (a) how we POOL the imagined frames we
    # store and (b) how finish_step is mapped back to env-steps. Detect it ONCE
    # here, before the rollout, so the imagined "video" can be stored at the
    # classifier's native granularity: a chunk classifier only ever looks at one
    # pooled frame per chunk (``chunk_pool``), so storing every imagined env-step
    # wastes K× the memory. Pooling each chunk as it is generated is identical to
    # the classifier's internal ``_chunk_aggregate`` (same chunk_pool), so
    # success detection is unchanged — see predict_success(pre_pooled=...).
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
    cls_gran = (
        str(cfg_override)
        if cfg_override is not None
        else str(getattr(classifier_cfg, "granularity", "action"))
    )
    chunk_granular = cls_gran == "chunk"
    if chunk_granular:
        chunk_pool = str(
            wmpo_cfg.get("classifier_chunk_pool", None)
            or getattr(classifier_cfg, "chunk_pool", "last")
        )
        finish_offset = (
            K - 1 if chunk_pool == "last" else (0 if chunk_pool == "first" else K // 2)
        )
    else:
        chunk_pool = None
        finish_offset = 0

    # Store the per-chunk actor features in the dtype the policy actually consumes
    # them in (its action head casts the hidden to ``lm_head.weight.dtype``), not
    # an unconditional float32. With a bf16 action head this halves the
    # accumulated ``actor_feats`` (num_chunks x [B_eff, token_count, dim]) with
    # ZERO change to the logits/log-probs (verified: bf16-in == float32-in once
    # the head casts). Falls back to float32 for actors without an ``lm_head``.
    _policy_module = getattr(policy, "module", policy)
    _lm_head = getattr(_policy_module, "lm_head", None)
    feat_dtype = (
        _lm_head.weight.dtype
        if _lm_head is not None and hasattr(_lm_head, "weight")
        else torch.float32
    )

    # ─── Phase 1: imagine each group-aligned slice into its (CPU) host buffer ──
    # Pure no_grad data collection (the imagination is data; the PPO gradient
    # flows only through the Phase-3 re-eval). The WM stays frozen for the sweep.
    slices: list[dict[str, Any]] = []
    with _temporarily_freeze(chunk_world_model):
        for s_lo, s_hi in slice_bounds:
            slices.append(
                _imagine_and_score_slice(
                    policy=policy,
                    chunk_world_model=chunk_world_model,
                    classifier_module=classifier_module,
                    classifier_threshold=classifier_threshold,
                    current=_slice_latent(
                        current, s_lo * group_size, s_hi * group_size
                    ),
                    device=device,
                    chunk_size=K,
                    num_chunks=num_chunks,
                    chunk_granular=chunk_granular,
                    chunk_pool=chunk_pool,
                    finish_offset=finish_offset,
                    feat_dtype=feat_dtype,
                    use_ref=use_ref,
                    ref_policy=ref_policy,
                    imag_mb=imag_mb,
                    eval_micro_batch=eval_micro_batch,
                    classifier_min_steps=classifier_min_steps,
                )
            )

    # ─── Phase 2: assemble GLOBAL scoring tensors from the slices ──────────
    # Concatenating the per-slice results reproduces the full-batch tensors
    # exactly (slices are whole-group contiguous blocks), so the advantage and
    # every scalar metric below are identical to the original full-batch path.
    # finish_step is already mapped to env-step units inside the slice helper.
    complete = torch.cat([d["complete"] for d in slices], dim=0)
    finish_step = torch.cat([d["finish_step"] for d in slices], dim=0)

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
    if use_ref:
        # Stack ref_kls per chunk across the slices into the [num_chunks, B_eff]
        # global layout (each slice holds num_chunks tensors of shape [mb]).
        ref_kl_stack = torch.stack(
            [
                torch.cat([d["ref_kls"][c] for d in slices], dim=0)
                for c in range(num_chunks)
            ],
            dim=0,
        )  # [num_chunks, B_eff]
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
    mask_sum_total = float(chunk_mask.sum().item())  # PPO signal / entropy denom
    bc_mask_sum_total = float(bc_chunk_mask.sum().item())  # BC signal
    # RLinf masked_mean_ratio: per-rollout valid-chunk counts (clamp ≥1 so empty
    # rollouts, whose terms are masked to 0, do not divide by zero). Each rollout
    # is then weighted equally over B_eff regardless of episode length.
    ppo_per_rollout_count = chunk_mask.sum(dim=0).clamp(min=1.0)  # [B_eff]
    bc_per_rollout_count = bc_chunk_mask.sum(dim=0).clamp(min=1.0)  # [B_eff]
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

    # ─── Phase 3: multi-epoch, micro-batched policy update (RLinf pattern) ──
    # Each epoch re-evaluates the SAME stored trajectory (fixed actions / old_lp)
    # under the current params and accumulates per-slice gradients before ONE
    # optimizer step — the standard RLinf/verl PPO epoch, here with one mini-batch
    # = the whole effective batch spread over group-aligned micro-batches. The
    # normalization passes the GLOBAL B_eff in every slice (per-rollout
    # term/mask/count come from the slice), so summing the per-slice gradients
    # reproduces the full-batch gradient exactly. The host buffer (Phase 1) keeps
    # the rollout fixed across epochs while GPU only ever holds one slice / chunk.
    for epoch_idx in range(update_epochs):
        actor_optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
        epoch_actor_loss = 0.0
        epoch_bc_ref_loss_sum = 0.0
        epoch_bc_ref_count = 0
        if epoch_idx == update_epochs - 1:
            last_epoch_ratio_records = []
        for slice_idx, (s_lo, s_hi) in enumerate(slice_bounds):
            lo, hi = s_lo * group_size, s_hi * group_size
            data = slices[slice_idx]
            adv_slice = advantages[lo:hi]
            ppo_count_slice = ppo_per_rollout_count[lo:hi]
            bc_count_slice = bc_per_rollout_count[lo:hi]
            for c in range(num_chunks):
                actor_feat = data["actor_feats"][c].to(device)  # streamed from CPU
                action_detached = data["actions"][c]
                token_ids = data["action_token_ids"][c]
                old_lp = data["old_log_probs"][c]
                eval_batch = {
                    "mode": "evaluate",
                    "hidden": actor_feat,
                    "action": action_detached,
                }
                if token_ids is not None:
                    eval_batch["action_token_ids"] = token_ids
                new_lp, entropy_t, _ = policy(eval_batch)
                mask_c = chunk_mask[c, lo:hi]  # [mb], 0/1 per rollout
                ratio = _ppo_ratio(new_lp, old_lp, clip_log_ratio=clip_log_ratio)
                ppo_clip = _ppo_clip_term(
                    ratio, adv_slice, clip_low, clip_high, clip_ratio_c=clip_ratio_c
                )  # [mb]
                # Backprop chunk-by-chunk (within a slice) instead of accumulating
                # all chunk graphs. kl_coef is folded into advantages above.
                # masked_mean_ratio passes the GLOBAL B_eff so the per-slice terms
                # sum to the full-batch mean — each rollout weighted equally.
                ppo_term = masked_mean_ratio_chunk_term(
                    ppo_clip, mask_c, ppo_count_slice, B_eff
                )
                ent_term = masked_mean_ratio_chunk_term(
                    entropy_t, mask_c, ppo_count_slice, B_eff
                )
                loss_c = ppo_term - entropy_coef * ent_term
                total_entropy_sum += float((entropy_t.detach() * mask_c).sum().item())

                if epoch_idx == update_epochs - 1:
                    valid = mask_c > 0
                    if valid.any():
                        last_epoch_ratio_records.append(
                            ratio.detach()[valid].reshape(-1)
                        )

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
                            # chunks. Normalizer is per-rollout masked_mean_ratio
                            # like PPO, so ``actor_bc_to_ref_scale`` carries the
                            # literal relative weight against PPO inside the
                            # active region.
                            bc_mask_c = bc_chunk_mask[c, lo:hi]
                            bc_per_rollout = (
                                (action_chunk.float() - ref_action_chunk.detach().float())
                                .square()
                                .mean(dim=(-1, -2))
                            )  # [mb]
                            bc_term = masked_mean_ratio_chunk_term(
                                bc_per_rollout, bc_mask_c, bc_count_slice, B_eff
                            )
                            loss_c = loss_c + actor_bc_ref_scale * bc_term
                            epoch_bc_ref_loss_sum += float(
                                (bc_per_rollout * bc_mask_c).sum().detach().item()
                            )
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
    # avg_entropy metric: mean entropy per valid (chunk, rollout) pair across all
    # epochs. (Reporting granularity only; the loss now uses the per-rollout
    # masked_mean_ratio normalization above.)
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
