# MEM-RL-01 — micro-batch the WMPO outcome update (bound peak GPU memory)

## Problem (GPU-verified 2026-06-22)
`dino_wmpo_outcome_step` imagines the whole trajectory for the FULL effective batch
`B_eff = n_starts × group_size` (measured ~715), holds it on GPU, then runs predict_success +
GRPO advantage + multi-epoch PPO loss. This pins an 80GB H100 (the `video` gather alone ≈24GB;
the imagination forward + loss at full B_eff keep it at ~80GB). WM warmup is already fixed
(gradient checkpointing); the real-env rollout already lives on host (`OnlineReplay`). Reducing
`ppo_rollouts_per_start` or `episode_max_steps` did NOT help (the per-forward at full B_eff is
the wall). The partial offload (actor_feats/video→CPU, predict_success micro-batched) only
cleared the 24GB `video` gather; the loss + imagination forward still run at full B_eff.

## Goal
Bound the update's peak GPU memory to ONE group-aligned micro-batch instead of full B_eff,
**numerically equivalent** to the current full-batch update. Process the imagination as data
that never sits on GPU at full B_eff (RLinf/WoVR pattern).

## Key facts that make it correct
- PPO is score-function: gradient flows ONLY through the policy log-prob re-eval
  (`new_lp = policy.evaluate(actor_feat, action)`), NOT through the WM dynamics (rollout is
  `no_grad`). So the imagination is pure data — safe to slice/stream.
- GRPO groups are CONTIGUOUS blocks of `group_size` in the B_eff dim
  (`_repeat_latent(..., group_size)` + `_group_advantage` reshape `(-1, group_size)`).
  → micro-batch boundaries MUST be multiples of `group_size` (whole groups per slice), so the
  group-relative advantage is identical to the full-batch one.
- Per-rollout loss normalization (`masked_mean_ratio`, per-rollout counts) is independent across
  rollouts → summing per-slice gradients and dividing by the global rollout count reproduces the
  full-batch gradient.

## Design
Restructure the body into a per-micro-batch loop over group-aligned start slices:
```
zero_grad()
for slice in group_aligned_slices(current, mb_starts):     # mb = mb_starts × group_size rollouts
    imagine(slice) -> per-chunk (actor_feat, action, old_lp[, token_ids, ref_kl]) + video_latents
    returns/complete <- predict_success(video_slice)        # per-rollout, micro-batched already
    adv <- group_advantage(returns_slice, group_size)        # groups whole within the slice
    for epoch in update_epochs:
        for chunk: ppo loss term (new_lp vs old_lp, clip, masked_mean) ; backward (accumulate)
optimizer.step()
metrics = mean over slices (weighted by valid mask)
```
- New knob `algorithm.wmpo.update_micro_batch_starts` (default sized to ~fit; e.g. starts giving
  ~96 rollouts/slice). `mb_starts <= n_starts`; falls back to full batch when unset/large.
- Each slice's imagination is the explicit, transient "imagination buffer" content, separate
  from `OnlineReplay`. (The persistent, fully-separate imagination replay = MEM-RL-02 WM-as-env.)
- Drop the partial offload once per-slice memory is small enough (revert the CPU `.to("cpu")`
  shuttles), OR keep a thin host-buffer for the slice — decide by what the GPU re-test needs.

## Steps (each verifiable)
1. ✅ DONE — helpers read. Findings:
   - `_repeat_latent` = `repeat_interleave(g, dim=0)` → B_eff layout is CONTIGUOUS group blocks
     `[start0×g, start1×g, …]`; group i = rollouts `[i*g:(i+1)*g]`. `B_eff = n_starts × g`,
     `g = ppo_rollouts_per_start`, `n_starts` from `_flatten_strided_steps`.
   - Slice = `mb_starts × g` rollouts, `[lo*g : hi*g]` along dim 0 — group-aligned by construction.
     Need a `_slice_latent(latent, lo, hi)` (tensor/dict/dataclass) mirroring `_repeat_latent`.
   - `_group_advantage(score, g, eps)` raises unless `numel % g == 0` → slices MUST be group-aligned.
   - `masked_mean_ratio_chunk_term(term, mask_c, per_rollout_count, b_eff)` divides by the GLOBAL
     `b_eff` → pass the FULL B_eff in every slice; per-rollout `term/mask/count` come from the
     slice. Summed over slices = the full-batch mean → identical gradient.
   - CORRECTNESS RULE: PPO multi-epoch must reuse the SAME sampled trajectory (fixed actions/
     old_lp). So do NOT re-imagine per epoch. → imagine once into a host buffer, then multi-epoch
     micro-batched loss. (For `update_epochs==1`, per-slice imagine+loss is equivalent and simpler;
     start there, assert/guard `update_epochs>1` until the host-buffer path lands.)
   - Equivalence test is a SAFETY gate, not a RED-driver (the change is memory-only): with the
     knob set, loss+grad must be UNCHANGED. RNG aligns full-vs-micro because `torch.randn([N])`'s
     first m == `torch.randn([m])` slice-0 (sequential RNG) when only the policy sample draws RNG.
2. TDD: numerical-equivalence test on the tiny WM fixture (`test_chunk_wm_autoregressive` style):
   `dino_wmpo_outcome_step` with the whole batch in 1 slice vs split into N group-aligned slices
   → loss + policy `.grad` identical within atol. RED first (knob absent).
3. Implement the micro-batch loop + group-aligned slicing + grad accumulation + identical
   normalization + metric averaging. → GREEN the equivalence test.
4. Remove the temporary `DVLA_MEMDIAG` print in `online_cotrain_runner`; reconcile the partial
   offload edits in `outcome.py` (keep or revert) so the code is clean.
5. `ruff check` + full unit suite green (dreamervla env).
6. GPU re-test: full `episode_max_steps=300`, `batch_size=96`, on a free GPU →
   reach `320/320`, no OOM, peak well under 80GB; tune `update_micro_batch_starts`.

## Out of scope (separate TODO)
MEM-RL-02 — WM-as-env so imagination becomes a real rollout into a persistent, explicit host
replay buffer + standard micro-batched PPO (full RLinf/WoVR structure).
