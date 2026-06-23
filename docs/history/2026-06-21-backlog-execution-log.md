# Backlog execution log (pass 3, 2026-06-21)

Branch: `chore/backlog-execution` (off `main`). Source: the open-items backlog
`docs/superpowers/plans/2026-06-21-todo-backlog.md` (since moved to the superpowers planning workspace).
Constraint honored: **behaviour-preserving**
(every commit ran the full unit suite green in the `dreamervla` env), except the two
maintainer-approved `changes-numerics` flips, which are flagged below. Suite: baseline
584 → **593 passed, 7 skipped** (+9 new guard/regression tests).

Items below are DONE and removed from the open backlog. Verification gaps that remain
(GPU cotrain smoke; the deferred items) stay in the open backlog.

## P0 — correctness

- **A1 / ALG-03 (entropy-key)** — was already resolved in code (all three PPO routes read
  entropy via `grpo._entropy_coef`, `actent` → `entropy_coef` → 0.0); corrected the stale
  backlog + audit + cleanup-log notes.
- **A4 (replay bootstrap)** — *changes numerics, maintainer-approved.* Replay λ-return now
  bootstraps with the critic's per-state value on the replay posterior states (standard
  DreamerV3 repval), not the single imagined-return scalar `raw_returns[:,0]`. Wired the
  previously-dead `repl_loss.slowtar` (target vs fast critic). Deep path (inside
  `imagine_actor_critic_step`) covered by code review + the unchanged
  `compute_replay_lambda_returns` recurrence test; **the GPU cotrain smoke is the remaining
  verification gate.**
- **outcome loss normalization** — *changes numerics, maintainer-approved.* Replaced the
  global per-(chunk,rollout) masked-sum denominator with RLinf `masked_mean_ratio`
  (per-rollout / episode-length normalization) via the new
  `grpo.masked_mean_ratio_chunk_term`; PPO term + BC anchor both use it (relative weight
  preserved). Identical at `B_eff=1`. New `test_masked_mean_ratio.py` (per-rollout weighting,
  RLinf-formula equivalence, empty-rollout safety).

## P2 — migration design

- **DIAG-01** — new opt-in `load_world_model_state_from_dict(remap_reward_head=,
  skip_shape_mismatch=, reset_reward_head=)`; the path-based `load_world_model_state` and
  `visualize_dreamervla_reward` route through it (the dict loader now strips both
  `_fsdp_wrapped_module.` and `module.`). Plain `strict=False` / generic multi-module loaders
  left intentionally (different job — flagged, not silently unified). New
  `test_wm_state_loader.py`.
- **X-03** — `dreamervla/constants.py:DEFAULT_ACTION_TOKEN_ID` single-sources the `10004`
  literal across all first-party sites; the 5 hardcoded `embodied_eval_runner` token
  insertions now read `eval.target_token_id` (default = constant) via `self._action_token_id`
  → adjustable + consistent. Vendored `chameleon_model` (13 occurrences) left by design.
  `model_dim 4106` is already a per-config value.
- **X-01 (partial)** — `CHECKPOINT_FORMAT_VERSION` (=1) stamped on both the BaseRunner and
  online_dreamervla writers; online load routed through the shared `load_runner_payload`;
  the `checkpoints/` (canonical) vs `ckpt/` (legacy read-only) dirs were already unified.
  New `test_checkpoint_format_version.py` (versioned + legacy-unversioned roundtrip). The
  3-payload-SCHEMA → 1-writer collapse stays open (format-breaking; GPU resume smoke).

## P3 — structural

- **pretokenize_dataset.py** — extracted the 7 stateless path/IO static helpers + `_FRAME_RE`
  into `dreamervla/dataset/_pretokenize_helpers.py`; the dataset classes keep them as
  static-method delegators (call sites + subclasses unchanged). Dropped the now-unused
  `import re`. The high-coupling init/windowing core stays.
- **embodied_eval_runner.py (Tier-1)** — extracted the 9 zero-coupling static methods
  (encoder-state normalize, ckpt-cfg, real-relabel rewards, numpy/array/action-stats utils,
  clip-bounds, prefix-strip, image-resize) into `dreamervla/runners/_embodied_eval_helpers.py`
  with static-method delegators. god-file 2522 → 2431 LOC. The higher-coupling rollout /
  action-decoding / vla-encoding / dreamer-latent tiers remain in the open backlog.
- **pixel-WM loss scaffolding** — ASSESSED: **genuinely diverges, not unified.** Shared core
  shape, but token WM uses categorical CE (configurable reduction) vs pixel-WM fixed-mean
  MSE, and `dreamer_v3_pixel_backbone_world_model` adds two extra terms (hidden_mse,
  full_hidden_loss). Unifying needs CE/MSE + reduction-flag + optional-hidden branches —
  abstraction overhead exceeds the value.

## P4 — config dedup

- **CFG-05** — extracted `configs/dreamervla/_base_wmpo_outcome.yaml`; all 6 affected
  resolved configs (2 parents + 4 inheriting children) proven semantically byte-identical
  before/after via `tests/unit_tests/_cfg_resolve_snapshot.py` (resolve=False, sorted-key
  JSON diff).
- **CFG-08** — the 9 `OpenVLA_Onetraj_*` / `RynnVLA_LIBERO` task configs renamed to snake_case
  filenames (the `task=` selection token); every selection ref updated (experiment defaults,
  task-config inheritance defaults, preprocess/collect case-arms, test selections, tutorials).
  Internal `name:`/`artifact_name:` kept CamelCase so on-disk preprocessed data/checkpoints
  are not orphaned (behaviour-preserving). 21 experiment/task combos verified to hydra-compose.

## Maintainer follow-up (docs)

- **All parameters adjustable + one reference** — `docs/PARAMETERS.md` (launcher keys, config
  groups, training/dataloader/dataset/world_model/algorithm/optim/critic/policy/init/eval/
  logger blocks with defaults, constants, env vars, interdependencies).
- **Tutorials → step-only + one explanation file** — every `docs/experiment_tutorials/` recipe
  slimmed to commands-only with snake_case `task=` tokens; all prose moved into
  `docs/experiment_tutorials/EXPLAINED.md`. Cross-linked from AGENTS.md / configs/README.md.

## GPU-box execution (2026-06-21 → 06-22, branch `chore/backlog-execution`, 8×H100)

Moved here from the open TODO once completed + pushed (`e3edf4e..c309cf0`).

- **Verification gate (was the #1 open gap).** `tests/e2e_tests` 43 passed / 3 gated-skip;
  unit suite 597 passed / 7 skipped. GPU online-RL cotrain smoke
  (`online_cotrain_pipeline_oft_action_hidden`, `training.debug=true`, resolved cfg
  `update_type=wmpo_outcome` + `repval_loss=true`) ran warmup→online→ckpt with no NaN/crash →
  the two landed numerics flips (**A4** critic-value replay bootstrap; **outcome masked_mean**)
  are GPU-exercised; A4's named GPU-smoke gate satisfied. **save→resume→continue**: `global_step`
  2→4, clean exit-0 on real disk.
- **Two real resume bugs found + fixed** (commit `099e3d6`, regression tests in
  `test_checkpoint_format_version.py`): (1) `is_hf_checkpoint` mis-detected the torch
  `latest.ckpt` as HF when sibling per-module `latest_hf_*/` dirs existed (default
  `checkpoint_format=both`) → `resolve_hf_checkpoint_dir` no longer scans a file's sibling
  subdirs; (2) `load_runner_payload(mmap=True)` left resumed optimizer tensors as mmap views of
  `latest.ckpt` → the next overwrite silently corrupted them / SIGBUS'd → eager load.
- **P3 embodied_eval split DONE** (2431→1351): four sibling mixins
  `_embodied_eval_{export,image_token,action,latent}_mixin.py` (commits `6cdd9e7`,`bedc9c2`).
  **imagine_actor_critic_step → leave** (cohesive ~40-local DreamerV3 update; no clean-bounded
  extraction worth the dropped-variable hazard).
- **P3 online_dreamervla seams extracted** (1861→1679, commits `767c774`,`b9a23f2`):
  `_online_dreamervla_dist.py` (torchrun/NCCL helpers — the RUN-01 seam) and
  `_online_dreamervla_checkpoint.py` (save/load — the X-01 seam), pure relocation + re-export.
  `parse_args` + the 1264-line `main()` stay (text-pinned / unverifiable).
- **Docs aligned.** EXPLAINED OFT-fork env corrected (fork is in the main `dreamervla` env,
  `60_verify.sh`-checked, not `dvla_oft`); action-hidden §7 sidecar requirement fixed; AGENTS.md
  polished 306→154 (big-picture, repo-aligned); CLAUDE.md Scheme-A note + tutorial §1 made
  checkpoint-relative. **Scheme-A `history` resolved = h1 (per-checkpoint, not a fixed scheme):**
  OFT `history` = `num_images_in_input ÷ #cameras`; all bundled OFT ckpts are 1-image (h1),
  derived from the single source `task.openvla_oft.expected_history`.

## RLinf-alignment + decoupling (2026-06-22, landed)

Moved here from the open TODO once completed. RLINF-01/02 landed their core but keep a
**deferred remainder** (GPU-gated wiring), which stays in the open backlog.

- **RLINF-01 — capture/restore RNG on the online DreamerVLA checkpoint path.** Canonical
  `capture_rng_state()` / `restore_rng_state()` in `dreamervla/utils/seed.py` (python `random` +
  torch + cuda) wired into `_online_dreamervla_checkpoint.save_checkpoint` (adds `payload["rng"]`)
  and `load_training_checkpoint` (restored last). Additive + backward-compatible (old ckpts lack
  the key → no-op). Cover: `tests/unit_tests/test_rng_checkpoint.py`. *Consolidation (core-req#2):*
  DreamerV3's inline RNG now routes through the same helper — two flagged, approved behaviour
  changes (also restores python `random`; warns via `warnings.warn` on CUDA-RNG restore failure).
  *Remaining (open backlog):* (a) multi-GPU RynnVLA save→resume bit-exact smoke (GPU-gated);
  (b) fold `rng` into the canonical envelope when X-01 rewrites it.
- **RLINF-02 — structured `Timers` helper + opt-in `torch.profiler`.** `dreamervla/utils/timers.py`
  — `Timers` (context-manager timing, mean/sum/min/max, `to_metrics(prefix="time")`, optional
  `cuda_sync`) + `Profiler` (config-gated, default-off `torch.profiler` wrapper, no-op when
  disabled). Cover: `tests/unit_tests/test_timers.py` (6 tests). Unblocks the deferred P5 kernel
  tuning. *Remaining (open backlog, GPU-gated):* wire into the training loops — reroute the
  scattered `f"time/..."` points through `Timers` and add the default-off `Profiler`.
- **RLINF-03 — `.github/workflows/ci.yml`.** A `lint` job (`ruff check dreamervla tests`, ruff
  pinned `0.15.14`) on push/PR. Made the tree repo-wide ruff-clean first (one pre-existing `I001`
  auto-fixed, behaviour-neutral). Pytest intentionally not run on stock runners (needs the
  hand-built `dreamervla` conda env); `ruff format --check` and the `__init__.py` check omitted to
  avoid a false-red gate.
- **RLINF-04 — validate checkpoint `format_version` on load.** Added `_check_format_version` inside
  `dreamervla/utils/hf_checkpoint.load_runner_payload` (the single chokepoint for all four
  runner-payload consumers). Only the unsafe direction hard-fails (ckpt newer than code);
  missing/older versions stay loadable (dual-read backward-compat holds). Cover:
  `tests/unit_tests/test_checkpoint_version_guard.py`.
- **DECOUPLE-01 — success classifier via a `_target_`-aware builder.**
  `dreamervla.models.reward.build_classifier` honors a Hydra `_target_`, else falls back to the
  default `LatentSuccessClassifier` (byte-identical for legacy ckpts). Routed the three hardcoded
  sites (`online_dreamervla`, `dreamervla_runner`, `online_cotrain_runner`) through it. Cover:
  `tests/unit_tests/test_build_classifier.py`.

## WMPO imagination memory (2026-06-23, landed)

- **MEM-RL-01 (immediate fix) — group-aligned micro-batch of the WMPO outcome update.** Commit
  `816dd33` (`feat(rl): micro-batch the WMPO outcome update to bound peak GPU memory`).
  `dino_wmpo_outcome_step` now (Phase 1) imagines + scores each group-aligned start slice into a
  transient per-slice CPU host buffer, (Phase 2) assembles the GLOBAL scoring tensors, then (Phase
  3) runs the multi-epoch policy update streaming one slice / one chunk back to GPU at a time and
  backprops chunk-by-chunk. Every slice is normalized by the GLOBAL `B_eff`
  (`masked_mean_ratio_chunk_term(..., B_eff)`), so summing the per-slice gradients reproduces the
  full-batch gradient exactly (incl. the BC anchor). Knob `algorithm.wmpo.update_micro_batch_starts`
  (`<= 0` or `>= n_starts` ⇒ one full-batch slice = bit-for-bit the original path; default off).
  Peak GPU measured ~82GB (full-batch OOM) → ~11GB at 96 rollouts/slice. Cover:
  `tests/unit_tests/test_wmpo_microbatch_equivalence.py` (per-slice grad == full-batch grad incl. BC
  anchor + unequal slices) and `test_wmpo_slice_latent.py`. *Remaining (open backlog):* the
  imagination host data is still a local `slices` list, not promoted to an explicit
  imagination-buffer abstraction separate from `OnlineReplay` — that structural step overlaps
  MEM-RL-02 and stays open.

## RUN-01 — online DDP routed through the shared distributed helper (2026-06-23, landed)

- **RUN-01 (code) — `online_dreamervla.main` DDP now goes through `NopretokenizeSFTDistributedHelper`.**
  Commit `85788fc` (`feat(distributed): add default-off DDP opt-ins; route online_dreamervla through
  helper (RUN-01)`). Three default-off opt-ins were added to the shared helper so the standalone RynnVLA
  online path stops hand-rolling its own DDP: `find_unused_parameters` + `broadcast_buffers` as per-call
  kwargs on `wrap_trainable_module`/`_wrap_module_with_ddp` (`None` ⇒ the historical hardcoded `False`, so
  every existing OFT caller stays byte-identical), and `nccl_timeout_seconds` on `initialize` (`None` ⇒ no
  timeout). `main()` now builds the helper via `initialize(nccl_timeout_seconds=DVLA_DDP_TIMEOUT_SEC)` and
  wraps wm/policy/critic (`find_unused_parameters=True`) and the classifier (`False`) with
  `broadcast_buffers=True`, reproducing the old DDP kwargs on the multi-GPU path. The custom all-reduce
  error-wrapping (`_dist_all_reduce_flag/int`) is kept; the now-dead `_init_distributed` seam and the
  orphaned `DDP` import were removed. Cover: `tests/unit_tests/test_distributed_ddp_opt_ins.py` (7 tests,
  TDD red→green); full suite 739 passed, zero regressions. *Remaining (open backlog):* a RynnVLA multi-GPU
  save→resume GPU smoke (suite-green ≠ verified) + confirming the two flagged `WORLD_SIZE=1`-only
  divergences — stays open as RUN-01 in the todo backlog.
