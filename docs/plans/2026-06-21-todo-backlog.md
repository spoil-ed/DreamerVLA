# TODO backlog (open items)

Open work only. Done items live in `../history/2026-06-21-backlog-execution-log.md`
(passes 1–3 + the 2026-06-21→22 GPU-box execution) and the other `docs/history/`
logs. This file is the only live open-items list.

## Core requirements (核心思想 — govern every item below)

1. **维持功能 / behaviour-preserving** — the #1 red line. Only merge code proven
   equivalent (AST/diff for identical, algebra/0.0-diff for math, seeded-batch for
   models); where implementations genuinely diverge, **flag it, do not silently
   unify**. Full unit suite stays green after every commit; anything that changes
   numerics is marked "changes numerics" and needs an explicit decision.
2. **统一实现 / one implementation per job** — the same functionality lives in ONE
   place; no competing or copy-pasted schemes. Make one canonical helper/interface,
   route all consumers through it.
3. **对齐 RLinf** — the upstream `RLinf` repo (workspace sibling) is the reference for
   RL correctness and overall code-tree alignment; diverge only deliberately.
4. **正确合理的接口** — algorithm/helper primitives have one correct, extensible
   interface (opt-in, default-off options); no lying/dead parameters.
5. **干净 + 简短可读** — minimal, surgical changes; no speculative/bloated code.

## Open items — all gated on the RynnVLA online path

The three remaining items all touch the **standalone RynnVLA `online_dreamervla` path**,
whose behaviour-changing parts the **unit suite cannot verify** (no multi-GPU DDP test) and
which **this box cannot run** (its WM/classifier ckpts — `outputs/worldmodel/
rynn_dino_wm_action_hidden/chunkaware_pinned/step_00017000.ckpt` + the outcome classifier — are
absent). Per core-req#1 they were not shipped blind. The clean seams already exist (extracted
2026-06-22, see history log): `_online_dreamervla_dist.py` and `_online_dreamervla_checkpoint.py`.

- [ ] **RUN-01 — route `online_dreamervla.main` DDP through the base helper.** Code-solvable:
  online_dreamervla wraps each *whole* module → maps to `helper.wrap_trainable_module`
  (not the per-child `wrap_world_model`). Needs **three default-off opt-ins** on
  `NopretokenizeSFTDistributedHelper` to preserve genuine divergences (so the mainline OFT
  callers stay byte-identical):
  1. `find_unused_parameters` — `True` for world_model/policy/critic, `False` for the classifier
     (helper currently hardcodes `False`).
  2. `broadcast_buffers` — online_dreamervla uses the DDP default **`True`**; the helper's
     `_wrap_module_with_ddp` hardcodes **`False`** (easy to miss — naive routing silently flips it).
  3. NCCL **timeout** — `DVLA_DDP_TIMEOUT_SEC` (helper's `initialize` has none).
  Keep the custom all-reduce error-wrapping (`_dist_all_reduce_flag/int`) — a deliberate divergence.
  **Verify with a RynnVLA multi-GPU save→resume smoke** before relying on it; suite-green ≠ verified.

- [ ] **X-01 (②, format-breaking remainder) — unify `online_dreamervla.save_checkpoint`.**
  Collapse its `{format_version, env_step, update_step (top-level), cfg, state_dicts}` into the
  canonical BaseRunner envelope `{format_version, cfg, state_dicts, pickles}`. This is a
  **multi-site format break**: top-level `env_step`/`update_step` is a consumer contract for
  `load_training_checkpoint`, `frozen_wm_actor_critic`, and three diagnostics
  (`measure_reward_and_drift` reads `ckpt["env_step"]` directly, `measure_wm_imagine_actor`,
  `measure_wm_imagine_fidelity`). Needs a dual-read loader + a **RynnVLA GPU save→resume** to prove
  old + new ckpts resume. (Resolved already: ① BaseRunner is the canonical writer and is
  GPU-verified; ③ the WM-only/classifier `{model,...}` payload is a genuinely-divergent inference
  artifact, flagged **not** unified per core-req#1. The shared dual-read `load_runner_payload` is
  in place.) Entangled with RUN-01 (BaseRunner-ifying `main()` is the clean route).

- [ ] **`online_dreamervla.py` `main()` split (P3).** The dist + checkpoint seams are already
  extracted (1861→1679). The remainder is `parse_args` (text-pinned by
  `test_online_env_episode_end`) and the 1264-line `main()` loop (unverifiable without the RynnVLA
  run). Do **after** RUN-01 + X-01② settle the DDP / save-load regions they rewrite.

## RLinf-alignment learnings (open enhancements)

Surfaced 2026-06-22 by re-surveying the sibling `RLinf` repo against the current tree
(core-req#3). Each was **verified to be a genuine gap** (not already present) and fits the
single-machine scope; the deliberately-out-of-scope RLinf features (collocated/disaggregated/
hybrid placement modes, vLLM/SGLang, Megatron TP+PP, multi-node, VRAM auto-sizing, Channel
key-routing) stay non-targets per `../ray_rlinf_alignment_todo.md` and are not listed here.

- [x] **RLINF-01 — capture/restore RNG state on the online DreamerVLA checkpoint path.**
  *Landed 2026-06-22 (TDD, suite green):* canonical `capture_rng_state()` / `restore_rng_state()`
  in `dreamervla/utils/seed.py` (python `random` + torch + cuda; numpy is outside `set_seed`'s
  contract) wired into `_online_dreamervla_checkpoint.save_checkpoint` (adds `payload["rng"]`) and
  `load_training_checkpoint` (restored last, after all state-dict loads). **Additive +
  backward-compatible** — old ckpts lack the key → restore is a no-op (core-req#1 preserved).
  Unit cover: `tests/unit_tests/test_rng_checkpoint.py` (helper round-trip + bit-exact
  save→load draws + None/partial tolerance).
  *Consolidation (done 2026-06-22, core-req#2):* DreamerV3's inline RNG
  (`_dreamer_runner_common.py` `_save_ckpt`/`_maybe_resume`) now routes through the same shared
  helper. Two **flagged behaviour changes** (approved): DreamerV3 resume now (i) also restores
  python `random` (it previously snapshotted torch+cuda only — strictly more deterministic) and
  (ii) warns via `warnings.warn` instead of a `[tag]` log line on a CUDA-RNG restore failure.
  Unit cover added (`test_dreamerv3_*` in the same file). RNG capture/restore is now single-source.
  *Remaining (deferred):* (a) the multi-GPU RynnVLA save→resume bit-exact smoke is GPU-gated
  (this box lacks the ckpts; same gate as RUN-01/X-01); (b) when **X-01** rewrites the envelope,
  fold `rng` into the canonical BaseRunner envelope.

- [x] **RLINF-02 — structured `Timers` helper + opt-in `torch.profiler`.**
  *Landed 2026-06-22 (TDD, suite green):* `dreamervla/utils/timers.py` — `Timers` (context-manager
  timing, mean/sum/min/max reduction, `to_metrics(prefix="time")` namespacing, optional `cuda_sync`)
  + `Profiler` (config-gated, **default-off** `torch.profiler` wrapper with schedule + chrome-trace
  export; a safe no-op when disabled). Unit cover: `tests/unit_tests/test_timers.py` (6 tests, incl.
  CPU trace emission). **Unblocks the deferred P5 "kernel tuning gated on benchmark data"** —
  hotspot selection now has a profiler.
  *Remaining (GPU-gated):* wire into the training loops — reroute the scattered `f"time/..."`
  points (e.g. `online_cotrain_ray_runner.py`) through `Timers` (core-req#2) and add the default-off
  `Profiler` to the loop. Deferred because every integration site is in the GPU/Ray-gated loops this
  box cannot run; the helper is staged ahead of wiring (mirrors the `_online_dreamervla_*` seams).

- [x] **RLINF-03 — add `.github/workflows/` CI.**
  *Landed 2026-06-22:* `.github/workflows/ci.yml` — a `lint` job (`ruff check dreamervla tests`,
  ruff pinned `0.15.14`) on push/PR. Made the tree repo-wide ruff-clean first: one pre-existing
  `I001` import-sort in `tests/unit_tests/test_actor_file_split_imports.py` was auto-fixed
  (behaviour-neutral). Lint command verified green locally.
  *Intentionally scoped down (documented in the workflow):* the pytest suite is **not** run on
  stock runners — it needs the hand-built `dreamervla` conda env (transformers 4.40.1 fork,
  robosuite/third_party) that isn't reproducible from PyPI; `ruff format --check` (≈244 files are
  historically unformatted) and the `__init__.py`-presence check (several PEP 420 namespace
  packages by design) are omitted to avoid a false-red gate.

- [x] **RLINF-04 — validate checkpoint `format_version` on load.**
  *Landed 2026-06-22 (TDD, suite green):* `format_version` was written by every writer
  (`_online_dreamervla_checkpoint`, `base_runner`) but **never checked**, so a newer-format ckpt
  loaded by older code was silently mishandled. Added `_check_format_version` inside
  `dreamervla/utils/hf_checkpoint.load_runner_payload` — the single chokepoint for all four
  runner-payload consumers (online resume, BaseRunner, embodied eval, rynn preprocess). Only the
  unsafe direction hard-fails (ckpt newer than code); missing/older versions stay loadable, so the
  dual-read backward-compat contract holds (core-req#1). RLinf stores+validates its version on
  load; CLAUDE.md calls for early resume-checkpoint validation. Unit cover:
  `tests/unit_tests/test_checkpoint_version_guard.py` (future rejected / current + legacy accepted).

## Hydra-core decoupling roadmap

Goal: every model/dataset/impl built via `hydra.utils.instantiate(cfg.<x>)`, swappable from
config alone (AGENTS.md §1/§2 + the Hydra-core construction rules added there).

Coverage: a deterministic full-tree sweep (2026-06-22) over all `dreamervla/` subdirs +
three antipattern classes (cross-module concrete imports, runtime `_target_` mutation,
`isinstance`-on-sibling). No `isinstance`-on-concrete-sibling exists; the items below are the
complete set. (The earlier Explore audit only scoped `models/dataset/runners/algorithms/
workers`, so it missed the `preprocess/` and `envs/` sites now folded into DECOUPLE-02/04.)

- [x] **DECOUPLE-01 — success classifier via a `_target_`-aware builder.** *Landed 2026-06-22
  (TDD, suite green):* `dreamervla.models.reward.build_classifier` honors a Hydra `_target_`,
  else falls back to the default `LatentSuccessClassifier` — byte-identical for legacy ckpts.
  Routed the three hardcoded sites (`online_dreamervla`, `dreamervla_runner`,
  `online_cotrain_runner`) through it; `latent_classifier_runner` already used the pattern.
  Cover: `tests/unit_tests/test_build_classifier.py`.
- [ ] **DECOUPLE-02 — env / encoder / policy construction (GPU-gated).** `OpenVLAOFTPolicy`,
  `RynnVLAEncoder`, `DreamerVLAOnlineTrainEnv` are built with hardcoded params across
  `runners/online_utils`, `runners/oft_collect_common`, `envs/train_env.py:694`, and
  `preprocess/preprocess_oft_action_hidden.py:273` + `preprocess/preprocess_rynn_pixel_hidden.py:430`;
  route through `instantiate(cfg.<x>)` and move the baked "contract" params into config. Deferred:
  these run only on GPU/LIBERO and the box cannot E2E-verify; refactoring blind risks breaking real
  training (core-req#1).
- [ ] **DECOUPLE-03 — action head injection.** `L1RegressionActionHead` is hardcoded in three
  actors + an encoder; inject via a protocol + config `_target_`. Deep model-internal, GPU-gated.
- [ ] **DECOUPLE-04 — small impls.** `ReturnPercentileTracker` direct instantiation (low value);
  `BalancedTerminalDataset` runtime `cfg._target_` mutation (`frozen_wm_actor_critic.py:240`,
  `diagnostics/finetune_reward_head_sparse.py:160`) → move selection into config; the HF-save
  `target=` string in `online_cotrain_runner` → derive from config; `soft_update` lives under
  `models/critic` but is used by `algorithms/` → move to a shared util.
- Won't-fix: `ChunkAwareDinoWMWorldModel(DinoWMWorldModel)` inheritance is code reuse and is
  already swappable via `world_model._target_` — not a coupling violation.

## WMPO imagination memory (GPU-verified 2026-06-22)

The online RL update (`dino_wmpo_outcome_step`) imagines the whole trajectory for the FULL
effective batch (B_eff ≈ batch × rollout-starts, measured ~715) and holds it on GPU, then
computes the loss — pinning an 80GB H100 (`video` gather alone ≈ 24GB). WM warmup is fixed
(gradient checkpointing) and the real-env rollout already lives on host (`OnlineReplay`), but
the imagination does not. Two open items:

- [ ] **MEM-RL-01 — explicit, separate imagination replay buffer + micro-batch (immediate fix).**
  The imagination data (`state/actor_feat, action, old_log_prob, advantage`) must live in its
  OWN host buffer object, explicitly separate from `OnlineReplay` (which holds real-env
  transitions): two distinct buffers, two distinct lifetimes (persistent real replay vs
  per-update imagination buffer). The PPO update then samples that imagination buffer in
  group-aligned micro-batches over B_eff (imagination → predict_success → GRPO advantage → PPO
  loss, gradient-accumulated), moving one micro-batch to GPU at a time (RLinf
  `split_dict_to_chunk` + `put_tensor_device`). Valid because PPO is score-function — gradient
  flows only through the policy log-prob re-eval, NOT the WM dynamics (rollout is `no_grad`) —
  so the imagination is pure data. Partial work landed (actor_feats/video offloaded to CPU,
  predict_success micro-batched); the loss + imagination forward still run at full B_eff and
  must be split, and the host data should be promoted to an explicit buffer abstraction.
- [ ] **MEM-RL-02 — WM-as-env (structural, RLinf/WoVR alignment).** Make the world model a gym
  env (cf. `RLinf/rlinf/envs/world_model/`), so WM-imagination becomes a normal rollout that
  writes trajectories to a (separate) host replay buffer, and the policy update is a standard
  micro-batched PPO sampling from it. This removes the in-update imagination entirely and
  matches WoVR. Bigger refactor; do after MEM-RL-01.

## Won't-fix / intentional (record only)

**DIAG-06** (16 doc-only diagnostics) and **MOD-07** (`official` OFT action-model) — kept by
maintainer decision: not zero-import dead code (the diagnostics carry README rows + hygiene
test pins; `official` is called by `diagnostics/eval_openvla_oft_libero.py`) and they hold
paper/diagnostic value.
**Pixel-WM loss scaffolding** — assessed, genuinely diverges (CE vs MSE + extra backbone
hidden terms); not unified (see the pass-3 log).
ALG-02 (return assembly differs by rank/discount), UDA-06/04, MOD-05 (vendored OFT loader),
HF `register()` triplets (different classes per site), JSONL logging (`JsonLogger` drops
non-numeric fields — different job), RUN-09 (`build_optimizer` filters `requires_grad`),
`_decode_bpe` vs reconstructor, divergent diagnostics device-resolution groups, KL k1
signed estimator. See review/audit docs for rationale.
