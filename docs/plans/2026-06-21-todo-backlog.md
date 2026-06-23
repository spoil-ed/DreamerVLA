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

- [ ] **RLINF-01 (remainder) — multi-GPU RynnVLA save→resume bit-exact RNG smoke.** The RNG
  capture/restore + DreamerV3 consolidation landed 2026-06-22 (see
  `../history/2026-06-21-backlog-execution-log.md`). Two open follow-ups stay GPU-gated: (a) prove
  the multi-GPU RynnVLA save→resume is bit-exact (this box lacks the ckpts — same gate as
  RUN-01/X-01); (b) when **X-01** rewrites the envelope, fold `rng` into the canonical BaseRunner
  envelope.

- [ ] **RLINF-02 (remainder) — wire `Timers`/`Profiler` into the training loops.** The helper
  `dreamervla/utils/timers.py` landed 2026-06-22 (see history log). Remaining (GPU/Ray-gated):
  reroute the scattered `f"time/..."` points (e.g. `online_cotrain_ray_runner.py`) through `Timers`
  (core-req#2) and add the default-off `Profiler` to the loop — every integration site is in the
  GPU/Ray loops this box cannot run.

## Hydra-core decoupling roadmap

Goal: every model/dataset/impl built via `hydra.utils.instantiate(cfg.<x>)`, swappable from
config alone (AGENTS.md §1/§2 + the Hydra-core construction rules added there).

Coverage: a deterministic full-tree sweep (2026-06-22) over all `dreamervla/` subdirs +
three antipattern classes (cross-module concrete imports, runtime `_target_` mutation,
`isinstance`-on-sibling). No `isinstance`-on-concrete-sibling exists; the items below are the
complete set. (The earlier Explore audit only scoped `models/dataset/runners/algorithms/
workers`, so it missed the `preprocess/` and `envs/` sites now folded into DECOUPLE-02/04.)

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

## WMPO imagination memory

The online RL update (`dino_wmpo_outcome_step`) imagined the whole trajectory for the FULL
effective batch (B_eff ≈ batch × rollout-starts, measured ~715) and held it on GPU, then computed
the loss — pinning an 80GB H100 (`video` gather alone ≈ 24GB). **MEM-RL-01's micro-batch immediate
fix landed 2026-06-23 (`816dd33`, see `../history/2026-06-21-backlog-execution-log.md`):** the
imagination forward + loss now run one group-aligned slice / one chunk at a time, streamed from a
transient per-slice CPU host buffer, normalized by the global `B_eff` so the gradient is bit-for-bit
the full-batch one (knob `update_micro_batch_starts`, default off). One structural item remains, plus
the bigger refactor:

- [ ] **MEM-RL-01 (remainder) — promote the imagination host data to an explicit buffer
  abstraction.** The micro-batch immediate fix is done (above). What is NOT done: the imagination
  data (`actor_feat, action, old_log_prob, advantage`) still lives in a local `slices` list inside
  `dino_wmpo_outcome_step`, not in its OWN host buffer object explicitly separate from `OnlineReplay`
  (two distinct buffers / lifetimes: persistent real replay vs per-update imagination buffer). This
  structural promotion **overlaps MEM-RL-02** (which subsumes it) — do them together, or fold this
  into MEM-RL-02.
- [ ] **MEM-RL-02 — WM-as-env (structural, RLinf/WoVR alignment).** Make the world model a gym
  env (cf. `RLinf/rlinf/envs/world_model/`), so WM-imagination becomes a normal rollout that
  writes trajectories to a (separate) host replay buffer, and the policy update is a standard
  micro-batched PPO sampling from it. This removes the in-update imagination entirely and
  matches WoVR. Bigger refactor; subsumes the MEM-RL-01 remainder above.

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
