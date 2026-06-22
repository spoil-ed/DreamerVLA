# TODO backlog (open items)

Open work only. Done items: `../history/2026-06-21-backlog-execution-log.md` (pass 3,
branch `chore/backlog-execution`) and `../history/2026-06-21-cleanup-execution-log.md`
(passes 1–2). Detail: `../history/2026-06-21-codebase-cleanup-review.md` (cleanup),
`../history/2026-06-21-rlinf-alignment-correctness-audit.md` (RLinf).
(Completed plans/designs/audits were archived to `docs/history/` on 2026-06-21;
this file is the only live open-items list.)

## Core requirements (核心思想 — govern every item below)

1. **维持功能 / behaviour-preserving** — the #1 red line. Only merge code proven
   equivalent (AST/diff for identical, algebra/0.0-diff for math, seeded-batch for
   models); where implementations genuinely diverge, **flag it, do not silently
   unify**. Full unit suite stays green after every commit; anything that changes
   numerics is marked "changes numerics" and needs an explicit decision.
2. **统一实现 / one implementation per job** — the same functionality lives in ONE
   place; no competing or copy-pasted schemes ("front does it one way, back another").
   Make one canonical helper/interface, route all consumers through it (e.g. progress
   reporter, PPO primitives in `grpo.py`).
3. **对齐 RLinf** — the upstream `RLinf` repo (workspace sibling) is the reference for
   RL correctness and for overall code-tree alignment; diverge only deliberately.
4. **正确合理的接口** — algorithm primitives have one correct, extensible interface
   (opt-in, default-off options) so future PPO / other-algorithm calls stay correct;
   no lying/dead parameters.
5. **干净 + 简短可读** — minimal, surgical changes; no speculative/bloated code; keep
   structure short and readable (code and docs).

Constraint shorthand below: **behaviour-preserving** unless flagged "changes numerics".

## P2 — migration design (verification-gated)

- [~] **X-01 (scheme-unify, remaining half)** — analysed 2026-06-21; partially resolved,
  format-breaking remainder **deferred** per core-req#1 (the #1 red line) + verification gate.
  - **① BaseRunner `{format_version,cfg,state_dicts,pickles}`** is the canonical writer (OFT
    cotrain via `online_cotrain_runner` + offline runners already use it; step counters live in
    `pickles`). Its save→resume→continue is now **GPU-verified** (see Verification gaps). The
    shared dual-read load path (`load_runner_payload`, reads versioned + legacy) is in place.
  - **③ WM-only / classifier `{model,threshold,config,f1}`** is a genuinely-divergent
    standalone *inference artifact* (loaded directly via `cls_payload["model"]`, classifier
    eval, warmup reuse), **not** a runner-resume payload → per core-req#1 **flag, do NOT unify**.
  - **② `online_dreamervla.save_checkpoint` `{...,env_step,update_step,...}`** (RynnVLA standalone
    `main()`) keeps step counters at the **top level**, and that is a *consumer contract*:
    `load_training_checkpoint`, `frozen_wm_actor_critic`, and three diagnostics
    (`measure_reward_and_drift` reads `ckpt["env_step"]` directly, `measure_wm_imagine_actor`,
    `measure_wm_imagine_fidelity`) read it top-level. Collapsing ② into ①'s pickled envelope is
    therefore a **multi-site format break** on a path only reachable through the standalone
    RynnVLA `main()`, which this box's RynnVLA online setup cannot GPU-verify. Deferring it (it is
    also entangled with RUN-01, which would BaseRunner-ify that `main()` and is the clean route).
- [~] **RUN-01** — analysed 2026-06-21; **deferred** (verification-gated). `online_dreamervla.main`
  is a standalone argparse script (`python -m dreamervla.runners.online_dreamervla`, `__main__`;
  only `scripts/README.md` references it) — the RynnVLA online path, **not** the mainline OFT
  cotrain. The mainline `online_cotrain_runner` is a `BaseRunner` and **already** routes DDP
  through the base helper (`self.distributed.wrap_trainable_module`). Routing `online_dreamervla`
  through the same helper requires extending it with an **opt-in `find_unused_parameters`**
  (base helper hard-codes `False`; online_dreamervla needs `True` for the outcome branch — a
  genuine divergence) and an opt-in NCCL timeout (`DVLA_DDP_TIMEOUT_SEC`). Adding that opt-in
  without also routing `online_dreamervla` would be a dead parameter (core-req#4), and the
  routing is a real `main()` restructure that needs a **RynnVLA multi-GPU** save→resume smoke to
  prove behaviour-preservation — not safely doable on this box's RynnVLA setup. Left as the
  documented next step; the helper API (`_wrap_module_with_ddp`) is the seam.

## P3 — structural (god-file splits; behaviour-preserving, suite-verifiable)

Seam details: the archived execution plan `../history/2026-06-21-backlog-execution.md`
(Tasks 10–13). The clean approach for a coupled god-class is mixins (move cohesive
method groups to sibling mixin classes the runner inherits — zero call-site change).

- [x] **`algorithms/dreamervla.imagine_actor_critic_step`** — assessed → **leave** (Task 10).
  Single cohesive DreamerV3 actor-critic update threading ~40 config-derived scalars + many
  accumulator lists; the only cleanly-bounded block (config parsing) would trade ~50 fewer
  lines for ~40 attribute-access renames and the "dropped variable" hazard the item warns of —
  not a net win. The nested helpers (`_flat_grad`/`_norm`/`_sequence_field`) are already
  extracted. Left intact per the item's own "extract cleanly-bounded, or leave" guidance.
- [x] **`embodied_eval_runner.py`** (2431 → **1351**) — **done** (Task 13). All five remaining
  groups extracted into four sibling mixins the runner inherits (zero call-site change, MRO
  resolves all self-calls): `_embodied_eval_export_mixin` (real-relabel + policy-trace export),
  `_embodied_eval_image_token_mixin` (WM IO-mode + image-BPE tokens), `_embodied_eval_action_mixin`
  (action decode/unnorm + TDMPC + hidden-vs-recon compare), `_embodied_eval_latent_mixin`
  (VLA-hidden encoding + dreamer latent/observation). Behaviour-preserving; suite green (597).
  Commits `6cdd9e7`, `bedc9c2`.
- [ ] **`online_dreamervla.py`** (1856) — deferred: gated AFTER RUN-01 + X-01 (they rewrite the
  same DDP / save-load regions, both deferred below for verification reasons).

## Verification gaps — DONE (2026-06-21, GPU box, 8×H100)

- [x] **GPU cotrain smoke (save→resume→continue) + `tests/e2e_tests`.** Ran on GPU 4–7.
  - `tests/e2e_tests`: **43 passed, 3 skipped** (the 3 skips are `DVLA_GPU_E2E` / real-OFT-ckpt
    gated). Unit baseline **597 passed, 7 skipped** (was 593 + 4 new regression tests).
  - **GPU online-RL cotrain smoke** (`online_cotrain_pipeline_oft_action_hidden`,
    `training.debug=true`, resolved cfg has `update_type=wmpo_outcome` + `repval_loss=true`)
    ran warmup → online RL → ckpt with **no NaN/crash**, exercising both landed numerics flips
    (**A4** critic-value replay bootstrap on the `repval_loss=true` path; **outcome masked_mean**
    on the wmpo_outcome route) — A4's named GPU-smoke gate is satisfied.
  - **save→resume→continue**: Run1 saved at `global_step=2`; Run2 `training.resume=true` resumed
    and continued to `global_step=4`, **clean exit-0** on real disk.
  - The smoke surfaced + fixed **two real resume bugs** (commit `099e3d6`, regression tests in
    `test_checkpoint_format_version.py`): (1) `is_hf_checkpoint(latest.ckpt)` mis-detected the
    torch ckpt as HF when sibling `latest_hf_*/` sidecars existed (default
    `checkpoint_format=both`) → `resolve_hf_checkpoint_dir` no longer scans a file's sibling
    subdirs; (2) `load_runner_payload(mmap=True)` left resumed optimizer tensors as views of
    `latest.ckpt`, which the next overwrite corrupted (silent) or SIGBUS'd → eager load.

## Docs — `experiment_tutorials` (2026-06-21)

Audited every tutorial against the repo (experiment/task tokens, script paths, Hydra keys,
module paths, links — all resolve). Concrete fixes made:
- **EXPLAINED.md** (commit `418c167`): the OFT transformers-fork note said "use the dedicated
  `dvla_oft` env" — **wrong**. Verified by dist-info: the fork
  (`github.com/moojink/transformers-openvla-oft`) is installed as the single authoritative
  transformers **in the main `dreamervla` env** (`scripts/install/40_third_party.sh`;
  `60_verify.sh` FATAL-checks it); `dvla_oft` is now vanilla PyPI. Corrected.
- **action-hidden tutorial §7** (commit `631acdd`): the "verified smoke" pinned
  `SC=..._oft_official_legacy_action_hidden_vla_policy_h2`, but that on-disk sidecar is the
  **L1-regression** route (`oft_l1_regression`, history=2, include_state=true) and the discrete
  WM (`task=openvla_onetraj_libero` expects `oft_discrete_token`, history=1, include_state=false)
  aborts on the metadata mismatch. Replaced the broken pin with the explicit metadata-match
  requirement. The offline WM route itself was re-verified to a `latest.ckpt` against a
  metadata-matching discrete sidecar.

**Scheme-A `history` — RESOLVED (2026-06-22, maintainer decision = option (a)): h1 discrete.**
Scheme-A is the h1 discrete action-query route, matching `task.openvla_oft.expected_*`
(`expected_history=1`, `expected_action_head_type=oft_discrete_token`, single agentview camera →
`token_count=56`, `wm_obs_dim=229376`; `action_hidden_dir=..._h1`). The action-hidden WM
(`oft_world_model_dinowm_chunk` → `worldmodel/openvla_oft_action_chunk`) inherits those, so the
sidecar is `..._oft_legacy_action_hidden_vla_policy_h1`. Fixed: action-hidden tutorial §1 now
preprocesses with `OFT_HISTORY=1 OFT_IMAGE_KEYS=agentview_rgb` (was h2 + two cameras), matching
the discrete recipe §3. **TODO (manual):** `CLAUDE.md` lines 23–24 still say `..._h2` — it is
write-protected from agents (ARS scope guard), so the maintainer must change `h2` → `h1` there.
The on-disk `*_h2` action-hidden dumps remain the separate (not-yet-validated) L1-regression
route and are intentionally not used by the discrete Scheme-A recipe.

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
