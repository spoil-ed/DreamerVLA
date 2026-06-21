# TODO backlog (open items)

Open work only. Done items: `../history/2026-06-21-cleanup-execution-log.md`. Detail:
`../history/2026-06-21-codebase-cleanup-review.md` (cleanup),
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

## P0 — correctness (highest value)

- [x] **A4-intent** — DONE (changes numerics, maintainer-approved). Replay λ-return now
  bootstraps with the critic's per-state value (DreamerV3 repval); `repl_loss.slowtar`
  selects target vs fast critic (was a dead key). Verify deep path via GPU smoke.
- [x] **outcome loss normalization** — DONE (changes numerics, maintainer-approved).
  Switched to RLinf `masked_mean_ratio` (per-rollout/episode-length normalization) via
  `grpo.masked_mean_ratio_chunk_term`; PPO + BC both use it. Test: `test_masked_mean_ratio`.
- [x] **A1 / ALG-03 (entropy-key)** — RESOLVED in code: all three PPO routes
  (`outcome.py:196`, `dense.py`, `dense_chunk.py`) read entropy via `grpo._entropy_coef`
  (`actent` → `entropy_coef` → 0.0); no silent `actent` no-op remains. Siblings A2/A3
  (log-ratio clamp + dual-clip) already landed — see the cleanup-execution-log pass 2.

## P2 — needs migration design (feature, not pure refactor)

- [~] **X-01** — PARTIAL. Done: `CHECKPOINT_FORMAT_VERSION` stamped on both writers;
  online load routed through the shared `load_runner_payload`; dirs already unified
  (`checkpoints/` canonical, `ckpt/` legacy read-only). DEFERRED: collapsing the 3
  payload SCHEMAS into one writer — format-breaking, needs the GPU cotrain
  save→resume→continue smoke to verify (cannot run here).
- [x] **DIAG-01** — DONE. New opt-in `load_world_model_state_from_dict(remap_reward_head=,
  skip_shape_mismatch=, reset_reward_head=)`; path-based loader + `visualize_dreamervla_reward`
  route through it. Plain `strict=False` / generic multi-module loaders intentionally left
  (different job — see commit). Test: `test_wm_state_loader`.
- [ ] **RUN-01** — DEFERRED. `online_dreamervla.main` hand-rolls `dist.init_process_group`
  + per-module DDP wrap (with `find_unused_parameters`, `DVLA_DDP_TIMEOUT_SEC`, all-reduce
  flag helpers); it is a standalone `main()`, not a `BaseRunner` subclass, so routing
  through `NopretokenizeSFTDistributedHelper` is a real restructure. **DDP-sensitive — needs
  a multi-GPU test that cannot run on this box.** Genuine divergences must be preserved, not
  unified.
- [x] **X-03** — DONE. `dreamervla/constants.py:DEFAULT_ACTION_TOKEN_ID` single-sources the
  `10004` literal across all first-party sites; the 5 hardcoded eval-runner token insertions
  now read `eval.target_token_id` (adjustable). Vendored chameleon code left by design.
  `model_dim 4106` is already a per-config value (adjustable). Documented in PARAMETERS.md;
  covered by the existing config-validation + eval-runner import tests.

## P3 — structural (smell, not duplication)

- [~] Split god-files: PARTIAL. `pretokenize_dataset.py` — pure path/IO static helpers
  extracted to `_pretokenize_helpers.py` (behaviour-preserving; the high-coupling
  init/windowing core stays). DEFERRED (high-coupling / fragile, lowest-priority structural
  smell): `embodied_eval_runner.py` (2514), `algorithms/dreamervla.imagine_actor_critic_step`
  (819, tightly threads ~50 locals), `online_dreamervla.main` (1856; must follow RUN-01 +
  the X-01 scheme-unify — same regions). Seams documented in `2026-06-21-backlog-execution.md`
  (Tasks 10–13). Not rushed given the behaviour-preserving + everything-runnable red lines.
- [x] Pixel WMs vs token WMs loss scaffolding — ASSESSED: **genuinely diverges, not
  unified** (per the do-not-force-unify rule). Shared core shape (rec + dyn/rep KL + reward +
  continue) but: token WM = categorical CE with configurable `rec_reduction`; pixel WMs =
  fixed-mean MSE; `dreamer_v3_pixel_backbone_world_model` adds two extra terms (hidden_mse,
  full_hidden_loss). Unifying needs CE-vs-MSE + reduction-flag + optional-hidden branches —
  abstraction overhead > value. Recorded.

## P4 — config dedup (low risk; verify resolved config byte-identical)

- [x] **CFG-05** — DONE. `_base_wmpo_outcome.yaml` extracted; all 6 affected resolved
  configs (2 parents + 4 children) proven semantically byte-identical via
  `tests/unit_tests/_cfg_resolve_snapshot.py`.
- [x] **CFG-08** — DONE. 9 task configs → snake_case filenames (`task=` selection token);
  all selection refs (experiment defaults, inheritance defaults, script case-arms, test
  selections, tutorials) updated. Internal `name:`/`artifact_name:` kept CamelCase so
  on-disk data is not orphaned (behaviour-preserving). 21 experiment/task combos verified
  to hydra-compose.

## Verification gaps (not yet run)

- [ ] No GPU run / `tests/e2e_tests` not run. Cotrain GPU smoke (save→resume→continue) not run.
  **Cannot run on this box (no GPU).** Commands are in the cold-start warmup/cotrain tutorial
  and `EXPLAINED.md`; this smoke is the verification gate for A4, the X-01 scheme-unify, and
  RUN-01. Run on a GPU box / by the maintainer.

## Won't-fix / intentional (record only)

**DIAG-06** (16 doc-only diagnostics) and **MOD-07** (`official` OFT action-model) — kept by
maintainer decision: not zero-import dead code (the diagnostics carry README rows + hygiene
test pins; `official` is called by `diagnostics/eval_openvla_oft_libero.py`) and they hold
paper/diagnostic value.
ALG-02 (return assembly differs by rank/discount), UDA-06/04, MOD-05 (vendored OFT loader),
HF `register()` triplets (different classes per site), JSONL logging (`JsonLogger` drops
non-numeric fields — different job), RUN-09 (`build_optimizer` filters `requires_grad`),
`_decode_bpe` vs reconstructor, divergent diagnostics device-resolution groups, KL k1
signed estimator. See review/audit docs for rationale.
