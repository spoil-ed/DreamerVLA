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

- [ ] **X-01 (scheme-unify, remaining half)** — collapse the 3 checkpoint payload SCHEMAS
  (BaseRunner `{cfg,state_dicts,pickles}` / online `{env_step,update_step,cfg,state_dicts}`
  / WM-only `{model}`) into one writer. The format_version stamp + shared load path already
  landed (pass 3). This is the **format-breaking** half: needs a dual-read loader keyed on
  `format_version` and the **GPU cotrain save→resume→continue smoke** to prove old + new
  ckpts resume. *(IO; verification-gated — see GPU smoke below)*
- [ ] **RUN-01** — route the dreamer runners through the base distributed helper instead of
  the hand-rolled DDP. `online_dreamervla.main` hand-rolls `dist.init_process_group` +
  per-module DDP wrap (`find_unused_parameters`, `DVLA_DDP_TIMEOUT_SEC`, the all-reduce flag
  helpers); it is a standalone `main()`, not a `BaseRunner` subclass, so this is a real
  restructure and genuine divergences must be preserved, not unified. *(DDP-sensitive —
  needs a multi-GPU test; cannot verify on this box)*

## P3 — structural (god-file splits; behaviour-preserving, suite-verifiable)

- [ ] **`algorithms/dreamervla.imagine_actor_critic_step`** (819 lines) — single cohesive
  function threading ~50 locals; decomposing into helpers risks dropping a variable and can
  hurt readability. Extract only cleanly-bounded sub-computations with explicit in/out, or
  leave. Seams: `2026-06-21-backlog-execution.md` Task 10.
- [ ] **`embodied_eval_runner.py`** (2514) — extract the low-coupling / static helper methods
  to sibling modules (pretokenize-style delegators), keep the tightly-`self`-coupled rollout
  core. Seams: Task 13.
- [ ] **`online_dreamervla.py`** (1856) — split AFTER RUN-01 + the X-01 scheme-unify (they
  rewrite the same DDP / save-load regions). Seams: Task 12.

## Verification gaps (not yet run)

- [ ] **GPU cotrain smoke (save→resume→continue) + `tests/e2e_tests`.** Cannot run on this
  box (no GPU). This is the verification gate for the two landed numerics flips (A4,
  outcome masked_mean), the X-01 scheme-unify, and RUN-01. Commands: the cold-start
  warmup/cotrain tutorial + `docs/experiment_tutorials/EXPLAINED.md`. Run on a GPU box.

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
