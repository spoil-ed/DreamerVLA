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

- [ ] **A4-intent** — replay λ-return `boot=raw_returns[:,0]` (imagined return) vs the
  standard DreamerV3 critic-value bootstrap. Dead param/compute already removed.
  *Introduced to maintainer; kept as-is (imagined-return boot) — flip to critic-value
  bootstrap only on request.* *(changes numerics — needs design call)*
- [ ] **outcome loss normalization** — global mask-sum denominator biases gradient toward
  long/failed rollouts vs RLinf `masked_mean_ratio`. *Introduced; kept as-is — open the
  RLinf-aligned `masked_mean` only on request.* *(numerics)*
- [x] **A1 / ALG-03 (entropy-key)** — RESOLVED in code: all three PPO routes
  (`outcome.py:196`, `dense.py`, `dense_chunk.py`) read entropy via `grpo._entropy_coef`
  (`actent` → `entropy_coef` → 0.0); no silent `actent` no-op remains. Siblings A2/A3
  (log-ratio clamp + dual-clip) already landed — see the cleanup-execution-log pass 2.

## P2 — needs migration design (feature, not pure refactor)

- [ ] **X-01** — unify the 3 checkpoint payload schemes + 2 dirs (`checkpoints/` vs `ckpt/`).
  Needs a format-version + dual-read loader (else breaks resume of old ckpts). *(numerics/IO)*
- [ ] **DIAG-01** — route the ~9 diagnostics WM-loaders through one helper via a new opt-in
  `load_world_model_state_from_dict(remap_reward_head=, skip_shape_mismatch=)`.
- [ ] **RUN-01** — dreamer runners use `self.distributed`/base loader instead of hand-rolled
  DDP/loader. *(DDP-sensitive — needs multi-GPU test)*
- [ ] **X-03** — magic token id `10004` (45 sites) + `model_dim 4106` (8 configs) → config. *(config-resolution sensitive)*

## P3 — structural (smell, not duplication)

- [ ] Split god-files: `embodied_eval_runner.py` (2514), `online_dreamervla.main` (1265),
  `algorithms/dreamervla.imagine_actor_critic_step` (834), `pretokenize_dataset.py` (822).
- [ ] Pixel WMs share the same loss scaffolding as the token WMs (noted by the token-WM
  unification pass) — a larger follow-on unification, not yet started.

## P4 — config dedup (low risk; verify resolved config byte-identical)

- [ ] **CFG-05** `_base_wmpo_outcome.yaml` for the 2 parallel `*_wmpo_outcome` configs.
- [ ] **CFG-08** task naming snake_case (`OpenVLA_Onetraj_*`, `RynnVLA_LIBERO` → snake_case;
  update ~15 `task=` refs + script case-arms + test asserts). *(CFG-06 already done: the
  `OpenVLA_Onetraj_*` + rynnvla classifier configs are already thin `defaults:`.)*

## Verification gaps (not yet run)

- [ ] No GPU run / `tests/e2e_tests` not run. Cotrain GPU smoke (save→resume→continue) not run.

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
