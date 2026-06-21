# TODO backlog (open items)

Open work only. Done items: `2026-06-21-cleanup-execution-log.md`. Detail:
`2026-06-21-codebase-cleanup-review.md` (cleanup), `2026-06-21-rlinf-alignment-correctness-audit.md` (RLinf).
Constraint on all code work: **behaviour-preserving** unless flagged "changes numerics".

## P0 — correctness (highest value)

- [ ] **A4-intent** — replay λ-return `boot=raw_returns[:,0]` (imagined return) vs the
  standard DreamerV3 critic-value bootstrap. Dead param/compute already removed; decide
  if `boot` should be the per-state critic value. *(changes numerics — needs design call)*
- [ ] **Enable A2/A3 guards** — `_ppo_ratio(clip_log_ratio=)` + `_ppo_clip_term(clip_ratio_c=)`
  now exist (default-off). Decide whether to set them in the RL configs (RLinf uses `clip_ratio_c: 3.0`). *(changes numerics if enabled)*
- [ ] **Fix the 6 pre-existing failing tests** — 3× `test_online_cotrain_pipeline` (mock
  doesn't set `self.policy`), 1× `test_online_cotrain_ray_runner` (`_metric_logger`), 2×
  `test_setup_scripts` (script-curation). Restores a green suite.
- [ ] **outcome loss normalization** — global mask-sum denominator biases gradient toward
  long/failed rollouts vs RLinf `masked_mean_ratio`. Decide intended weighting. *(numerics)*
- [ ] **Verify `ppo_gamma`/`lam` in YAML** — default `ppo_gamma=1.0`; RLinf embodiment uses
  0.99/0.95. Confirm configs set these deliberately.

## P1 — keep/delete decisions (need your call, then I act)

- [ ] **PRE-02** — delete orphan scripts `pretoken_world_model.py`,
  `world_model_bi_views_conv_generation.py` (0 refs, ~384 LOC).
- [ ] **PRE-03** — drop unused `FlexARItemProcessor` (base) + `FlexARItemProcessorActionFast` (test-only).
- [ ] **MOD-10** — `encoder/oft_action_hidden_encoder.py` placeholder (always raises) — delete or keep as scaffold.
- [ ] **RUN-14** — `online_dreamervla_multiproc.py` unwired fork — archive/delete/wire.
- [ ] **DIAG-06** — 16 doc-only diagnostics scripts (4 test-pinned) — archive or prune as a batch.
- [ ] **MOD-07** — two OFT action-model impls (`official` vs `dreamervla`) — retire one?
- [ ] **Rename `real_relabel_ppo_loss`** — it's a frozen-old-logprob BC anchor, not PPO; name overstates fidelity.

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

## P4 — config dedup (low risk; verify resolved config byte-identical)

- [ ] **CFG-04** `configs/task/libero_{goal,object,spatial,10}.yaml` → `_base_libero.yaml` + thin `defaults:`.
- [ ] **CFG-05** `_base_wmpo_outcome.yaml` for the 2 parallel `*_wmpo_outcome` configs.
- [ ] **CFG-06** thin `defaults:` for `OpenVLA_Onetraj_*` + rynnvla classifier variants. **CFG-08** task naming snake_case.

## Verification gaps (not yet run)

- [ ] No GPU run / `tests/e2e_tests` not run. Cotrain GPU smoke (save→resume→continue) not run.

## Won't-fix / intentional (record only)

ALG-02 (return assembly differs by rank/discount), UDA-06/04, MOD-05 (vendored OFT loader),
HF `register()` triplets (different classes per site), JSONL logging (`JsonLogger` drops
non-numeric fields — different job), RUN-09 (`build_optimizer` filters `requires_grad`),
`_decode_bpe` vs reconstructor, divergent diagnostics device-resolution groups, KL k1
signed estimator. See review/audit docs for rationale.
