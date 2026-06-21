# Backlog execution plan (2026-06-21)

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans`.
> Steps use checkbox (`- [ ]`) syntax. Source spec:
> `docs/plans/2026-06-21-todo-backlog.md`. Maintainer decisions baked in below.

**Goal:** Clear every open item in the TODO backlog while honoring the #1 red line
(behaviour-preserving; full unit suite green after every commit) and the two
maintainer-approved numerics flips (A4 → critic-value bootstrap; outcome-norm →
masked_mean).

**Architecture / order:** Risk-ascending. Behaviour-preserving refactors first
(suite stays trivially green), then the interface rename, then the two isolated
numerics flips (each its own `changes-numerics` commit + test update), then the
migration-design items (checkpoint format, DDP), then the structural god-file
splits last (highest churn, lowest semantic risk). Cross-file dependencies:
`online_dreamervla.py` is touched by RUN-01 → X-01 → P3-split, in that order.

**Tooling rules (from memory + AGENTS.md):**
- Run unit tests in the **`dreamervla` conda env** (base env → ~13 spurious fails).
  Clean baseline = **582 passed, 7 skipped**.
- Commits: `--signoff`; conventional subject; **no `===` or `/` in subject**; ruff
  runs on changed Python. Co-author trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Test command: `conda run -n dreamervla python -m pytest tests/unit_tests -q`.

**Maintainer decisions (from the scoping question):**
- Scope = **all four buckets** (safe refactors, CFG-08, X-01/RUN-01, P3 splits).
- Numerics = **flip both** (A4 → critic-value bootstrap; outcome-norm → masked_mean).
- A1/ALG-03 is **already fixed in code** (all routes use `grpo._entropy_coef`);
  only the stale docs need correcting.

**Maintainer follow-up directives (full autonomy granted; original code preserved by git):**
1. Free to modify code without per-change approval.
2. **Everything must stay runnable; `docs/experiment_tutorials` is the source of
   truth for "runnable".** Every config/code change that a tutorial references
   (esp. CFG-08 `task=` rename, X-03 token id) MUST be propagated into the tutorials.
   Verification without GPU = hydra compose resolves + unit suite green + imports OK;
   GPU e2e is the flagged gap (Task 15).
3. **All parameters must be adjustable (Hydra config, no in-code magic literals)** and
   documented in **one** parameter-reference doc → `docs/PARAMETERS.md`.
4. **Tutorials converge to a minimal step-only form** (commands, no prose). All
   explanation moves to **one** unified file → `docs/experiment_tutorials/EXPLAINED.md`.
   This is Phase 7, run AFTER CFG-08 + X-03 so tutorials are rewritten once.

---

## Phase 0 — baseline

- [ ] **0.1** Branch off `main`: `git switch -c chore/backlog-execution`.
- [ ] **0.2** Capture baseline: `conda run -n dreamervla python -m pytest
  tests/unit_tests -q` → expect `582 passed, 7 skipped`. Record the number; this is
  the green bar every later commit must hold (numerics commits update specific
  expected values, never the pass count).

---

## Phase 1 — safe behaviour-preserving batch

### Task 1: A1 / ALG-03 doc correction (already fixed in code)

**Files:**
- Modify: `docs/plans/2026-06-21-todo-backlog.md` (P0 A1 bullet)
- Modify: `docs/history/2026-06-21-rlinf-alignment-correctness-audit.md` (A1 section)
- Modify: `docs/history/2026-06-21-cleanup-execution-log.md` (ALG-03 "flagged" note)

- [ ] **1.1** Verify in code that `outcome.py`, `dense.py`, `dense_chunk.py` all read
  entropy via `grpo._entropy_coef` (actent→entropy_coef→0.0). (Confirmed: outcome.py:196.)
- [ ] **1.2** Mark A1/ALG-03 resolved in the three docs (one line each: "resolved —
  all routes route through `grpo._entropy_coef`"). Do **not** rewrite history prose.
- [ ] **1.3** Commit: `docs(backlog): mark A1/ALG-03 entropy-key resolved`.

### Task 2: CFG-05 — `_base_wmpo_outcome.yaml`

**Files:**
- Create: `configs/dreamervla/_base_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/openvla_oft_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/rynnvla_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/openvla_oft_input_token_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/rynnvla_input_token_wmpo_outcome.yaml`
- Modify: `configs/dreamervla/online_wmpo_outcome_libero_goal.yaml` (if it shares the shape)

- [ ] **2.1** Snapshot resolved configs BEFORE change for each affected config:
  `conda run -n dreamervla python -m dreamervla.train --cfg job -p
  experiment=<...>` (or the repo's existing compose-dump used by
  `test_config_validation`). Save to `/tmp/cfg_before/<name>.yaml`.
- [ ] **2.2** Extract the byte-identical shared blocks (critic / optim / eval /
  checkpoint / algorithm.update_type) into `_base_wmpo_outcome.yaml`; leave only the
  diverging keys (output dir, ckpt paths, dataset paths, model dims, policy target,
  wmpo batch knobs, the rynnvla-only `real_rollout_relabel`) in each variant via
  `defaults: [_base_wmpo_outcome]`.
- [ ] **2.3** Re-dump resolved configs to `/tmp/cfg_after/<name>.yaml`.
- [ ] **2.4** Assert byte-identical: `diff -r /tmp/cfg_before /tmp/cfg_after` → empty.
  This is the behaviour-preserving proof.
- [ ] **2.5** Run suite (config-validation tests must stay green). Commit:
  `refactor(config): CFG-05 shared _base_wmpo_outcome`.

### Task 3: DIAG-01 — opt-in `load_world_model_state_from_dict`

**Files:**
- Modify: `dreamervla/runners/online_utils.py` (split helper)
- Modify: the ~9 diagnostic loaders (route through the new helper with opts that
  reproduce each site's current behaviour) — enumerated at execution time from the
  exploration list (visualize_dreamervla_reward, analyze_rynn_hidden_action_metrics,
  measure_reward_and_drift, measure_wm_imagine_actor, measure_recon_and_action_delta,
  measure_real_vs_imagine, reward_landscape_sweep, measure_wm_imagine_fidelity,
  measure_wm_closed_loop, eval_chunkwm_closeloop).
- Test: `tests/unit_tests/test_wm_state_loader.py` (new)

- [ ] **3.1** Write failing test `test_wm_state_loader.py`: build a tiny `nn.Module`
  with a `reward_head.net.net.*` submodule; feed a state dict keyed
  `module.reward_head.net.*` + one shape-mismatched tensor; assert
  `load_world_model_state_from_dict(model, state, remap_reward_head=True,
  skip_shape_mismatch=True)` strips `module.`, remaps the reward head, skips the
  mismatch; and assert `remap_reward_head=False, skip_shape_mismatch=False` does
  neither (raises / keeps). Run → FAIL (function missing).
- [ ] **3.2** Refactor `online_utils.py`: extract the dict-cleaning core (lines
  86–112) into `load_world_model_state_from_dict(model, state, *,
  remap_reward_head=True, skip_shape_mismatch=True, reset_reward_head=False)` and make
  the existing path-based `load_world_model_state` call it. Keep its current defaults
  so the 4 existing callers are byte-identical in behaviour.
- [ ] **3.3** Run test → PASS. Run suite → green.
- [ ] **3.4** Route each diagnostic loader through the new helper, choosing opts to
  match its CURRENT behaviour exactly (sites doing plain `strict=False` →
  `remap_reward_head=False`; sites doing the reward remap → `remap_reward_head=True`).
  `eval_chunkwm_closeloop` keeps building its model but uses the helper for the load.
  This is behaviour-preserving per site; **flag in the commit any site whose current
  behaviour is ambiguous** rather than silently normalizing.
- [ ] **3.5** Run suite → green. Commit: `refactor(diagnostics): DIAG-01 route WM
  loaders through load_world_model_state_from_dict`.

### Task 4: X-03 — magic token id + model_dim → config

**Files:**
- Modify: `configs/**` (introduce `target_token_id` / `model_dim` where literal today;
  they are mostly already config values — the work is removing in-code `10004`/`4106`
  literals and threading the config value).
- Modify: code sites with literal `10004` (~32) and `4106` (config-only, 8 files).
- Test: `tests/unit_tests/test_magic_token_config.py` (new) — assert the resolved
  config carries `target_token_id` and that the default equals the historical literal
  `10004` (alignment assertion, not a bare literal pin elsewhere).

- [ ] **4.1** Enumerate the literal sites: `grep -rn '10004' dreamervla/` and
  `grep -rn '4106' configs/`. Classify each `10004` as (a) a default arg already
  overridable by config, (b) a hard literal needing a config read, (c) a comment.
- [ ] **4.2** For `4106`: it is already a config value (`model_dim: 4106` in 8 files).
  Leave the value; only act if the backlog intent is a shared anchor — fold it into
  `_base_*` if a base exists, else **no change** (record: already config). Document in
  commit.
- [ ] **4.3** For `10004`: ensure every consumer reads `target_token_id` from config
  (default `10004`), removing in-code literals. Single source for the default constant
  (e.g. `DEFAULT_ACTION_TOKEN_ID = 10004` in one module) so the 32 sites reference it,
  not a copy-pasted literal.
- [ ] **4.4** Write/extend `test_magic_token_config.py`: assert default resolves to
  `10004` and that overriding config flows to the consumer. Run → green.
- [ ] **4.5** Confirm resolved configs byte-identical (same dump/diff as Task 2) for
  any config touched. Commit: `refactor(config): X-03 single-source action token id`.

---

## Phase 2 — interface rename

### Task 5: CFG-08 — task configs → snake_case

**Files:**
- Rename: `configs/task/OpenVLA_Onetraj_*.yaml` (8) + `RynnVLA_LIBERO.yaml` → snake_case.
- Modify: ~40 `task=` refs across `scripts/`, `tests/unit_tests/`,
  `dreamervla/**` docstrings/CLI examples, `scripts/README.md`.
- Modify: test asserts in `test_coldstart_suite_configs.py`,
  `test_openvla_traj1_libero_matrix.py`, `test_config_validation.py`,
  `test_setup_scripts.py`.

- [ ] **5.1** Decide the snake_case mapping (e.g. `OpenVLA_Onetraj_LIBERO` →
  `openvla_onetraj_libero`, `RynnVLA_LIBERO` → `rynnvla_libero`). Record the full map
  in the commit body. **Flag:** this changes the user-facing `task=` token and any
  artifact/output path that embeds the config name — confirm artifact_name derivation
  still resolves and update any golden-path test that pins the old name.
- [ ] **5.2** `git mv` each file; grep-replace every `task=<Old>` reference; update
  case-arms in scripts and the test assert lists.
- [ ] **5.3** Run suite → green (the suite-config + matrix tests are the guard).
  Commit: `refactor(config): CFG-08 snake_case task configs`.

---

## Phase 3 — numerics flips (each its own `changes-numerics` commit)

### Task 6: A4 → critic-value bootstrap

**Files:**
- Modify: `dreamervla/algorithms/dreamervla.py` (replay λ-return bootstrap ~:1124–1149)
- Test: `tests/unit_tests/test_replay_lambda_returns.py` (new or extend)

- [ ] **6.1** Read `dreamervla.py:1100–1160` + `compute_replay_lambda_returns` +
  `_lambda_return_recurrence` (:402–466). Confirm `replay_target_values` is computed
  (the slow/fast critic branch) and currently discarded; `boot = raw_returns[:,0]`.
- [ ] **6.2** Write a characterization test that asserts the NEW behaviour: with a
  seeded batch, `compute_replay_lambda_returns(..., boot=replay_target_values)` differs
  from the old `boot=raw_returns[:,0]` and matches the DreamerV3 per-state critic
  bootstrap (hand-compute the expected recurrence for a 2-step toy case). Run → FAIL.
- [ ] **6.3** Wire `replay_boot = replay_target_values` (per-state critic value), not
  `raw_returns[:,0]`. Keep the slow/fast-target selection that previously fed
  `replay_target_values` — it is now live, not discarded. Remove any compute that is
  now genuinely dead.
- [ ] **6.4** Run test → PASS. Run full suite; update any imagination-numerics test
  whose expected value legitimately shifts (justify each in the commit body). Pass
  count unchanged.
- [ ] **6.5** Commit: `fix(algorithms): A4 critic-value replay bootstrap (changes
  numerics)`.

### Task 7: outcome-norm → masked_mean

**Files:**
- Modify: `dreamervla/algorithms/ppo/outcome.py` (loss normalization ~:467–548, 667)
- Test: `tests/unit_tests/test_outcome_loss_norm.py` (new)

- [ ] **7.1** Read `outcome.py:460–680`. Current: `loss_c = (...) / max(1.0,
  mask_sum_total)` with a single global denominator. RLinf `masked_mean_ratio`:
  per-rollout episode-length normalization (mean over valid steps, then mean over
  rollouts) — confirm the exact RLinf formula in the sibling repo
  (`RLinf/.../losses.py` `masked_mean` / `masked_mean_ratio`) before coding.
- [ ] **7.2** Write a test asserting masked_mean: a 2-rollout toy batch where rollout
  A finishes early and rollout B runs long; assert the new normalization up-weights A
  relative to the old global-sum denominator (hand-computed expected ratio). Run → FAIL.
- [ ] **7.3** Replace the global `mask_sum_total` denominator with the RLinf
  masked_mean_ratio reduction (PPO loss, BC loss, and the entropy denom at :667 must
  all use the consistent normalization). One canonical helper, applied to all three.
- [ ] **7.4** Run test → PASS. Full suite; update outcome-route numerics tests whose
  expected values shift (justify each). Pass count unchanged.
- [ ] **7.5** Commit: `fix(algorithms): outcome masked_mean normalization (changes
  numerics)`.

---

## Phase 4 — migration-design items

### Task 8: X-01 — unify checkpoint payload + dirs

**Files:**
- Modify: `dreamervla/runners/base_runner.py` (:769–930 save/load + :821–852 resolve)
- Modify: `dreamervla/runners/online_dreamervla.py` (:476–580 save/load)
- Test: `tests/unit_tests/test_checkpoint_roundtrip.py` (new)

- [ ] **8.1** Read both schemes fully. BaseRunner payload `{cfg, state_dicts,
  pickles}`; OnlineDreamerVLA `{env_step, update_step, cfg, state_dicts}`. Goal: one
  payload schema with a `format_version` field + a dual-read loader that still loads
  BOTH legacy schemes and the legacy `ckpt/latest.ckpt` dir. **Do not break resume of
  existing checkpoints** — that is the whole point of the migration design.
- [ ] **8.2** Write tests: (a) save→load roundtrip under the new schema restores all
  state; (b) a hand-built legacy `{cfg, state_dicts, pickles}` payload still loads;
  (c) a hand-built legacy online payload (`env_step/update_step`) still loads; (d)
  `ckpt/latest.ckpt` fallback still resolves. Run → FAIL.
- [ ] **8.3** Implement: add `format_version`; make the writer emit one schema
  (superset: include `env_step/update_step` as optional metadata); make the reader
  branch on presence of keys / version. Keep `checkpoints/` canonical, `ckpt/`
  read-only fallback (already the case). The online runner's extra fields become
  top-level optional metadata rather than a separate scheme.
- [ ] **8.4** Run tests → PASS. Full suite green (the cotrain pipeline test pins
  `checkpoint_format`; keep it working). Commit: `feat(checkpoint): X-01 unified
  payload + dual-read loader`.

### Task 9: RUN-01 — route dreamer runners through the base distributed helper

**Files:**
- Modify: `dreamervla/runners/online_dreamervla.py` (init :32–103, wrap :772–824)
- Modify: `dreamervla/runners/dreamerv3_pixel_runner.py` (:303–318)
- Modify: `dreamervla/runners/dreamerv3_token_runner.py` (:253)
- Reference: `dreamervla/runners/distributed.py` `NopretokenizeSFTDistributedHelper`

- [ ] **9.1** Read the base helper's API (init, `wrap_module_with_ddp`,
  `maybe_make_sampler`, `resolve_device`, rank props) and each runner's hand-rolled
  equivalent. **Flag genuine divergences** (e.g. online's `find_unused_parameters=True`,
  the `DVLA_DDP_TIMEOUT_SEC` env, the all-reduce flag helpers) — only unify what is
  truly equivalent; preserve real differences by parameterizing the helper.
- [ ] **9.2** Single-process guard test: assert the runners construct and run one
  training step under `world_size=1` (no DDP) identically before/after. Run.
- [ ] **9.3** Route init + wrap through the helper, parameterized to preserve each
  runner's real options. **CANNOT multi-GPU test here (no GPU / single box)** — record
  this verification gap explicitly in the commit body and the backlog. Keep changes
  minimal and DDP-semantics-preserving.
- [ ] **9.4** Run suite → green. Commit: `refactor(runners): RUN-01 use base
  distributed helper (multi-GPU test pending)`.

---

## Phase 5 — structural god-file splits (behaviour-preserving)

Each split: extract cohesive seams into sibling modules under the same package, keep
the public class/entry import path stable (re-export from the original module),
verify `python -c "import ..."` resolves and the full suite stays green. **No logic
changes** — pure moves. Commit per file.

### Task 10: split `algorithms/dreamervla.py::imagine_actor_critic_step` (819 ln)

- [ ] **10.1** Extract the 7 seams identified (init starts, imagination horizon,
  returns, actor update, critic update, replay value, metrics) into private helpers
  `_imagine_*` in the same module (or a `_imagine/` subpackage if cleaner). Keep
  `imagine_actor_critic_step` as the thin orchestrator. NB: Task 6 already touched the
  replay/return region — do this AFTER Task 6 lands.
- [ ] **10.2** Suite green. Commit: `refactor(algorithms): split imagine_actor_critic_step`.

### Task 11: split `pretokenize_dataset.py` (822 ln)

- [ ] **11.1** Extract init/manifest, windowing, tokenization, collation seams into
  sibling modules; keep `PretokenizeDataset` import path stable.
- [ ] **11.2** Suite green. Commit: `refactor(dataset): split pretokenize_dataset`.

### Task 12: split `online_dreamervla.py` (1856 ln)

- [ ] **12.1** Do AFTER Tasks 8 (X-01) and 9 (RUN-01) so the save/load + DDP code is
  already in its final form. Extract model setup, replay/env setup, train loop,
  logging/checkpoint seams into sibling modules; keep `main` thin.
- [ ] **12.2** Suite green. Commit: `refactor(runners): split online_dreamervla`.

### Task 13: split `embodied_eval_runner.py` (2514 ln)

- [ ] **13.1** Extract rollout loop, metrics/video, encoder/state munging,
  dreamer-eval seams; keep `EmbodiedEvalRunner` import path stable.
- [ ] **13.2** Suite green. Commit: `refactor(runners): split embodied_eval_runner`.

### Task 14: P3 pixel-WM loss scaffolding unification

- [ ] **14.1** Assess whether the pixel WMs can share the token-WM
  `_dreamer_v3_token_common` loss scaffolding (noted as a larger follow-on). If the
  loss math genuinely diverges, **flag and stop** — do not force-unify. Only unify the
  proven-equivalent parts. Commit if any unification lands; else record findings in
  the backlog and close the item as "assessed, diverges".

---

## Phase 6 — verification gaps

### Task 15: GPU smoke (needs a GPU box)

- [ ] **15.1** Prepare the cotrain GPU smoke command (save→resume→continue) and the
  `tests/e2e_tests` invocation. **Cannot run without a GPU** — either the maintainer
  runs them, or run on a GPU box. Record the exact commands in the backlog so the gap
  is actionable, not lost.

---

## Phase 7 — docs (param reference + tutorial slimming) — AFTER Phases 1–6

### Task 16: `docs/PARAMETERS.md` — single parameter reference

- [ ] **16.1** Enumerate the tunable Hydra surface from the config groups
  (`experiment/`, `VLA/`, `worldmodel/`, `classifier/`, `dreamervla/`, `evaluation/`,
  `task/`, `logger/`) plus the common overrides (`gpus`, `ngpu`, `batch_size`,
  `num_workers`, `num_epochs`, `out_dir`, `logger=`, `algorithm.*` RL knobs incl. the
  new `clip_ratio_c`/`clip_log_ratio`, `imag_last`, `ppo_rollouts_per_start`,
  `target_token_id`, `model_dim`). Group by subsystem; one row per param: name,
  default, what it does, valid range/effect.
- [ ] **16.2** Cross-link from `docs/experiment_tutorials/README.md` and `AGENTS.md`.
  Commit: `docs(params): single PARAMETERS.md reference`.

### Task 17: slim tutorials + unified `EXPLAINED.md`

**Files:**
- Create: `docs/experiment_tutorials/EXPLAINED.md` (all moved prose: memory/OOM,
  scheme rationale, WM sizing, logging deep-dive, validation notes).
- Modify: the 8 tutorial files → step-only (install → download → preprocess → train →
  eval as terse command blocks; a one-line link to EXPLAINED.md for context).
- Modify: `docs/experiment_tutorials/README.md` → keep the recipe table; move the
  Memory/OOM + WM-predictor + Validation prose into EXPLAINED.md.

- [ ] **17.1** Create `EXPLAINED.md` holding every explanatory section currently
  scattered across the tutorials (de-duplicated). Each topic once.
- [ ] **17.2** Rewrite each tutorial to commands-only, **with the CFG-08 snake_case
  `task=` names and any X-03 config keys already applied** (so they run as written).
- [ ] **17.3** Sanity: every command's `task=`/`experiment=`/config name resolves via
  `python -m dreamervla.train --cfg job ...` (hydra compose), and referenced configs
  exist. Commit: `docs(tutorials): slim to step-only + unified EXPLAINED.md`.

## Closeout

- [ ] Update `docs/plans/2026-06-21-todo-backlog.md`: tick every landed item, move the
  done detail to `docs/history/2026-06-21-cleanup-execution-log.md`, leave only true
  remaining gaps (e.g. GPU verification, any flagged "diverges" item).
- [ ] Final full suite run in `dreamervla` env; confirm pass count ≥ baseline.
- [ ] Use `superpowers:finishing-a-development-branch` to decide merge/PR.

## Self-review notes
- Spec coverage: P0 (A1 doc, A4, outcome-norm) ✓; P2 (X-01, DIAG-01, RUN-01, X-03) ✓;
  P3 (4 god-files + pixel-WM) ✓; P4 (CFG-05, CFG-08) ✓; verification gap ✓.
- Numerics flips isolated to one commit each with characterization tests (TDD).
- Cross-file ordering: RUN-01 → X-01 → online_dreamervla split; A4 → imagine split.
- Risk flags carried into commit bodies (DDP multi-GPU gap; CFG-08 interface change;
  any ambiguous DIAG-01 site; any pixel-WM divergence).
