# Algorithm Content Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden and accurately document every public DreamerVLA algorithm family without changing accepted valid-path numerics or the frozen imagined-RL mainline.

**Architecture:** Put reusable hyperparameter contracts in `dreamervla.algorithms.validation`, call them from Hydra validation and direct APIs, and convert benchmark-changing fallbacks to explicit failures. Consolidate the public explanation in `docs/algorithms.md` and reconcile specifications with effective behavior.

**Tech Stack:** Python 3.11, PyTorch, OmegaConf/Hydra, pytest, Ruff.

## Global Constraints

- Keep `failure_imagined_rl` frozen-WM/classifier and imagined-only.
- Keep `manual_cotrain.initial_condition_selector=failed_episode_start`.
- Do not change valid default hyperparameters or valid-path equations.
- Do not edit or stage `third_party/`.
- New behavior requires a failing regression test before production code.

---

### Task 1: Algorithm hyperparameter contracts

**Files:**
- Create: `dreamervla/algorithms/validation.py`
- Modify: `dreamervla/config.py`
- Test: `tests/unit_tests/test_algorithm_hyperparameter_validation.py`

**Interfaces:**
- Produces: `validate_ppo_hyperparameters(config, *, prefix) -> None` and `validate_tdmpc_hyperparameters(config, *, prefix) -> None`.
- Consumes: mapping-like resolved sections without selecting defaults.

- [ ] Write parametrized failing tests for probability ranges, positive integer geometry, clipping relationships, reward bounds, and classifier thresholds.
- [ ] Run the focused tests and confirm they fail because the validators do not exist.
- [ ] Implement pure validation and invoke it for `algorithm`, `actor.train_cfg.algorithm_cfg`, `learner.train_cfg.algorithm_cfg`, and TD-MPC config blocks.
- [ ] Re-run focused tests and all existing config validation tests.

### Task 2: Numerical API boundary hardening

**Files:**
- Modify: `dreamervla/algorithms/ppo/grpo.py`
- Modify: `dreamervla/algorithms/reward/{sparse_outcome,probability_outcome}.py`
- Modify: `dreamervla/algorithms/critic/twohot_critic.py`
- Modify: `dreamervla/algorithms/tdmpc_mpc.py`
- Test: `tests/unit_tests/test_algorithm_numerical_contracts.py`

**Interfaces:**
- Preserves: existing valid-input return values and tensor layouts.
- Adds: `ValueError` for empty/non-finite/malformed geometry before PyTorch emits NaNs or opaque scatter/top-k errors.

- [ ] Write failing tests for empty GRPO scores, non-positive eps, invalid PPO clipping, malformed reward tensors, invalid two-hot bins/percentiles, and invalid TD-MPC CEM geometry.
- [ ] Run tests and confirm each new contract fails for the intended missing guard.
- [ ] Add the minimum guards without changing the numerical formulas.
- [ ] Run new tests plus existing golden, microbatch, reward, critic, and TD-MPC tests.

### Task 3: Fail-fast model and evaluation paths

**Files:**
- Modify: `dreamervla/runtime/world_model_training_base.py`
- Modify: `dreamervla/runners/libero_vla_evaluation_runner.py`
- Test: `tests/unit_tests/test_algorithm_runtime_fail_fast.py`

**Interfaces:**
- Spatial-codec setup raises `RuntimeError` with the original exception chained when mapping attachment fails.
- Evaluation falls back only when `generate_action_head` is absent; an implemented but failing action head raises `RuntimeError`.

- [ ] Write failing tests for missing/failed image-token mapping and a raising action-head method.
- [ ] Confirm current code swallows the failures.
- [ ] Implement contextual fail-fast behavior while retaining capability-based absence fallback.
- [ ] Run the focused runtime and evaluation test modules.

### Task 4: Canonical algorithm documentation

**Files:**
- Create: `docs/algorithms.md`
- Modify: `docs/README.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `spec/02_naming.md`
- Modify: `spec/04_complete_loop.md`
- Test: `tests/unit_tests/test_algorithm_documentation.py`

**Interfaces:**
- `docs/algorithms.md` is the public route/equation/status reference.
- `spec/04_complete_loop.md` remains authoritative for mainline ordering and transient replay semantics.

- [ ] Write a failing documentation-contract test for route coverage, canonical package paths, selector semantics, and replay-not-checkpointed wording.
- [ ] Add the reference and reconcile contradictory/stale text.
- [ ] Run documentation and repository-hygiene tests.

### Task 5: Full verification and handoff

**Files:**
- Modify: `.planning/2026-07-18-algorithm-content-polish/{task_plan,findings,progress}.md` (ignored working records only)

- [ ] Run all focused algorithm/config/runtime tests.
- [ ] Compose, resolve, and validate all experiment recipes.
- [ ] Run the full unit suite.
- [ ] Run Ruff check, Ruff format check, shell syntax checks, and `git diff --check`.
- [ ] Review the complete diff against this design, commit with sign-off, and report exact evidence and any deliberately excluded scope.
