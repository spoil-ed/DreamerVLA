# OSMesa Real Rollout Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 25 one-slot OSMesa real environment workers through eight shared GPU rollout workers without changing imagined-rollout parallelism.

**Architecture:** Real environment requests use one shared FIFO request key, allowing eight rollout workers to consume work dynamically, while responses remain keyed by the originating environment rank. A dedicated `real_envs_per_worker` configuration separates CPU real-rollout geometry from WM batch geometry, and stage-local WM ranks preserve the existing rank-keyed imagined route.

**Tech Stack:** Python 3.11, Ray actors/channels, Hydra/OmegaConf, pytest

---

### Task 1: Specify configuration and placement geometry

**Files:**
- Modify: `tests/unit_tests/test_cotrain_placement.py`
- Modify: `tests/unit_tests/test_cotrain_config_validation.py`
- Modify: `tests/unit_tests/test_dreamer_mainline.py`
- Modify: `dreamervla/workers/cotrain/placement.py`
- Modify: `dreamervla/config.py`
- Modify: `configs/dreamervla/wmcls_cotrain.yaml`

- [ ] Add failing tests asserting 25 CPU real workers, seven unchanged WM workers,
  a one-slot Dreamer real-worker configuration, and `[2] * 7 + [1] * 18` epoch
  distribution for a target of 32 trajectories.
- [ ] Run the focused tests and confirm they fail because placement is capped at
  one and `real_envs_per_worker` is not implemented.
- [ ] Remove the default placement cap, add and validate
  `real_envs_per_worker`, and set the Dreamer recipe to 25 workers by one slot.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Add shared request and rank-scoped response routing

**Files:**
- Modify: `tests/unit_tests/test_multistep_rollout_worker.py`
- Modify: `tests/unit_tests/test_trajectory_env_worker.py`
- Modify: `dreamervla/workers/rollout/multistep_rollout_worker.py`
- Modify: `dreamervla/workers/env/trajectory_env_worker.py`

- [ ] Add a failing rollout-worker test where rollout rank zero consumes a batch
  with `env_rank=24` from a shared key and returns the result on key `24`.
- [ ] Add a failing real-env-worker test proving requests use the configured
  shared key while responses are still read from the worker's rank key.
- [ ] Run both tests and confirm the old rank equality/request-key assumptions fail.
- [ ] Let the batch generator accept an explicit shared input key, validate batch
  internal rank consistency, and preserve the original rank in its response.
- [ ] Add an optional request key to trajectory env workers and use it only for
  outbound observation batches.
- [ ] Re-run both focused test modules and confirm they pass.

### Task 3: Wire Dreamer phase orchestration and progress accounting

**Files:**
- Modify: `tests/unit_tests/test_cotrain_stage_order.py`
- Modify: `tests/unit_tests/test_dreamer_mainline.py`
- Modify: `dreamervla/runners/dreamer_runner.py`

- [ ] Add failing tests asserting that Dreamer real generation and all eight stop
  messages use the shared real request key, while WM stop messages stay rank-keyed.
- [ ] Add a failing progress-budget test using the real-specific slot and max-step
  values.
- [ ] Run the tests and confirm current orchestration uses legacy rank keys and
  generic real geometry.
- [ ] Launch real env workers with the shared request key, start rollout generation
  with `real_envs_per_worker`, stop it through the shared key, reset WM rank offset
  to zero, and use real-specific geometry in progress totals.
- [ ] Re-run the stage and Dreamer tests and confirm they pass.

### Task 4: Regression verification and delivery

**Files:**
- Verify only; no additional production files are planned.

- [ ] Run focused placement, configuration, worker-routing, runner-stage, and
  Dreamer-mainline tests.
- [ ] Run the broader unit-test subset covering cotrain resume, channels,
  environment workers, rollout workers, and checkpoint configuration.
- [ ] Compile all modified Python files and resolve the Dreamer Hydra config.
- [ ] Review `git diff` and `git status`, ensuring pre-existing user changes are
  neither overwritten nor staged.
- [ ] Commit only this feature's files with a signed Conventional Commit and push
  the current branch to its configured upstream.
