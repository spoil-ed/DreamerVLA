# Resident Eval Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate every aggressive Dreamer step and log trajectory-level WM cosine plus CLS F1/accuracy from resident physical eval trajectories.

**Architecture:** Evaluation Env retains but never replays the current eval episodes. Actor re-encodes raw frames with the resident VLA, Learner runs the existing read-only cotrain WM/CLS diagnostic, and Runner merges its scalar results into the same eval log step.

**Tech Stack:** Python 3.11, PyTorch, Ray worker groups, Hydra/OmegaConf, pytest, Docker BuildKit.

## Global Constraints

- Hydra is the source of truth; original experiment recipes remain unchanged.
- Metrics go through `BaseRunner.log_metrics` under `eval/`.
- Evaluation must not write replay, update optimizers, or recalibrate the classifier threshold.
- New behavior requires focused unit tests before implementation.
- Docker uses the repository's existing `docker/Dockerfile` and public tag scheme.

---

### Task 1: Read-only learner trajectory diagnostics

**Files:**
- Modify: `dreamervla/runtime/cotrain_eval.py`
- Modify: `dreamervla/workers/actor/learner_worker.py`
- Test: `tests/unit_tests/test_cotrain_transaction_eval.py`

**Interfaces:**
- Consumes: `RealTrajectoryBatch` with every transition containing `obs_embedding`, `action`, and `proprio`, plus trajectory language sidecars.
- Produces: `LearnerWorker.evaluate_cotrain_trajectories(batch: RealTrajectoryBatch) -> dict[str, Any]`.

- [ ] Add a failing test that builds two encoded `RealTrajectory` values and asserts `eval/wm_closed_loop_cosine`, `eval/classifier_real_f1`, and `eval/classifier_real_accuracy` are returned.
- [ ] Run `pytest -q tests/unit_tests/test_cotrain_transaction_eval.py` and verify failure because the learner evaluation interface is absent.
- [ ] Add a strict conversion helper and the no-gradient learner method, preserving/restoring module train modes and using the checkpoint threshold.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Resident eval orchestration

**Files:**
- Modify: `dreamervla/workers/env/trajectory_env_worker.py`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Test: `tests/unit_tests/test_cotrain_resident_eval.py`
- Test: `tests/unit_tests/test_step_local_real_batch.py`

**Interfaces:**
- Consumes: `EvaluationEnvGroup.drain_real_trajectories(global_step)`, Actor re-encoding, and Learner diagnostics from Task 1.
- Produces: resident eval metrics merged with policy success metrics at the current global step.

- [ ] Add failing tests proving eval trajectories are retained without replay writes and that runner calls drain -> Actor reencode -> Learner evaluate.
- [ ] Run the two focused test files and verify expected failures.
- [ ] Retain `eval_env` completed trajectories, drain them exactly once, and add runner orchestration after rollout completion.
- [ ] Re-run both focused test files and verify they pass.

### Task 3: Aggressive Hydra recipe

**Files:**
- Modify: `configs/experiment/openvla_libero_aggressive.yaml`
- Test: `tests/unit_tests/test_aggressive_dreamer_config.py`

**Interfaces:**
- Produces: `eval_interval_global_steps=1`, actor LR `2e-6`, two PPO epochs, `algorithm.kl_beta=0.005`, and `max_policy_kl=0.05` only for the isolated experiment.

- [ ] Strengthen the config assertions first and run the test to observe the old values fail.
- [ ] Update the aggressive recipe with the exact values above.
- [ ] Re-run the config test and verify the original recipe composition remains unchanged.

### Task 4: Verification, Git, and Docker publication

**Files:**
- Modify: `README.md` only if the aggressive recipe description is stale.

**Interfaces:**
- Consumes: all prior task changes.
- Produces: signed Git commit, pushed branch, and published Docker Hub tags resolving to the new image digest.

- [ ] Run focused tests for runtime diagnostics, resident eval, env drain, and aggressive composition.
- [ ] Run the related cotrain/config regression set and Ruff on changed Python files.
- [ ] Inspect `git diff --check`, `git diff`, and `git status` for scope and secret safety.
- [ ] Commit with `git commit -s -m "feat: add resident wm classifier eval metrics"` and push the current branch to `origin`.
- [ ] Build `docker/Dockerfile` with the committed SHA and push `spoil/dreamervla:sha-<12>`, `:cu124-h100-v1`, and `:v1`.
- [ ] Run `docker buildx imagetools inspect` for all three tags and verify they resolve to the pushed digest.

