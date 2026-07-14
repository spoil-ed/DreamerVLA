# Failure-Conditioned Imagined RL Implementation Plan

**Goal:** Collect real failures, repeatedly initialize frozen-WM imagination from
failed episode starts, and update only the actor through imagined PPO.

**Architecture:** Keep the existing Ray Actor/Rollout/Env/Replay group boundaries.
Add an explicit replay initial-condition selector, append current real trajectories
to bounded historical replay, and select a dedicated cotrain step implementation by
Hydra `training_mode`. LearnerGroup stays resident for checkpoint/state ownership but
does not train in this mode.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, Ray worker groups, PyTorch/FSDP, pytest.

---

### Task 1: Define replay failure-anchor behavior

- [x] Add failing tests in `tests/unit_tests/test_online_replay_task_balanced.py` and
  `tests/unit_tests/test_step_local_wm_classifier_update.py`.
- [x] Add explicit outcome override, append-real-batch API, eligible-count API, and
  `failed_episode_start` selection in `dreamervla/runtime/online_replay.py` and
  `dreamervla/workers/replay/replay_worker.py`.
- [x] Run focused replay tests.

### Task 2: Forward selector through WM bootstrap

- [x] Add failing selector/group tests in `tests/unit_tests/test_wm_env_bootstrap.py`.
- [x] Forward `env.wm.cfg.initial_condition_selector` from
  `dreamervla/workers/env/trajectory_env_worker.py`.
- [x] Run WM bootstrap tests.

### Task 3: Add imagined-only cotrain step

- [x] Add failing stage-order and no-failure tests in
  `tests/unit_tests/test_cotrain_stage_order.py`.
- [x] Select `failure_imagined_rl` in `dreamervla/runners/cotrain_runner.py`; append
  real trajectories, skip encoder/re-encode/learner phases, refresh failure anchors,
  and execute imagined PPO only when failures exist.
- [x] Update validation in `dreamervla/config.py`.
- [x] Run cotrain stage and config tests.

### Task 4: Activate recipe and verify

- [x] Update `configs/dreamervla/openvla_onetraj_libero_cotrain.yaml` and
  `configs/dreamervla/wmcls_cotrain.yaml` with explicit mode/selector/frozen flags.
- [x] Run focused unit tests, Hydra composition validation, and compile checks.
- [x] Review the diff, commit with a signed Conventional Commit, and provide the
  existing cotrain launch command.
