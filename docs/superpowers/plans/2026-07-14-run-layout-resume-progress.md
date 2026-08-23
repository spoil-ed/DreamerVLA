# Unified Run Layout, Resume, and Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use an execution workflow and TDD to implement this plan task-by-task.

**Goal:** Give every active experiment one shallow Hydra-owned run root, resume every trainable route in place, and report cotrain real/eval progress by completed trajectories.

**Architecture:** Add a shared Hydra `run` group that owns the output base, experiment name, and timestamp; keep `training.out_dir` as the exact resolved run root. Centralize resume-path discovery in a small utility used by `BaseRunner` and the public launchers. Keep canonical writes below `checkpoints/` while retaining legacy `ckpt/` reads. Make cotrain progress helpers aggregate completed trajectories across workers and treat action chunks as status only.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch, Ray worker callbacks, pytest, Bash.

---

### Task 1: Pin run-root and artifact-layout contracts

**Files:**
- Create: `tests/unit_tests/test_run_paths.py`
- Modify: `tests/unit_tests/test_runner_artifacts.py`
- Modify: `tests/unit_tests/test_runner_public_api.py`

- [ ] Add failing tests for run-root inference from a run directory, canonical checkpoint file/directory, and legacy `ckpt/` path.
- [ ] Add failing tests asserting `BaseRunner` uses sibling `checkpoints/`, `wandb/`, `tensorboard/`, `logs/`, `video/`, `diagnostics/` directories.
- [ ] Add composition tests proving active recipes resolve to `<output-base>/<run.name>/<timestamp>` without repeated route folders.
- [ ] Run the focused tests and verify they fail for the missing helper and current nested paths.

### Task 2: Implement the shared Hydra run layout

**Files:**
- Create: `configs/run/default.yaml`
- Modify: `configs/train.yaml`
- Modify: active files under `configs/experiment/`, `configs/dreamervla/`, `configs/worldmodel/`, `configs/classifier/`, and `configs/evaluation/`
- Modify: `configs/logger/*.yaml`
- Modify: `dreamervla/runners/base_runner.py`

- [ ] Add the `run` config group with output-base, static recipe name, timestamp, and common resume fields.
- [ ] Give each public leaf experiment a stable `run.name`; remove scattered `training.out_dir` nesting from active recipes.
- [ ] Flatten BaseRunner logger/artifact helpers while preserving the exact run root for Hydra `.hydra` output.
- [ ] Run the Task 1 tests until green, then run all config composition tests.

### Task 3: Pin and implement the common resume adapter

**Files:**
- Create: `dreamervla/utils/run_paths.py`
- Modify: `dreamervla/launchers/train.py`
- Modify: `dreamervla/launchers/cotrain.py`
- Modify: `dreamervla/runners/base_runner.py`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`
- Modify: `tests/unit_tests/test_cotrain_launcher.py`
- Test: `tests/unit_tests/test_run_paths.py`

- [ ] Add failing launcher tests for `--resume PATH`, exact original-run reuse, missing paths, file/directory inference, and conflicting output/resume overrides.
- [ ] Implement canonical run-root discovery and latest-checkpoint selection with legacy read compatibility.
- [ ] Translate the friendly option into `training.resume=true`, `training.resume_dir`, `training.resume_path`, and the original `training.out_dir`.
- [ ] Make BaseRunner honor `resume_path` first and derive its output directory from `resume_dir` when resuming directly through Hydra.
- [ ] Run launcher, run-path, and BaseRunner tests until green.

### Task 4: Resume every trainable runner and canonicalize checkpoint writes

**Files:**
- Modify: `dreamervla/runners/dino_token_world_model_training_runner.py`
- Modify: `dreamervla/runners/world_model_training_runner.py`
- Modify: `dreamervla/runners/success_classifier_training_runner.py`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `dreamervla/runners/rollout_collection_runner.py`
- Modify: corresponding focused runner tests under `tests/unit_tests/`

- [ ] Add failing round-trip tests for DINO epoch/global-step/optimizer state, Dreamer warmup progress, classifier best-metric state, and cotrain consolidated state.
- [ ] Move all new WM/classifier aliases and warmup progress/top-k writes to `checkpoints/`; search canonical paths before legacy `ckpt/` paths when loading.
- [ ] Extend classifier resume keys to preserve loop/best-checkpoint state.
- [ ] Map cotrain common resume fields to the consolidated checkpoint, write one `global_step_<N>/` checkpoint level plus a canonical latest pointer, and retain old manual-step reads.
- [ ] Make collection reuse its original run root and persisted shard/manifest completion set.
- [ ] Run focused runner/checkpoint tests until green.

### Task 5: Report cotrain real/eval progress by total trajectories

**Files:**
- Modify: `tests/unit_tests/test_cotrain_phase_progress.py`
- Modify: `tests/unit_tests/test_cotrain_stage_order.py`
- Modify: `dreamervla/runners/cotrain_runner.py`

- [ ] Add failing tests proving callback progress uses aggregate `completed / configured trajectory target` for both real rollout and evaluation.
- [ ] Keep successes, success rate, and chunks in the status text; never use chunks or worker callback ordinals as the bar numerator.
- [ ] Aggregate completed/success/chunk totals across all participating workers before rendering.
- [ ] Run both cotrain progress suites until green.

### Task 6: Document, verify, and commit

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/data_layout.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `configs/README.md`
- Modify: relevant route/tutorial docs and repository contract tests

- [ ] Document the canonical run tree, `--resume` forms, legacy read-only compatibility, and trajectory-based progress semantics.
- [ ] Run targeted unit suites, all unit tests, shell syntax checks, Ruff on changed Python files, and `git diff --check` in the `dreamervla` environment.
- [ ] Inspect the final diff and ensure untracked workspace planning notes are not staged.
- [ ] Commit with `git commit -s -m "feat: unify run resume and progress"`.
