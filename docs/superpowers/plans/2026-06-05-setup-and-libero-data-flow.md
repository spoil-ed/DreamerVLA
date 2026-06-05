# Setup And LIBERO Data Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the verbose setup flow with stable shell entry points and a LIBERO preprocessing pipeline that marks no-op steps first, then filters by option while preserving current training paths.

**Architecture:** Shell launchers share one `scripts/common_env.sh` for project roots, rendering, Python, and LIBERO config. Installation, downloads, and LIBERO preprocessing are separate stable scripts. Python preprocessing exposes small, testable helpers for no-op masks and HDF5 filtering, while the top-level data script composes the existing replay, reward, image-tree, pretokenize, and sidecar stages.

**Tech Stack:** Bash, Hydra launch wrappers, Python 3.11, h5py, pytest, LIBERO/robosuite.

---

### Task 1: No-Op Marking Helpers

**Files:**
- Modify: `dreamer_vla/preprocess/libero_utils/regenerate_libero_dataset_filter_no_op.py`
- Test: `tests/unit_tests/test_libero_noop_marking.py`

- [ ] Add tests for `compute_noop_mask` and HDF5 demo filtering.
- [ ] Run the new test and verify it fails before implementation.
- [ ] Add pure helpers that can be tested without importing LIBERO simulator code.
- [ ] Run the new test and verify it passes.

### Task 2: Shared Shell Environment

**Files:**
- Create: `scripts/common_env.sh`
- Modify: `scripts/train_vla.sh`
- Modify: `scripts/train_wm.sh`
- Modify: `scripts/train_dreamervla.sh`
- Modify: `scripts/eval_libero_vla.sh`

- [ ] Add a test that formal shell scripts source `scripts/common_env.sh`.
- [ ] Run the test and verify it fails before implementation.
- [ ] Implement `common_env.sh` with `DVLA_ROOT`, `PROJECT_ROOT`, `LIBERO_CONFIG_PATH`, `PYTHON`, `MUJOCO_GL`, tokenizer, CUDA allocator, and LIBERO config creation.
- [ ] Source it from formal training and eval wrappers.
- [ ] Run the test and shell syntax checks.

### Task 3: Install And Download Scripts

**Files:**
- Create: `scripts/install_env.sh`
- Create: `scripts/download_assets.sh`
- Test: `tests/unit_tests/test_setup_scripts.py`

- [ ] Add tests for documented script existence and key commands.
- [ ] Run the test and verify it fails before implementation.
- [ ] Implement install flow: apt tools, conda env, uv, torch, requirements, flash-attn wheel, egl_probe, third_party clone/install, LIBERO config/fix, verification.
- [ ] Implement download flow: Hugging Face weights, LIBERO datasets, CALVIN datasets.
- [ ] Run tests and shell syntax checks.

### Task 4: Formal LIBERO Data Script

**Files:**
- Create: `scripts/preprocess/prepare_libero_data.sh`
- Modify: `scripts/preprocess/process_all_libero_data.sh`
- Test: `tests/unit_tests/test_setup_scripts.py`

- [ ] Add tests for `TASK` defaulting and `HIS=1`, `ACTION_HORIZON=1` defaults.
- [ ] Run the test and verify it fails before implementation.
- [ ] Implement the one-command LIBERO flow with `TASK`, `FILTER_NOOPS=1`, `RUN_PRETOKENIZE=1`, `RUN_REWARD=1`, and `RUN_ACTION_HIDDEN=1`.
- [ ] Keep current final paths: `data/processed_data/${TASK}_no_noops_t_256`, reward dir, and legacy action-hidden dir.
- [ ] Run tests and shell syntax checks.

### Task 5: Documentation

**Files:**
- Modify: `SETUP.md`
- Modify: `README.md`
- Modify: `scripts/README.md`

- [ ] Replace verbose setup prose with short script-oriented setup.
- [ ] Move config details to training/eval examples only.
- [ ] Document that formal VLA data defaults to `his=1`, `len_action=1`; action-hidden sidecar keeps `history=2`.
- [ ] Run repository hygiene tests.
