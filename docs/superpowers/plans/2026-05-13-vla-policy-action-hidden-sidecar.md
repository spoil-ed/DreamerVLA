# VLA Policy Action Hidden Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a new pi0 action-hidden sidecar whose target exactly matches the action hidden produced by the pure VLA policy eval path.

**Architecture:** Extend `scripts/preprocess_rynn_pixel_hidden.py` with a `vla_policy` prompt style. The new mode tokenizes each HDF5 timestep as `Finish the task: ...` plus proprio state plus two-frame image history, using the same rotated image order as pure LIBERO VLA eval.

**Tech Stack:** Python, PyTorch, h5py, existing `RynnVLAEncoder` and LIBERO HDF5 datasets.

---

### Task 1: Add VLA-Policy Input Construction

**Files:**
- Modify: `scripts/preprocess_rynn_pixel_hidden.py`

- [ ] **Step 1: Add CLI controls**

Add `--prompt-style {task_only,vla_policy}`, `--history`, `--include-state`, and `--rotate-images-180` arguments. Defaults keep existing behavior: `task_only`, `history=1`, no state, no rotation.

- [ ] **Step 2: Build HDF5 state and history helpers**

Add helpers that read `obs/ee_pos`, `obs/ee_ori`, and `obs/gripper_states`, and that assemble `[prev_third, prev_wrist, curr_third, curr_wrist]` with first-frame padding.

- [ ] **Step 3: Route `vla_policy` through eval-style tokenization**

For `prompt_style=vla_policy`, call the encoder processor with `training_mode=False`, run the backbone with all `-100` labels, then extract pi0 action hidden. This matches the trace path used for pure VLA policy action hidden.

### Task 2: Preserve Metadata and Shell Entry

**Files:**
- Modify: `scripts/preprocess_rynn_pixel_hidden.py`
- Modify: `scripts/preprocess_rynn_pixel_hidden.sh`

- [ ] **Step 1: Write metadata**

Store `prompt_style`, `history`, `include_state`, and `rotate_images_180` in both `preprocess_config.json` and each sidecar HDF5 attrs.

- [ ] **Step 2: Expose env vars in shell wrapper**

Add `PROMPT_STYLE`, `HISTORY`, `INCLUDE_STATE`, and `ROTATE_IMAGES_180` env controls so the 4-GPU preprocessing job can be launched without a custom command.

### Task 3: Verify and Launch

**Files:**
- Read: `data/outputs/eval/eval_libero_vla/trace_compare_summary_20260513_133000.json`

- [ ] **Step 1: Syntax check**

Run `python -m py_compile scripts/preprocess_rynn_pixel_hidden.py`.

- [ ] **Step 2: Generate a small one-file sample**

Run the preprocessor with `MAX_FILES=1`, `MAX_DEMOS_PER_FILE=1`, `PROMPT_STYLE=vla_policy`, `HISTORY=2`, `INCLUDE_STATE=1`, and `ROTATE_IMAGES_180=1`.

- [ ] **Step 3: Compare first hidden against pure VLA trace**

Load the new sample `action_hidden_states[0]` and compare it to the pure VLA trace `action_hidden`; expected MSE should be near zero for the same file/task frame.

- [ ] **Step 4: Launch full sidecar on GPUs 4,5,6,7**

Write to a new output directory named `libero_goal_no_noops_t_256_pi0_action_hidden_vla_policy_h2`, leaving the old sidecar intact.
