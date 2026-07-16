# Safe Quality Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the existing Ruff CI gate and correct demonstrably stale documentation without changing training behavior.

**Architecture:** Preserve the vendored RLinf vector-env implementation and exempt only its upstream modernization rules while retaining correctness/import checks. Apply mechanical fixes to owned source and tests, then align active documentation with files that actually exist.

**Tech Stack:** Python 3.11, Ruff 0.15.14, pytest, Markdown.

---

### Task 1: Preserve the replay-free resume contract

**Files:**
- Modify: `spec/99_manual_notes.md`
- Modify: `spec/00_overview.md`
- Modify: `spec/02_naming.md`
- Modify: `spec/05_ray_runtime.md`
- Modify: `dreamervla/runtime/online_replay.py`
- Modify: `dreamervla/workers/replay/replay_worker.py`
- Test: `tests/unit_tests/test_spec_docs.py`

- [x] Add a failing documentation-contract test.
- [x] Document global-step-boundary, replay-free cotrain resume.
- [x] Run replay and resume tests; expect all selected tests to pass.

### Task 2: Restore the Ruff CI gate

**Files:**
- Modify: `.pre-commit-config.yaml`
- Modify: `pyproject.toml`
- Modify: `dreamervla/envs/libero/libero_env.py`
- Modify: `dreamervla/envs/libero/venv.py`
- Modify: Ruff-reported owned source and unit-test import lists only

- [x] Update pre-commit Ruff to the CI-pinned `v0.15.14`.
- [x] Replace the stale deleted-file exclusion with narrow per-file ignores for the vendored RLinf modernization rules.
- [x] Consolidate late module imports without changing definitions or control flow.
- [x] Apply Ruff's safe fixes to the remaining import and unused-import findings.
- [x] Run `ruff check dreamervla tests`; expect `All checks passed!`.
- [x] Run LIBERO env, replay, and affected unit tests; expect all selected tests to pass.

### Task 3: Correct stale documentation indexes

**Files:**
- Modify: `tests/e2e_tests/README.md`
- Modify: `docs/README.md`

- [x] Replace the obsolete empty-E2E statement with the current flat opt-in test layout.
- [x] Remove index links to absent `docs/debug_log.md` and `docs/reports/`.
- [x] Verify every remaining direct link in `docs/README.md` exists.

### Task 4: Final verification

- [x] Run focused pytest suites for all touched behavior.
- [x] Run Ruff 0.15.14 over `dreamervla` and `tests`.
- [x] Run `git diff --check`.
- [x] Inspect the final diff and confirm unrelated W&B changes remain untouched.

### Task 5: Align active route documentation

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `spec/01_goal.md`
- Modify: `spec/06_routes.md`
- Modify: `spec/README.md`
- Modify: `docs/reference/routes.md`
- Modify: `docs/tutorials/experiments/OpenVLA_Onetraj_LIBERO.md`
- Test: `tests/unit_tests/test_spec_docs.py`

- [x] Assert active docs map `openvla_libero` to `DreamerRunner`.
- [x] Document `openvla_onetraj_libero_cotrain` as the supporting full staged `CotrainRunner` route.
- [x] Remove active references to the deleted `frozen_model_pre_mainline` launcher.
- [x] Run documentation and repository-hygiene tests.
