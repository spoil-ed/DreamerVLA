# Runner Setup Failure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee cleanup after a constructed Runner fails during `setup()` without masking the setup exception or introducing a distributed barrier hang.

**Architecture:** `dreamervla.train.run` owns lifecycle orchestration and calls a dedicated `teardown_after_setup_failure()` hook only when setup raises. `BaseRunner` provides a safe default that tears down local resources and then destroys an initialized distributed group; `SuccessClassifierTrainingRunner` overrides it to omit its normal barrier.

**Tech Stack:** Python 3.11, Hydra, pytest, Ruff.

---

### Task 1: Specify setup-failure lifecycle behavior

**Files:**
- Create: `tests/unit_tests/test_train_lifecycle.py`

- [x] Write a test whose dummy Runner raises from `setup()`, records cleanup-hook use, and proves `execute()` is not called.
- [x] Write a test whose cleanup hook also raises and prove the original setup exception remains the propagated exception.
- [x] Run the two tests and verify they fail against the current entrypoint.

### Task 2: Add partial-setup cleanup hooks

**Files:**
- Modify: `dreamervla/train.py`
- Modify: `dreamervla/runners/base_runner.py`
- Modify: `dreamervla/runners/success_classifier_training_runner.py`
- Test: `tests/unit_tests/test_train_lifecycle.py`

- [x] Wrap setup in lifecycle error handling and call `teardown_after_setup_failure()` before re-raising.
- [x] Add the BaseRunner hook using `try/finally` so distributed cleanup still runs if local teardown fails.
- [x] Add the classifier override that invokes `BaseRunner.teardown()` and distributed cleanup without a barrier.
- [x] Test BaseRunner distributed cleanup and classifier no-barrier cleanup directly.
- [x] Run lifecycle tests and verify they pass.

### Task 3: Verify the repository

- [x] Run the complete unit-test suite.
- [x] Run Ruff 0.15.14 over `dreamervla` and `tests`.
- [x] Run `git diff --check` and inspect the lifecycle diff.
