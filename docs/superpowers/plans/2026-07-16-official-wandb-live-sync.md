# Official W&B Live Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove DreamerVLA's custom offline W&B uploader and make the official `wandb beta sync --live` workflow the only active documented upload path.

**Architecture:** GPU workers retain the existing W&B offline logger and canonical `${training.out_dir}/wandb` output. A networked CPU host runs W&B's native live sync directly against that shared directory; DreamerVLA owns no upload process, polling, stream parsing, marker creation, or append logic.

**Tech Stack:** Python 3.11, pytest, Markdown, W&B CLI 0.24.1+

---

### Task 1: Establish the repository contract

**Files:**
- Modify: `tests/unit_tests/test_repository_hygiene.py`

- [x] **Step 1: Write the failing hygiene test**

Add `test_offline_wandb_uses_official_live_sync()` that asserts the custom Python
launcher, shell wrapper, and launcher-specific test file do not exist. Read
`README.md`, `README.zh-CN.md`, `configs/README.md`, `scripts/README.md`, and
`docs/tutorials/experiments/EXPLAINED.md`; assert they contain
`wandb beta sync --live` and do not contain `scripts/utils/wandb_sync.sh`.

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
python -m pytest tests/unit_tests/test_repository_hygiene.py::test_offline_wandb_uses_official_live_sync -q
```

Expected: FAIL because the repository-owned launcher and wrapper still exist.

### Task 2: Remove the custom uploader and document the official command

**Files:**
- Delete: `dreamervla/launchers/wandb_sync.py`
- Delete: `scripts/utils/wandb_sync.sh`
- Delete: `tests/unit_tests/test_wandb_sync_launcher.py`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `configs/README.md`
- Modify: `scripts/README.md`
- Modify: `docs/tutorials/experiments/EXPLAINED.md`
- Modify: `dreamervla/utils/metric_logger.py`

- [x] **Step 1: Delete the repository-owned upload surface**

Remove the Python launcher, thin shell wrapper, and launcher-specific tests. Do not
change W&B initialization, stable run identity, offline resume, or artifact layout.

- [x] **Step 2: Replace active documentation commands**

Use this canonical CPU-host command after `${OUT_DIR}/wandb/offline-run-*` exists:

```bash
wandb login
wandb beta sync --live "${OUT_DIR}/wandb"
```

Document that the GPU host remains in offline mode, CPU and GPU share the run
directory, W&B 0.24.1+ is required, 0.24.0 must not be used, a clean writer exit
ends live sync, and an unclean writer crash can require stopping live sync and
running `wandb beta sync "${OUT_DIR}/wandb"`. For legacy parents containing
unrelated run IDs, document passing the exact active `offline-run-*` directory.
Remove the script-registry entry for the deleted wrapper.

- [x] **Step 3: Correct the metric logger comment**

Replace the comment that instructs users to append segments with the custom
uploader. State only that offline resume creates multiple physical segments with
one stable ID and the official W&B sync command consumes them.

- [x] **Step 4: Run the focused test to verify GREEN**

Run the Task 1 pytest command. Expected: `1 passed`.

### Task 3: Verify documentation and logger regressions

**Files:**
- Test: `tests/unit_tests/test_repository_hygiene.py`
- Test: `tests/unit_tests/test_metric_logger.py`

- [x] **Step 1: Search active sources for stale custom commands**

Run `rg` across active sources while excluding `docs/superpowers/**`. Expected: no
references to `scripts/utils/wandb_sync.sh`, `dreamervla.launchers.wandb_sync`, or
`test_wandb_sync_launcher`.

- [x] **Step 2: Run focused unit tests**

Run:

```bash
python -m pytest -q tests/unit_tests/test_repository_hygiene.py tests/unit_tests/test_metric_logger.py
```

Expected: all selected tests pass.

- [x] **Step 3: Validate the installed official CLI contract**

Run `wandb beta sync --help`. Expected: output contains `--live` and
`Sync a run while it's still being logged`.

- [x] **Step 4: Review the final diff and worktree ownership**

Run `git status --short`, `git diff --check`, and `git diff --stat`. Expected: only
task files plus pre-existing user-owned changes are present, with no whitespace
errors.
