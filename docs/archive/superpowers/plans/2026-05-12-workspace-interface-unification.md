# Runner Interface Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one public workspace API and consistent workspace names while preserving old Hydra targets as compatibility aliases.

**2026-05-13 update:** The large-module workspace direction was reverted.
Public configs again target route-specific workspace classes directly.

**Architecture:** Add a `dreamer_vla.runners` public package with unified class names and metadata. Keep existing implementation files in place, because they contain large training loops and active uncommitted work. Update active configs and docs to target the public classes; keep old implementation class names runnable for archive configs.

**Tech Stack:** Python namespace package, Hydra `_target_`, pytest.

---

### Task 1: Public Runner API

**Files:**
- Create: `dreamer_vla/runners/__init__.py`
- Modify: `dreamer_vla/runners/base_runner.py`
- Test: `tests/test_runner_public_api.py`

- [ ] **Step 1: Write public API tests**

Test that `dreamer_vla.runners` exports the canonical workspace names and every class has the lifecycle methods `setup`, `execute`, `run`, and `teardown`.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/test_runner_public_api.py -q
```

Expected: fails because `dreamer_vla.runners` has no public canonical API yet.

- [ ] **Step 3: Implement canonical workspace exports**

Add route-specific public classes:

```text
ActionHiddenWMRunner
PooledHiddenWMRunner
PixelWMRunner
TokenWMRunner
PretokenizedWMRunner
VLASFTRunner
JointDreamerVLARunner
LiberoEvalRunner
ChameleonLatentWMRunner
SemanticBottleneckWMRunner
```

- [ ] **Step 4: Add lifecycle methods to `BaseRunner`**

Expose `setup`, `execute`, and `teardown` on every workspace through inheritance. Existing `run` methods remain the execution body.

### Task 2: Config Target Unification

**Files:**
- Modify active root configs under `configs/*.yaml`
- Do not modify `configs/archive/libero10_legacy/*.yaml` in this pass

- [ ] **Step 1: Update active config `_target_` values**

Point active configs to public `dreamer_vla.runners.<CanonicalName>` targets.

- [ ] **Step 2: Verify Hydra composes representative configs**

Compose the current mainline action-hidden config plus representative secondary configs.

### Task 3: Documentation And Script Naming

**Files:**
- Modify: `README.md`
- Modify: `configs/README.md`
- Modify: `scripts/README.md`
- Modify: `docs/repository_structure.md`
- Modify: `docs/wm_training_routes.md`

- [ ] **Step 1: Replace old workspace names in docs**

Use canonical workspace names in current docs while mentioning old classes only as compatibility aliases.

- [ ] **Step 2: Verify tests**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests -q
```

Expected: all project tests pass.
