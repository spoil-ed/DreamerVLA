# Manual Cotrain Periodic Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make periodic manual-cotrain segments resume through Hydra for both WM/CLS and frozen-model evaluation recipes.

**Architecture:** Keep resume state launcher-injected rather than widening recipe schemas. Normalize override keys for replacement and emit the optional checkpoint with Hydra `++` force-add syntax.

**Tech Stack:** Python 3.11, Hydra, OmegaConf, pytest.

---

### Task 1: Resume override composition regression

**Files:**
- Modify: `tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py`
- Modify: `dreamervla/launchers/manual_cotrain_vla_eval.py`

- [x] **Step 1: Write the failing composition test**

Add a parametrized test for `dreamervla_wmcls_cotrain_ray_eval` and `dreamervla_frozen_models_rl_ray_eval`. Build each production launch, construct a resumed segment, compose it with `_compose_training_config()`, and assert the resolved resume path equals the checkpoint. Also require the generated argument to begin with `++manual_cotrain.resume_ckpt=`.

- [x] **Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py::test_periodic_eval_resume_segment_composes_for_supported_recipe_schemas -q
```

Expected: the WM/CLS case fails with `ConfigCompositionException: Could not override 'manual_cotrain.resume_ckpt'`.

- [x] **Step 3: Implement normalized replacement and force-add emission**

Change `_override_key()` so it accepts either a complete override or a bare mapping key, strips Hydra `+`/`~` prefixes, and returns the normalized key. Change `_replace_overrides()` to normalize its mapping keys before filtering the existing command. Emit the resume entry as:

```python
"++manual_cotrain.resume_ckpt": _hydra_string(resume_ckpt)
```

- [x] **Step 4: Run the focused test to verify GREEN**

Run the Step 2 command. Expected: `2 passed`.

- [x] **Step 5: Run the complete launcher test module**

Run:

```bash
PYTHONPATH=. pytest tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py -q
```

Expected: all tests pass.

- [x] **Step 6: Run related launcher/config tests**

Run:

```bash
PYTHONPATH=. pytest tests/unit_tests/test_manual_cotrain_async_launcher.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q
```

Expected: all tests pass.

- [x] **Step 7: Inspect the final diff**

Run `git diff --check` and `git diff -- dreamervla/launchers/manual_cotrain_vla_eval.py tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py`. Expected: no whitespace errors and only the intended key normalization, force-add emission, and regression test changes.
