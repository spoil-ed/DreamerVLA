# Dreamer Frozen Imagined-RL Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route `openvla_libero` through a dedicated `DreamerRunner` that disables encoder/WM/CLS updates, keeps the frozen LearnerGroup off GPU, and avoids PPO activation OOM with micro-batch 8.

**Architecture:** Preserve `dreamervla/runners/cotrain_runner.py` unchanged. Rename the copied runner class in `dreamervla/runners/dreamer_runner.py` to `DreamerRunner`, retain its failure-imagined-RL control flow, and specialize its placement so the checkpoint-owning but non-updating LearnerGroup runs on CPU. Select the runner and memory-safe micro-batch through Hydra.

**Tech Stack:** Python 3.11, PyTorch, Ray, Hydra/OmegaConf, pytest, Ruff

---

## File map

- `dreamervla/runners/dreamer_runner.py`: dedicated frozen imagined-RL runner; copied implementation is retained while the public class is renamed and its Learner placement is specialized.
- `dreamervla/runners/__init__.py`: lazy public export for `DreamerRunner`; the existing `CotrainRunner` export remains unchanged.
- `configs/dreamervla/wmcls_cotrain.yaml`: select `DreamerRunner` and override the active PPO micro-batch to 8.
- `tests/unit_tests/test_dreamer_runner.py`: focused public-export, placement, and preservation contracts.
- `tests/unit_tests/test_openvla_traj1_libero_matrix.py`: composed mainline target and batch-size contracts.
- `tests/unit_tests/test_cotrain_stage_order.py`: execute the failure-imagined stage-order contract against both retained and new runners.

### Task 1: Add failing DreamerRunner contracts

**Files:**
- Create: `tests/unit_tests/test_dreamer_runner.py`
- Modify: `tests/unit_tests/test_openvla_traj1_libero_matrix.py`
- Modify: `tests/unit_tests/test_cotrain_stage_order.py`

- [ ] **Step 1: Write the failing export and placement tests**

Create `tests/unit_tests/test_dreamer_runner.py` with:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.runners import CotrainRunner, DreamerRunner


def test_dreamer_runner_preserves_original_cotrain_runner() -> None:
    assert CotrainRunner.__module__ == "dreamervla.runners.cotrain_runner"
    assert DreamerRunner.__module__ == "dreamervla.runners.dreamer_runner"
    assert DreamerRunner is not CotrainRunner


def test_dreamer_runner_places_frozen_learner_on_cpu() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=openvla_libero"])
    OmegaConf.resolve(cfg)
    runner = DreamerRunner(cfg)
    plan = runner._placement_plan()
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[gpu] for gpu in range(8)]
    assert plan.learner_spec is not None
    assert plan.learner_spec.gpu_ids == []
```

- [ ] **Step 2: Extend the composed mainline assertions**

In `test_cotrain_components_are_selected_from_worldmodel_and_classifier_groups`, add:

```python
assert cfg._target_ == "dreamervla.runners.DreamerRunner"
assert cfg.actor.train_cfg.global_batch_size == 16384
assert cfg.actor.train_cfg.micro_batch_size == 8
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/unit_tests/test_dreamer_runner.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py::test_cotrain_components_are_selected_from_worldmodel_and_classifier_groups
```

Expected: collection/import failure because `DreamerRunner` is not exported, followed by target/batch assertion failures once import is temporarily isolated.

- [ ] **Step 4: Parameterize the stage-order regression**

Import `DreamerRunner` in `test_cotrain_stage_order.py` and parameterize
`test_failure_imagined_rl_skips_encoder_and_learner_updates` over
`(CotrainRunner, DreamerRunner)`. Instantiate `runner_cls(cfg)` inside the test.
The assertions remain:

```python
assert "encoder_sft" not in events
assert "reencode" not in events
assert "wm_cls_update" not in events
assert "wm_cls_sync" not in events
assert "wm_generate" in events
assert "ppo" in events
```

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/unit_tests/test_dreamer_runner.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py \
  tests/unit_tests/test_cotrain_stage_order.py
git commit -s -m "test: define Dreamer imagined RL runner contracts"
```

### Task 2: Implement and export DreamerRunner

**Files:**
- Modify: `dreamervla/runners/dreamer_runner.py`
- Modify: `dreamervla/runners/__init__.py`

- [ ] **Step 1: Rename the copied public runner**

In `dreamervla/runners/dreamer_runner.py`, rename only the copied public class and its public metadata/error text:

```python
class DreamerRunner(BaseRunner):
    """Frozen latent-imagination RL runner for the DreamerVLA mainline."""

    runner_name = "dreamer"
    runner_status = "current"
    runner_family = "cotrain"
```

Update the module `__all__` to:

```python
__all__ = ["DreamerRunner"]
```

Do not edit `dreamervla/runners/cotrain_runner.py`.

- [ ] **Step 2: Put the non-updating LearnerGroup on CPU**

Import `replace` from `dataclasses` and change `DreamerRunner._placement_plan` to retain every existing placement except the Learner GPU list:

```python
def _placement_plan(self) -> ManualCotrainPlacementPlan:
    plan = build_manual_cotrain_placement(
        self._ngpu(),
        real_env_workers=self._real_env_workers(),
        include_learner=True,
        component_gpu_groups=self._component_gpu_groups(),
    )
    if plan.learner_spec is None:
        raise ValueError("DreamerRunner requires a checkpoint-owning LearnerGroup")
    return replace(
        plan,
        learner_spec=replace(plan.learner_spec, gpu_ids=[]),
    )
```

This keeps the inherited state-load/checkpoint protocol intact while ensuring the disabled WM/CLS optimizers cannot consume Actor rank-0 CUDA memory.

- [ ] **Step 3: Export the new runner without replacing CotrainRunner**

Add this mapping to `_RUNNER_MODULES` in `dreamervla/runners/__init__.py`:

```python
"DreamerRunner": "dreamervla.runners.dreamer_runner",
```

Leave this existing mapping unchanged:

```python
"CotrainRunner": "dreamervla.runners.cotrain_runner",
```

- [ ] **Step 4: Run the new runner tests and verify GREEN**

Run:

```bash
pytest -q tests/unit_tests/test_dreamer_runner.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the runner implementation**

```bash
git add dreamervla/runners/dreamer_runner.py dreamervla/runners/__init__.py
git commit -s -m "feat: add frozen imagined RL DreamerRunner"
```

### Task 3: Select the runner and bound PPO activation memory

**Files:**
- Modify: `configs/dreamervla/wmcls_cotrain.yaml`

- [ ] **Step 1: Change only the frozen mainline config**

Replace its target and add a route-local actor override:

```yaml
_target_: dreamervla.runners.DreamerRunner

actor:
  train_cfg:
    # Keep the 16384 effective global batch while bounding per-forward Llama activations.
    micro_batch_size: 8
```

Keep these existing flags unchanged:

```yaml
manual_cotrain:
  training_mode: failure_imagined_rl
  learner_updates_enabled: false
  staged_policy_update: false
```

- [ ] **Step 2: Run the composed config and stage-order tests**

Run:

```bash
pytest -q \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py::test_cotrain_components_are_selected_from_worldmodel_and_classifier_groups \
  tests/unit_tests/test_cotrain_stage_order.py::test_failure_imagined_rl_skips_encoder_and_learner_updates
```

Expected: both contracts pass for the mainline config and both runner classes.

- [ ] **Step 3: Commit the route switch**

```bash
git add configs/dreamervla/wmcls_cotrain.yaml \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py \
  tests/unit_tests/test_cotrain_stage_order.py
git commit -s -m "fix: bound Dreamer actor PPO memory"
```

### Task 4: Verify preservation and regression scope

**Files:**
- Verify: `dreamervla/runners/cotrain_runner.py`
- Verify: all files above

- [ ] **Step 1: Prove the retained runner was not edited**

Run:

```bash
git diff --exit-code HEAD~3 -- dreamervla/runners/cotrain_runner.py
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Run focused regression tests**

Run:

```bash
pytest -q \
  tests/unit_tests/test_dreamer_runner.py \
  tests/unit_tests/test_cotrain_stage_order.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py \
  tests/unit_tests/test_cotrain_placement.py \
  tests/unit_tests/test_cotrain_config_validation.py
```

Expected: all tests pass.

- [ ] **Step 3: Run static validation**

Run:

```bash
ruff check dreamervla/runners/dreamer_runner.py dreamervla/runners/__init__.py \
  tests/unit_tests/test_dreamer_runner.py \
  tests/unit_tests/test_cotrain_stage_order.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py
ruff format --check dreamervla/runners/dreamer_runner.py dreamervla/runners/__init__.py \
  tests/unit_tests/test_dreamer_runner.py \
  tests/unit_tests/test_cotrain_stage_order.py \
  tests/unit_tests/test_openvla_traj1_libero_matrix.py
git diff --check
```

Expected: every command exits 0 with no lint, format, or whitespace errors.

- [ ] **Step 4: Inspect the final diff and status**

Run:

```bash
git status --short
git diff --stat HEAD~3..HEAD
git log -4 --oneline
```

Expected: only the planned runner, config, tests, and documentation are changed; the user's pre-existing `task_plan.md`, `findings.md`, and `progress.md` remain untracked and untouched.
