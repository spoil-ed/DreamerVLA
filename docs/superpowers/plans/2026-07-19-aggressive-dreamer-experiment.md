# Aggressive Dreamer Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately selected 20-step aggressive Dreamer Hydra experiment without changing the existing `openvla_libero` configuration.

**Architecture:** Compose the same task, frozen Dreamer route, WM, classifier, and cotrain launcher in a new experiment YAML. Override only the approved training/evaluation hyperparameters, then prove both the new values and original-route isolation through one focused composition test.

**Tech Stack:** Hydra, OmegaConf, pytest, DreamerVLA configuration validation.

## Global Constraints

- Python 3.11.
- Hydra remains the source of truth.
- Do not modify `configs/experiment/openvla_libero.yaml` or runtime code.
- Select the new route explicitly with `--config openvla_libero_aggressive`.
- Preserve 1,024 imagined trajectories, global batch 16,384, microbatch 8, replay capacity 80,000, and all inherited reward/clip/GAE settings.

---

### Task 1: Isolated aggressive Hydra experiment

**Files:**
- Create: `configs/experiment/openvla_libero_aggressive.yaml`
- Create: `tests/unit_tests/test_aggressive_dreamer_config.py`
- Modify: `configs/README.md`

**Interfaces:**
- Consumes: Hydra `train.yaml`, `dreamervla=wmcls_cotrain`, `launch=cotrain`, and `dreamervla.config.validate_cfg`.
- Produces: explicit `experiment=openvla_libero_aggressive` composition with no changes to `experiment=openvla_libero`.

- [ ] **Step 1: Write the failing composition and isolation test**

Create `tests/unit_tests/test_aggressive_dreamer_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from dreamervla.config import validate_cfg


def _compose(experiment: str):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="train", overrides=[f"experiment={experiment}"])


def test_aggressive_dreamer_experiment_is_explicit_and_isolated() -> None:
    aggressive = _compose("openvla_libero_aggressive")
    original = _compose("openvla_libero")

    validate_cfg(aggressive)
    validate_cfg(original)

    assert aggressive._target_ == "dreamervla.runners.DreamerRunner"
    assert aggressive.run.name == "openvla_libero_aggressive"
    assert aggressive.manual_cotrain.global_steps == 20
    assert aggressive.manual_cotrain.checkpoint_every == 5
    assert aggressive.manual_cotrain.eval_interval_global_steps == 5
    assert aggressive.manual_cotrain.eval_initial_global_step is True
    assert aggressive.manual_cotrain.eval_protocol.num_episodes_per_task == 10
    assert aggressive.manual_cotrain.real_rollout_target_trajectories == 64
    assert aggressive.manual_cotrain.wm_rollout_target_trajectories == 1024
    assert aggressive.manual_cotrain.max_policy_kl == 0.03
    assert aggressive.algorithm.group_size == 16
    assert aggressive.algorithm.entropy_bonus == 1.0e-3
    assert aggressive.actor.train_cfg.algorithm_cfg.group_size == 16
    assert aggressive.actor.train_cfg.algorithm_cfg.entropy_coef == 1.0e-3
    assert aggressive.actor.train_cfg.algorithm_cfg.ppo_update_epochs == 2
    assert aggressive.actor.train_cfg.lr == 1.0e-6
    assert aggressive.actor.train_cfg.optimizers.policy.lr == 1.0e-6
    assert aggressive.actor.train_cfg.global_batch_size == 16384
    assert aggressive.actor.train_cfg.micro_batch_size == 8
    assert aggressive.replay.cfg.capacity == 80000

    assert original.run.name == "openvla_libero"
    assert original.manual_cotrain.global_steps == 20_000
    assert original.manual_cotrain.checkpoint_every == 10
    assert original.manual_cotrain.eval_interval_global_steps == 10
    assert original.manual_cotrain.real_rollout_target_trajectories == 32
    assert original.manual_cotrain.max_policy_kl == 0.1
    assert original.algorithm.group_size == 8
    assert original.algorithm.entropy_bonus == 0.0
    assert original.actor.train_cfg.algorithm_cfg.ppo_update_epochs == 1
    assert original.actor.train_cfg.lr == 5.0e-7
    assert original.actor.train_cfg.optimizers.policy.lr == 5.0e-7
```

- [ ] **Step 2: Run the focused test and verify the missing config fails**

Run:

```bash
conda run -n dreamervla pytest -q tests/unit_tests/test_aggressive_dreamer_config.py
```

Expected: FAIL because Hydra cannot find `experiment/openvla_libero_aggressive`.

- [ ] **Step 3: Add the minimal experiment YAML**

Create `configs/experiment/openvla_libero_aggressive.yaml`:

```yaml
# @package _global_
defaults:
  - /task: openvla_onetraj_libero
  - override /dreamervla: wmcls_cotrain
  - override /worldmodel: dreamer-wm
  - override /classifier: dreamer-cls
  - override /launch: cotrain
  - _self_

run:
  name: openvla_libero_aggressive

runner:
  logger:
    logger_backends: [tensorboard, wandb]
    wandb_mode: online
    wandb_proxy: null

manual_cotrain:
  global_steps: 20
  checkpoint_every: 5
  eval_interval_global_steps: 5
  eval_initial_global_step: true
  real_rollout_target_trajectories: 64
  max_policy_kl: 0.03
  eval_protocol:
    num_episodes_per_task: 10

algorithm:
  group_size: 16
  entropy_bonus: 1.0e-3

ray_actor_optimizer:
  lr: 1.0e-6

actor:
  train_cfg:
    lr: 1.0e-6
    algorithm_cfg:
      ppo_update_epochs: 2
```

- [ ] **Step 4: Add the opt-in route to the config registry**

Add this row after `openvla_libero` in `configs/README.md`:

```markdown
| `openvla_libero_aggressive` | opt-in 20-step aggressive frozen-WM/CLS effect-validation route |
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
conda run -n dreamervla pytest -q \
  tests/unit_tests/test_aggressive_dreamer_config.py \
  tests/unit_tests/test_cotrain_launcher.py \
  tests/unit_tests/test_cotrain_debug_config.py
```

Expected: PASS.

- [ ] **Step 6: Render and validate both resolved Hydra configs**

Run:

```bash
conda run -n dreamervla python -m dreamervla.train \
  experiment=openvla_libero_aggressive --cfg job --resolve
conda run -n dreamervla python -m dreamervla.train \
  experiment=openvla_libero --cfg job --resolve
```

Expected: both commands exit 0; the aggressive output contains
`global_steps: 20`, and the original contains `global_steps: 20000`.

- [ ] **Step 7: Run formatting and diff checks**

Run:

```bash
conda run -n dreamervla ruff check tests/unit_tests/test_aggressive_dreamer_config.py
conda run -n dreamervla ruff format --check tests/unit_tests/test_aggressive_dreamer_config.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 8: Commit the implementation**

```bash
git add \
  configs/experiment/openvla_libero_aggressive.yaml \
  configs/README.md \
  tests/unit_tests/test_aggressive_dreamer_config.py
git commit -s -m "feat(config): add aggressive Dreamer experiment"
```
