# Cotrain P2 Adaptive Imagination Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make imagined PPO groups bounded, variance-aware, and clearly logged as actor-training diagnostics rather than real success metrics.

**Architecture:** Keep imagined rollouts as a short-lived in-update buffer inside `dino_lumos_step()`. Generate at most `algorithm.lumos.ppo_rollouts_per_start_max` per start, choose the first prefix with non-zero return variance, skip zero-variance groups, and expose actor-signal metrics under `rl/` and `LUMOS/` only.

**Tech Stack:** Python 3.11, PyTorch, pytest, Hydra/OmegaConf, DreamerVLA PPO/LUMOS algorithm registry.

---

## File Structure

- Modify: `dreamervla/algorithms/ppo/outcome.py:166`
  - Keep adaptive group prefix helper bounded by min/max.
- Modify: `dreamervla/runners/online_cotrain_runner.py:1106`
  - Factor actor metrics into a helper so namespace rules are testable.
- Modify: `dreamervla/config.py`
  - Validate `algorithm.lumos.ppo_rollouts_per_start_min <= max`.
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:219`
  - Keep min/max rollout bounds in Hydra.
- Modify: `tests/unit_tests/test_lumos_signal.py:201`
  - Extend adaptive group tests.
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py`
  - Add namespaced actor metric tests.
- Verify: `docs/online_cotrain_metrics_inventory.md`
  - Inventory remains consistent with emitted metrics.

## Task 1: Lock Bounded Adaptive Group Semantics

**Files:**
- Modify: `tests/unit_tests/test_lumos_signal.py:201`
- Verify: `dreamervla/algorithms/ppo/outcome.py:166`

- [ ] **Step 1: Add direct helper tests**

Add to `tests/unit_tests/test_lumos_signal.py`:

```python
def test_adaptive_group_advantage_uses_first_prefix_with_variance() -> None:
    from dreamervla.algorithms.ppo.outcome import _adaptive_group_advantage_and_mask

    returns = torch.tensor([0.0, 0.0, 1.0, 1.0])

    advantages, active_mask, has_variance, effective_counts = (
        _adaptive_group_advantage_and_mask(
            returns,
            group_size_min=2,
            group_size_max=4,
            eps=1.0e-6,
        )
    )

    assert has_variance.tolist() == [True]
    assert effective_counts.tolist() == [3]
    assert active_mask.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert advantages[3].item() == 0.0


def test_adaptive_group_advantage_marks_zero_variance_at_max_bound() -> None:
    from dreamervla.algorithms.ppo.outcome import _adaptive_group_advantage_and_mask

    returns = torch.tensor([0.0, 0.0, 0.0, 0.0])

    advantages, active_mask, has_variance, effective_counts = (
        _adaptive_group_advantage_and_mask(
            returns,
            group_size_min=2,
            group_size_max=4,
            eps=1.0e-6,
        )
    )

    assert has_variance.tolist() == [False]
    assert effective_counts.tolist() == [4]
    assert active_mask.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert advantages.abs().sum().item() == 0.0
```

- [ ] **Step 2: Run direct helper tests**

Run:

```bash
pytest tests/unit_tests/test_lumos_signal.py::test_adaptive_group_advantage_uses_first_prefix_with_variance tests/unit_tests/test_lumos_signal.py::test_adaptive_group_advantage_marks_zero_variance_at_max_bound -q
```

Expected: PASS if `_adaptive_group_advantage_and_mask()` already matches the bounded contract.

- [ ] **Step 3: Keep helper behavior unchanged**

If tests fail, update `_adaptive_group_advantage_and_mask()` so it:

```python
    for idx in range(groups.shape[0]):
        chosen = int(group_size_max)
        std = torch.zeros((), dtype=groups.dtype, device=groups.device)
        for count in range(int(group_size_min), int(group_size_max) + 1):
            vals = groups[idx, :count]
            std = vals.std(unbiased=False)
            if bool(std > eps):
                chosen = count
                has_variance[idx] = True
                break
        active_mask[idx, :chosen] = 1.0
        effective_counts[idx] = chosen
        if bool(has_variance[idx]):
            vals = groups[idx, :chosen]
            advantages[idx, :chosen] = (vals - vals.mean()) / (std + eps)
```

- [ ] **Step 4: Verify LUMOS signal tests**

Run:

```bash
pytest tests/unit_tests/test_lumos_signal.py -q
```

Expected: all LUMOS signal tests pass.

## Task 2: Validate Rollout Bounds from Config

**Files:**
- Modify: `dreamervla/config.py`
- Modify: `tests/unit_tests/test_config_validation.py`
- Verify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:219`

- [ ] **Step 1: Add config validation tests**

Add to `tests/unit_tests/test_config_validation.py`:

```python
def test_validate_cfg_rejects_bad_lumos_rollout_bounds():
    import pytest
    from omegaconf import OmegaConf
    from dreamervla.config import validate_cfg

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "lumos": {
                    "ppo_rollouts_per_start_min": 8,
                    "ppo_rollouts_per_start_max": 4,
                }
            }
        }
    )

    with pytest.raises(ValueError, match="ppo_rollouts_per_start_min"):
        validate_cfg(cfg)
```

- [ ] **Step 2: Run validation test**

Run:

```bash
pytest tests/unit_tests/test_config_validation.py::test_validate_cfg_rejects_bad_lumos_rollout_bounds -q
```

Expected before implementation: FAIL because invalid rollout bounds are accepted.

- [ ] **Step 3: Add validator**

In `dreamervla/config.py`, add a helper:

```python
def _validate_lumos_rollout_bounds(cfg: DictConfig) -> None:
    min_value = OmegaConf.select(cfg, "algorithm.lumos.ppo_rollouts_per_start_min", default=None)
    max_value = OmegaConf.select(cfg, "algorithm.lumos.ppo_rollouts_per_start_max", default=None)
    if min_value is None and max_value is None:
        return
    legacy = int(OmegaConf.select(cfg, "algorithm.ppo_rollouts_per_start", default=4))
    min_rollouts = int(min_value if min_value is not None else legacy)
    max_rollouts = int(max_value if max_value is not None else legacy)
    if min_rollouts < 1 or max_rollouts < min_rollouts:
        raise ValueError(
            "algorithm.lumos.ppo_rollouts_per_start_min/max must satisfy "
            f"1 <= min <= max, got min={min_rollouts} max={max_rollouts}"
        )
```

Call `_validate_lumos_rollout_bounds(cfg)` from `validate_cfg()`.

- [ ] **Step 4: Verify config validation**

Run:

```bash
pytest tests/unit_tests/test_config_validation.py::test_validate_cfg_rejects_bad_lumos_rollout_bounds -q
```

Expected: PASS.

## Task 3: Factor Actor-Signal Metric Namespace

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py:1106`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py`

- [ ] **Step 1: Write metric helper test**

Add to `tests/unit_tests/test_online_cotrain_pipeline.py`:

```python
def test_actor_signal_metrics_keep_imagined_success_out_of_rollout_namespace():
    from dreamervla.runners.online_cotrain_runner import build_actor_signal_metrics

    metrics = build_actor_signal_metrics(
        {
            "actor_loss": 1.25,
            "returns_mean": 0.5,
            "returns_std": 0.25,
            "advantage_std": 0.2,
            "advantage_mag": 0.3,
            "actor_grad_norm": 4.0,
            "ppo_step_applied": 1.0,
            "LUMOS/success_rate": 0.75,
            "LUMOS/score_mean": 0.6,
            "LUMOS/score_std": 0.1,
            "LUMOS/group_var_keep_frac": 0.5,
            "LUMOS/skipped_zero_variance_groups": 2.0,
        }
    )

    assert metrics["rl/actor_loss"] == 1.25
    assert metrics["rl/returns_std"] == 0.25
    assert metrics["rl/policy_grad_norm"] == 4.0
    assert metrics["rl/skipped_zero_variance_groups"] == 2.0
    assert metrics["LUMOS/score_mean"] == 0.6
    assert metrics["LUMOS/group_var_keep_frac"] == 0.5
    assert "rollout/success_rate" not in metrics
    assert "rollout/recent_success_rate" not in metrics
```

- [ ] **Step 2: Run the helper test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_actor_signal_metrics_keep_imagined_success_out_of_rollout_namespace -q
```

Expected before implementation: FAIL because `build_actor_signal_metrics` does not exist.

- [ ] **Step 3: Add helper**

In `dreamervla/runners/online_cotrain_runner.py`, near `build_rollout_progress_metrics()`, add:

```python
def build_actor_signal_metrics(ac_metrics: dict[str, Any]) -> dict[str, float]:
    """Return PPO/LUMOS diagnostics without real-rollout success keys."""
    out = {
        "rl/actor_loss": float(ac_metrics.get("actor_loss", 0.0)),
        "rl/returns_mean": float(ac_metrics.get("returns_mean", 0.0)),
        "rl/returns_std": float(ac_metrics.get("returns_std", 0.0)),
        "rl/advantage_std": float(ac_metrics.get("advantage_std", 0.0)),
        "rl/advantage_mag": float(ac_metrics.get("advantage_mag", 0.0)),
        "rl/policy_grad_norm": float(ac_metrics.get("actor_grad_norm", 0.0)),
        "rl/skipped_zero_variance_groups": float(
            ac_metrics.get("LUMOS/skipped_zero_variance_groups", 0.0)
        ),
        "rl/ppo_step_applied": float(ac_metrics.get("ppo_step_applied", 0.0)),
    }
    for key in (
        "LUMOS/score_mean",
        "LUMOS/score_std",
        "LUMOS/group_var_keep_frac",
        "LUMOS/num_mixed_groups",
        "LUMOS/skipped_zero_variance_groups",
    ):
        if key in ac_metrics:
            out[key] = float(ac_metrics[key])
    return out
```

- [ ] **Step 4: Use helper in the online training burst**

Replace the inline actor metric assignments around `online_cotrain_runner.py:1106` with:

```python
                metrics.update(build_actor_signal_metrics(ac_metrics))
```

Do not copy `LUMOS/success_rate` into `rollout/`.

- [ ] **Step 5: Verify actor metric namespace**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_actor_signal_metrics_keep_imagined_success_out_of_rollout_namespace tests/unit_tests/test_lumos_signal.py -q
```

Expected: selected tests pass.

## Task 4: Run Adaptive Imagination Regression Suite

**Files:**
- Verify: all files modified in this plan.

- [ ] **Step 1: Run CPU tests**

Run:

```bash
pytest tests/unit_tests/test_lumos_signal.py tests/unit_tests/test_online_cotrain_pipeline.py::test_actor_signal_metrics_keep_imagined_success_out_of_rollout_namespace tests/unit_tests/test_config_validation.py::test_validate_cfg_rejects_bad_lumos_rollout_bounds -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Inspect metric inventory consistency**

Run:

```bash
rg -n "rollout/success_rate|LUMOS/success_rate|rl/skipped_zero_variance_groups|LUMOS/score_std" docs/online_cotrain_metrics_inventory.md dreamervla/runners/online_cotrain_runner.py dreamervla/algorithms/ppo/outcome.py
```

Expected: `rollout/success_rate` appears only in real rollout metric code/docs; `LUMOS/success_rate` appears only as imagined diagnostic.

- [ ] **Step 3: Commit**

```bash
git add dreamervla/algorithms/ppo/outcome.py dreamervla/runners/online_cotrain_runner.py dreamervla/config.py configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml tests/unit_tests/test_lumos_signal.py tests/unit_tests/test_online_cotrain_pipeline.py tests/unit_tests/test_config_validation.py
git commit -s -m "feat(cotrain): bound adaptive imagined rollout groups"
```
