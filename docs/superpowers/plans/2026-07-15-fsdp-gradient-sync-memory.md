# FSDP Gradient-Sync Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disable FSDP `no_sync()` by default during PPO microbatch accumulation so each rank retains only sharded gradients.

**Architecture:** Add an explicit `enable_gradient_accumulation` option to the existing FSDP manager, defaulting to `false` like RLinf. The Actor uses `no_sync()` only when this option is explicitly enabled and the microbatch is not the last one; loss scaling and optimizer boundaries are unchanged.

**Tech Stack:** Python 3.11, PyTorch FSDP1, pytest, Hydra static YAML.

---

### Task 1: Specify Default and Opt-In Synchronization Behavior

**Files:**
- Modify: `tests/unit_tests/test_embodied_fsdp_actor.py`
- Test: `tests/unit_tests/test_embodied_fsdp_actor.py`

- [ ] **Step 1: Replace the existing unconditional no-sync test with a failing default test**

```python
def test_actor_fsdp_syncs_every_accumulated_backward_by_default(monkeypatch) -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["global_batch_size"] = 8
    cfg["train_cfg"]["micro_batch_size"] = 2
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    policy = _NoSyncProbePolicy()
    actor.policy = policy
    actor.optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    actor.load_trajectory_shards([_shard(0.0, 1.0), _shard(2.0, 3.0)])
    actor.compute_advantages_and_returns()
    monkeypatch.setattr(embodied_fsdp_actor, "_is_fsdp_module", lambda _policy: True)

    actor.run_training()

    assert policy.no_sync_calls == 0
    assert policy.forward_sync_states == [True, True, True, True]
```

- [ ] **Step 2: Add an opt-in compatibility test**

```python
def test_actor_fsdp_no_syncs_nonfinal_backward_when_explicitly_enabled(
    monkeypatch,
) -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["global_batch_size"] = 8
    cfg["train_cfg"]["micro_batch_size"] = 2
    cfg["train_cfg"]["fsdp"]["enable_gradient_accumulation"] = True
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    policy = _NoSyncProbePolicy()
    actor.policy = policy
    actor.optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    actor.load_trajectory_shards([_shard(0.0, 1.0), _shard(2.0, 3.0)])
    actor.compute_advantages_and_returns()
    monkeypatch.setattr(embodied_fsdp_actor, "_is_fsdp_module", lambda _policy: True)

    actor.run_training()

    assert policy.no_sync_calls == 3
    assert policy.forward_sync_states == [False, False, False, True]
```

- [ ] **Step 3: Run the default test and verify RED**

Run:

```bash
pytest -q tests/unit_tests/test_embodied_fsdp_actor.py::test_actor_fsdp_syncs_every_accumulated_backward_by_default
```

Expected: FAIL because the current Actor calls `no_sync()` three times.

### Task 2: Make FSDP Gradient Accumulation Explicit

**Files:**
- Modify: `dreamervla/hybrid_engines/fsdp/fsdp_model_manager.py`
- Modify: `dreamervla/hybrid_engines/fsdp/strategy/base.py`
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py:740-788`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain.yaml:141-147`

- [ ] **Step 1: Add the default-off manager and strategy option**

Add this dataclass field and pass it through `FSDPModelManager.make_strategy()`:

```python
enable_gradient_accumulation: bool = False
```

Accept and store the same boolean in `FSDPStrategyBase.__init__()`. This keeps
construction compatible for every registered strategy while making the FSDP
contract explicit.

- [ ] **Step 2: Gate Actor `no_sync()` on the option**

Before the PPO loops, derive:

```python
fsdp_gradient_accumulation = bool(
    fsdp_policy
    and self.fsdp_manager is not None
    and self.fsdp_manager.enable_gradient_accumulation
)
```

Then select the context with:

```python
backward_context = (
    policy.no_sync()
    if fsdp_gradient_accumulation and not is_last_micro
    else nullcontext()
)
```

- [ ] **Step 3: Declare the mainline default in static Hydra YAML**

Under `actor.train_cfg.fsdp`, add:

```yaml
enable_gradient_accumulation: false
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
pytest -q \
  tests/unit_tests/test_embodied_fsdp_actor.py::test_actor_fsdp_syncs_every_accumulated_backward_by_default \
  tests/unit_tests/test_embodied_fsdp_actor.py::test_actor_fsdp_no_syncs_nonfinal_backward_when_explicitly_enabled \
  tests/unit_tests/test_fsdp_model_manager.py \
  tests/unit_tests/test_fsdp_strategy.py
```

Expected: all tests PASS.

### Task 3: Regression Verification and Commit

**Files:**
- Verify: `dreamervla/workers/actor/embodied_fsdp_actor.py`
- Verify: `dreamervla/hybrid_engines/fsdp/`
- Verify: `configs/dreamervla/openvla_onetraj_libero_cotrain.yaml`
- Verify: `tests/unit_tests/test_embodied_fsdp_actor.py`

- [ ] **Step 1: Run the complete Actor unit-test module**

```bash
pytest -q tests/unit_tests/test_embodied_fsdp_actor.py
```

Expected: all tests PASS.

- [ ] **Step 2: Run formatting and diff checks on only the owned files**

```bash
ruff check dreamervla/hybrid_engines/fsdp/fsdp_model_manager.py \
  dreamervla/hybrid_engines/fsdp/strategy/base.py \
  dreamervla/workers/actor/embodied_fsdp_actor.py \
  tests/unit_tests/test_embodied_fsdp_actor.py
ruff format --check dreamervla/hybrid_engines/fsdp/fsdp_model_manager.py \
  dreamervla/hybrid_engines/fsdp/strategy/base.py \
  dreamervla/workers/actor/embodied_fsdp_actor.py \
  tests/unit_tests/test_embodied_fsdp_actor.py
git diff --check -- \
  dreamervla/hybrid_engines/fsdp/fsdp_model_manager.py \
  dreamervla/hybrid_engines/fsdp/strategy/base.py \
  dreamervla/workers/actor/embodied_fsdp_actor.py \
  configs/dreamervla/openvla_onetraj_libero_cotrain.yaml \
  tests/unit_tests/test_embodied_fsdp_actor.py
```

Expected: every command exits successfully.

- [ ] **Step 3: Commit only the owned files**

```bash
git add \
  docs/superpowers/plans/2026-07-15-fsdp-gradient-sync-memory.md \
  dreamervla/hybrid_engines/fsdp/fsdp_model_manager.py \
  dreamervla/hybrid_engines/fsdp/strategy/base.py \
  dreamervla/workers/actor/embodied_fsdp_actor.py \
  configs/dreamervla/openvla_onetraj_libero_cotrain.yaml \
  tests/unit_tests/test_embodied_fsdp_actor.py
git commit -s -m "fix: bound FSDP gradient accumulation memory"
```

Expected: a signed Conventional Commit containing no concurrent runner changes.
