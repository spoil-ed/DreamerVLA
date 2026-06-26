# Cotrain P0 Action Chunk and Real Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure real online rollout executes every low-level action in each actor action chunk and reports success metrics only from completed real episodes.

**Architecture:** Introduce one tiny action-chunk queue helper and wire it into no-Ray single-env, no-Ray vectorized, and Ray inference rollout paths. Keep metrics in `rollout/` limited to real completed episodes; imagined classifier scores remain in `rl/` and `LUMOS/`.

**Tech Stack:** Python 3.11, NumPy, PyTorch, pytest, Hydra/OmegaConf, DreamerVLA runners/workers.

---

## File Structure

- Create: `dreamervla/runners/action_chunk_queue.py`
  - Single-purpose queue for open-loop execution of action chunks.
- Create: `tests/unit_tests/test_action_chunk_queue.py`
  - Pure CPU unit tests for queue refill, pop, clear, and short-chunk rejection.
- Modify: `dreamervla/runners/online_cotrain_runner.py:74`
  - Import and use `ActionChunkQueue` in single-env and vectorized cotrain rollout.
- Modify: `dreamervla/workers/inference/rollout_inference_worker.py:32`
  - Use the same queue in Ray OFT fixed-base inference.
- Modify: `tests/unit_tests/test_cotrain_vec_rollout.py:239`
  - Keep vectorized rollout test focused on full-chunk open-loop behavior.
- Modify: `tests/unit_tests/test_rollout_inference_worker.py:71`
  - Keep Ray inference test focused on `action_steps` full chunk execution.
- Modify: `tests/unit_tests/test_online_cotrain_ray_runner.py:252`
  - Guard real rollout metric names.
- Verify only: `docs/online_cotrain_metrics_inventory.md`
  - Metric inventory should already state that `rollout/` is real-only.

## Task 1: Shared Action Chunk Queue

**Files:**
- Create: `dreamervla/runners/action_chunk_queue.py`
- Create: `tests/unit_tests/test_action_chunk_queue.py`

- [ ] **Step 1: Write the failing queue tests**

Add `tests/unit_tests/test_action_chunk_queue.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from dreamervla.runners.action_chunk_queue import ActionChunkQueue


def test_action_chunk_queue_executes_full_chunk_before_refill() -> None:
    queue = ActionChunkQueue(action_dim=7, action_steps=3)
    first_chunk = np.arange(28, dtype=np.float32).reshape(4, 7)

    queue.refill(first_chunk)

    np.testing.assert_array_equal(queue.pop(), first_chunk[0])
    np.testing.assert_array_equal(queue.pop(), first_chunk[1])
    np.testing.assert_array_equal(queue.pop(), first_chunk[2])
    assert queue.has_pending is False


def test_action_chunk_queue_clear_drops_pending_actions() -> None:
    queue = ActionChunkQueue(action_dim=7, action_steps=2)
    queue.refill(np.ones((2, 7), dtype=np.float32))

    queue.clear()

    assert queue.has_pending is False
    with pytest.raises(IndexError, match="empty action chunk queue"):
        queue.pop()


def test_action_chunk_queue_rejects_short_chunk() -> None:
    queue = ActionChunkQueue(action_dim=7, action_steps=4)

    with pytest.raises(ValueError, match="need action_steps=4"):
        queue.refill(np.zeros((3, 7), dtype=np.float32))
```

- [ ] **Step 2: Run the queue tests and confirm the missing module failure**

Run:

```bash
pytest tests/unit_tests/test_action_chunk_queue.py -q
```

Expected before implementation: `ModuleNotFoundError: No module named 'dreamervla.runners.action_chunk_queue'`.
If this already passes, keep the existing helper and continue with Task 2.

- [ ] **Step 3: Implement the helper**

Create `dreamervla/runners/action_chunk_queue.py`:

```python
"""Open-loop action chunk queue shared by cotrain rollout paths."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


class ActionChunkQueue:
    """Queue exactly `action_steps` low-level actions from one actor chunk."""

    def __init__(self, *, action_dim: int = 7, action_steps: int | None = None) -> None:
        self.action_dim = int(action_dim)
        self.action_steps = None if action_steps is None else max(1, int(action_steps))
        self._pending: Deque[np.ndarray] = deque()

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def refill(self, action_chunk: np.ndarray) -> None:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.ndim != 2:
            raise ValueError(f"action chunk must be [K,A], got shape {tuple(chunk.shape)}")
        if chunk.shape[1] < self.action_dim:
            raise ValueError(
                f"action chunk dim {chunk.shape[1]} < action_dim={self.action_dim}"
            )
        steps = self.action_steps if self.action_steps is not None else int(chunk.shape[0])
        if chunk.shape[0] < steps:
            raise ValueError(
                f"policy returned {chunk.shape[0]} actions, need action_steps={steps}"
            )
        self._pending.clear()
        for row in chunk[:steps, : self.action_dim]:
            self._pending.append(np.asarray(row, dtype=np.float32).copy())

    def pop(self) -> np.ndarray:
        if not self._pending:
            raise IndexError("empty action chunk queue")
        return self._pending.popleft()

    def clear(self) -> None:
        self._pending.clear()


__all__ = ["ActionChunkQueue"]
```

- [ ] **Step 4: Verify the helper**

Run:

```bash
pytest tests/unit_tests/test_action_chunk_queue.py -q
```

Expected: `3 passed`.

## Task 2: Wire No-Ray Real Rollout

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py:74`
- Modify: `tests/unit_tests/test_cotrain_vec_rollout.py:265`

- [ ] **Step 1: Strengthen the vectorized rollout test**

In `tests/unit_tests/test_cotrain_vec_rollout.py`, keep `test_vectorized_rollout_isolated_queues_and_episode_grouping` and ensure the action pattern asserts full chunk execution:

```python
    for slot in range(2):
        slot_actions = vec.actions_seen[slot::2]
        assert len(slot_actions) == 6
        expected_scalars = [0.25, 0.75, 0.25, 0.25, 0.75, 0.25]
        for action, expected in zip(slot_actions, expected_scalars, strict=True):
            np.testing.assert_array_equal(action, np.full(7, expected, dtype=np.float32))
```

- [ ] **Step 2: Run the targeted rollout test**

Run:

```bash
pytest tests/unit_tests/test_cotrain_vec_rollout.py::test_vectorized_rollout_isolated_queues_and_episode_grouping -q
```

Expected before wiring: FAIL if any rollout path refills the actor every env step and repeatedly executes `chunk[0]`.

- [ ] **Step 3: Replace ad-hoc lists with `ActionChunkQueue`**

In `dreamervla/runners/online_cotrain_runner.py`, import the helper near the runner imports:

```python
from dreamervla.runners.action_chunk_queue import ActionChunkQueue
```

In the single-env loop around `online_cotrain_runner.py:878`, replace the list queue with:

```python
pending_actions = ActionChunkQueue(action_dim=7)
```

Replace the pop/refill block around `online_cotrain_runner.py:895` with:

```python
            if not pending_actions.has_pending:
                chunk = self._sample_actor_action_chunk(self.world_model, self.policy, latent)
                pending_actions.refill(chunk)
            policy_action = pending_actions.pop()
```

Keep the episode reset clear:

```python
                pending_actions.clear()
```

In `_vectorized_cotrain_rollout` around `online_cotrain_runner.py:1302`, replace:

```python
slot_pending_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
```

with:

```python
slot_pending_actions = [ActionChunkQueue(action_dim=7) for _ in range(num_envs)]
```

Replace the vectorized pop/refill block around `online_cotrain_runner.py:1348` with:

```python
                if not slot_pending_actions[k].has_pending:
                    chunk = self._sample_actor_action_chunk(
                        self.world_model, self.policy, slot_latent[k]
                    )
                    slot_pending_actions[k].refill(chunk)
                act = slot_pending_actions[k].pop()
```

Keep `_start_slot` clearing the queue:

```python
            slot_pending_actions[k].clear()
```

- [ ] **Step 4: Verify no-Ray rollout tests**

Run:

```bash
pytest tests/unit_tests/test_action_chunk_queue.py tests/unit_tests/test_cotrain_vec_rollout.py -q
```

Expected: all selected tests pass.

## Task 3: Wire Ray OFT Fixed-Base Inference

**Files:**
- Modify: `dreamervla/workers/inference/rollout_inference_worker.py:32`
- Modify: `tests/unit_tests/test_rollout_inference_worker.py:71`

- [ ] **Step 1: Keep the Ray open-loop test explicit**

Ensure `tests/unit_tests/test_rollout_inference_worker.py` contains:

```python
def test_forward_batch_executes_action_chunk_open_loop() -> None:
    cfg = _cfg()
    cfg["action_steps"] = 3
    w = RolloutInferenceWorker(cfg, {}, num_envs=1)
    w.init()

    first = w.forward_batch([{"seed": 0}], [0])
    second = w.forward_batch([{"seed": 10}], [0])
    third = w.forward_batch([{"seed": 20}], [0])

    assert [int(out["actions"][0][0]) for out in (first, second, third)] == [0, 1, 2]
```

- [ ] **Step 2: Run the Ray inference test**

Run:

```bash
pytest tests/unit_tests/test_rollout_inference_worker.py::test_forward_batch_executes_action_chunk_open_loop -q
```

Expected before wiring: FAIL if `RolloutInferenceWorker.forward_batch()` discards pending chunk actions.

- [ ] **Step 3: Use `ActionChunkQueue` in `RolloutInferenceWorker`**

In `dreamervla/workers/inference/rollout_inference_worker.py`, import:

```python
from dreamervla.runners.action_chunk_queue import ActionChunkQueue
```

Replace `_action_queues` initialization with:

```python
        self._action_queues = [
            ActionChunkQueue(action_dim=self._action_dim, action_steps=self._action_steps)
            for _ in range(self._num_envs)
        ]
```

Replace the queue handling in `forward_batch()` with:

```python
            queue = self._action_queues[env_index]
            if not queue.has_pending:
                queue.refill(np.asarray(action_chunk, dtype=np.float32))
            action = process_action(queue.pop())[: self._action_dim]
```

Replace reset handling with:

```python
            self._action_queues[int(env_id)].clear()
```

- [ ] **Step 4: Verify Ray inference tests**

Run:

```bash
pytest tests/unit_tests/test_rollout_inference_worker.py -q
```

Expected: all selected tests pass.

## Task 4: Guard Real Success Metrics

**Files:**
- Modify: `tests/unit_tests/test_cotrain_vec_rollout.py:118`
- Modify: `tests/unit_tests/test_online_cotrain_ray_runner.py:252`
- Verify: `dreamervla/runners/online_cotrain_runner.py:170`
- Verify: `dreamervla/runners/online_cotrain_ray_runner.py:656`

- [ ] **Step 1: Add a no-legacy-metric assertion for sync rollout metrics**

In `tests/unit_tests/test_cotrain_vec_rollout.py`, extend `test_rollout_progress_metrics_reports_recent_success_rate`:

```python
    assert "rollout/current_success_rate" not in metrics
    assert "rollout/avg_success_rate" not in metrics
    assert "LUMOS/success_rate" not in metrics
```

- [ ] **Step 2: Ensure Ray history excludes legacy metric names**

In `tests/unit_tests/test_online_cotrain_ray_runner.py`, keep these assertions in
`test_ray_runner_prints_episode_success_rate`:

```python
    assert "rollout/current_success_rate" not in history
    assert "rollout/avg_success_rate" not in history
```

- [ ] **Step 3: Run metric tests**

Run:

```bash
pytest tests/unit_tests/test_cotrain_vec_rollout.py::test_rollout_progress_metrics_reports_recent_success_rate tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_prints_episode_success_rate -q
```

Expected: both tests pass. If they fail, remove legacy `rollout/current_success_rate` and
`rollout/avg_success_rate` from runner metric dictionaries, not from console-only text.

- [ ] **Step 4: Run P0 regression suite**

Run:

```bash
pytest tests/unit_tests/test_action_chunk_queue.py tests/unit_tests/test_cotrain_vec_rollout.py tests/unit_tests/test_rollout_inference_worker.py tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_prints_episode_success_rate -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/action_chunk_queue.py dreamervla/runners/online_cotrain_runner.py dreamervla/workers/inference/rollout_inference_worker.py tests/unit_tests/test_action_chunk_queue.py tests/unit_tests/test_cotrain_vec_rollout.py tests/unit_tests/test_rollout_inference_worker.py tests/unit_tests/test_online_cotrain_ray_runner.py
git commit -s -m "fix(cotrain): execute full action chunks in real rollout"
```
