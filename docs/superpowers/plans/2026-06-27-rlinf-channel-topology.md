# RLinf Channel Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the DreamerVLA Ray online cotrain driver-mediated rollout loop with an RLinf-style channel topology: `EnvWorker.interact` feeds observations to `MultiStepRolloutWorker.generate`, and `LearnerWorker.recv_rollout_trajectories` ingests completed trajectories.

**Architecture:** RLinf still has an `EnvWorker`; the topology to copy is not "delete EnvWorker", but the channel control flow in `rlinf/runners/embodied_runner.py`: env, rollout, and actor/learner run concurrently and exchange batches through channels. DreamerVLA should keep its worker-level EGL `EnvWorker` and spawn children, move policy inference into a new `MultiStepRolloutWorker`, and make `OnlineCotrainRayRunner` orchestrate handles instead of performing per-step `infer.forward_batch -> env.step` calls in the driver.

**Tech Stack:** Ray actors, DreamerVLA `WorkerGroup`, `Channel`, Hydra config, existing `EnvWorker`, existing `RolloutInferenceWorker`, existing `LearnerWorker`, pytest.

---

## Source Comparison

RLinf source topology:

- `rlinf/runners/embodied_runner.py:284-303`
  - `env.interact(input_channel=env_channel, rollout_channel=rollout_channel, actor_channel=actor_channel)`
  - `rollout.generate(input_channel=rollout_channel, output_channel=env_channel)`
  - `actor.recv_rollout_trajectories(input_channel=actor_channel)`
- `rlinf/workers/rollout/hf/huggingface_worker.py:391-456`
  - `MultiStepRolloutWorker.generate_one_epoch` receives env output, predicts actions, sends rollout result.
- `rlinf/workers/env/env_worker.py:50`
  - `EnvWorker` is still an independent Ray worker; env subprocesses are below it.

Current DreamerVLA topology:

- `dreamervla/runners/online_cotrain_ray_runner.py:219-224`
  - launches a separate `RolloutInferenceWorker`.
- `dreamervla/runners/online_cotrain_ray_runner.py:374-416`
  - driver calls `infer.forward_batch`, then calls `EnvWorker.step`.
- `dreamervla/runners/online_cotrain_ray_runner.py:718-781`
  - overlap loop is still driver-mediated.

Target DreamerVLA topology:

```text
OnlineCotrainRayRunner
  ├── Channel("ray_cotrain_env")
  ├── Channel("ray_cotrain_rollout")
  ├── Channel("ray_cotrain_actor")
  ├── EnvWorker.interact(...)
  ├── MultiStepRolloutWorker.generate(...)
  └── LearnerWorker.recv_rollout_trajectories(...)
```

`EnvWorker` keeps owning the LIBERO env slots and spawn children. `MultiStepRolloutWorker` owns the rollout model/inference bundle. The runner no longer sends every observation/action pair itself in the RLinf topology.

## File Structure

- Create `dreamervla/workers/rollout/channel_contracts.py`
  - Message constructors and validation for env-to-rollout, rollout-to-env, and env-to-learner traffic.
- Create `dreamervla/workers/rollout/multistep_rollout_worker.py`
  - RLinf-style rollout worker that wraps `RolloutInferenceWorker`/`InferenceWorker` behavior behind `generate`.
- Modify `dreamervla/workers/rollout/__init__.py`
  - Export `MultiStepRolloutWorker`.
- Modify `dreamervla/workers/env/env_worker.py`
  - Add `interact`.
  - Add an episode sink path so channel topology sends completed episodes to `actor_channel` instead of directly adding to replay.
- Modify `dreamervla/workers/actor/learner_worker.py`
  - Add `recv_rollout_trajectories`.
- Modify `dreamervla/runners/online_cotrain_ray_runner.py`
  - Build channels and launch `MultiStepRolloutWorker`.
  - Add `topology=rlinf_channel` path.
  - Keep legacy driver loop behind `ray_topology=legacy_driver_loop` until the new path is verified.
- Modify configs under `configs/dreamervla/ray_online_cotrain_*.yaml`
  - Add explicit `ray_topology: rlinf_channel`.
  - Add rollout worker target and channel settings.
- Add tests:
  - `tests/unit_tests/test_rollout_channel_contracts.py`
  - `tests/unit_tests/test_multistep_rollout_worker.py`
  - Extend `tests/unit_tests/test_env_worker_record_builder.py`
  - Extend `tests/unit_tests/test_learner_worker_manual_precision.py`
  - Extend `tests/unit_tests/test_online_cotrain_ray_runner.py`

---

### Task 1: Channel Message Contracts

**Files:**
- Create: `dreamervla/workers/rollout/channel_contracts.py`
- Test: `tests/unit_tests/test_rollout_channel_contracts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/test_rollout_channel_contracts.py`:

```python
from dreamervla.workers.rollout.channel_contracts import (
    ACTION_MESSAGE,
    EPISODE_MESSAGE,
    OBS_MESSAGE,
    STOP_MESSAGE,
    action_key,
    episode_message,
    obs_message,
    stop_message,
    validate_action_message,
    validate_obs_message,
)


def test_obs_and_action_messages_round_trip() -> None:
    obs = {"task_description": "open drawer", "step": 3}
    msg = obs_message(env_id=7, worker_rank=1, slot_id=3, obs=obs, env_step=42)

    assert msg["type"] == OBS_MESSAGE
    assert msg["env_id"] == 7
    assert msg["worker_rank"] == 1
    assert msg["slot_id"] == 3
    assert msg["env_step"] == 42
    assert msg["obs"] == obs
    assert action_key(7) == "env:7"
    validate_obs_message(msg)

    action = {
        "type": ACTION_MESSAGE,
        "env_id": 7,
        "action": [0.0] * 7,
        "obs_embedding": None,
        "lang_emb": None,
        "policy_version": 5,
    }
    validate_action_message(action)


def test_episode_and_stop_messages_are_explicit() -> None:
    episode = [{"rewards": 1.0, "dones": True}]

    ep_msg = episode_message(
        env_id=2,
        worker_rank=0,
        slot_id=2,
        episode=episode,
        env_steps=16,
    )
    assert ep_msg["type"] == EPISODE_MESSAGE
    assert ep_msg["episode"] == episode
    assert ep_msg["env_steps"] == 16

    stop = stop_message(worker_rank=0, env_steps=16)
    assert stop == {"type": STOP_MESSAGE, "worker_rank": 0, "env_steps": 16}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_rollout_channel_contracts.py
```

Expected: fail with `ModuleNotFoundError: No module named 'dreamervla.workers.rollout.channel_contracts'`.

- [ ] **Step 3: Implement the contract module**

Create `dreamervla/workers/rollout/channel_contracts.py`:

```python
"""RLinf-style channel message contracts for Ray online rollout."""

from __future__ import annotations

from typing import Any

OBS_MESSAGE = "obs"
ACTION_MESSAGE = "action"
EPISODE_MESSAGE = "episode"
STOP_MESSAGE = "stop"


def action_key(env_id: int) -> str:
    """Return the channel key an EnvWorker slot listens on for actions."""
    return f"env:{int(env_id)}"


def obs_message(
    *,
    env_id: int,
    worker_rank: int,
    slot_id: int,
    obs: dict[str, Any],
    env_step: int,
) -> dict[str, Any]:
    return {
        "type": OBS_MESSAGE,
        "env_id": int(env_id),
        "worker_rank": int(worker_rank),
        "slot_id": int(slot_id),
        "obs": dict(obs),
        "env_step": int(env_step),
    }


def episode_message(
    *,
    env_id: int,
    worker_rank: int,
    slot_id: int,
    episode: list[dict[str, Any]],
    env_steps: int,
) -> dict[str, Any]:
    return {
        "type": EPISODE_MESSAGE,
        "env_id": int(env_id),
        "worker_rank": int(worker_rank),
        "slot_id": int(slot_id),
        "episode": list(episode),
        "env_steps": int(env_steps),
    }


def stop_message(*, worker_rank: int, env_steps: int) -> dict[str, Any]:
    return {
        "type": STOP_MESSAGE,
        "worker_rank": int(worker_rank),
        "env_steps": int(env_steps),
    }


def validate_obs_message(message: dict[str, Any]) -> None:
    if message.get("type") != OBS_MESSAGE:
        raise ValueError(f"expected obs message, got {message.get('type')!r}")
    for key in ("env_id", "worker_rank", "slot_id", "obs", "env_step"):
        if key not in message:
            raise ValueError(f"obs message missing {key!r}")
    if not isinstance(message["obs"], dict):
        raise TypeError("obs message field 'obs' must be a dict")


def validate_action_message(message: dict[str, Any]) -> None:
    if message.get("type") != ACTION_MESSAGE:
        raise ValueError(f"expected action message, got {message.get('type')!r}")
    for key in ("env_id", "action", "policy_version"):
        if key not in message:
            raise ValueError(f"action message missing {key!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_rollout_channel_contracts.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/workers/rollout/channel_contracts.py tests/unit_tests/test_rollout_channel_contracts.py
git commit -s -m "feat: add rollout channel message contracts"
```

---

### Task 2: MultiStepRolloutWorker Skeleton And Batch Inference

**Files:**
- Create: `dreamervla/workers/rollout/multistep_rollout_worker.py`
- Modify: `dreamervla/workers/rollout/__init__.py`
- Test: `tests/unit_tests/test_multistep_rollout_worker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/test_multistep_rollout_worker.py`:

```python
from dreamervla.workers.rollout.channel_contracts import (
    ACTION_MESSAGE,
    STOP_MESSAGE,
    action_key,
    obs_message,
    stop_message,
)


class _MemoryChannel:
    def __init__(self) -> None:
        self.items: dict[str, list[dict]] = {"default": []}

    def put(self, item: dict, *, key: str = "default") -> None:
        self.items.setdefault(key, []).append(item)

    def get(self, *, key: str = "default") -> dict:
        queue = self.items.setdefault(key, [])
        if not queue:
            raise AssertionError(f"empty channel key {key!r}")
        return queue.pop(0)


class _FakeInference:
    def __init__(self) -> None:
        self.initialized = False
        self.calls: list[tuple[list[dict], list[int]]] = []

    def init(self) -> None:
        self.initialized = True

    def forward_batch(self, obs_batch: list[dict], env_ids: list[int]) -> dict:
        self.calls.append((obs_batch, env_ids))
        return {
            "actions": [[float(env_id)] * 7 for env_id in env_ids],
            "obs_embedding": [None for _ in env_ids],
            "lang_emb": [None for _ in env_ids],
        }


def test_multistep_rollout_worker_generates_actions_until_all_env_workers_stop(monkeypatch) -> None:
    from dreamervla.workers.rollout import multistep_rollout_worker as module
    from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker

    created: list[_FakeInference] = []

    def fake_build_worker(*_args, **_kwargs):
        worker = _FakeInference()
        created.append(worker)
        return worker

    monkeypatch.setattr(module, "_build_inference_worker", fake_build_worker)

    input_channel = _MemoryChannel()
    output_channel = _MemoryChannel()
    input_channel.put(obs_message(env_id=0, worker_rank=0, slot_id=0, obs={"x": 1}, env_step=0))
    input_channel.put(obs_message(env_id=1, worker_rank=0, slot_id=1, obs={"x": 2}, env_step=0))
    input_channel.put(stop_message(worker_rank=0, env_steps=2))

    worker = MultiStepRolloutWorker(
        inference_cfg={"kind": "fake"},
        inference_init_ckpt={},
        num_envs=2,
        num_env_workers=1,
        batch_size=2,
    )
    worker.init()
    metrics = worker.generate(input_channel=input_channel, output_channel=output_channel)

    assert created[0].initialized is True
    assert created[0].calls == [([{"x": 1}, {"x": 2}], [0, 1])]
    assert output_channel.get(key=action_key(0))["type"] == ACTION_MESSAGE
    assert output_channel.get(key=action_key(1))["action"] == [1.0] * 7
    assert metrics["rollout/generated_actions"] == 2
    assert metrics["rollout/stop_messages"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_multistep_rollout_worker.py
```

Expected: fail with missing `MultiStepRolloutWorker`.

- [ ] **Step 3: Implement the worker**

Create `dreamervla/workers/rollout/multistep_rollout_worker.py`:

```python
"""RLinf-style rollout worker for DreamerVLA Ray online cotrain."""

from __future__ import annotations

from typing import Any

from dreamervla.scheduler.worker import Worker
from dreamervla.workers.inference.inference_worker import InferenceWorker
from dreamervla.workers.inference.rollout_inference_worker import RolloutInferenceWorker
from dreamervla.workers.rollout.channel_contracts import (
    ACTION_MESSAGE,
    OBS_MESSAGE,
    STOP_MESSAGE,
    action_key,
    validate_obs_message,
)


def _build_inference_worker(
    inference_cfg: dict[str, Any],
    inference_init_ckpt: dict[str, Any],
    num_envs: int,
) -> Any:
    target = str(inference_cfg.get("worker_target", inference_cfg.get("worker", "")))
    if target.endswith("RolloutInferenceWorker"):
        return RolloutInferenceWorker(inference_cfg, inference_init_ckpt, num_envs=num_envs)
    return InferenceWorker(inference_cfg, inference_init_ckpt)


class MultiStepRolloutWorker(Worker):
    """Consume env observations from a channel and send actions back by env key."""

    def __init__(
        self,
        inference_cfg: dict[str, Any],
        inference_init_ckpt: dict[str, Any],
        num_envs: int,
        num_env_workers: int,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.inference_cfg = dict(inference_cfg)
        self.inference_init_ckpt = dict(inference_init_ckpt)
        self.num_envs = int(num_envs)
        self.num_env_workers = int(num_env_workers)
        self.batch_size = int(batch_size or num_envs)
        self.inference: Any | None = None
        self.policy_version = 0

    def init(self) -> None:
        self.inference = _build_inference_worker(
            self.inference_cfg,
            self.inference_init_ckpt,
            self.num_envs,
        )
        self.inference.local_rank = getattr(self, "local_rank", 0)
        self.inference.rank = getattr(self, "rank", 0)
        self.inference.world_size = getattr(self, "world_size", 1)
        self.inference.device = getattr(self, "device", "cpu")
        self.inference.init()

    def generate(self, input_channel: Any, output_channel: Any) -> dict[str, float | int]:
        """Run until every EnvWorker sends one stop message."""
        stop_messages = 0
        generated_actions = 0
        pending: list[dict[str, Any]] = []
        while stop_messages < self.num_env_workers:
            message = input_channel.get()
            message_type = str(message.get("type"))
            if message_type == STOP_MESSAGE:
                stop_messages += 1
                continue
            if message_type != OBS_MESSAGE:
                raise ValueError(f"unexpected rollout channel message: {message_type!r}")
            validate_obs_message(message)
            pending.append(message)
            if len(pending) >= self.batch_size:
                generated_actions += self._flush(pending, output_channel)
                pending.clear()
        if pending:
            generated_actions += self._flush(pending, output_channel)
        return {
            "rollout/generated_actions": int(generated_actions),
            "rollout/stop_messages": int(stop_messages),
        }

    def _flush(self, messages: list[dict[str, Any]], output_channel: Any) -> int:
        inference = self._require_inference()
        env_ids = [int(message["env_id"]) for message in messages]
        obs_batch = [dict(message["obs"]) for message in messages]
        result = inference.forward_batch(obs_batch, env_ids)
        actions = list(result["actions"])
        hidden_batch = list(result.get("obs_embedding", [None] * len(env_ids)))
        lang_batch = list(result.get("lang_emb", [None] * len(env_ids)))
        while len(hidden_batch) < len(env_ids):
            hidden_batch.append(None)
        while len(lang_batch) < len(env_ids):
            lang_batch.append(None)
        for env_id, action, hidden, lang_emb in zip(
            env_ids,
            actions,
            hidden_batch,
            lang_batch,
            strict=True,
        ):
            output_channel.put(
                {
                    "type": ACTION_MESSAGE,
                    "env_id": int(env_id),
                    "action": action,
                    "obs_embedding": hidden,
                    "lang_emb": lang_emb,
                    "policy_version": int(self.policy_version),
                },
                key=action_key(env_id),
            )
        return len(env_ids)

    def _require_inference(self) -> Any:
        if self.inference is None:
            raise RuntimeError("MultiStepRolloutWorker.init() has not been called")
        return self.inference
```

Modify `dreamervla/workers/rollout/__init__.py`:

```python
"""Rollout workers and helpers."""

from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker

__all__ = ["MultiStepRolloutWorker"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_multistep_rollout_worker.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/workers/rollout/multistep_rollout_worker.py dreamervla/workers/rollout/__init__.py tests/unit_tests/test_multistep_rollout_worker.py
git commit -s -m "feat: add multistep rollout worker"
```

---

### Task 3: EnvWorker Channel Interaction

**Files:**
- Modify: `dreamervla/workers/env/env_worker.py`
- Test: `tests/unit_tests/test_env_worker_record_builder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/test_env_worker_record_builder.py`:

```python
from dreamervla.workers.rollout.channel_contracts import (
    ACTION_MESSAGE,
    EPISODE_MESSAGE,
    STOP_MESSAGE,
    action_key,
)


class _MemoryChannel:
    def __init__(self) -> None:
        self.items: dict[str, list[dict]] = {"default": []}

    def put(self, item: dict, *, key: str = "default") -> None:
        self.items.setdefault(key, []).append(item)

    def get(self, *, key: str = "default") -> dict:
        queue = self.items.setdefault(key, [])
        if not queue:
            raise AssertionError(f"empty channel key {key!r}")
        return queue.pop(0)


def test_env_worker_interact_uses_channels_instead_of_driver_step(monkeypatch) -> None:
    from dreamervla.workers.env.env_worker import EnvWorker

    env_cfg = {
        "target": "dreamervla.workers.env._test_envs:CounterEnv",
        "kwargs": {"horizon": 2, "image_shape": (4, 4, 3), "embedding_dim": 4},
    }
    worker = EnvWorker(env_cfg=env_cfg, task_id=0, replay=None)
    worker.local_rank = 0
    worker.init()

    env_channel = _MemoryChannel()
    rollout_channel = _MemoryChannel()
    actor_channel = _MemoryChannel()

    def fake_rollout_get(*, key: str = "default") -> dict:
        obs_request = rollout_channel.get()
        env_id = int(obs_request["env_id"])
        env_channel.put(
            {
                "type": ACTION_MESSAGE,
                "env_id": env_id,
                "action": [0.0] * 7,
                "obs_embedding": None,
                "lang_emb": None,
                "policy_version": 0,
            },
            key=action_key(env_id),
        )
        return env_channel.get(key=key)

    monkeypatch.setattr(env_channel, "get", fake_rollout_get)

    metrics = worker.interact(
        input_channel=env_channel,
        rollout_channel=rollout_channel,
        actor_channel=actor_channel,
        max_env_steps=2,
        base_env_id=0,
    )

    episode_msg = actor_channel.get()
    stop_msg = actor_channel.get()
    rollout_stop_msg = rollout_channel.get()

    assert metrics["rollout/env_steps"] == 2
    assert episode_msg["type"] == EPISODE_MESSAGE
    assert len(episode_msg["episode"]) == 2
    assert stop_msg["type"] == STOP_MESSAGE
    assert rollout_stop_msg["type"] == STOP_MESSAGE
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_env_worker_record_builder.py::test_env_worker_interact_uses_channels_instead_of_driver_step
```

Expected: fail with `AttributeError: 'EnvWorker' object has no attribute 'interact'`.

- [ ] **Step 3: Implement `EnvWorker.interact`**

Modify imports in `dreamervla/workers/env/env_worker.py`:

```python
from dreamervla.workers.rollout.channel_contracts import (
    EPISODE_MESSAGE,
    STOP_MESSAGE,
    action_key,
    episode_message,
    obs_message,
    stop_message,
    validate_action_message,
)
```

Add methods inside `EnvWorker`:

```python
    def interact(
        self,
        input_channel: Any,
        rollout_channel: Any,
        actor_channel: Any,
        max_env_steps: int,
        base_env_id: int = 0,
    ) -> dict[str, float | int]:
        """RLinf-style env loop: send obs, receive actions, step env slots."""
        del input_channel  # action messages are keyed in this channel by env id.
        max_steps = int(max_env_steps)
        env_steps = 0
        episodes = 0
        while env_steps < max_steps:
            for slot_id in range(self.num_envs):
                if env_steps >= max_steps:
                    break
                env_id = int(base_env_id) + int(slot_id)
                obs = self._slot_obs(slot_id)
                rollout_channel.put(
                    obs_message(
                        env_id=env_id,
                        worker_rank=int(self.local_rank),
                        slot_id=slot_id,
                        obs=obs,
                        env_step=env_steps,
                    )
                )
                action_message = input_channel.get(key=action_key(env_id))
                validate_action_message(action_message)
                _next_obs, done, _info = self.step_slot(
                    slot_id,
                    action_message["action"],
                    action_message.get("obs_embedding"),
                    action_message.get("lang_emb"),
                    {
                        "global_step": 0,
                        "env_step": env_steps + 1,
                        "policy_version": int(action_message.get("policy_version", 0)),
                    },
                    episode_sink=actor_channel,
                    env_id=env_id,
                )
                env_steps += 1
                episodes += int(done)
        rollout_channel.put(
            stop_message(worker_rank=int(self.local_rank), env_steps=env_steps)
        )
        actor_channel.put(stop_message(worker_rank=int(self.local_rank), env_steps=env_steps))
        return {"rollout/env_steps": int(env_steps), "rollout/episodes": int(episodes)}

    def _slot_obs(self, slot_id: int) -> dict[str, Any]:
        if self.num_envs > 1:
            obs = self._obs_by_slot[int(slot_id)]
            if obs is None:
                raise RuntimeError("EnvWorker.init() has not been called")
            return dict(obs)
        if self.obs is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        return dict(self.obs)
```

Change `step` and `step_slot` signatures to accept an optional sink:

```python
    def step_slot(
        self,
        slot_id: int,
        action: Any,
        obs_embedding: Any = None,
        lang_emb: Any | None = None,
        step_metadata: dict[str, Any] | None = None,
        episode_sink: Any | None = None,
        env_id: int | None = None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
```

Update `_step_spawn_slot` and `_step_inproc` to call:

```python
self._finish_episode(slot_id=slot_id, episode_sink=episode_sink, env_id=env_id)
```

Add:

```python
    def _finish_episode(
        self,
        *,
        slot_id: int = 0,
        episode_sink: Any | None = None,
        env_id: int | None = None,
    ) -> None:
        episode = self._episodes_by_slot[int(slot_id)] if self._spawned else self.episode
        if episode_sink is None:
            self._push_episode(self.replay, episode)
        else:
            episode_sink.put(
                episode_message(
                    env_id=int(env_id if env_id is not None else slot_id),
                    worker_rank=int(self.local_rank),
                    slot_id=int(slot_id),
                    episode=list(episode),
                    env_steps=sum(len(ep) for ep in self._episodes_by_slot)
                    if self._spawned
                    else len(self.episode),
                )
            )
        self._push_episode(self.dump, episode)
```

Keep `_add_episode_to_replay` for legacy driver code, but implement it through `_finish_episode`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_env_worker_record_builder.py::test_env_worker_interact_uses_channels_instead_of_driver_step
```

Expected: pass.

- [ ] **Step 5: Run legacy EnvWorker tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q \
  tests/unit_tests/test_env_worker_record_builder.py \
  tests/unit_tests/test_env_worker_world_model_sync.py \
  tests/unit_tests/test_env_worker_spawn_recovery.py
```

Expected: pass. Legacy `step` behavior must still push completed episodes to replay.

- [ ] **Step 6: Commit**

```bash
git add dreamervla/workers/env/env_worker.py tests/unit_tests/test_env_worker_record_builder.py
git commit -s -m "feat: add channel interaction to env worker"
```

---

### Task 4: LearnerWorker Trajectory Receiver

**Files:**
- Modify: `dreamervla/workers/actor/learner_worker.py`
- Test: `tests/unit_tests/test_learner_worker_manual_precision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/test_learner_worker_manual_precision.py`:

```python
from dreamervla.workers.rollout.channel_contracts import episode_message, stop_message


class _MemoryChannel:
    def __init__(self, items: list[dict]) -> None:
        self.items = list(items)

    def get(self, *, key: str = "default") -> dict:
        del key
        if not self.items:
            raise AssertionError("empty channel")
        return self.items.pop(0)


class _Replay:
    def __init__(self) -> None:
        self.episodes: list[list[dict]] = []

    def add_episode(self, episode: list[dict]) -> None:
        self.episodes.append(list(episode))


def test_learner_worker_receives_rollout_trajectories_into_replay() -> None:
    from dreamervla.workers.actor.learner_worker import LearnerWorker

    replay = _Replay()
    worker = LearnerWorker.__new__(LearnerWorker)
    worker.replay = replay

    channel = _MemoryChannel(
        [
            episode_message(
                env_id=0,
                worker_rank=0,
                slot_id=0,
                episode=[{"rewards": 1.0, "dones": True}],
                env_steps=1,
            ),
            stop_message(worker_rank=0, env_steps=1),
        ]
    )

    metrics = worker.recv_rollout_trajectories(
        input_channel=channel,
        expected_stop_messages=1,
    )

    assert replay.episodes == [[{"rewards": 1.0, "dones": True}]]
    assert metrics["rollout/received_episodes"] == 1
    assert metrics["rollout/receiver_stop_messages"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_learner_worker_manual_precision.py::test_learner_worker_receives_rollout_trajectories_into_replay
```

Expected: fail with missing `recv_rollout_trajectories`.

- [ ] **Step 3: Implement receiver**

Modify `dreamervla/workers/actor/learner_worker.py`:

```python
    def recv_rollout_trajectories(
        self,
        input_channel: Any,
        expected_stop_messages: int = 1,
    ) -> dict[str, float | int]:
        """RLinf-compatible trajectory receiver.

        EnvWorker sends completed episodes here in channel topology. The learner
        writes them into the existing replay so current update code remains unchanged.
        """
        stop_messages = 0
        received_episodes = 0
        while stop_messages < int(expected_stop_messages):
            message = input_channel.get()
            message_type = str(message.get("type"))
            if message_type == "stop":
                stop_messages += 1
                continue
            if message_type != "episode":
                raise ValueError(f"unexpected learner channel message: {message_type!r}")
            episode = list(message["episode"])
            add_episode = self.replay.add_episode
            remote = getattr(add_episode, "remote", None)
            if remote is not None:
                ray.get(remote(episode))
            else:
                add_episode(episode)
            received_episodes += 1
        return {
            "rollout/received_episodes": int(received_episodes),
            "rollout/receiver_stop_messages": int(stop_messages),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_learner_worker_manual_precision.py::test_learner_worker_receives_rollout_trajectories_into_replay
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/workers/actor/learner_worker.py tests/unit_tests/test_learner_worker_manual_precision.py
git commit -s -m "feat: receive rollout trajectories through learner worker"
```

---

### Task 5: Runner Builds RLinf Channel Topology

**Files:**
- Modify: `dreamervla/runners/online_cotrain_ray_runner.py`
- Modify: `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml`
- Modify: `configs/dreamervla/ray_online_cotrain_oft_backbone_latent.yaml`
- Modify: `configs/dreamervla/ray_online_cotrain_rynn_action_hidden.yaml`
- Test: `tests/unit_tests/test_online_cotrain_ray_runner.py`

- [ ] **Step 1: Write the failing topology build test**

Append to `tests/unit_tests/test_online_cotrain_ray_runner.py`:

```python
def test_ray_runner_builds_multistep_rollout_worker_for_rlinf_channel_topology(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners import online_cotrain_ray_runner as module
    from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker

    launched: list[type] = []

    class _Group:
        def __init__(self, cls, *args, **kwargs):
            self.cls = cls
            self.args = args
            self.kwargs = kwargs
            self.workers = [_ReadyActor()]

        def launch(self, *_args, **_kwargs):
            launched.append(self.cls)
            return self

    class _Cluster:
        def require_single_node(self):
            return None

    monkeypatch.setattr(module, "WorkerGroup", _Group)
    monkeypatch.setattr(module, "NodePlacementStrategy", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "PackedPlacementStrategy", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "FlexiblePlacementStrategy", lambda *_args, **_kwargs: object())

    runner = module.OnlineCotrainRayRunner.__new__(module.OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "ray_topology": "rlinf_channel",
            "rollout": {"steps": 4},
            "env": {"num_workers": 1, "envs_per_worker": 1},
            "inference": {"worker_target": "dreamervla.workers.inference.rollout_inference_worker:RolloutInferenceWorker"},
        }
    )
    runner._restore_ray_resume_state = lambda _groups: None
    runner._build_rollout_dump_group = lambda *_args, **_kwargs: (None, None, None)
    runner._oft_worker_plan = lambda: {
        "env": {"target": "dreamervla.workers.env._test_envs:CounterEnv", "kwargs": {}},
        "inference": {"worker_target": "dreamervla.workers.inference.rollout_inference_worker:RolloutInferenceWorker"},
    }

    groups = runner._build_components(_Cluster())

    assert MultiStepRolloutWorker in launched
    assert "rollout" in groups
    assert "infer" not in groups
```

`_ReadyActor` can reuse the existing ready fake actor class in the same test file. If it is not available at module scope, add:

```python
class _ReadyActor:
    def init(self):
        return _Ready([None])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_builds_multistep_rollout_worker_for_rlinf_channel_topology
```

Expected: fail because `_build_components` still launches `RolloutInferenceWorker` under `"infer"`.

- [ ] **Step 3: Modify component build**

In `dreamervla/runners/online_cotrain_ray_runner.py`, import:

```python
from dreamervla.scheduler.channel import Channel
from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker
```

Add:

```python
    def _ray_topology(self) -> str:
        return str(self._select_first(("ray_topology", "runner.ray_topology"), "rlinf_channel"))

    def _uses_rlinf_channel_topology(self) -> bool:
        return self._ray_topology() == "rlinf_channel"
```

In `_build_components`, replace the inference worker launch block with a branch:

```python
        if self._uses_rlinf_channel_topology():
            rollout_group = WorkerGroup(
                MultiStepRolloutWorker,
                infer_cfg,
                infer_init_ckpt,
                num_envs=num_envs,
                num_env_workers=num_env_workers,
                batch_size=int(self._select_first(("rollout.infer_batch_size",), num_envs)),
            ).launch(cluster, inference_placement)
        else:
            infer_group = WorkerGroup(
                infer_worker_cls,
                infer_cfg,
                infer_init_ckpt,
                num_envs=num_envs,
            ).launch(cluster, inference_placement)
```

Set `groups` like this:

```python
        groups = {
            "replay": replay_group,
            "envs": env_group,
            "learner": learner_group,
            "store_name": store_name,
            "num_envs": num_envs,
            "num_env_workers": num_env_workers,
            "envs_per_worker": self._effective_envs_per_worker(),
            "ray_topology": self._ray_topology(),
        }
        if self._uses_rlinf_channel_topology():
            groups["rollout"] = rollout_group
        else:
            groups["infer"] = infer_group
```

Update configs:

```yaml
ray_topology: rlinf_channel
rollout:
  infer_batch_size: ${env.total_num_envs}
```

If `env.total_num_envs` is absent in a config, use the explicit logical count already implied by `env.num_workers * env.envs_per_worker`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_builds_multistep_rollout_worker_for_rlinf_channel_topology
```

Expected: pass.

- [ ] **Step 5: Run config composition tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py -k "experiment_accepts_render_backend_override or backbone_latent or rlinf_channel"
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add dreamervla/runners/online_cotrain_ray_runner.py configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml configs/dreamervla/ray_online_cotrain_oft_backbone_latent.yaml configs/dreamervla/ray_online_cotrain_rynn_action_hidden.yaml tests/unit_tests/test_online_cotrain_ray_runner.py
git commit -s -m "feat: build rlinf channel rollout topology"
```

---

### Task 6: Runner Executes Channel Handles

**Files:**
- Modify: `dreamervla/runners/online_cotrain_ray_runner.py`
- Test: `tests/unit_tests/test_online_cotrain_ray_runner.py`

- [ ] **Step 1: Write failing execution test**

Append:

```python
def test_ray_runner_channel_loop_invokes_rlinf_style_worker_methods(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    calls: list[str] = []

    class _Handle:
        def __init__(self, value):
            self.value = value

        def wait(self):
            return self.value

    class _Env:
        def interact(self, **kwargs):
            calls.append("env.interact")
            assert {"input_channel", "rollout_channel", "actor_channel"} <= set(kwargs)
            return _Handle({"rollout/env_steps": 4, "rollout/episodes": 1})

    class _Rollout:
        def generate(self, **kwargs):
            calls.append("rollout.generate")
            assert {"input_channel", "output_channel"} <= set(kwargs)
            return _Handle({"rollout/generated_actions": 4})

    class _Learner:
        def recv_rollout_trajectories(self, **kwargs):
            calls.append("learner.recv_rollout_trajectories")
            assert {"input_channel", "expected_stop_messages"} <= set(kwargs)
            return _Handle({"rollout/received_episodes": 1})

        def update(self, phase, num_steps):
            calls.append(f"learner.update:{phase}:{num_steps}")
            return _Handle({"train/rl_loss": 0.25, "train/ppo_updates": 1})

    class _Replay:
        def ready(self, **_kwargs):
            return _Handle([True])

    class _Channel:
        @classmethod
        def create(cls, name, maxsize=0):
            del name, maxsize
            return cls()

    monkeypatch.setattr("dreamervla.runners.online_cotrain_ray_runner.Channel", _Channel)

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "ray_topology": "rlinf_channel",
            "rollout": {"steps": 4, "min_replay_episodes": 1},
            "training": {"max_steps": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )

    metrics = runner._run_loop(
        {
            "ray_topology": "rlinf_channel",
            "envs": type("EnvGroup", (), {"workers": [_Env()]})(),
            "rollout": type("RolloutGroup", (), {"workers": [_Rollout()]})(),
            "learner": type("LearnerGroup", (), {"workers": [_Learner()]})(),
            "replay": _Replay(),
            "num_envs": 1,
            "num_env_workers": 1,
            "envs_per_worker": 1,
        }
    )

    assert calls[:3] == [
        "env.interact",
        "rollout.generate",
        "learner.recv_rollout_trajectories",
    ]
    assert "learner.update:cotrain:1" in calls
    assert metrics["rollout/generated_actions"] == 4
    assert metrics["rollout/received_episodes"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_channel_loop_invokes_rlinf_style_worker_methods
```

Expected: fail because `_run_loop` does not dispatch channel handles.

- [ ] **Step 3: Implement channel loop**

In `_run_loop`, branch before legacy loops:

```python
            if groups.get("ray_topology") == "rlinf_channel":
                result = self._run_loop_rlinf_channel(groups)
            elif not _uses_ray_worker_groups(groups):
                result = self._run_loop_sync(groups)
            else:
                result = self._run_loop_overlap(groups)
```

Add:

```python
    def _run_loop_rlinf_channel(self, groups: dict[str, Any]) -> dict[str, float | int]:
        env_channel = Channel.create("ray_cotrain_env_channel")
        rollout_channel = Channel.create("ray_cotrain_rollout_channel")
        actor_channel = Channel.create("ray_cotrain_actor_channel")

        env_worker = groups["envs"].workers[0]
        rollout_worker = groups["rollout"].workers[0]
        learner_worker = groups["learner"].workers[0]
        replay = groups["replay"]
        target_env_steps = int(self._select_first(("rollout.steps",), 1))
        target_updates = int(self._select_first(("training.max_steps",), 1))
        learner_phase = self._learner_phase()

        env_handle = env_worker.interact(
            input_channel=env_channel,
            rollout_channel=rollout_channel,
            actor_channel=actor_channel,
            max_env_steps=target_env_steps,
            base_env_id=0,
        )
        rollout_handle = rollout_worker.generate(
            input_channel=rollout_channel,
            output_channel=env_channel,
        )
        recv_handle = learner_worker.recv_rollout_trajectories(
            input_channel=actor_channel,
            expected_stop_messages=int(groups.get("num_env_workers", 1)),
        )

        env_metrics = dict(env_handle.wait())
        rollout_metrics = dict(rollout_handle.wait())
        recv_metrics = dict(recv_handle.wait())

        learner_updates = 0
        train_metrics: dict[str, Any] = {}
        replay_ready_kwargs = self._replay_ready_kwargs()
        while learner_updates < target_updates and replay.ready(**replay_ready_kwargs).wait()[0]:
            update_metrics = learner_worker.update(learner_phase, 1).wait()
            train_metrics.update(dict(update_metrics or {}))
            learner_updates += 1

        metrics: dict[str, float | int] = {}
        metrics.update(env_metrics)
        metrics.update(rollout_metrics)
        metrics.update(recv_metrics)
        metrics.update(train_metrics)
        metrics["rollout/steps"] = int(env_metrics.get("rollout/env_steps", target_env_steps))
        metrics["train/learner_updates"] = int(learner_updates)
        metrics["sync/policy_version"] = int(getattr(self, "_policy_version", 0))
        return metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_channel_loop_invokes_rlinf_style_worker_methods
```

Expected: pass.

- [ ] **Step 5: Run online cotrain Ray runner unit subset**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add dreamervla/runners/online_cotrain_ray_runner.py tests/unit_tests/test_online_cotrain_ray_runner.py
git commit -s -m "feat: execute rlinf channel topology in ray runner"
```

---

### Task 7: Topology Comparison Guard And Documentation

**Files:**
- Modify: `docs/architecture/ray_online_cotrain_backend.md`
- Modify: `docs/superpowers/TODO_IMPLEMENTATION_SUMMARY.zh-CN.md`
- Test: `tests/unit_tests/test_online_cotrain_ray_runner.py`

- [ ] **Step 1: Add a source-level topology assertion test**

Append:

```python
def test_rlinf_channel_topology_has_no_driver_forward_then_step_loop() -> None:
    import inspect

    import dreamervla.runners.online_cotrain_ray_runner as module

    channel_src = inspect.getsource(module.OnlineCotrainRayRunner._run_loop_rlinf_channel)
    assert ".interact(" in channel_src
    assert ".generate(" in channel_src
    assert ".recv_rollout_trajectories(" in channel_src
    assert "forward_batch(" not in channel_src
    assert "_env_step(" not in channel_src
```

- [ ] **Step 2: Run test to verify it passes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_online_cotrain_ray_runner.py::test_rlinf_channel_topology_has_no_driver_forward_then_step_loop
```

Expected: pass.

- [ ] **Step 3: Update Ray backend docs**

In `docs/architecture/ray_online_cotrain_backend.md`, add a section:

```markdown
## RLinf Channel Topology

The default Ray online cotrain topology mirrors RLinf's embodied runner control
flow:

```text
EnvWorker.interact
  -> rollout channel
MultiStepRolloutWorker.generate
  -> env channel
LearnerWorker.recv_rollout_trajectories
  -> replay
LearnerWorker.update
```

`EnvWorker` still owns real LIBERO env slots and EGL spawn children. The key change
from the legacy DreamerVLA Ray loop is that the driver no longer performs every
`forward_batch -> env.step` transition itself; it starts worker handles and waits
for the channelized rollout epoch to complete.
```

- [ ] **Step 4: Update summary doc**

In `docs/superpowers/TODO_IMPLEMENTATION_SUMMARY.zh-CN.md`, add a short note under Ray backend tasks:

```markdown
Ray online cotrain 的目标拓扑是 RLinf channel topology:
`EnvWorker.interact -> MultiStepRolloutWorker.generate -> LearnerWorker.recv_rollout_trajectories`。
旧 driver-mediated `infer.forward_batch -> env.step` loop 只保留为验证期 fallback。
```

- [ ] **Step 5: Run docs/link checks**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re
files = [p for p in Path("docs").rglob("*.md")]
link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
missing = []
for path in files:
    text = path.read_text(encoding="utf-8")
    for m in link_re.finditer(text):
        target = m.group(1).split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if "://" in target or any(ch in target for ch in "$*?"):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            missing.append((path, target))
if missing:
    for path, target in missing:
        print(f"{path}: missing {target}")
    raise SystemExit(1)
print("docs links ok")
PY
```

Expected: `docs links ok`.

- [ ] **Step 6: Commit**

```bash
git add docs/architecture/ray_online_cotrain_backend.md docs/superpowers/TODO_IMPLEMENTATION_SUMMARY.zh-CN.md tests/unit_tests/test_online_cotrain_ray_runner.py
git commit -s -m "docs: document rlinf channel ray topology"
```

---

### Task 8: Verification Matrix

**Files:**
- No new source files.
- Runs tests and static checks.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q \
  tests/unit_tests/test_rollout_channel_contracts.py \
  tests/unit_tests/test_multistep_rollout_worker.py \
  tests/unit_tests/test_env_worker_record_builder.py \
  tests/unit_tests/test_learner_worker_manual_precision.py \
  tests/unit_tests/test_online_cotrain_ray_runner.py
```

Expected: pass.

- [ ] **Step 2: Run repository hygiene**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m pytest -q tests/unit_tests/test_repository_hygiene.py
```

Expected: pass.

- [ ] **Step 3: Run static old-topology search**

Run:

```bash
rg -n "_run_loop_rlinf_channel|MultiStepRolloutWorker|recv_rollout_trajectories|ray_topology" \
  dreamervla tests configs docs
```

Expected: output includes the new worker, runner branch, tests, and docs.

- [ ] **Step 4: Run a CPU/synthetic Ray smoke**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python \
  -m dreamervla.train \
  experiment=online_cotrain_ray_world_model_env_tiny \
  ray_topology=rlinf_channel \
  logger=tensorboard \
  training.max_steps=1 \
  rollout.steps=4 \
  training.out_dir=/tmp/dvla_rlinf_channel_topology_smoke
```

Expected: run completes and logs metrics containing:

```text
rollout/generated_actions
rollout/received_episodes
train/learner_updates
```

- [ ] **Step 5: Run EGL gated smoke when a GPU is free**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_NVLS_ENABLE=0 PYTHONPATH=. \
  /home/user01/miniconda3/envs/dreamervla/bin/python -m dreamervla.train \
  experiment=online_cotrain_ray_oft_action_hidden \
  ray_topology=rlinf_channel \
  logger=tensorboard \
  render_backend=egl \
  env.num_workers=1 \
  env.envs_per_worker=4 \
  rollout.steps=32 \
  rollout.min_replay_episodes=999999 \
  rollout.min_replay_transitions=999999 \
  training.max_steps=1 \
  training.out_dir=/tmp/dvla_rlinf_channel_egl_smoke
```

Expected:

```text
ONLINE COTRAIN (ray)
rollout/generated_actions
FINAL METRICS
```

Negative grep on the run log:

```text
EOFError
egl spawn child died
SIGABRT
Traceback
```

None of those strings should appear.

- [ ] **Step 6: Final commit**

```bash
git status --short
git add dreamervla tests configs docs .gitignore
git commit -s -m "feat: align ray cotrain topology with rlinf channels"
```

---

## Self-Review

- Spec coverage: The plan covers source comparison, contracts, rollout worker, EnvWorker channel interaction, LearnerWorker trajectory ingestion, runner channel orchestration, config enablement, docs, and verification.
- Placeholder scan: The plan avoids unspecified implementation markers and gives concrete file paths, tests, commands, and expected outcomes.
- Type consistency: Message fields are consistent across tasks: `type`, `env_id`, `worker_rank`, `slot_id`, `obs`, `action`, `obs_embedding`, `lang_emb`, `policy_version`, `episode`, `env_steps`.
- Boundary check: `EnvWorker` remains the owner of real env slots and EGL spawn children. `MultiStepRolloutWorker` owns rollout inference. `LearnerWorker` writes received episodes into existing replay so PPO/WM/classifier update internals remain unchanged.
