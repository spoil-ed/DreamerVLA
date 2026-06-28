# manual cotrain Cotrain Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and launch the target Ray cotrain topology described in `spec/99_manual_notes.md`, with separated LearnerGroup, ActorGroup, RolloutGroup, and EnvGroup, and with a runnable short cotrain cycle for 0-5 GPUs.

**Architecture:** Add a new target route instead of renaming the existing `OnlineCotrainRayRunner`: the old runner remains available for compatibility, while `ManualCotrainRayRunner` builds the four manual-notes groups and drives a RLinf/manual cotrain-style channel loop. `LearnerWorker` is restricted to world-model/classifier updates in this route; VLA PPO lives only in `EmbodiedFSDPActor`; rollout inference uses non-FSDP `MultiStepRolloutWorker`; environment interaction is done by `RealEnvWorker` and `WMEnvWorker`.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, Ray, DreamerVLA `WorkerGroup`/`Channel`, PyTorch/FSDP via `FSDPModelManager`, existing OpenVLA-OFT actor/world-model/classifier configs, pytest.

---

## Source Of Truth

- `spec/99_manual_notes.md` is the implementation target.
- `spec/02_ray.md` defines the same target graph and explicitly lists the current-code gap.
- Existing `OnlineCotrainRayRunner` is only a reference for config loading, checkpoints, logging, rollout dump, and Ray startup patterns.
- Existing `dreamervla/algorithms/ppo/grpo.py` PPO helpers are reused for ratio/clip/group advantage; do not duplicate PPO math.
- Existing `dreamervla/hybrid_engines/weight_syncer/patch.py` is reused for Actor -> Rollout patch sync.

## Required Current-vs-Target Correction

The current Ray route is not target-compliant because `LearnerWorker` owns policy RL updates, rollout uses `RolloutInferenceWorker.forward_batch`, EnvWorkers are all real-env style, and Actor -> Rollout patch sync is a no-op. The implementation below must make the target route observable in code and logs:

```text
LearnerGroup   -> LearnerWorker(mode=wm_classifier_only), no policy optimizer
ActorGroup     -> EmbodiedFSDPActor, policy optimizer/backward/FSDP
RolloutGroup   -> MultiStepRolloutWorker, no_grad HF/BasePolicy/actor-copy inference
EnvGroup       -> RealEnvWorker on the first device, WMEnvWorker on remaining devices
```

## File Map

- Create: `dreamervla/workers/cotrain/__init__.py`
- Create: `dreamervla/workers/cotrain/messages.py`
  - Typed channel messages and trajectory collation helpers.
- Create: `dreamervla/workers/cotrain/placement.py`
  - Manual-notes placement planner for 0-5 GPUs and any larger local count.
- Create: `dreamervla/workers/rollout/multistep_rollout_worker.py`
  - Non-FSDP rollout policy copy, OFT encoder/hidden extraction adapter, policy sampling/evaluation inputs, patch sync.
- Create: `dreamervla/workers/env/trajectory_env_worker.py`
  - EnvWorker interface plus `RealEnvWorker` and `WMEnvWorker` target interaction loop.
- Create: `dreamervla/workers/actor/embodied_fsdp_actor.py`
  - FSDP actor worker, trajectory receive, advantage/return computation, PPO update, Actor -> Rollout sync.
- Modify: `dreamervla/workers/actor/learner_worker.py`
  - Add `wm_classifier_only` mode and validation that this mode never builds or trains a policy.
- Create: `dreamervla/runners/manual_cotrain_ray_runner.py`
  - Target runner that builds the four groups and drives the global-step loop.
- Modify: `dreamervla/runners/__init__.py`
  - Export `ManualCotrainRayRunner`.
- Create: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`
  - Target route config aligned with the current OFT backbone latent recipe and RLinf manual cotrain hyperparameters.
- Create: `configs/experiment/manual_cotrain_ray_oft_backbone_latent.yaml`
  - Experiment entry.
- Modify: `configs/scripts/coldstart_warmup_cotrain.yaml`
  - Set `cotrain_async_experiment` to the target route after implementation.
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`
  - Generate target placement overrides for 0-5 GPUs and keep the old route selectable only by explicit override.
- Create: `tests/unit_tests/test_cotrain_messages.py`
- Create: `tests/unit_tests/test_manual_cotrain_placement.py`
- Create: `tests/unit_tests/test_multistep_rollout_worker.py`
- Create: `tests/unit_tests/test_trajectory_env_worker.py`
- Create: `tests/unit_tests/test_embodied_fsdp_actor.py`
- Create: `tests/unit_tests/test_manual_cotrain_ray_runner.py`
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`

---

### Task 1: Typed Cotrain Messages

**Files:**
- Create: `dreamervla/workers/cotrain/__init__.py`
- Create: `dreamervla/workers/cotrain/messages.py`
- Test: `tests/unit_tests/test_cotrain_messages.py`

- [ ] **Step 1: Write failing message tests**

Add `tests/unit_tests/test_cotrain_messages.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from dreamervla.workers.cotrain.messages import (
    ObservationMsg,
    RolloutResultMsg,
    StopMsg,
    TrajectoryShard,
    collate_trajectory_shards,
)


def test_rollout_result_keeps_forward_inputs_and_versions() -> None:
    msg = RolloutResultMsg(
        env_rank=2,
        slot_id=3,
        task_id=4,
        episode_id=5,
        step=6,
        actions=np.zeros((2, 7), dtype=np.float32),
        prev_logprobs=np.array([0.1], dtype=np.float32),
        prev_values=None,
        forward_inputs={"hidden": np.ones((1, 4), dtype=np.float32)},
        versions={"policy": 9},
    )

    assert msg.key == "2:3"
    assert msg.versions["policy"] == 9
    assert msg.forward_inputs["hidden"].shape == (1, 4)


def test_collate_trajectory_shards_stacks_steps_and_batch() -> None:
    shards = [
        TrajectoryShard(
            env_rank=0,
            slot_id=0,
            task_id=0,
            episode_ids=[10],
            actions=torch.ones(2, 1, 3),
            rewards=torch.tensor([[0.0], [1.0]]),
            dones=torch.tensor([[False], [True]]),
            prev_logprobs=torch.zeros(2, 1),
            prev_values=None,
            forward_inputs={"hidden": torch.ones(2, 1, 4)},
            versions={"policy": torch.ones(2, 1, dtype=torch.long)},
        ),
        TrajectoryShard(
            env_rank=1,
            slot_id=0,
            task_id=1,
            episode_ids=[20],
            actions=torch.full((2, 1, 3), 2.0),
            rewards=torch.tensor([[1.0], [0.0]]),
            dones=torch.tensor([[False], [True]]),
            prev_logprobs=torch.zeros(2, 1),
            prev_values=None,
            forward_inputs={"hidden": torch.full((2, 1, 4), 2.0)},
            versions={"policy": torch.full((2, 1), 2, dtype=torch.long)},
        ),
    ]

    batch = collate_trajectory_shards(shards)

    assert batch.actions.shape == (2, 2, 3)
    assert batch.rewards.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert batch.forward_inputs["hidden"].shape == (2, 2, 4)
    assert batch.versions["policy"].tolist() == [[1, 2], [1, 2]]


def test_stop_msg_is_distinct_control_message() -> None:
    assert StopMsg(reason="unit-test").reason == "unit-test"
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_cotrain_messages.py -q
```

Expected: FAIL with import error for `dreamervla.workers.cotrain.messages`.

- [ ] **Step 3: Implement message dataclasses**

Create `dreamervla/workers/cotrain/messages.py` with these public names:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ObservationMsg:
    env_rank: int
    slot_id: int
    task_id: int
    episode_id: int
    step: int
    obs: dict[str, Any]
    versions: dict[str, int]

    @property
    def key(self) -> str:
        return f"{int(self.env_rank)}:{int(self.slot_id)}"


@dataclass(frozen=True)
class RolloutResultMsg:
    env_rank: int
    slot_id: int
    task_id: int
    episode_id: int
    step: int
    actions: Any
    prev_logprobs: Any
    prev_values: Any | None
    forward_inputs: dict[str, Any]
    versions: dict[str, int]

    @property
    def key(self) -> str:
        return f"{int(self.env_rank)}:{int(self.slot_id)}"


@dataclass(frozen=True)
class TrajectoryShard:
    env_rank: int
    slot_id: int
    task_id: int
    episode_ids: list[int]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    prev_logprobs: torch.Tensor
    prev_values: torch.Tensor | None
    forward_inputs: dict[str, torch.Tensor]
    versions: dict[str, torch.Tensor]


@dataclass(frozen=True)
class TrajectoryBatch:
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    prev_logprobs: torch.Tensor
    prev_values: torch.Tensor | None
    forward_inputs: dict[str, torch.Tensor]
    versions: dict[str, torch.Tensor]
    task_ids: torch.Tensor
    episode_ids: torch.Tensor


@dataclass(frozen=True)
class StopMsg:
    reason: str


def as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _cat_step_batch(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([as_tensor(value) for value in values], dim=1)


def collate_trajectory_shards(shards: list[TrajectoryShard]) -> TrajectoryBatch:
    if not shards:
        raise ValueError("collate_trajectory_shards requires at least one shard")
    steps = int(shards[0].actions.shape[0])
    for shard in shards:
        if int(shard.actions.shape[0]) != steps:
            raise ValueError("all trajectory shards must have the same step dimension")
    forward_keys = set(shards[0].forward_inputs)
    version_keys = set(shards[0].versions)
    for shard in shards:
        if set(shard.forward_inputs) != forward_keys:
            raise ValueError("trajectory shards must share forward_input keys")
        if set(shard.versions) != version_keys:
            raise ValueError("trajectory shards must share version keys")
    prev_values = None
    if all(shard.prev_values is not None for shard in shards):
        prev_values = _cat_step_batch([shard.prev_values for shard in shards if shard.prev_values is not None])
    return TrajectoryBatch(
        actions=_cat_step_batch([shard.actions for shard in shards]),
        rewards=_cat_step_batch([shard.rewards for shard in shards]).float(),
        dones=_cat_step_batch([shard.dones for shard in shards]).bool(),
        prev_logprobs=_cat_step_batch([shard.prev_logprobs for shard in shards]).float(),
        prev_values=prev_values,
        forward_inputs={
            key: _cat_step_batch([shard.forward_inputs[key] for shard in shards])
            for key in sorted(forward_keys)
        },
        versions={
            key: _cat_step_batch([shard.versions[key] for shard in shards])
            for key in sorted(version_keys)
        },
        task_ids=torch.tensor([int(shard.task_id) for shard in shards], dtype=torch.long),
        episode_ids=torch.tensor(
            [int(ep) for shard in shards for ep in shard.episode_ids],
            dtype=torch.long,
        ),
    )
```

Create `dreamervla/workers/cotrain/__init__.py` exporting the dataclasses and `collate_trajectory_shards`.

- [ ] **Step 4: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_cotrain_messages.py -q
```

Expected: `3 passed`.

---

### Task 2: Manual-Notes Placement Planner

**Files:**
- Create: `dreamervla/workers/cotrain/placement.py`
- Test: `tests/unit_tests/test_manual_cotrain_placement.py`

- [ ] **Step 1: Write failing placement tests**

Add `tests/unit_tests/test_manual_cotrain_placement.py`:

```python
from __future__ import annotations

import pytest

from dreamervla.workers.cotrain.placement import build_manual_cotrain_placement


@pytest.mark.parametrize("ngpu", [0, 1, 2, 3, 4, 5])
def test_manual_cotrain_placement_supports_zero_to_five_gpus(ngpu: int) -> None:
    plan = build_manual_cotrain_placement(ngpu)
    assert plan.ngpu == ngpu
    assert plan.real_env_ranks == [0]
    assert len(plan.env_specs) == max(1, ngpu)
    assert plan.learner_spec.kind == "learner"
    assert plan.rollout_specs
    assert plan.actor_specs


def test_gpu_placement_matches_manual_notes_for_five_gpus() -> None:
    plan = build_manual_cotrain_placement(5)
    assert plan.env_specs[0].role == "real_env"
    assert plan.env_specs[0].gpu_ids == [0]
    assert [spec.gpu_ids for spec in plan.env_specs[1:]] == [[1], [2], [3], [4]]
    assert [spec.role for spec in plan.env_specs[1:]] == ["wm_env"] * 4
    assert plan.learner_spec.gpu_ids == [0]
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[1], [2], [3], [4]]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[0], [1], [2], [3], [4]]


def test_zero_gpu_placement_is_cpu_target_topology() -> None:
    plan = build_manual_cotrain_placement(0)
    assert plan.env_specs[0].gpu_ids == []
    assert plan.learner_spec.gpu_ids == []
    assert plan.actor_specs[0].gpu_ids == []
    assert plan.rollout_specs[0].gpu_ids == []
    assert plan.actor_fsdp_strategy == "none"
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_manual_cotrain_placement.py -q
```

Expected: FAIL with import error for `dreamervla.workers.cotrain.placement`.

- [ ] **Step 3: Implement placement planner**

Create `dreamervla/workers/cotrain/placement.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolePlacement:
    kind: str
    role: str
    rank: int
    gpu_ids: list[int]

    @property
    def resource_map(self) -> str:
        if not self.gpu_ids:
            return "node"
        return ",".join(str(gpu) for gpu in self.gpu_ids)


@dataclass(frozen=True)
class ManualCotrainPlacementPlan:
    ngpu: int
    env_specs: list[RolePlacement]
    rollout_specs: list[RolePlacement]
    actor_specs: list[RolePlacement]
    learner_spec: RolePlacement
    actor_fsdp_strategy: str

    @property
    def real_env_ranks(self) -> list[int]:
        return [spec.rank for spec in self.env_specs if spec.role == "real_env"]

    @property
    def wm_env_ranks(self) -> list[int]:
        return [spec.rank for spec in self.env_specs if spec.role == "wm_env"]


def build_manual_cotrain_placement(ngpu: int) -> ManualCotrainPlacementPlan:
    count = int(ngpu)
    if count < 0:
        raise ValueError(f"ngpu must be >= 0, got {ngpu!r}")
    if count == 0:
        cpu = RolePlacement(kind="env", role="real_env", rank=0, gpu_ids=[])
        return ManualCotrainPlacementPlan(
            ngpu=0,
            env_specs=[cpu],
            rollout_specs=[RolePlacement(kind="rollout", role="rollout", rank=0, gpu_ids=[])],
            actor_specs=[RolePlacement(kind="actor", role="actor", rank=0, gpu_ids=[])],
            learner_spec=RolePlacement(kind="learner", role="learner", rank=0, gpu_ids=[]),
            actor_fsdp_strategy="none",
        )
    env_specs = [
        RolePlacement(kind="env", role=("real_env" if gpu == 0 else "wm_env"), rank=gpu, gpu_ids=[gpu])
        for gpu in range(count)
    ]
    rollout_specs = [
        RolePlacement(kind="rollout", role="rollout", rank=gpu, gpu_ids=[gpu])
        for gpu in range(count)
    ]
    actor_gpus = list(range(1, count)) or [0]
    actor_specs = [
        RolePlacement(kind="actor", role="actor", rank=rank, gpu_ids=[gpu])
        for rank, gpu in enumerate(actor_gpus)
    ]
    return ManualCotrainPlacementPlan(
        ngpu=count,
        env_specs=env_specs,
        rollout_specs=rollout_specs,
        actor_specs=actor_specs,
        learner_spec=RolePlacement(kind="learner", role="learner", rank=0, gpu_ids=[0]),
        actor_fsdp_strategy="fsdp" if actor_gpus else "none",
    )
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_manual_cotrain_placement.py -q
```

Expected: `3 passed`.

---

### Task 3: MultiStepRolloutWorker

**Files:**
- Create: `dreamervla/workers/rollout/multistep_rollout_worker.py`
- Modify: `dreamervla/workers/rollout/__init__.py`
- Test: `tests/unit_tests/test_multistep_rollout_worker.py`

- [ ] **Step 1: Write failing rollout worker tests**

Add `tests/unit_tests/test_multistep_rollout_worker.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from dreamervla.workers.cotrain.messages import ObservationMsg, RolloutResultMsg
from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker


def _policy_cfg() -> dict:
    return {
        "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
        "kwargs": {"hidden_dim": 4, "action_dim": 3, "chunk_size": 2},
    }


def test_generate_once_accepts_obs_embedding_and_returns_forward_inputs() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()
    obs = ObservationMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        obs={"obs_embedding": np.ones(4, dtype=np.float32)},
        versions={"policy": 0},
    )

    out = worker.generate_once(obs)

    assert isinstance(out, RolloutResultMsg)
    assert out.actions.shape == (2, 3)
    assert out.prev_logprobs.shape == (1,)
    assert out.forward_inputs["hidden"].shape == (1, 4)
    assert out.forward_inputs["action"].shape == (1, 2, 3)
    assert out.versions["policy"] == 0


def test_sync_model_from_actor_applies_patch_syncer() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu", "syncer": {"store_name": "test_rollout_patch_sync"}},
    )
    worker.init()
    state = worker.state_dict()
    changed = {key: value + 1.0 for key, value in state.items()}
    worker._syncer().push("policy", changed, 1)

    assert worker.sync_model_from_actor("policy", local_version=0) == 1
    synced = worker.state_dict()
    assert torch.allclose(next(iter(synced.values())), next(iter(changed.values())))
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_multistep_rollout_worker.py -q
```

Expected: FAIL with import error for `MultiStepRolloutWorker`.

- [ ] **Step 3: Implement minimal target rollout worker**

Create `dreamervla/workers/rollout/multistep_rollout_worker.py` with these methods:

```python
class MultiStepRolloutWorker(Worker):
    def __init__(self, policy_cfg, encoder_cfg=None, init_ckpt=None, train_cfg=None): ...
    def init(self) -> None: ...
    @torch.no_grad()
    def generate_once(self, obs_msg: ObservationMsg) -> RolloutResultMsg: ...
    def generate(self, input_channel_name: str, output_channel_name: str) -> dict[str, float]: ...
    def sync_model_from_actor(self, key: str = "policy", local_version: int = 0) -> int | None: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...
```

Implementation requirements:

```python
hidden = obs_msg.obs.get("obs_embedding", obs_msg.obs.get("latent"))
hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.torch_device).reshape(1, -1)
action, log_prob, extra = self.policy(
    {"mode": "sample", "hidden": hidden_t, "return_chunk": True, "deterministic": False}
)
forward_inputs = {"hidden": hidden_t.detach().cpu(), "action": action.detach().cpu()}
for key in ("action_token_ids", "input_ids", "attention_mask", "hidden_states"):
    if key in extra:
        forward_inputs[key] = extra[key].detach().cpu()
```

`generate()` must read `ObservationMsg` from a `Channel`, stop on `StopMsg`, write `RolloutResultMsg`, and return `{"rollout/generated": float(count)}`.

- [ ] **Step 4: Export the worker**

In `dreamervla/workers/rollout/__init__.py`:

```python
from dreamervla.workers.rollout.multistep_rollout_worker import MultiStepRolloutWorker

__all__ = ["MultiStepRolloutWorker"]
```

- [ ] **Step 5: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_multistep_rollout_worker.py -q
```

Expected: `2 passed`.

---

### Task 4: Target EnvWorker Interface, RealEnvWorker, WMEnvWorker

**Files:**
- Create: `dreamervla/workers/env/trajectory_env_worker.py`
- Modify: `dreamervla/workers/env/__init__.py`
- Test: `tests/unit_tests/test_trajectory_env_worker.py`

- [ ] **Step 1: Write failing EnvGroup tests**

Add `tests/unit_tests/test_trajectory_env_worker.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from dreamervla.workers.cotrain.messages import ObservationMsg, RolloutResultMsg, TrajectoryShard
from dreamervla.workers.env.trajectory_env_worker import (
    BaseTrajectoryEnvWorker,
    RealEnvWorker,
    WMEnvWorker,
)


class _MemoryChannel:
    def __init__(self) -> None:
        self.items = []

    def put(self, item, *, key="default") -> None:
        self.items.append(item)

    def get(self, *, key="default"):
        assert self.items
        return self.items.pop(0)


def test_real_env_worker_buffers_rollout_result_into_trajectory() -> None:
    worker = BaseTrajectoryEnvWorker(
        role="real_env",
        env_cfg={"target": "dreamervla.workers.env._test_envs:TinyCounterEnv", "kwargs": {"horizon": 2, "action_dim": 3}},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    worker.init()
    obs = worker.bootstrap_obs()[0]
    assert isinstance(obs, ObservationMsg)
    result = RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        actions=np.zeros((2, 3), dtype=np.float32),
        prev_logprobs=np.array([0.0], dtype=np.float32),
        prev_values=None,
        forward_inputs={"hidden": np.ones((1, 4), dtype=np.float32), "action": np.zeros((1, 2, 3), dtype=np.float32)},
        versions={"policy": 1},
    )

    shard = worker.apply_rollout_result(result)

    assert isinstance(shard, TrajectoryShard)
    assert shard.actions.shape == (2, 1, 3)
    assert shard.forward_inputs["hidden"].shape == (2, 1, 4)
    assert shard.versions["policy"].shape == (2, 1)


def test_real_and_wm_worker_classes_are_distinct_roles() -> None:
    assert RealEnvWorker.role_name == "real_env"
    assert WMEnvWorker.role_name == "wm_env"
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_trajectory_env_worker.py -q
```

Expected: FAIL with import error.

- [ ] **Step 3: Implement base trajectory env worker**

Create `dreamervla/workers/env/trajectory_env_worker.py`:

```python
class BaseTrajectoryEnvWorker(Worker):
    role_name = "env"
    def __init__(self, role, env_cfg, num_slots, rollout_epoch, max_steps_per_rollout_epoch, num_action_chunks, task_id=0, replay=None, dump=None): ...
    def init(self) -> None: ...
    def bootstrap_obs(self) -> list[ObservationMsg]: ...
    def apply_rollout_result(self, result: RolloutResultMsg) -> TrajectoryShard: ...
    def interact(self, env_channel_name: str, rollout_channel_name: str, actor_channel_name: str) -> dict[str, float]: ...
    def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None: ...
    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None: ...
```

Implementation requirements:

- `bootstrap_obs()` resets each slot and emits `ObservationMsg`.
- `apply_rollout_result()` executes every action in `result.actions` sequentially.
- It records `actions`, `rewards`, `dones`, `prev_logprobs`, `forward_inputs`, and `versions` into a `TrajectoryShard`.
- For each action in an action chunk, repeat the single rollout-level `prev_logprobs`, `forward_inputs`, and policy version across the produced env steps.
- On done, push the complete episode to replay/dump using the existing `EnvWorker._push_episode` static behavior or an equivalent local helper.

Add subclasses:

```python
class RealEnvWorker(BaseTrajectoryEnvWorker):
    role_name = "real_env"


class WMEnvWorker(BaseTrajectoryEnvWorker):
    role_name = "wm_env"
```

`WMEnvWorker` must use the same constructor and rely on config target `dreamervla.envs.world_model.latent_world_model_env:LatentWorldModelEnv` or another Hydra-selected WM env.

- [ ] **Step 4: Export classes**

In `dreamervla/workers/env/__init__.py`:

```python
from dreamervla.workers.env.trajectory_env_worker import BaseTrajectoryEnvWorker, RealEnvWorker, WMEnvWorker
```

- [ ] **Step 5: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_trajectory_env_worker.py -q
```

Expected: `2 passed`.

---

### Task 5: EmbodiedFSDPActor Owns VLA PPO

**Files:**
- Create: `dreamervla/workers/actor/embodied_fsdp_actor.py`
- Modify: `dreamervla/workers/actor/__init__.py`
- Test: `tests/unit_tests/test_embodied_fsdp_actor.py`

- [ ] **Step 1: Write failing ActorGroup tests**

Add `tests/unit_tests/test_embodied_fsdp_actor.py`:

```python
from __future__ import annotations

import torch

from dreamervla.workers.actor.embodied_fsdp_actor import EmbodiedFSDPActor
from dreamervla.workers.cotrain.messages import TrajectoryShard


def _actor_cfg() -> dict:
    return {
        "policy_cfg": {
            "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
            "kwargs": {"hidden_dim": 4, "action_dim": 3, "chunk_size": 2},
        },
        "init_ckpt": {},
        "train_cfg": {
            "device": "cpu",
            "lr": 1e-3,
            "fsdp": {"strategy": "none", "precision": "fp32"},
            "algorithm_cfg": {
                "group_size": 2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 3.0,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ppo_update_epochs": 1,
                "entropy_coef": 0.0,
            },
        },
    }


def _shard(reward0: float, reward1: float) -> TrajectoryShard:
    return TrajectoryShard(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_ids=[0, 1],
        actions=torch.zeros(2, 2, 3),
        rewards=torch.tensor([[reward0, reward1], [reward0, reward1]], dtype=torch.float32),
        dones=torch.zeros(2, 2, dtype=torch.bool),
        prev_logprobs=torch.zeros(2, 2),
        prev_values=None,
        forward_inputs={"hidden": torch.ones(2, 2, 4), "action": torch.zeros(2, 2, 2, 3)},
        versions={"policy": torch.zeros(2, 2, dtype=torch.long)},
    )


def test_actor_group_computes_group_advantages_from_trajectory_rewards() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()

    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    metrics = actor.compute_advantages_and_returns()

    assert metrics["actor/trajectory_count"] == 2.0
    assert metrics["actor/advantage_std"] > 0.0


def test_actor_run_training_updates_policy_parameters() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    before = {key: value.clone() for key, value in actor.state_dict().items()}

    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    actor.compute_advantages_and_returns()
    metrics = actor.run_training()

    after = actor.state_dict()
    assert metrics["actor/ppo_updates"] == 1.0
    assert any(not torch.equal(before[key], after[key]) for key in before)
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_embodied_fsdp_actor.py -q
```

Expected: FAIL with import error.

- [ ] **Step 3: Implement Actor worker**

Create `dreamervla/workers/actor/embodied_fsdp_actor.py`:

```python
class EmbodiedFSDPActor(Worker):
    def __init__(self, policy_cfg: dict[str, Any], init_ckpt: dict[str, Any], train_cfg: dict[str, Any]) -> None: ...
    def init(self) -> None: ...
    def set_global_step(self, global_step: int) -> None: ...
    def load_trajectory_shards(self, shards: list[TrajectoryShard]) -> None: ...
    def recv_rollout_trajectories(self, actor_channel_name: str, expected_shards: int | None = None) -> dict[str, float]: ...
    def compute_advantages_and_returns(self) -> dict[str, float]: ...
    def run_training(self) -> dict[str, float]: ...
    def sync_model_to_rollout(self, key: str = "policy", version: int | None = None) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...
```

Implementation requirements:

- Build policy from config, load `init_ckpt["policy"]` if present.
- Build optimizer over trainable policy params only.
- Use `FSDPModelManager` from `train_cfg.fsdp`; call `ensure_process_group()` and `prepare_model()`.
- Store raw trajectory shards, collate with `collate_trajectory_shards`.
- Compute trajectory-level returns as `batch.rewards.sum(dim=0)`.
- Compute advantages with `_group_advantage(returns, group_size, eps=1e-6)` from `dreamervla.algorithms.ppo.grpo`.
- During `run_training()`, flatten `[steps, batch]` to minibatch records; for each step evaluate:

```python
log_prob, entropy, _ = policy({"mode": "evaluate", **forward_inputs_step, "action": actions_step})
ratio = _ppo_ratio(log_prob, old_log_prob, clip_log_ratio=train_cfg.get("clip_log_ratio", 20.0))
loss = _ppo_clip_term(ratio, advantage, clip_low, clip_high, clip_ratio_c=clip_ratio_c).mean()
loss = loss - entropy_coef * entropy.mean()
```

- Support `forward_inputs` keys `hidden`, `action_token_ids`, `input_ids`, `attention_mask`, and `hidden_states`.
- Run `ppo_update_epochs` from config.
- Push actor weights through `PatchWeightSyncer` in `sync_model_to_rollout`.

- [ ] **Step 4: Export class**

In `dreamervla/workers/actor/__init__.py`:

```python
from dreamervla.workers.actor.embodied_fsdp_actor import EmbodiedFSDPActor
```

- [ ] **Step 5: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_embodied_fsdp_actor.py -q
```

Expected: `2 passed`.

---

### Task 6: LearnerWorker WM/Classifier-Only Mode

**Files:**
- Modify: `dreamervla/workers/actor/learner_worker.py`
- Test: `tests/unit_tests/test_learner_worker_manual_precision.py`

- [ ] **Step 1: Add failing LearnerGroup separation test**

Append to `tests/unit_tests/test_learner_worker_manual_precision.py`:

```python
def test_learner_worker_wm_classifier_only_does_not_require_policy() -> None:
    worker = LearnerWorker(
        model_cfg={
            "world_model": {
                "target": "dreamervla.workers.actor._test_models:TinyLumosWorldModel",
                "kwargs": {"hidden_dim": 4, "action_dim": 3},
            },
            "classifier": {
                "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
                "kwargs": {"hidden_dim": 4, "window": 2},
            },
        },
        init_ckpt={},
        train_cfg={"mode": "wm_classifier_only", "device": "cpu", "precision": "fp32"},
        replay=None,
    )

    worker.init()

    assert worker.policy is None
    assert "policy" not in worker.optimizers
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_learner_worker_manual_precision.py::test_learner_worker_wm_classifier_only_does_not_require_policy -q
```

Expected: FAIL because `_build_components()` currently requires policy.

- [ ] **Step 3: Implement `wm_classifier_only` mode**

In `LearnerWorker._build_components()`, replace unconditional policy requirement:

```python
mode = str(self.train_cfg.get("mode", "synthetic_ppo"))
if mode != "wm_classifier_only" and "policy" not in components:
    raise ValueError("LearnerWorker model_cfg must include a policy component")
if mode == "wm_classifier_only":
    missing = [name for name in ("world_model", "classifier") if name not in components]
    if missing:
        raise ValueError(f"wm_classifier_only requires components: {missing}")
```

In `update()`:

```python
if mode == "wm_classifier_only":
    if phase == "cotrain":
        metrics = {}
        metrics.update(self._dreamervla_wm_update_once())
        metrics.update(self._dreamervla_classifier_update_once())
        return metrics
    if phase == "wm":
        return self._dreamervla_wm_update_once()
    if phase == "classifier":
        return self._dreamervla_classifier_update_once()
    raise ValueError("wm_classifier_only supports only wm, classifier, cotrain")
```

Modify `_dreamervla_wm_update_once()` so it passes `policy=None` when `self.policy is None` and the selected world-model update function can accept it; if the current `world_model_pretrain_step` requires policy, add `train_cfg["wm_update_fn"]` support and use a config-selected updater. For target route, configure `wm_update_fn` to existing `world_model_pretrain_step` only when policy is present, otherwise to a small `phase_updater` compatible with replay/hidden tensors.

- [ ] **Step 4: Run focused test**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_learner_worker_manual_precision.py::test_learner_worker_wm_classifier_only_does_not_require_policy -q
```

Expected: PASS.

---

### Task 7: Target manual cotrain Runner Loop

**Files:**
- Create: `dreamervla/runners/manual_cotrain_ray_runner.py`
- Modify: `dreamervla/runners/__init__.py`
- Test: `tests/unit_tests/test_manual_cotrain_ray_runner.py`

- [ ] **Step 1: Write failing runner topology tests**

Add `tests/unit_tests/test_manual_cotrain_ray_runner.py`:

```python
from __future__ import annotations

from omegaconf import OmegaConf

from dreamervla.runners.manual_cotrain_ray_runner import ManualCotrainRayRunner


def _cfg(ngpu: int = 2):
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.ManualCotrainRayRunner",
            "seed": 7,
            "training": {"out_dir": "/tmp/dvla-manual_cotrain-test", "seed": 7},
            "cluster": {"num_nodes": 1},
            "manual_cotrain": {
                "ngpu": ngpu,
                "global_steps": 1,
                "learner_update_step": 1,
                "rollout_epoch": 1,
                "max_steps_per_rollout_epoch": 2,
                "num_action_chunks": 2,
                "envs_per_worker": 1,
                "sync_every": 1,
            },
            "actor": {"train_cfg": {"algorithm_cfg": {"group_size": 2}}},
        }
    )


def test_runner_plans_manual_notes_groups() -> None:
    runner = ManualCotrainRayRunner(_cfg(ngpu=5))
    plan = runner._placement_plan()
    assert [spec.role for spec in plan.env_specs] == ["real_env", "wm_env", "wm_env", "wm_env", "wm_env"]
    assert len(plan.actor_specs) == 4
    assert len(plan.rollout_specs) == 5
    assert plan.learner_spec.gpu_ids == [0]


def test_runner_loop_order_names_actor_before_learner_update() -> None:
    runner = ManualCotrainRayRunner(_cfg(ngpu=2))
    order = runner._global_step_operation_names()
    assert order[:4] == [
        "set_global_step",
        "actor_to_rollout_sync",
        "env_interact_and_rollout_generate",
        "actor_recv_trajectories",
    ]
    assert "actor_run_training" in order
    assert "learner_update_wm_classifier" in order
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_manual_cotrain_ray_runner.py -q
```

Expected: FAIL with import error.

- [ ] **Step 3: Implement runner skeleton and real loop**

Create `dreamervla/runners/manual_cotrain_ray_runner.py`:

```python
class ManualCotrainRayRunner(BaseRunner):
    runner_name = "manual_cotrain_ray"
    runner_status = "current"
    runner_family = "actor"

    def _placement_plan(self) -> ManualCotrainPlacementPlan: ...
    def _global_step_operation_names(self) -> list[str]: ...
    def _build_groups(self, cluster: Cluster) -> dict[str, Any]: ...
    def _run_global_step(self, groups: dict[str, Any], global_step: int) -> dict[str, float]: ...
    def run(self) -> dict[str, float | int]: ...
```

`_global_step_operation_names()` returns exactly:

```python
[
    "set_global_step",
    "actor_to_rollout_sync",
    "env_interact_and_rollout_generate",
    "actor_recv_trajectories",
    "actor_compute_advantages_and_returns",
    "actor_run_training",
    "learner_update_wm_classifier",
    "learner_to_wm_env_sync",
    "checkpoint_and_metrics",
]
```

`_run_global_step()` must follow `99_manual_notes.md`:

```python
actor.set_global_step(global_step).wait()
rollout.set_global_step(global_step).wait()
if global_step % sync_every == 0:
    actor.sync_model_to_rollout("policy", global_step).wait()
    rollout.sync_model_from_actor("policy", local_version).wait()
env_refs = envs.interact(env_channel_name, rollout_channel_name, actor_channel_name)
rollout_refs = rollouts.generate(env_channel_name, rollout_channel_name)
env_metrics = env_refs.wait()
rollout_metrics = rollout_refs.wait()
actor.recv_rollout_trajectories(actor_channel_name, expected_shards).wait()
adv_metrics = actor.compute_advantages_and_returns().wait()
train_metrics = actor.run_training().wait()
if global_step % learner_update_step == 0:
    learner.update("cotrain", 1).wait()
    learner.sync_weights("world_model", global_step).wait()
    learner.sync_weights("classifier", global_step).wait()
    wm_envs.load_world_model_state(...).wait()
    wm_envs.load_classifier_state(...).wait()
```

The runner may use `WorkerGroup.execute_on()` to target real env rank 0 and WM env ranks separately.

- [ ] **Step 4: Export runner**

In `dreamervla/runners/__init__.py`, add `_RunnerSpec` for `ManualCotrainRayRunner` and include it in `__all__`/lazy exports following the existing pattern.

- [ ] **Step 5: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_manual_cotrain_ray_runner.py -q
```

Expected: `2 passed`.

---

### Task 8: Hydra Config And Launcher Route

**Files:**
- Create: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`
- Create: `configs/experiment/manual_cotrain_ray_oft_backbone_latent.yaml`
- Modify: `configs/scripts/coldstart_warmup_cotrain.yaml`
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`

- [ ] **Step 1: Add failing launcher test for target experiment**

Add:

```python
def test_async_cotrain_defaults_to_manual_cotrain_target_experiment(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = dict(_launcher_cfg())
    cfg["cotrain_engine"] = "async"
    plan = build_pipeline_plan(
        mode="ray",
        run_root=tmp_path,
        python="python",
        profile="smoke",
        ngpu=2,
        launcher_cfg=cfg,
    )

    assert "experiment=manual_cotrain_ray_oft_backbone_latent" in plan.cotrain_online_cmd
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_cotrain_defaults_to_manual_cotrain_target_experiment -q
```

Expected: FAIL because current default is `online_cotrain_ray_oft_backbone_latent`.

- [ ] **Step 3: Add config files**

`configs/experiment/manual_cotrain_ray_oft_backbone_latent.yaml`:

```yaml
# @package _global_
defaults:
  - /dreamervla: manual_cotrain_ray_oft_backbone_latent
  - _self_
```

`configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml` must:

- Set `_target_: dreamervla.runners.ManualCotrainRayRunner`.
- Import `/task: openvla_onetraj_coldstart_libero`.
- Copy `ray_components.world_model`, `ray_components.policy`, and `ray_components.classifier` from `ray_online_cotrain_oft_backbone_latent.yaml`.
- Set RLinf manual cotrain PPO hyperparameters:

```yaml
algorithm:
  group_size: 8
  rollout_epoch: 16
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  clip_ratio_c: 3.0
  gamma: 0.99
  gae_lambda: 0.95
  kl_beta: 0.0
  entropy_bonus: 0.0
```

- Set target loop controls:

```yaml
manual_cotrain:
  ngpu: 1
  global_steps: 1
  learner_update_step: 1
  sync_every: 1
  rollout_epoch: ${algorithm.rollout_epoch}
  max_steps_per_rollout_epoch: 256
  num_action_chunks: ${task.openvla_oft.input_tokens.chunk_size}
  envs_per_worker: 8
  real_env_workers: 1
```

- Configure `actor.policy_cfg: ${ray_components.policy}` and `actor.train_cfg.fsdp.strategy: fsdp`.
- Configure `learner.train_cfg.mode: wm_classifier_only`.
- Configure `rollout.policy_cfg: ${ray_components.policy}`.
- Configure real env from current OFT `env.cfg`.
- Configure WM env target as `dreamervla.envs.world_model.latent_world_model_env:LatentWorldModelEnv`.

- [ ] **Step 4: Change launcher default**

In `configs/scripts/coldstart_warmup_cotrain.yaml`:

```yaml
cotrain_async_experiment: manual_cotrain_ray_oft_backbone_latent
```

In `dreamervla/launchers/coldstart_warmup_cotrain.py`, ensure async overrides include:

```text
manual_cotrain.ngpu=<ngpu>
manual_cotrain.envs_per_worker=<profile online_rollout_envs_per_gpu or 8>
manual_cotrain.rollout_epoch=16
algorithm.group_size=8
```

For `ngpu=0`, force:

```text
render_backend=osmesa
actor.train_cfg.fsdp.strategy=none
actor.train_cfg.device=cpu
learner.train_cfg.device=cpu
rollout.train_cfg.device=cpu
```

- [ ] **Step 5: Run config and launcher tests**

Run:

```bash
PYTHONPATH=$PWD pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_cotrain_defaults_to_manual_cotrain_target_experiment -q
python -m dreamervla.train experiment=manual_cotrain_ray_oft_backbone_latent task=goal --cfg job
```

Expected: test PASS and Hydra config prints without missing interpolation errors.

---

### Task 9: Target Startup Verification And Short Run

**Files:**
- Create: `tests/e2e_tests/test_manual_cotrain_ray_startup.py`
- Modify if needed: `scripts/e2e_coldstart_warmup_cotrain_ray.sh`

- [ ] **Step 1: Add gated e2e startup test**

Add `tests/e2e_tests/test_manual_cotrain_ray_startup.py`:

```python
from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    os.environ.get("DVLA_RUN_RAY_E2E") != "1",
    reason="set DVLA_RUN_RAY_E2E=1 to run Ray cotrain startup",
)
def test_manual_cotrain_ray_starts_one_global_step(tmp_path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "dreamervla.train",
        "experiment=manual_cotrain_ray_oft_backbone_latent",
        "task=goal",
        f"training.out_dir={tmp_path}",
        "manual_cotrain.global_steps=1",
        "manual_cotrain.max_steps_per_rollout_epoch=2",
        "manual_cotrain.rollout_epoch=1",
        "manual_cotrain.envs_per_worker=1",
        "logger=tensorboard",
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    assert result.returncode == 0, result.stdout
    assert "LearnerGroup" in result.stdout
    assert "ActorGroup" in result.stdout
    assert "RolloutGroup" in result.stdout
    assert "EnvGroup" in result.stdout
```

- [ ] **Step 2: Run unit suite for target route**

Run:

```bash
PYTHONPATH=$PWD pytest \
  tests/unit_tests/test_cotrain_messages.py \
  tests/unit_tests/test_manual_cotrain_placement.py \
  tests/unit_tests/test_multistep_rollout_worker.py \
  tests/unit_tests/test_trajectory_env_worker.py \
  tests/unit_tests/test_embodied_fsdp_actor.py \
  tests/unit_tests/test_manual_cotrain_ray_runner.py \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run short 0-GPU startup**

Run:

```bash
CUDA_VISIBLE_DEVICES= \
DVLA_RUN_RAY_E2E=1 \
PYTHONPATH=$PWD \
python -m dreamervla.train \
  experiment=manual_cotrain_ray_oft_backbone_latent \
  task=goal \
  render_backend=osmesa \
  manual_cotrain.ngpu=0 \
  manual_cotrain.global_steps=1 \
  manual_cotrain.max_steps_per_rollout_epoch=2 \
  manual_cotrain.rollout_epoch=1 \
  manual_cotrain.envs_per_worker=1 \
  actor.train_cfg.fsdp.strategy=none \
  actor.train_cfg.device=cpu \
  learner.train_cfg.device=cpu \
  rollout.train_cfg.device=cpu
```

Expected: process exits 0 after one global step and logs all four groups.

- [ ] **Step 4: Run short available-GPU startup**

For 1-5 GPUs, run the same command with `CUDA_VISIBLE_DEVICES` and `manual_cotrain.ngpu=N`. On the current 6-GPU host, the required verification command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
PYTHONPATH=$PWD \
python -m dreamervla.train \
  experiment=manual_cotrain_ray_oft_backbone_latent \
  task=goal \
  render_backend=egl \
  manual_cotrain.ngpu=5 \
  manual_cotrain.global_steps=1 \
  manual_cotrain.max_steps_per_rollout_epoch=2 \
  manual_cotrain.rollout_epoch=1 \
  manual_cotrain.envs_per_worker=1
```

Expected: process starts Ray, logs group placement, completes one global step, writes checkpoint/metrics under `training.out_dir`.

---

## Completion Checklist

- [ ] `LearnerGroup` exists in the target runner and uses `LearnerWorker(mode=wm_classifier_only)`.
- [ ] `ActorGroup` exists in the target runner and uses `EmbodiedFSDPActor`.
- [ ] `RolloutGroup` exists in the target runner and uses `MultiStepRolloutWorker`.
- [ ] `EnvGroup` exists in the target runner and rank 0 is `RealEnvWorker`; remaining ranks are `WMEnvWorker`.
- [ ] Actor PPO consumes trajectory shards from `actor_channel`; it does not sample policy PPO batches from replay.
- [ ] Learner updates WM/cls only and syncs WM/cls to `WMEnvWorker`.
- [ ] Rollout syncs policy weights from Actor via `PatchWeightSyncer`.
- [ ] PPO hyperparameters match RLinf manual cotrain: `group_size=8`, `rollout_epoch=16`, `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `clip_ratio_c=3.0`, `gamma=0.99`, `gae_lambda=0.95`, no KL/entropy bonus by default.
- [ ] Startup supports `manual_cotrain.ngpu=0,1,2,3,4,5`.
- [ ] A one-global-step run completes successfully.
