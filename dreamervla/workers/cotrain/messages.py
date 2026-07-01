from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ObservationMsg:
    """Observation payload sent from EnvGroup to RolloutGroup."""

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
class ObservationBatchMsg:
    """Rank-scoped observation batch sent from one EnvWorker to RolloutWorker."""

    env_rank: int
    observations: list[ObservationMsg]

    @property
    def key(self) -> str:
        return str(int(self.env_rank))


@dataclass(frozen=True)
class RolloutResultMsg:
    """Rollout policy output plus ActorGroup training inputs."""

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
class RolloutResultBatchMsg:
    """Rank-scoped rollout result batch returned to one EnvWorker."""

    env_rank: int
    results: list[RolloutResultMsg]

    @property
    def key(self) -> str:
        return str(int(self.env_rank))


@dataclass(frozen=True)
class TrajectoryShard:
    """Step-major trajectory fragment produced by one EnvWorker slot batch."""

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
    loss_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class TrajectoryBatch:
    """Trajectory shards collated as [steps, batch, ...] tensors."""

    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    prev_logprobs: torch.Tensor
    prev_values: torch.Tensor | None
    forward_inputs: dict[str, torch.Tensor]
    versions: dict[str, torch.Tensor]
    loss_mask: torch.Tensor
    task_ids: torch.Tensor
    episode_ids: torch.Tensor


@dataclass(frozen=True)
class StopMsg:
    """Control message used to stop cotrain worker loops."""

    reason: str


def as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return value as a detached tensor, optionally cast to dtype."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _cat_step_batch(values: list[Any]) -> torch.Tensor:
    return torch.cat([as_tensor(value) for value in values], dim=1)


def _validate_step_batch_dim(
    name: str,
    value: Any,
    steps: int,
    *,
    batch_size: int | None = None,
) -> int:
    tensor = as_tensor(value)
    if tensor.ndim < 2:
        raise ValueError(f"{name} must be a [steps, batch, ...] tensor")
    if int(tensor.shape[0]) != steps:
        raise ValueError("all trajectory shards must have the same step dimension")
    value_batch_size = int(tensor.shape[1])
    if batch_size is not None and value_batch_size != batch_size:
        raise ValueError(
            "all tensors in a trajectory shard must share the same batch dimension"
        )
    return value_batch_size


def _validate_shard_shape(shard: TrajectoryShard) -> tuple[int, int]:
    steps = int(as_tensor(shard.actions).shape[0])
    batch_size = _validate_step_batch_dim("actions", shard.actions, steps)
    _validate_step_batch_dim("rewards", shard.rewards, steps, batch_size=batch_size)
    _validate_step_batch_dim("dones", shard.dones, steps, batch_size=batch_size)
    _validate_step_batch_dim(
        "prev_logprobs", shard.prev_logprobs, steps, batch_size=batch_size
    )
    if shard.prev_values is not None:
        _validate_step_batch_dim(
            "prev_values", shard.prev_values, steps, batch_size=batch_size
        )
    for key, value in shard.forward_inputs.items():
        _validate_step_batch_dim(
            f"forward_inputs[{key!r}]", value, steps, batch_size=batch_size
        )
    for key, value in shard.versions.items():
        _validate_step_batch_dim(
            f"versions[{key!r}]", value, steps, batch_size=batch_size
        )
    if shard.loss_mask is not None:
        _validate_step_batch_dim(
            "loss_mask", shard.loss_mask, steps, batch_size=batch_size
        )
    if len(shard.episode_ids) != batch_size:
        raise ValueError(
            "episode_ids length must match trajectory shard batch dimension"
        )
    return steps, batch_size


def _pad_step_batch(value: Any, steps: int, *, pad_value: bool | float = 0.0) -> torch.Tensor:
    tensor = as_tensor(value).detach().cpu()
    current_steps = int(tensor.shape[0])
    if current_steps == int(steps):
        return tensor
    if current_steps > int(steps):
        raise ValueError("cannot pad trajectory tensor to a shorter step dimension")
    pad_shape = (int(steps) - current_steps, *tuple(tensor.shape[1:]))
    pad = torch.full(
        pad_shape,
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=0)


def _shard_loss_mask(shard: TrajectoryShard, steps: int, batch_size: int) -> torch.Tensor:
    if shard.loss_mask is not None:
        return _pad_step_batch(shard.loss_mask, steps).float()
    actual_steps = int(as_tensor(shard.actions).shape[0])
    mask = torch.zeros(int(steps), int(batch_size), dtype=torch.float32)
    if actual_steps <= 0:
        return mask
    dones = as_tensor(shard.dones).detach().cpu().bool()
    if dones.ndim > 2:
        done_by_step = dones.reshape(actual_steps, int(batch_size), -1).any(dim=2)
    else:
        done_by_step = dones.reshape(actual_steps, int(batch_size))
    alive = torch.ones(int(batch_size), dtype=torch.bool)
    for step in range(actual_steps):
        mask[step] = alive.to(dtype=torch.float32)
        alive = alive & ~done_by_step[step]
    return mask


def collate_trajectory_shards(shards: list[TrajectoryShard]) -> TrajectoryBatch:
    """Collate step-major shards by concatenating their batch dimension."""

    if not shards:
        raise ValueError("collate_trajectory_shards requires at least one shard")

    forward_keys = set(shards[0].forward_inputs)
    version_keys = set(shards[0].versions)

    batch_sizes: list[int] = []
    steps_by_shard: list[int] = []
    for shard in shards:
        steps, batch_size = _validate_shard_shape(shard)
        steps_by_shard.append(steps)
        batch_sizes.append(batch_size)
        if set(shard.forward_inputs) != forward_keys:
            raise ValueError("trajectory shards must share forward_input keys")
        if set(shard.versions) != version_keys:
            raise ValueError("trajectory shards must share version keys")

    has_prev_values = [shard.prev_values is not None for shard in shards]
    if any(has_prev_values) and not all(has_prev_values):
        raise ValueError("trajectory shards must consistently include prev_values")

    prev_values = None
    max_steps = max(steps_by_shard)
    if all(has_prev_values):
        prev_values = _cat_step_batch(
            [
                _pad_step_batch(shard.prev_values, max_steps)
                for shard in shards
                if shard.prev_values is not None
            ]
        )

    return TrajectoryBatch(
        actions=_cat_step_batch(
            [_pad_step_batch(shard.actions, max_steps) for shard in shards]
        ),
        rewards=_cat_step_batch(
            [_pad_step_batch(shard.rewards, max_steps) for shard in shards]
        ).float(),
        dones=_cat_step_batch(
            [_pad_step_batch(shard.dones, max_steps, pad_value=True) for shard in shards]
        ).bool(),
        prev_logprobs=_cat_step_batch(
            [_pad_step_batch(shard.prev_logprobs, max_steps) for shard in shards]
        ).float(),
        prev_values=prev_values,
        forward_inputs={
            key: _cat_step_batch(
                [
                    _pad_step_batch(shard.forward_inputs[key], max_steps)
                    for shard in shards
                ]
            )
            for key in sorted(forward_keys)
        },
        versions={
            key: _cat_step_batch(
                [_pad_step_batch(shard.versions[key], max_steps) for shard in shards]
            )
            for key in sorted(version_keys)
        },
        loss_mask=_cat_step_batch(
            [
                _shard_loss_mask(shard, max_steps, batch_size)
                for shard, batch_size in zip(shards, batch_sizes, strict=True)
            ]
        ),
        task_ids=torch.tensor(
            [
                int(shard.task_id)
                for shard, batch_size in zip(shards, batch_sizes, strict=True)
                for _ in range(batch_size)
            ],
            dtype=torch.long,
        ),
        episode_ids=torch.tensor(
            [int(ep) for shard in shards for ep in shard.episode_ids],
            dtype=torch.long,
        ),
    )
