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
    batched_obs: dict[str, Any] | None = None

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
    slot_ids: list[int] | None = None
    task_ids: list[int] | None = None
    episode_ids: list[int] | None = None
    steps: list[int] | None = None
    actions: Any | None = None
    prev_logprobs: Any | None = None
    prev_values: Any | None = None
    forward_inputs: dict[str, Any] | None = None
    versions: dict[str, Any] | None = None

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


def pack_rollout_result_batch(
    *,
    env_rank: int,
    results: list[RolloutResultMsg],
) -> RolloutResultBatchMsg:
    """Pack per-slot rollout results into rank-level batch tensors."""

    if not results:
        return RolloutResultBatchMsg(env_rank=int(env_rank), results=[])
    expected_rank = int(env_rank)
    forward_keys = set(results[0].forward_inputs)
    version_keys = set(results[0].versions)
    has_prev_values = results[0].prev_values is not None
    for result in results:
        if int(result.env_rank) != expected_rank:
            raise ValueError(
                "rollout result env_rank mismatch: "
                f"got {int(result.env_rank)}, expected {expected_rank}"
            )
        if set(result.forward_inputs) != forward_keys:
            raise ValueError("rollout results must share forward_input keys")
        if set(result.versions) != version_keys:
            raise ValueError("rollout results must share version keys")
        if (result.prev_values is not None) != has_prev_values:
            raise ValueError("rollout results must consistently include prev_values")

    return RolloutResultBatchMsg(
        env_rank=expected_rank,
        results=[],
        slot_ids=[int(result.slot_id) for result in results],
        task_ids=[int(result.task_id) for result in results],
        episode_ids=[int(result.episode_id) for result in results],
        steps=[int(result.step) for result in results],
        actions=_batch_result_values([result.actions for result in results]),
        prev_logprobs=_batch_result_values(
            [result.prev_logprobs for result in results],
        ),
        prev_values=(
            _batch_result_values(
                [
                    result.prev_values
                    for result in results
                    if result.prev_values is not None
                ],
            )
            if has_prev_values
            else None
        ),
        forward_inputs={
            key: _batch_forward_input_values(
                [result.forward_inputs[key] for result in results],
            )
            for key in sorted(forward_keys)
        },
        versions={
            key: torch.as_tensor(
                [int(result.versions[key]) for result in results],
                dtype=torch.long,
            )
            for key in sorted(version_keys)
        },
    )


def rollout_result_batch_to_messages(
    msg: RolloutResultBatchMsg,
) -> list[RolloutResultMsg]:
    """Return per-slot rollout results from either single-slot or batched payloads."""

    if msg.results:
        return list(msg.results)
    if msg.slot_ids is None:
        return []
    slot_ids = [int(value) for value in msg.slot_ids]
    batch_size = len(slot_ids)
    task_ids = _required_id_list(msg.task_ids, "task_ids", batch_size)
    episode_ids = _required_id_list(msg.episode_ids, "episode_ids", batch_size)
    steps = _required_id_list(msg.steps, "steps", batch_size)
    if msg.actions is None or msg.prev_logprobs is None:
        raise ValueError("batched rollout result must include actions and prev_logprobs")
    actions = as_tensor(msg.actions).detach().cpu()
    prev_logprobs = as_tensor(msg.prev_logprobs).detach().cpu()
    prev_values = (
        None if msg.prev_values is None else as_tensor(msg.prev_values).detach().cpu()
    )
    if int(actions.shape[0]) != batch_size:
        raise ValueError("batched rollout actions batch size mismatch")
    if int(prev_logprobs.shape[0]) != batch_size:
        raise ValueError("batched rollout prev_logprobs batch size mismatch")
    if prev_values is not None and int(prev_values.shape[0]) != batch_size:
        raise ValueError("batched rollout prev_values batch size mismatch")
    forward_inputs = dict(msg.forward_inputs or {})
    versions = dict(msg.versions or {})
    for key, value in forward_inputs.items():
        if int(as_tensor(value).shape[0]) != batch_size:
            raise ValueError(f"batched rollout forward_inputs[{key!r}] size mismatch")
    for key, value in versions.items():
        if int(as_tensor(value).shape[0]) != batch_size:
            raise ValueError(f"batched rollout versions[{key!r}] size mismatch")

    return [
        RolloutResultMsg(
            env_rank=int(msg.env_rank),
            slot_id=slot_ids[index],
            task_id=task_ids[index],
            episode_id=episode_ids[index],
            step=steps[index],
            actions=actions[index],
            prev_logprobs=prev_logprobs[index],
            prev_values=None if prev_values is None else prev_values[index],
            forward_inputs={
                key: _unpack_forward_input_row(key, as_tensor(value)[index])
                for key, value in forward_inputs.items()
            },
            versions={
                key: int(as_tensor(value)[index].detach().cpu().reshape(-1)[0].item())
                for key, value in versions.items()
            },
        )
        for index in range(batch_size)
    ]


def _required_id_list(
    values: list[int] | None,
    name: str,
    batch_size: int,
) -> list[int]:
    if values is None:
        raise ValueError(f"batched rollout result must include {name}")
    out = [int(value) for value in values]
    if len(out) != int(batch_size):
        raise ValueError(f"batched rollout result {name} batch size mismatch")
    return out


def _batch_result_values(values: list[Any]) -> torch.Tensor:
    tensors = [as_tensor(value).detach().cpu() for value in values]
    shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != shape for tensor in tensors):
        raise ValueError("rollout result tensors must share shape for batching")
    return torch.cat([tensor.reshape(1, *shape) for tensor in tensors], dim=0)


def _batch_forward_input_values(values: list[Any]) -> torch.Tensor:
    rows = []
    for value in values:
        tensor = as_tensor(value).detach().cpu()
        if tensor.ndim > 0 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        rows.append(tensor)
    shape = tuple(rows[0].shape)
    if any(tuple(row.shape) != shape for row in rows):
        raise ValueError("rollout forward_inputs must share shape for batching")
    return torch.cat([row.reshape(1, *shape) for row in rows], dim=0)


def _unpack_forward_input_row(key: str, row: torch.Tensor) -> torch.Tensor:
    row = row.detach().cpu()
    if str(key) == "lang_emb":
        return row
    return row.reshape(1, *tuple(row.shape))


def _cat_step_batch(
    values: list[Any],
    *,
    name: str = "trajectory tensor",
) -> torch.Tensor:
    tensors = [as_tensor(value).detach().cpu() for value in values]
    if not tensors:
        raise ValueError(f"{name} requires at least one tensor")
    max_ndim = max(int(tensor.ndim) for tensor in tensors)
    normalized: list[torch.Tensor] = []
    for tensor in tensors:
        while tensor.ndim < max_ndim:
            tensor = tensor.unsqueeze(-1)
        normalized.append(tensor)

    expected = tuple(normalized[0].shape[:1]) + tuple(normalized[0].shape[2:])
    for tensor in normalized[1:]:
        shape = tuple(tensor.shape[:1]) + tuple(tensor.shape[2:])
        if shape != expected:
            shapes = [tuple(int(dim) for dim in item.shape) for item in tensors]
            raise ValueError(
                f"{name} tensors must share non-batch dimensions; got {shapes}"
            )
    if len(normalized) == 1:
        return normalized[0]
    return torch.cat(normalized, dim=1)


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
            ],
            name="prev_values",
        )

    return TrajectoryBatch(
        actions=_cat_step_batch(
            [_pad_step_batch(shard.actions, max_steps) for shard in shards],
            name="actions",
        ),
        rewards=_cat_step_batch(
            [_pad_step_batch(shard.rewards, max_steps) for shard in shards],
            name="rewards",
        ).float(),
        dones=_cat_step_batch(
            [_pad_step_batch(shard.dones, max_steps, pad_value=True) for shard in shards],
            name="dones",
        ).bool(),
        prev_logprobs=_cat_step_batch(
            [_pad_step_batch(shard.prev_logprobs, max_steps) for shard in shards],
            name="prev_logprobs",
        ).float(),
        prev_values=prev_values,
        forward_inputs={
            key: _cat_step_batch(
                [
                    _pad_step_batch(shard.forward_inputs[key], max_steps)
                    for shard in shards
                ],
                name=f"forward_inputs[{key!r}]",
            )
            for key in sorted(forward_keys)
        },
        versions={
            key: _cat_step_batch(
                [_pad_step_batch(shard.versions[key], max_steps) for shard in shards],
                name=f"versions[{key!r}]",
            )
            for key in sorted(version_keys)
        },
        loss_mask=_cat_step_batch(
            [
                _shard_loss_mask(shard, max_steps, batch_size)
                for shard, batch_size in zip(shards, batch_sizes, strict=True)
            ],
            name="loss_mask",
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
