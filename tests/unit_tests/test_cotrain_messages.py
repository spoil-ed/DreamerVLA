from __future__ import annotations

import numpy as np
import pytest
import torch

from dreamervla.workers.cotrain.messages import (
    RolloutResultMsg,
    StopMsg,
    TrajectoryShard,
    collate_trajectory_shards,
)


def _trajectory_shard(
    *,
    batch_size: int,
    task_id: int = 0,
    episode_start: int = 10,
    prev_values: torch.Tensor | None = None,
) -> TrajectoryShard:
    return TrajectoryShard(
        env_rank=task_id,
        slot_id=0,
        task_id=task_id,
        episode_ids=list(range(episode_start, episode_start + batch_size)),
        actions=torch.full((2, batch_size, 3), float(task_id + 1)),
        rewards=torch.zeros(2, batch_size),
        dones=torch.zeros(2, batch_size, dtype=torch.bool),
        prev_logprobs=torch.zeros(2, batch_size),
        prev_values=prev_values,
        forward_inputs={"hidden": torch.full((2, batch_size, 4), float(task_id + 1))},
        versions={"policy": torch.full((2, batch_size), task_id + 1, dtype=torch.long)},
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


def test_collate_chunk_level_trajectory_shards_keeps_chunk_axis() -> None:
    shards = [
        TrajectoryShard(
            env_rank=0,
            slot_id=0,
            task_id=0,
            episode_ids=[10],
            actions=torch.ones(1, 1, 2, 3),
            rewards=torch.tensor([[[0.0, 1.0]]], dtype=torch.float32),
            dones=torch.tensor([[[False, True]]], dtype=torch.bool),
            prev_logprobs=torch.tensor([[0.25]], dtype=torch.float32),
            prev_values=None,
            forward_inputs={
                "hidden": torch.ones(1, 1, 4),
                "action": torch.ones(1, 1, 2, 3),
            },
            versions={"policy": torch.ones(1, 1, dtype=torch.long)},
        ),
        TrajectoryShard(
            env_rank=1,
            slot_id=0,
            task_id=0,
            episode_ids=[20],
            actions=torch.full((1, 1, 2, 3), 2.0),
            rewards=torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
            dones=torch.tensor([[[False, False]]], dtype=torch.bool),
            prev_logprobs=torch.tensor([[0.5]], dtype=torch.float32),
            prev_values=None,
            forward_inputs={
                "hidden": torch.full((1, 1, 4), 2.0),
                "action": torch.full((1, 1, 2, 3), 2.0),
            },
            versions={"policy": torch.full((1, 1), 2, dtype=torch.long)},
        ),
    ]

    batch = collate_trajectory_shards(shards)

    assert batch.actions.shape == (1, 2, 2, 3)
    assert batch.rewards.shape == (1, 2, 2)
    assert batch.dones.shape == (1, 2, 2)
    assert batch.prev_logprobs.shape == (1, 2)
    assert batch.forward_inputs["hidden"].shape == (1, 2, 4)
    assert batch.forward_inputs["action"].shape == (1, 2, 2, 3)
    assert batch.versions["policy"].tolist() == [[1, 2]]


def test_collate_trajectory_shards_normalizes_trailing_singleton_rank() -> None:
    shards = [
        TrajectoryShard(
            env_rank=0,
            slot_id=0,
            task_id=0,
            episode_ids=[10],
            actions=torch.ones(2, 1, 3),
            rewards=torch.zeros(2, 1),
            dones=torch.zeros(2, 1, dtype=torch.bool),
            prev_logprobs=torch.zeros(2, 1),
            prev_values=None,
            forward_inputs={
                "hidden": torch.ones(2, 1, 4),
                "scalar_score": torch.tensor([[0.1], [0.2]]),
            },
            versions={"policy": torch.ones(2, 1, dtype=torch.long)},
        ),
        TrajectoryShard(
            env_rank=1,
            slot_id=0,
            task_id=0,
            episode_ids=[20],
            actions=torch.full((2, 1, 3), 2.0),
            rewards=torch.zeros(2, 1),
            dones=torch.zeros(2, 1, dtype=torch.bool),
            prev_logprobs=torch.zeros(2, 1),
            prev_values=None,
            forward_inputs={
                "hidden": torch.full((2, 1, 4), 2.0),
                "scalar_score": torch.tensor([[[0.3]], [[0.4]]]),
            },
            versions={"policy": torch.full((2, 1), 2, dtype=torch.long)},
        ),
    ]

    batch = collate_trajectory_shards(shards)

    assert batch.forward_inputs["scalar_score"].shape == (2, 2, 1)
    assert torch.allclose(
        batch.forward_inputs["scalar_score"].squeeze(-1),
        torch.tensor([[0.1, 0.3], [0.2, 0.4]]),
    )


def test_collate_trajectory_shards_repeats_task_ids_by_batch_size() -> None:
    shards = [
        _trajectory_shard(batch_size=2, task_id=0, episode_start=10),
        _trajectory_shard(batch_size=1, task_id=1, episode_start=20),
    ]

    batch = collate_trajectory_shards(shards)

    assert batch.actions.shape == (2, 3, 3)
    assert batch.task_ids.tolist() == [0, 0, 1]
    assert batch.episode_ids.tolist() == [10, 11, 20]
    assert len(batch.episode_ids) == 3


def test_collate_trajectory_shards_rejects_mismatched_batch_dimensions() -> None:
    shard = _trajectory_shard(batch_size=2)
    shard = TrajectoryShard(
        env_rank=shard.env_rank,
        slot_id=shard.slot_id,
        task_id=shard.task_id,
        episode_ids=shard.episode_ids,
        actions=shard.actions,
        rewards=torch.zeros(2, 1),
        dones=shard.dones,
        prev_logprobs=shard.prev_logprobs,
        prev_values=shard.prev_values,
        forward_inputs=shard.forward_inputs,
        versions=shard.versions,
    )

    with pytest.raises(ValueError, match="batch dimension"):
        collate_trajectory_shards([shard])


def test_collate_trajectory_shards_rejects_mixed_prev_values_presence() -> None:
    shards = [
        _trajectory_shard(batch_size=1, task_id=0, prev_values=torch.zeros(2, 1)),
        _trajectory_shard(batch_size=1, task_id=1, prev_values=None),
    ]

    with pytest.raises(ValueError, match="prev_values"):
        collate_trajectory_shards(shards)


def test_stop_msg_is_distinct_control_message() -> None:
    assert StopMsg(reason="unit-test").reason == "unit-test"
