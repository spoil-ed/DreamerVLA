from __future__ import annotations

import random

import numpy as np

from scripts.training.train_online_rynnvla_action_hidden_dreamervla import (
    OnlineReplay,
    pack_replay_task_stats_for_ddp,
    unpack_replay_task_stats_from_ddp,
)


def _step(task_id: int, t: int, *, success: bool = False, done: bool = False) -> dict:
    return {
        "image": np.full((1, 1, 3), t, dtype=np.uint8),
        "obs_embedding": np.full((2,), t, dtype=np.float32),
        "policy_action": np.zeros((1,), dtype=np.float32),
        "wm_action": np.full((1,), t, dtype=np.float32),
        "reward": np.float32(1.0 if success else 0.0),
        "done": np.float32(done or success),
        "is_first": t == 0,
        "is_terminal": np.float32(success),
        "is_last": np.float32(done or success),
        "task_id": task_id,
    }


def _episode(task_id: int, length: int, *, success: bool) -> list[dict]:
    return [
        _step(task_id, t, success=success and t == length - 1, done=t == length - 1)
        for t in range(length)
    ]


def test_online_replay_samples_failed_episodes_only_from_prefix() -> None:
    random.seed(0)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        failure_prefix_steps=4,
        task_balanced=True,
    )
    replay.add_episode(_episode(task_id=2, length=10, success=False))

    batch = replay.sample(16)

    assert set(batch["task_ids"].tolist()) == {2}
    assert batch["start_indices"].max().item() <= 1


def test_online_replay_balances_available_tasks() -> None:
    random.seed(1)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        failure_prefix_steps=4,
        task_balanced=True,
    )
    replay.add_episode(_episode(task_id=2, length=10, success=True))
    replay.add_episode(_episode(task_id=9, length=10, success=True))

    batch = replay.sample(6)

    assert batch["task_ids"].tolist().count(2) == 3
    assert batch["task_ids"].tolist().count(9) == 3


def test_online_replay_training_readiness_requires_each_task() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3)

    replay.add_episode(_episode(task_id=2, length=10, success=True))

    assert (
        replay.ready_for_training(
            min_transitions=3,
            task_ids=(2, 9),
            min_episodes_per_task=1,
        )
        is False
    )

    replay.add_episode(_episode(task_id=9, length=10, success=False))

    assert (
        replay.ready_for_training(
            min_transitions=3,
            task_ids=(2, 9),
            min_episodes_per_task=1,
        )
        is True
    )


def test_online_replay_reports_per_task_start_pool_stats() -> None:
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        failure_prefix_steps=4,
        failure_prefix_ratio=0.0,
    )

    replay.add_episode(_episode(task_id=2, length=10, success=True))
    replay.add_episode(_episode(task_id=2, length=10, success=False))
    replay.add_episode(_episode(task_id=9, length=5, success=False))

    stats = replay.task_stats(task_ids=(2, 9))

    assert stats["2"]["episodes"] == 2
    assert stats["2"]["successes"] == 1
    assert stats["2"]["failures"] == 1
    assert stats["2"]["sampleable_windows"] == 10
    assert stats["9"]["episodes"] == 1
    assert stats["9"]["successes"] == 0
    assert stats["9"]["failures"] == 1
    assert stats["9"]["sampleable_windows"] == 2


def test_online_replay_keeps_independent_capacity_per_requested_task() -> None:
    replay = OnlineReplay(
        capacity=12,
        sequence_length=3,
        task_ids=(0, 1),
        capacity_mode="per_task",
    )

    replay.add_episode(_episode(task_id=0, length=10, success=True))
    replay.add_episode(_episode(task_id=1, length=10, success=True))
    replay.add_episode(_episode(task_id=1, length=10, success=True))

    stats = replay.task_stats(task_ids=(0, 1))

    assert stats["0"]["episodes"] == 1
    assert stats["0"]["transitions"] == 10
    assert stats["1"]["episodes"] == 1
    assert stats["1"]["transitions"] == 10


def test_online_replay_can_report_global_ddp_task_stats() -> None:
    replay_rank0 = OnlineReplay(capacity=100, sequence_length=3, task_ids=(0, 1))
    replay_rank1 = OnlineReplay(capacity=100, sequence_length=3, task_ids=(0, 1))
    replay_rank0.add_episode(_episode(task_id=0, length=10, success=True))
    replay_rank1.add_episode(_episode(task_id=1, length=8, success=False))

    packed = pack_replay_task_stats_for_ddp(
        replay_rank0,
        task_ids=(0, 1),
        min_transitions=3,
        min_episodes_per_task=1,
    ) + pack_replay_task_stats_for_ddp(
        replay_rank1,
        task_ids=(0, 1),
        min_transitions=3,
        min_episodes_per_task=1,
    )

    stats, coverage_ready, all_ranks_ready = unpack_replay_task_stats_from_ddp(
        packed,
        task_ids=(0, 1),
        world_size=2,
        min_transitions=3,
        min_episodes_per_task=1,
    )

    assert coverage_ready is True
    assert all_ranks_ready is False
    assert stats["0"]["episodes"] == 1
    assert stats["0"]["successes"] == 1
    assert stats["1"]["episodes"] == 1
    assert stats["1"]["failures"] == 1
