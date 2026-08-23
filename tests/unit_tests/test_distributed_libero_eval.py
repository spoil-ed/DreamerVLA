"""Distributed standalone LIBERO evaluation sharding and aggregation."""

import pytest

from dreamervla.runtime.eval_metrics import (
    allocate_divisible_worker_budget,
    merge_libero_eval_rank_payloads,
    shard_libero_eval_tasks,
)


def test_eight_ranks_cover_ten_tasks_once() -> None:
    task_ids = list(range(10))

    shards = [shard_libero_eval_tasks(task_ids, rank=rank, world_size=8) for rank in range(8)]

    assert shards == [[0, 8], [1, 9], [2], [3], [4], [5], [6], [7]]
    assert sorted(task_id for shard in shards for task_id in shard) == task_ids


def test_global_environment_budget_preserves_complete_local_blocks() -> None:
    allocations = allocate_divisible_worker_budget(
        [20, 20, 10, 10, 10, 10, 10, 10],
        total_workers=25,
    )

    assert allocations == [5, 5, 5, 2, 2, 2, 2, 2]
    assert sum(allocations) == 25
    assert all(
        episodes % workers == 0
        for episodes, workers in zip([20, 20, 10, 10, 10, 10, 10, 10], allocations, strict=True)
    )


def test_rank_payloads_merge_task_records_and_additive_work() -> None:
    payloads = [
        {
            "records": {0: {0: True, 1: False}},
            "expected_episodes": 2,
            "env_chunk_steps": 4,
            "env_action_steps": 32,
            "elapsed_seconds": 10.0,
        },
        {
            "records": {1: {10: True, 11: True}},
            "expected_episodes": 2,
            "env_chunk_steps": 6,
            "env_action_steps": 48,
            "elapsed_seconds": 12.0,
        },
    ]

    metrics = merge_libero_eval_rank_payloads(payloads, episodes_per_task=2)

    assert metrics["eval_total_episodes"] == 4.0
    assert metrics["eval_total_successes"] == 3.0
    assert metrics["eval_success_rate"] == 0.75
    assert metrics["eval/env_chunk_steps"] == 10.0
    assert metrics["eval/env_action_steps"] == 80.0
    assert metrics["eval/elapsed_seconds"] == 12.0
    assert metrics["eval/env_chunk_per_s"] == 10.0 / 12.0


def test_rank_payload_merge_rejects_duplicate_or_missing_episodes() -> None:
    duplicate = {
        "records": {0: {0: True}},
        "expected_episodes": 1,
    }
    with pytest.raises(ValueError, match="duplicate eval result"):
        merge_libero_eval_rank_payloads(
            [duplicate, duplicate],
            episodes_per_task=1,
        )

    with pytest.raises(ValueError, match="rank 0.*expected 2, got 1"):
        merge_libero_eval_rank_payloads(
            [{"records": {0: {0: True}}, "expected_episodes": 2}],
            episodes_per_task=2,
        )
