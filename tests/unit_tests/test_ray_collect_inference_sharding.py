"""Unit tests for the multi-GPU ray-collection env->inference-worker partition."""

from __future__ import annotations

from dreamervla.runtime.rollout_collection_ray import _shard_env_ids_by_worker


def test_single_worker_owns_all_in_order():
    assert _shard_env_ids_by_worker([0, 1, 2, 3], 1) == {0: [0, 1, 2, 3]}


def test_round_robin_partition_stable_and_complete():
    shards = _shard_env_ids_by_worker([0, 1, 2, 3, 4, 5], 3)
    assert shards == {0: [0, 3], 1: [1, 4], 2: [2, 5]}
    flat = sorted(e for ids in shards.values() for e in ids)
    assert flat == [0, 1, 2, 3, 4, 5]


def test_empty_workers_omitted():
    # 2 workers but only even env ids -> worker 1 gets nothing and is omitted.
    assert _shard_env_ids_by_worker([0, 2, 4], 2) == {0: [0, 2, 4]}


def test_order_preserved_within_worker():
    shards = _shard_env_ids_by_worker([5, 2, 8, 1], 2)
    assert shards == {1: [5, 1], 0: [2, 8]}


def test_num_workers_floor_is_one():
    assert _shard_env_ids_by_worker([0, 1], 0) == {0: [0, 1]}
