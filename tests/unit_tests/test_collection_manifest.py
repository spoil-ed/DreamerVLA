"""Unified collected_rollouts space: manifest + episode-level resume helpers.

Collection now writes to a stable ``data/collected_rollouts/<task>/`` space with
a manifest (metadata + config) and episode-level resume: a relaunch tops up to
the target episode count by appending new shards instead of overwriting.
"""

import h5py

from dreamervla.dataset.collection_manifest import (
    count_collected_episodes,
    next_shard_index,
    read_manifest,
    resume_plan,
    write_manifest,
)


def _write_shard(path, num_demos: int) -> None:
    with h5py.File(str(path), "w") as f:
        data = f.create_group("data")
        data.attrs["num_demos"] = num_demos
        for i in range(num_demos):
            data.create_group(f"demo_{i}")


def test_count_collected_episodes_sums_num_demos_across_shards(tmp_path):
    _write_shard(tmp_path / "shard_000.hdf5", 3)
    _write_shard(tmp_path / "shard_001.hdf5", 2)
    assert count_collected_episodes(tmp_path) == 5


def test_count_collected_episodes_is_zero_when_empty(tmp_path):
    assert count_collected_episodes(tmp_path) == 0


def test_next_shard_index_returns_zero_when_empty(tmp_path):
    assert next_shard_index(tmp_path, prefix="shard") == 0


def test_next_shard_index_is_one_past_the_highest(tmp_path):
    (tmp_path / "shard_000.hdf5").touch()
    (tmp_path / "shard_002.hdf5").touch()
    assert next_shard_index(tmp_path, prefix="shard") == 3


def test_next_shard_index_respects_prefix(tmp_path):
    (tmp_path / "r0_shard_000.hdf5").touch()
    (tmp_path / "r0_shard_001.hdf5").touch()
    (tmp_path / "shard_000.hdf5").touch()  # different prefix, ignored
    assert next_shard_index(tmp_path, prefix="r0_shard") == 2


def test_resume_plan_full_collection_when_nothing_done():
    plan = resume_plan(target_total=500, num_tasks=10, collected=0)
    assert plan["complete"] is False
    assert plan["remaining"] == 500
    assert plan["episodes_per_task"] == 50


def test_resume_plan_tops_up_remaining_rounding_up_per_task():
    # 360 of 500 done across 10 tasks -> 140 remaining -> ceil(140/10)=14 per task.
    plan = resume_plan(target_total=500, num_tasks=10, collected=360)
    assert plan["complete"] is False
    assert plan["remaining"] == 140
    assert plan["episodes_per_task"] == 14


def test_resume_plan_complete_when_target_reached():
    plan = resume_plan(target_total=500, num_tasks=10, collected=500)
    assert plan["complete"] is True
    assert plan["remaining"] == 0
    assert plan["episodes_per_task"] == 0


def test_manifest_roundtrips(tmp_path):
    write_manifest(tmp_path, {"task": "libero_goal", "target": 500, "collected": 360})
    loaded = read_manifest(tmp_path)
    assert loaded["task"] == "libero_goal"
    assert loaded["target"] == 500
    assert loaded["collected"] == 360


def test_read_manifest_missing_returns_none(tmp_path):
    assert read_manifest(tmp_path) is None
