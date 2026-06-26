"""Unified collected_rollouts space: manifest + episode-level resume helpers.

Collection now writes to a stable ``data/collected_rollouts/<task>/`` space with
a manifest (metadata + config) and episode-level resume: a relaunch tops up to
the target episode count by appending new shards instead of overwriting.
"""

import h5py

from dreamervla.dataset.collection_manifest import (
    count_collected_episodes,
    count_episodes_per_task,
    format_collection_report,
    next_shard_index,
    quarantine_corrupt_shards,
    read_manifest,
    resume_plan,
    summarize_collection,
    write_manifest,
)


def _write_shard(path, num_demos: int) -> None:
    with h5py.File(str(path), "w") as f:
        data = f.create_group("data")
        data.attrs["num_demos"] = num_demos
        for i in range(num_demos):
            data.create_group(f"demo_{i}")


def _write_shard_with_task_ids(path, task_ids) -> None:
    with h5py.File(str(path), "w") as f:
        data = f.create_group("data")
        data.attrs["num_demos"] = len(task_ids)
        for i, tid in enumerate(task_ids):
            grp = data.create_group(f"demo_{i}")
            grp.attrs["task_id"] = int(tid)


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


def test_write_collection_manifest_records_hidden_schema(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import dreamervla.launchers.coldstart_warmup_cotrain as launcher

    del monkeypatch
    collected_root = tmp_path / "collected_rollouts" / "libero_goal"
    reward_dir = collected_root / "reward"
    hidden_dir = collected_root / "hidden"
    reward_dir.mkdir(parents=True)
    hidden_dir.mkdir(parents=True)
    collect_out = tmp_path / "run" / "collect"
    collect_out.mkdir(parents=True)
    resolved_config = "task:\n  suite: libero_goal\n"
    (collect_out / "resolved_config.yaml").write_text(resolved_config, encoding="utf-8")
    (hidden_dir / "preprocess_config.json").write_text(
        json.dumps(
            {
                "hidden_key": "obs_embedding",
                "hidden_dim": 229376,
                "chunk_size": 8,
                "token_count": 56,
                "token_dim": 4096,
                "output_dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    _write_shard_with_task_ids(reward_dir / "shard_000.hdf5", [0, 1])
    plan = SimpleNamespace(
        task="openvla_onetraj_coldstart_libero",
        mode="full",
        profile="release",
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        collected_root=collected_root,
        run_root=tmp_path / "run",
        collect_cmd=[
            "python",
            "-m",
            "dreamervla.train",
            "init.vla_ckpt_path=/ckpts/openvla",
        ],
    )

    launcher._write_collection_manifest(plan, target_episodes=10, num_tasks=2)

    manifest = json.loads((collected_root / "collection_manifest.json").read_text())
    assert manifest["suite"] == "libero_goal"
    assert manifest["target_episodes"] == 10
    assert manifest["collected_counts"] == {"total": 2, "per_task": {"0": 1, "1": 1}}
    assert manifest["policy_checkpoint"] == "/ckpts/openvla"
    assert manifest["hidden_schema"]["hidden_key"] == "obs_embedding"
    assert manifest["hidden_schema"]["hidden_dim"] == 229376
    assert manifest["hidden_schema"]["chunk_size"] == 8
    assert manifest["hidden_schema"]["token_count"] == 56
    assert manifest["hidden_schema"]["token_dim"] == 4096
    assert manifest["backend"] in {"unknown", "egl", "osmesa"}
    assert manifest["shards"] == ["shard_000.hdf5"]
    assert manifest["created_at"].endswith("Z")
    assert manifest["updated_at"].endswith("Z")
    assert manifest["resolved_config_snapshot"] == resolved_config
    assert manifest["resume_status"]["complete"] is False
    assert manifest["resume_status"]["remaining"] == 8


def test_read_manifest_missing_returns_none(tmp_path):
    assert read_manifest(tmp_path) is None


def test_count_episodes_per_task_buckets_by_task_id_attr(tmp_path):
    _write_shard_with_task_ids(tmp_path / "shard_000.hdf5", [0, 0, 1])
    _write_shard_with_task_ids(tmp_path / "shard_001.hdf5", [1, 2])
    assert count_episodes_per_task(tmp_path) == {0: 2, 1: 2, 2: 1}


def test_summarize_collection_reports_totals_per_task_and_remaining(tmp_path):
    _write_shard_with_task_ids(tmp_path / "shard_000.hdf5", [0, 0, 1])

    summary = summarize_collection(tmp_path, target_total=10, num_tasks=2)

    assert summary["total"] == 3
    assert summary["per_task"] == {0: 2, 1: 1}
    assert summary["target_total"] == 10
    assert summary["target_per_task"] == 5
    assert summary["remaining"] == 7
    assert summary["complete"] is False


def test_summarize_collection_complete_when_target_met(tmp_path):
    _write_shard_with_task_ids(tmp_path / "shard_000.hdf5", [0, 1, 2])

    summary = summarize_collection(tmp_path, target_total=3, num_tasks=3)

    assert summary["complete"] is True
    assert summary["remaining"] == 0


def test_summarize_collection_without_target_leaves_remaining_none(tmp_path):
    _write_shard_with_task_ids(tmp_path / "shard_000.hdf5", [0, 1])

    summary = summarize_collection(tmp_path, target_total=None, num_tasks=2)

    assert summary["total"] == 2
    assert summary["remaining"] is None
    assert summary["complete"] is False


def test_format_collection_report_mentions_counts_and_target(tmp_path):
    _write_shard_with_task_ids(tmp_path / "shard_000.hdf5", [0, 0, 1])
    summary = summarize_collection(tmp_path, target_total=10, num_tasks=2)

    report = format_collection_report(summary, root=tmp_path)

    assert "3" in report  # collected
    assert "10" in report  # target
    assert "7" in report  # remaining
    assert "task" in report.lower()


def test_quarantine_moves_corrupt_shard_and_keeps_good(tmp_path):
    reward = tmp_path / "reward"
    hidden = tmp_path / "hidden"
    reward.mkdir()
    hidden.mkdir()
    # valid shard in both dirs
    _write_shard(reward / "r0_shard_000.hdf5", 2)
    _write_shard(hidden / "r0_shard_000.hdf5", 2)
    # truncated/corrupt shard (e.g. left by a crashed collect) in both dirs
    (reward / "ray_shard_000.hdf5").write_bytes(b"\x00" * 96)
    (hidden / "ray_shard_000.hdf5").write_bytes(b"\x00" * 96)

    assert quarantine_corrupt_shards(reward, hidden) == ["ray_shard_000.hdf5"]

    # corrupt moved to .corrupt/ in BOTH dirs; valid shard untouched
    assert not (reward / "ray_shard_000.hdf5").exists()
    assert not (hidden / "ray_shard_000.hdf5").exists()
    assert (reward / ".corrupt" / "ray_shard_000.hdf5").exists()
    assert (hidden / ".corrupt" / "ray_shard_000.hdf5").exists()
    assert (reward / "r0_shard_000.hdf5").exists()
    assert (hidden / "r0_shard_000.hdf5").exists()


def test_quarantine_is_noop_when_all_shards_valid(tmp_path):
    reward = tmp_path / "reward"
    hidden = tmp_path / "hidden"
    reward.mkdir()
    hidden.mkdir()
    _write_shard(reward / "r0_shard_000.hdf5", 1)
    _write_shard(hidden / "r0_shard_000.hdf5", 1)

    assert quarantine_corrupt_shards(reward, hidden) == []
    assert not (reward / ".corrupt").exists()
