"""Unified collected_rollouts space: manifest + episode-level resume helpers.

Collection now writes to a stable ``data/collected_rollouts/<task>/`` space with
a manifest (metadata + config) and episode-level resume: a relaunch tops up to
the target episode count by appending new shards instead of overwriting.
"""

import json

import h5py
import numpy as np

from dreamervla.dataset.collection_manifest import (
    complete_episode_ids_per_task,
    count_collected_episodes,
    count_episodes_per_task,
    format_collection_report,
    online_rollout_episode_counts,
    read_online_rollout_manifest,
    record_online_rollout_episode,
    next_shard_index,
    quarantine_corrupt_shards,
    quarantine_incomplete_shards,
    read_manifest,
    resume_plan,
    summarize_collection,
    write_manifest,
    load_online_rollout_episodes,
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


def _touch_rollout_pair(root, name: str) -> tuple:
    reward = root / "reward" / name
    hidden = root / "hidden" / name
    reward.parent.mkdir(parents=True, exist_ok=True)
    hidden.parent.mkdir(parents=True, exist_ok=True)
    reward.write_bytes(b"reward")
    hidden.write_bytes(b"hidden")
    return reward, hidden


def _write_reward_hidden_pair(
    reward_path,
    hidden_path,
    episodes,
) -> None:
    import numpy as np

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(reward_path), "w") as rf, h5py.File(str(hidden_path), "w") as hf:
        rdata = rf.create_group("data")
        hdata = hf.create_group("data")
        for idx, spec in enumerate(episodes):
            key = f"demo_{idx}"
            length = int(spec.get("length", 3))
            rgrp = rdata.create_group(key)
            rgrp.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
            rgrp.create_dataset("dones", data=np.zeros((length,), dtype=np.uint8))
            rgrp.create_dataset("rewards", data=np.zeros((length,), dtype=np.float32))
            rgrp.create_dataset("sparse_rewards", data=np.zeros((length,), dtype=np.uint8))
            rgrp.create_dataset("robot_states", data=np.zeros((length, 9), dtype=np.float32))
            rgrp.create_dataset("states", data=np.zeros((length, 5), dtype=np.float32))
            obs = rgrp.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((length, 4, 4, 3), dtype=np.uint8))
            obs.create_dataset("ee_pos", data=np.zeros((length, 3), dtype=np.float32))
            obs.create_dataset("ee_ori", data=np.zeros((length, 3), dtype=np.float32))
            obs.create_dataset("ee_states", data=np.zeros((length, 6), dtype=np.float32))
            obs.create_dataset("gripper_states", data=np.zeros((length, 2), dtype=np.float32))
            obs.create_dataset("joint_states", data=np.zeros((length, 7), dtype=np.float32))
            rgrp.attrs["num_samples"] = str(length)
            rgrp.attrs["task_id"] = int(spec["task_id"])
            rgrp.attrs["episode_id"] = int(spec["episode_id"])
            if "complete" in spec:
                rgrp.attrs["complete"] = bool(spec["complete"])
            if spec.get("hidden", True):
                hgrp = hdata.create_group(key)
                hidden_length = int(spec.get("hidden_length", length))
                hgrp.create_dataset(
                    "obs_embedding",
                    data=np.zeros((hidden_length, 8), dtype=np.float16),
                )
        rdata.attrs["num_demos"] = len(episodes)
        hdata.attrs["num_demos"] = len(episodes)


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


def test_next_shard_index_accepts_metadata_named_shards(tmp_path):
    (tmp_path / "shard_000.hdf5").touch()
    (tmp_path / "shard_gs000040_success_002.hdf5").touch()
    assert next_shard_index(tmp_path, prefix="shard") == 3


def test_next_shard_index_respects_prefix(tmp_path):
    (tmp_path / "r0_shard_000.hdf5").touch()
    (tmp_path / "r0_shard_001.hdf5").touch()
    (tmp_path / "shard_000.hdf5").touch()  # different prefix, ignored
    assert next_shard_index(tmp_path, prefix="r0_shard") == 2


def test_online_rollout_manifest_prunes_to_recent_global_steps(tmp_path):
    root = tmp_path / "online_cotrain_hidden_token"
    for episode_id, global_step, success in (
        (1, 120, True),
        (2, 121, False),
        (3, 122, True),
    ):
        reward, hidden = _touch_rollout_pair(
            root, f"cotrain_episode_gs{global_step:06d}_{episode_id:03d}.hdf5"
        )
        record_online_rollout_episode(
            root,
            reward_path=reward,
            hidden_path=hidden,
            task_id=7,
            episode_id=episode_id,
            init_state_index=episode_id,
            success=success,
            complete=True,
            global_step=global_step,
            env_step=1000 + global_step,
            keep_last_global_steps=2,
        )

    entries = read_online_rollout_manifest(root)

    assert [int(item["global_step"]) for item in entries] == [121, 122]
    assert online_rollout_episode_counts(root, task_ids=(7,)) == {7: 4}
    assert not (
        root
        / "episodes"
        / "task_07"
        / "global_step000120_success_True"
        / "ep_000001.h5"
    ).exists()
    expected = (
        root
        / "episodes"
        / "task_07"
        / "global_step000122_success_True"
        / "ep_000003.h5"
    )
    assert expected.is_file()
    assert set(entries[-1]) == {
        "global_step",
        "env_step",
        "task_id",
        "episode_id",
        "init_state_index",
        "success",
        "complete",
        "episode_path",
        "reward_path",
        "hidden_path",
    }
    assert not (root / "reward" / "cotrain_episode_gs000120_001.hdf5").exists()
    assert not (root / "hidden" / "cotrain_episode_gs000120_001.hdf5").exists()


def test_load_online_rollout_episodes_rebuilds_eight_dim_proprio(tmp_path):
    root = tmp_path / "online_cotrain_hidden_token"
    reward = root / "reward" / "ep.hdf5"
    hidden = root / "hidden" / "ep.hdf5"
    _write_reward_hidden_pair(
        reward,
        hidden,
        [{"task_id": 7, "episode_id": 3, "length": 2, "complete": True}],
    )
    with h5py.File(reward, "r+") as f:
        demo = f["data"]["demo_0"]
        demo.attrs["success"] = True
        obs = demo["obs"]
        obs["ee_pos"][...] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        obs["ee_ori"][...] = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        obs["gripper_states"][...] = np.array([[13, 14], [15, 16]], dtype=np.float32)
        demo["robot_states"][...] = np.ones((2, 9), dtype=np.float32) * 99
    record_online_rollout_episode(
        root,
        reward_path=reward,
        hidden_path=hidden,
        task_id=7,
        episode_id=3,
        init_state_index=3,
        success=True,
        complete=True,
        global_step=40,
        env_step=123,
        keep_last_global_steps=8,
    )

    episodes = load_online_rollout_episodes(root)

    assert len(episodes) == 1
    assert episodes[0][0]["proprio"].shape == (8,)
    assert episodes[0][0]["proprio"].tolist() == [1, 2, 3, 7, 8, 9, 13, 14]


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
                "hidden_dim": 1_048_576,
                "chunk_size": 8,
                "token_count": 256,
                "token_dim": 4096,
                "obs_hidden_source": "hidden_token",
                "obs_embedding_shape": [256, 4096],
                "hidden_storage_format": "tokenized",
                "output_dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    _write_reward_hidden_pair(
        reward_dir / "shard_000.hdf5",
        hidden_dir / "shard_000.hdf5",
        [
            {"task_id": 0, "episode_id": 0, "complete": True},
            {"task_id": 1, "episode_id": 0, "complete": True},
        ],
    )
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
    assert manifest["hidden_schema"]["hidden_dim"] == 1_048_576
    assert manifest["hidden_schema"]["chunk_size"] == 8
    assert manifest["hidden_schema"]["token_count"] == 256
    assert manifest["hidden_schema"]["token_dim"] == 4096
    assert manifest["hidden_schema"]["obs_hidden_source"] == "hidden_token"
    assert manifest["hidden_schema"]["obs_embedding_shape"] == [256, 4096]
    assert manifest["hidden_schema"]["hidden_storage_format"] == "tokenized"
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


def test_complete_episode_counts_require_matching_hidden_sidecar(tmp_path):
    reward = tmp_path / "reward"
    hidden = tmp_path / "hidden"
    _write_reward_hidden_pair(
        reward / "shard_000.hdf5",
        hidden / "shard_000.hdf5",
        [
            {"task_id": 0, "episode_id": 0, "complete": True},
            {"task_id": 0, "episode_id": 1, "complete": True, "hidden": False},
            {"task_id": 1, "episode_id": 0, "complete": False},
            {"task_id": 1, "episode_id": 1, "complete": True, "hidden_length": 1},
        ],
    )

    assert count_collected_episodes(reward, hidden) == 1
    assert count_episodes_per_task(reward, hidden) == {0: 1}
    assert complete_episode_ids_per_task(reward, hidden) == {0: {0}}


def test_complete_episode_ids_preserve_gaps_for_resume(tmp_path):
    reward = tmp_path / "reward"
    hidden = tmp_path / "hidden"
    _write_reward_hidden_pair(
        reward / "shard_000.hdf5",
        hidden / "shard_000.hdf5",
        [
            {"task_id": 0, "episode_id": 0, "complete": True},
            {"task_id": 0, "episode_id": 1, "complete": False},
            {"task_id": 0, "episode_id": 2, "complete": True},
        ],
    )

    assert complete_episode_ids_per_task(reward, hidden) == {0: {0, 2}}


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


def test_quarantine_incomplete_moves_reward_shard_when_hidden_sidecar_missing(tmp_path):
    reward = tmp_path / "reward"
    hidden = tmp_path / "hidden"
    _write_reward_hidden_pair(
        reward / "ray_shard_000.hdf5",
        hidden / "ray_shard_000.hdf5",
        [{"task_id": 0, "episode_id": 0, "complete": True}],
    )
    (hidden / "ray_shard_000.hdf5").unlink()

    assert quarantine_incomplete_shards(reward, hidden) == ["ray_shard_000.hdf5"]
    assert not (reward / "ray_shard_000.hdf5").exists()
    assert (reward / ".incomplete" / "ray_shard_000.hdf5").exists()


def test_append_episode_index_record(tmp_path):
    from dreamervla.dataset.collection_manifest import (
        EPISODE_INDEX_NAME,
        append_episode_index_record,
    )

    reward_dir = tmp_path / "reward"
    rec1 = {"file": "traj_t00_ep000000.hdf5", "task_id": 0, "episode_id": 0, "success": True}
    rec2 = {"file": "traj_t01_ep000003.hdf5", "task_id": 1, "episode_id": 3, "success": False}
    path = append_episode_index_record(reward_dir, rec1)
    append_episode_index_record(reward_dir, rec2)
    assert path == reward_dir / EPISODE_INDEX_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [rec1, rec2]


def test_append_episode_index_record_is_safe_under_parallel_writers(tmp_path):
    import multiprocessing

    from dreamervla.dataset.collection_manifest import (
        EPISODE_INDEX_NAME,
        append_episode_index_record,
    )

    reward_dir = tmp_path / "reward"
    n_procs, n_records = 4, 25

    def _worker(rank):
        for i in range(n_records):
            append_episode_index_record(
                reward_dir,
                {"file": f"traj_t{rank:02d}_ep{i:06d}.hdf5", "task_id": rank, "episode_id": i},
            )

    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_worker, args=(r,)) for r in range(n_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0

    lines = (reward_dir / EPISODE_INDEX_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_procs * n_records
    parsed = [json.loads(line) for line in lines]  # every line is well-formed JSON
    assert {(r["task_id"], r["episode_id"]) for r in parsed} == {
        (r, i) for r in range(n_procs) for i in range(n_records)
    }
