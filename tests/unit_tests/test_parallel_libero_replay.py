from __future__ import annotations

import json
from argparse import Namespace
from concurrent.futures import Future
from pathlib import Path

import pytest


def test_task_replays_use_bounded_spawn_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamervla.preprocess.libero_utils import parallel_replay as module

    captured: dict[str, object] = {}

    class ImmediatePool:
        def __init__(self, *, max_workers: int, mp_context) -> None:
            captured["max_workers"] = max_workers
            captured["start_method"] = mp_context.get_start_method()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            return None

        def submit(self, function, request):
            future: Future[int] = Future()
            future.set_result(function(request))
            return future

    monkeypatch.setattr(module, "ProcessPoolExecutor", ImmediatePool)

    results = list(module.iter_task_results([1, 2, 3], num_workers=8, worker=lambda x: x * 2))

    assert sorted(results) == [2, 4, 6]
    assert captured == {"max_workers": 3, "start_method": "spawn"}


def test_one_worker_replay_stays_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamervla.preprocess.libero_utils import parallel_replay as module

    def reject_pool(**kwargs):
        raise AssertionError(f"pool must not be constructed: {kwargs}")

    monkeypatch.setattr(module, "ProcessPoolExecutor", reject_pool)
    visited: list[int] = []

    results = list(
        module.iter_task_results(
            [2, 1],
            num_workers=1,
            worker=lambda value: visited.append(value) or value,
        )
    )

    assert results == [2, 1]
    assert visited == [2, 1]


@pytest.mark.parametrize("num_workers", [0, -1])
def test_replay_worker_count_must_be_positive(num_workers: int) -> None:
    from dreamervla.preprocess.libero_utils.parallel_replay import iter_task_results

    with pytest.raises(ValueError, match="num_workers"):
        list(iter_task_results([1], num_workers=num_workers, worker=lambda value: value))


def test_resume_metadata_recovers_and_merges_task_shards(tmp_path: Path) -> None:
    from dreamervla.preprocess.libero_utils.parallel_replay import (
        load_resume_metadata,
        write_task_metadata_shard,
    )

    canonical = tmp_path / "metainfo.json"
    shard_dir = tmp_path / ".metainfo_shards"
    canonical.write_text(
        json.dumps({"task_a": {"demo_0": {"success": True}}}),
        encoding="utf-8",
    )
    write_task_metadata_shard(
        shard_dir,
        task_id=1,
        metadata={"task_b": {"demo_0": {"success": False}}},
    )

    metadata = load_resume_metadata(canonical, shard_dir)

    assert metadata == {
        "task_a": {"demo_0": {"success": True}},
        "task_b": {"demo_0": {"success": False}},
    }


def test_commit_result_atomically_updates_canonical_and_removes_shard(tmp_path: Path) -> None:
    from dreamervla.preprocess.libero_utils.parallel_replay import (
        ReplayTaskResult,
        commit_task_result,
        write_task_metadata_shard,
    )

    canonical = tmp_path / "metainfo.json"
    shard_dir = tmp_path / ".metainfo_shards"
    metadata = {"task_a": {"demo_0": {"success": True}}}
    write_task_metadata_shard(shard_dir, task_id=0, metadata=metadata)
    result = ReplayTaskResult(
        task_id=0,
        task_description="task a",
        output_path=str(tmp_path / "task_a.hdf5"),
        metadata=metadata,
        num_replays=2,
        num_successes=1,
        num_noops=3,
    )
    accumulated: dict[str, object] = {}

    commit_task_result(canonical, shard_dir, result, accumulated)

    assert json.loads(canonical.read_text(encoding="utf-8")) == metadata
    assert not (shard_dir / "task_000.json").exists()
    assert accumulated == metadata


def test_replay_totals_render_one_complete_summary_line() -> None:
    from dreamervla.preprocess.libero_utils.parallel_replay import ReplayTaskResult, ReplayTotals

    totals = ReplayTotals()
    totals.add(
        ReplayTaskResult(
            task_id=0,
            task_description="a",
            output_path="a.hdf5",
            metadata={},
            num_replays=151,
            num_successes=136,
            num_noops=20,
        )
    )
    totals.add(
        ReplayTaskResult(
            task_id=1,
            task_description="b",
            output_path="b.hdf5",
            metadata={},
            num_replays=7,
            num_successes=4,
            num_noops=6,
        )
    )

    assert totals.summary() == (
        "Total # episodes replayed: 158, Total # successes: 140 (88.6 %), "
        "Total # no-op actions filtered out: 26"
    )


def test_regeneration_main_dispatches_task_workers_and_parent_owns_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dreamervla.preprocess.libero_utils import (
        regenerate_libero_dataset_filter_no_op as module,
    )
    from dreamervla.preprocess.libero_utils.parallel_replay import ReplayTaskResult

    class FakeSuite:
        n_tasks = 2

    monkeypatch.setattr(
        module.benchmark,
        "get_benchmark_dict",
        lambda: {"libero_goal": FakeSuite},
    )
    captured: dict[str, object] = {}

    def fake_iter(requests, *, num_workers, worker):
        captured["task_ids"] = [request.task_id for request in requests]
        captured["num_workers"] = num_workers
        captured["worker"] = worker
        for task_id, success in ((0, 1), (1, 0)):
            metadata = {f"task_{task_id}": {"demo_0": {"success": bool(success)}}}
            yield ReplayTaskResult(
                task_id=task_id,
                task_description=f"task {task_id}",
                output_path=str(tmp_path / f"task_{task_id}.hdf5"),
                metadata=metadata,
                num_replays=1,
                num_successes=success,
                num_noops=task_id + 2,
            )

    monkeypatch.setattr(module, "iter_task_results", fake_iter)
    metainfo = tmp_path / "metainfo.json"

    module.main(
        Namespace(
            libero_task_suite="libero_goal",
            libero_raw_data_dir=str(tmp_path / "raw"),
            libero_target_dir=str(tmp_path / "target"),
            image_resolution=256,
            keep_noops=True,
            metainfo_json_out=str(metainfo),
            resume=False,
            num_workers=4,
        )
    )

    output = capsys.readouterr().out
    assert captured == {
        "task_ids": [0, 1],
        "num_workers": 4,
        "worker": module._replay_task,
    }
    assert output.count("Total # episodes replayed:") == 2
    assert "Total # episodes replayed: 2, Total # successes: 1 (50.0 %), " in output
    assert json.loads(metainfo.read_text(encoding="utf-8")) == {
        "task_0": {"demo_0": {"success": True}},
        "task_1": {"demo_0": {"success": False}},
    }


def test_preprocess_worker_count_is_hydra_owned_and_forwarded() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_config = (
        project_root / "configs/scripts/preprocess/regenerate_libero_dataset_filter_no_op.yaml"
    ).read_text(encoding="utf-8")
    suite_config = (project_root / "configs/scripts/preprocess/preprocess_suite.yaml").read_text(
        encoding="utf-8"
    )
    reward_script = (project_root / "scripts/preprocess/00_hdf5_reward.sh").read_text(
        encoding="utf-8"
    )

    assert "num_workers: 1" in script_config
    assert "LIBERO_REPLAY_WORKERS: ${num_procs}" in suite_config
    assert reward_script.count('num_workers="${LIBERO_REPLAY_WORKERS}"') == 2


def test_reproduction_preprocess_uses_selected_profile_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dreamervla.launchers import reproduce as module

    workflow = module.build_workflow(
        [
            "--config-name",
            "reproduce/prepare_assets",
            "--task",
            "libero_goal",
            "dry_run=true",
            f"data_root={tmp_path}",
        ]
    )
    commands: list[tuple[str, ...]] = []
    workflow.cfg.profile.task = "libero_object"
    monkeypatch.setattr(module, "_run", lambda command, **_: commands.append(tuple(command)))

    module._prepare_assets(workflow)

    preprocess = next(
        command for command in commands if "scripts/preprocess/prepare_libero_data.sh" in command[1]
    )
    assert f"task={workflow.cfg.profile.task}" in preprocess
