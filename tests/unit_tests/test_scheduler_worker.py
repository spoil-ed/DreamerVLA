from __future__ import annotations


def test_worker_reads_ray_rank_environment(monkeypatch) -> None:
    try:
        from dreamervla.scheduler.worker import Worker
    except ModuleNotFoundError as exc:
        raise AssertionError("Worker module should exist") from exc

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    worker = Worker()

    assert worker.rank == 2
    assert worker.local_rank == 1
    assert worker.world_size == 4
    assert worker.local_world_size == 4
    assert worker.visible_accelerators == ["3"]
    assert worker.device == "cuda:0"


def test_worker_defaults_to_single_cpu_rank(monkeypatch) -> None:
    try:
        from dreamervla.scheduler.worker import Worker
    except ModuleNotFoundError as exc:
        raise AssertionError("Worker module should exist") from exc

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    worker = Worker()

    assert worker.rank == 0
    assert worker.local_rank == 0
    assert worker.world_size == 1
    assert worker.local_world_size == 1
    assert worker.visible_accelerators == []
    assert worker.device == "cpu"
    assert worker.init() is None


def test_worker_create_group_returns_unlaunched_group() -> None:
    try:
        from dreamervla.scheduler.worker import Worker
    except ModuleNotFoundError as exc:
        raise AssertionError("Worker module should exist") from exc

    group = Worker.create_group("arg", named=True)

    assert group.worker_cls is Worker
    assert group.args == ("arg",)
    assert group.kwargs == {"named": True}
