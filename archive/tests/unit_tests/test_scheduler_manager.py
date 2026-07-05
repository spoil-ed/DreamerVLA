from __future__ import annotations


def test_worker_manager_registers_routes_and_sends_to_bound_actor() -> None:
    from dreamervla.scheduler.manager import WorkerManager

    class _Actor:
        def add(self, value: int) -> int:
            return int(value) + 2

    manager = WorkerManager()
    route = manager.register("learner", rank=1, node_rank=0, actor=_Actor())

    assert route.rank == 1
    assert manager.resolve("learner") is route
    assert manager.recv(manager.send("learner", "add", 3)) == 5


def test_device_lock_manager_serializes_named_resources() -> None:
    from dreamervla.scheduler.manager import DeviceLockManager

    manager = DeviceLockManager()

    assert manager.acquire("cuda:0", blocking=False) is True
    assert manager.acquire("cuda:0", blocking=False) is False
    manager.release("cuda:0")
    assert manager.acquire("cuda:0", blocking=False) is True
    manager.release("cuda:0")
