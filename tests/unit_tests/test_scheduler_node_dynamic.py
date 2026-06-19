from __future__ import annotations


def test_probe_local_node_returns_metadata() -> None:
    from dreamervla.scheduler.node import probe_local_node

    node = probe_local_node()

    assert node.node_id
    assert node.address
    assert node.alive is True


def test_component_scheduler_drains_ready_work() -> None:
    from dreamervla.scheduler.dynamic_scheduler import ComponentScheduler

    scheduler = ComponentScheduler(max_workers=1)
    try:
        work = scheduler.submit("learner", lambda value: value + 1, 2)

        assert work.wait() == 3
        assert scheduler.drain_ready() == {"learner": [3]}
        assert scheduler.drain_ready() == {}
    finally:
        scheduler.shutdown()
