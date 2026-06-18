from __future__ import annotations

import ray

from dreamervla.scheduler._test_workers import EchoWorker
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup


def test_worker_group_launches_workers_and_broadcasts_calls() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        group = WorkerGroup(EchoWorker, 10).launch(cluster, NodePlacementStrategy(3))

        infos = group.rank_info().wait()
        assert [info["rank"] for info in infos] == [0, 1, 2]
        assert [info["local_rank"] for info in infos] == [0, 1, 2]
        assert [info["world_size"] for info in infos] == [3, 3, 3]
        assert [info["device"] for info in infos] == ["cpu", "cpu", "cpu"]
        assert all(info["initialized"] for info in infos)

        result = group.add(5)
        assert result.done() is False or result.wait() == [15, 16, 17]
        assert result.wait() == [15, 16, 17]
    finally:
        cluster.shutdown()


def test_worker_group_execute_on_filters_next_call_only() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        group = WorkerGroup(EchoWorker, 20).launch(cluster, NodePlacementStrategy(2))

        assert group.execute_on(1).add(1).wait() == [22]
        assert group.add(1).wait() == [21, 22]
    finally:
        cluster.shutdown()
