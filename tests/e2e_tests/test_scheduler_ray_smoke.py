from __future__ import annotations

import uuid

import ray

from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup


def test_scheduler_primitives_move_items_between_ray_workers() -> None:
    try:
        from dreamervla.scheduler._test_workers import ChannelWorker
    except ImportError as exc:
        raise AssertionError("ChannelWorker test actor should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        name = f"scheduler-smoke-{uuid.uuid4().hex}"
        Channel.create(name)
        group = WorkerGroup(ChannelWorker, name).launch(cluster, NodePlacementStrategy(2))

        group.execute_on(0).put_many(["a", "b", "c"]).wait()
        assert group.execute_on(1).get_many(3).wait() == [["a", "b", "c"]]
    finally:
        cluster.shutdown()
