from __future__ import annotations

import os

import pytest
import ray

from dreamervla.scheduler._test_workers import EchoWorker
from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy, PackedPlacementStrategy
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


def test_cluster_defaults_ray_zero_gpu_env_override(monkeypatch) -> None:
    monkeypatch.delenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", raising=False)
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        assert os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"
    finally:
        cluster.shutdown()


@pytest.mark.parametrize("num_gpus", [2, 3, 4])
def test_worker_group_packed_gpu_env_maps_visible_devices(num_gpus: int) -> None:
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        if cluster.num_gpus < num_gpus:
            pytest.skip(f"requires at least {num_gpus} visible GPUs")
        group = WorkerGroup(EchoWorker, 0).launch(
            cluster,
            PackedPlacementStrategy(0, num_gpus - 1),
        )

        infos = group.rank_info().wait()

        parent = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        parent_devices = [part.strip() for part in parent.split(",") if part.strip()]
        expected = (
            parent_devices[:num_gpus]
            if len(parent_devices) >= num_gpus
            else [str(idx) for idx in range(num_gpus)]
        )
        assert [info["visible_accelerators"] for info in infos] == expected
        assert [info["device"] for info in infos] == ["cuda:0"] * num_gpus
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
