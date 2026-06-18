from __future__ import annotations

import socket

import ray


def test_cluster_initializes_ray_once_and_reports_single_node_resources() -> None:
    try:
        from dreamervla.scheduler.cluster import Cluster
    except ModuleNotFoundError as exc:
        raise AssertionError("Cluster module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()

    cluster = Cluster()
    again = Cluster()

    try:
        assert Cluster.has_initialized()
        assert ray.is_initialized()
        assert cluster.num_nodes == 1
        assert again.num_nodes == 1
        assert isinstance(cluster.num_gpus, int)
        assert cluster.num_gpus >= 0
    finally:
        cluster.shutdown()

    assert not ray.is_initialized()
    assert not Cluster.has_initialized()


def test_cluster_find_free_port_returns_bindable_port() -> None:
    try:
        from dreamervla.scheduler.cluster import Cluster
    except ModuleNotFoundError as exc:
        raise AssertionError("Cluster module should exist") from exc

    port = Cluster.find_free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))
