from __future__ import annotations

from pathlib import Path

import pytest


def test_scheduler_package_exports_public_primitives() -> None:
    from dreamervla.scheduler import (
        Channel,
        Cluster,
        NodePlacementStrategy,
        PackedPlacementStrategy,
        Placement,
        Worker,
        WorkerGroup,
    )

    assert Cluster.__name__ == "Cluster"
    assert Worker.__name__ == "Worker"
    assert WorkerGroup.__name__ == "WorkerGroup"
    assert Channel.__name__ == "Channel"
    assert Placement.__name__ == "Placement"
    assert PackedPlacementStrategy.__name__ == "PackedPlacementStrategy"
    assert NodePlacementStrategy.__name__ == "NodePlacementStrategy"


def test_ray_dependency_is_declared_for_fresh_installs() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")

    assert "[project.optional-dependencies]" in pyproject
    assert "ray = [" in pyproject
    assert '"ray==2.55.1"' in pyproject
    assert "ray[default]" not in pyproject
    assert "ray" not in requirements


def test_cluster_defaults_ray_task_event_capacity(monkeypatch) -> None:
    import os

    import dreamervla.scheduler.cluster as cluster_module
    from dreamervla.scheduler.cluster import Cluster

    monkeypatch.delenv("RAY_task_events_max_num_task_in_gcs", raising=False)
    monkeypatch.setattr(cluster_module.ray, "is_initialized", lambda: True)

    Cluster()

    assert os.environ["RAY_task_events_max_num_task_in_gcs"] == "1000000"


def test_cluster_respects_explicit_ray_task_event_capacity(monkeypatch) -> None:
    import os

    import dreamervla.scheduler.cluster as cluster_module
    from dreamervla.scheduler.cluster import Cluster

    monkeypatch.setenv("RAY_task_events_max_num_task_in_gcs", "42")
    monkeypatch.setattr(cluster_module.ray, "is_initialized", lambda: True)

    Cluster()

    assert os.environ["RAY_task_events_max_num_task_in_gcs"] == "42"


def test_packed_placement_rejects_non_positive_gpus_per_worker() -> None:
    from dreamervla.scheduler.placement import PackedPlacementStrategy

    with pytest.raises(ValueError, match="num_gpus_per_worker"):
        PackedPlacementStrategy(0, 1, num_gpus_per_worker=0)


def test_worker_group_gpu_env_vars_include_rlinf_egl_regime(monkeypatch) -> None:
    from dreamervla.scheduler.placement import Placement
    from dreamervla.scheduler.worker_group import WorkerGroup

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    placement = Placement(
        rank=0,
        local_rank=0,
        local_world_size=1,
        visible_accelerators=["2"],
        device="cuda:2",
    )

    env = WorkerGroup._env_vars(
        placement,
        extra_env_vars={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "2"
    assert env["MUJOCO_GL"] == "egl"
    assert env["PYOPENGL_PLATFORM"] == "egl"


def test_channel_get_timeout_reports_key() -> None:
    import ray

    from dreamervla.scheduler import Channel, Cluster

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        channel = Channel.create("test-channel-timeout")

        with pytest.raises(TimeoutError, match="missing-key"):
            channel.get(key="missing-key", timeout_s=0.01)
    finally:
        cluster.shutdown()


def test_channel_actor_disables_ray_task_events(monkeypatch) -> None:
    import dreamervla.scheduler.channel as channel_module

    captured: list[dict] = []

    class _RemoteMethod:
        def remote(self, *args, **kwargs):
            del args, kwargs
            return 0

    class _Actor:
        qsize = _RemoteMethod()

    class _RemoteClass:
        @staticmethod
        def options(**options):
            captured.append(options)
            return _RemoteClass

        @staticmethod
        def remote(*args, **kwargs):
            del args, kwargs
            return _Actor()

    monkeypatch.setattr(channel_module, "_ChannelActor", _RemoteClass)
    monkeypatch.setattr(channel_module.ray, "get", lambda ref: ref)

    channel_module.Channel.create("test-channel-task-events")

    assert captured[0]["enable_task_events"] is False
