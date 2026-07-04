"""Static no-GPU coverage for routing LIBERO env construction through one helper."""

from __future__ import annotations

import os
from typing import Any

import pytest

from dreamervla.launchers import coldstart_warmup_cotrain as coldstart
from dreamervla.workers.env import env_worker, trajectory_env_worker
from dreamervla.workers.env.trajectory_env_worker import RealEnvWorker


class _Pipe:
    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.closed = False

    def send(self, value: Any) -> None:
        self.sent.append(value)

    def recv(self) -> tuple[str, None]:
        return ("close", None)

    def close(self) -> None:
        self.closed = True


class _Env:
    def reset(self, *, task_id: int = 0, episode_id: int = 0):
        return {"task_id": int(task_id), "episode_id": int(episode_id)}, {}

    def close(self) -> None:
        return None


def test_collect_spawn_child_applies_libero_helper_before_env_build(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        events.append(("helper", backend, int(shard_id), list(gpu_pool)))

    def fake_build(cfg: dict[str, Any]) -> _Env:
        events.append(("build", cfg.get("render_backend")))
        return _Env()

    monkeypatch.setattr(
        env_worker, "apply_libero_render_regime", fake_apply, raising=False
    )
    monkeypatch.setattr(env_worker, "_build_env_from_cfg", fake_build)

    conn = _Pipe()
    env_worker._env_subprocess_main(
        conn,
        {
            "target": "unused",
            "render_backend": "egl",
            "render_devices": [3, 5],
            "_render_shard_id": 4,
        },
        0,
        None,
        None,
    )

    assert events[:2] == [("helper", "egl", 4, [3, 5]), ("build", "egl")]
    assert conn.sent[0][0] == "ready"


def test_collect_inproc_path_applies_libero_helper_before_env_build(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    worker = env_worker.EnvWorker(
        env_cfg={"target": "unused", "render_backend": "osmesa", "render_devices": [7]},
        task_id=0,
        replay=None,
    )
    worker.local_rank = 2

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        events.append(("helper", backend, int(shard_id), list(gpu_pool)))

    def fake_build(cfg: dict[str, Any]) -> _Env:
        events.append(("build", cfg.get("render_backend")))
        return _Env()

    monkeypatch.setattr(
        env_worker, "apply_libero_render_regime", fake_apply, raising=False
    )
    monkeypatch.setattr(env_worker, "_build_env_from_cfg", fake_build)

    worker._init_inproc()

    assert events[:2] == [("helper", "osmesa", 2, [7]), ("build", "osmesa")]


def test_manual_real_env_applies_libero_helper_before_env_build(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    worker = RealEnvWorker(
        env_cfg={"target": "unused", "render_backend": "egl", "render_devices": [6, 8]},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )
    worker.local_rank = 3

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        events.append(("helper", backend, int(shard_id), list(gpu_pool)))

    def fake_build(cfg: dict[str, Any]) -> _Env:
        events.append(("build", cfg.get("render_backend")))
        return _Env()

    monkeypatch.setattr(
        trajectory_env_worker,
        "apply_libero_render_regime",
        fake_apply,
        raising=False,
    )
    monkeypatch.setattr(trajectory_env_worker, "_build_env_from_cfg", fake_build)

    try:
        worker.init()
    finally:
        worker.close()

    assert events[:2] == [("helper", "egl", 3, [6, 8]), ("build", "egl")]


def test_manual_real_env_uses_worker_visible_gpu_when_cfg_pool_absent(monkeypatch) -> None:
    for key in (
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    worker = RealEnvWorker(
        env_cfg={"target": "unused", "render_backend": "egl"},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )

    monkeypatch.setattr(trajectory_env_worker, "_build_env_from_cfg", lambda cfg: _Env())

    try:
        worker.init()
    finally:
        worker.close()

    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"


def test_post_step_eval_env_uses_libero_helper(monkeypatch, tmp_path) -> None:
    if not hasattr(coldstart, "_post_step_eval_env"):
        pytest.skip("post-step eval env helper is not present in this checkout")
    calls: list[tuple[str, int, list[int]]] = []

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        calls.append((backend, int(shard_id), list(gpu_pool)))
        os.environ["MUJOCO_GL"] = backend
        os.environ["PYOPENGL_PLATFORM"] = backend
        if backend == "egl":
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_pool[int(shard_id) % len(gpu_pool)])
            os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        else:
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

    for key in (
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        coldstart, "apply_libero_render_regime", fake_apply, raising=False
    )
    plan = coldstart.PipelinePlan(
        mode="ray",
        profile="test",
        task="goal",
        run_root=tmp_path,
        collected_root=tmp_path / "collect",
        reward_dir=tmp_path / "reward",
        hidden_dir=tmp_path / "hidden",
        collect_cmd=[],
        cotrain_cmd=[],
        eval_cfg={"render_backend": "egl", "gpus": "7,0"},
    )

    env = coldstart._post_step_eval_env(plan)

    assert calls == [("egl", 0, [0])]
    assert env["MUJOCO_GL"] == "egl"
    assert env["PYOPENGL_PLATFORM"] == "egl"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
