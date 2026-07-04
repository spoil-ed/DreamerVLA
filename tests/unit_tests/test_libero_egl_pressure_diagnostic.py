import argparse

import numpy as np
import pytest

from dreamervla.diagnostics import libero_egl_pressure as diag


class _FakeEnv:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.steps = 0
        self.closed = False
        self.resets: list[tuple[int, int]] = []
        self.actions: list[np.ndarray] = []

    def reset(self, *, task_id: int, episode_id: int):
        self.resets.append((int(task_id), int(episode_id)))
        return {"step": 0}, {}

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action, dtype=np.float32))
        if self.fail_at is not None and self.steps == self.fail_at:
            raise RuntimeError("boom")
        return {"step": self.steps}, 0.0, False, False, {"success": False}

    def close(self):
        self.closed = True


def test_pressure_worker_applies_render_regime_before_building_env(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    fake_env = _FakeEnv()

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        events.append(("apply", backend, shard_id, tuple(gpu_pool)))

    def fake_build(config: diag.PressureWorkerConfig):
        events.append(("build", config.shard_id))
        assert events[0][0] == "apply"
        return fake_env

    monkeypatch.setattr(diag, "apply_libero_render_regime", fake_apply)
    monkeypatch.setattr(diag, "_build_online_env", fake_build)
    monkeypatch.setattr(diag, "_egl_device_count", lambda: 6)
    monkeypatch.setattr(diag, "_render_env_snapshot", lambda: {"MUJOCO_GL": "egl"})

    result = diag.run_pressure_worker(
        diag.PressureWorkerConfig(
            backend="egl",
            shard_id=1,
            gpu_pool=(2, 4),
            steps=3,
            task_id=5,
            seed=7,
            action_mode="zeros",
        )
    )

    assert events == [("apply", "egl", 1, (2, 4)), ("build", 1)]
    assert result["ok"] is True
    assert result["steps_completed"] == 3
    assert result["egl_device_count"] == 6
    assert fake_env.resets == [(5, 0)]
    assert len(fake_env.actions) == 3
    assert fake_env.closed is True


def test_pressure_worker_returns_structured_error_and_closes_env(monkeypatch) -> None:
    fake_env = _FakeEnv(fail_at=2)
    monkeypatch.setattr(diag, "apply_libero_render_regime", lambda *args: None)
    monkeypatch.setattr(diag, "_build_online_env", lambda config: fake_env)
    monkeypatch.setattr(diag, "_egl_device_count", lambda: None)
    monkeypatch.setattr(diag, "_render_env_snapshot", lambda: {})

    result = diag.run_pressure_worker(
        diag.PressureWorkerConfig(
            backend="egl",
            shard_id=0,
            gpu_pool=(0,),
            steps=4,
        )
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 1
    assert result["error_type"] == "RuntimeError"
    assert "boom" in result["error"]
    assert "Traceback" in result["traceback"]
    assert fake_env.closed is True


def test_pressure_config_from_args_normalizes_gpu_pool() -> None:
    args = argparse.Namespace(
        backend="egl",
        gpu_pool="0,2,5",
        steps=9,
        task_suite_name="libero_goal",
        task_id=3,
        seed=11,
        action_mode="random",
        action_scale=0.25,
        image_size=64,
        resolution=256,
        warmup_steps=2,
        cuda_stress_mb=256,
        cuda_stress_matmul_size=384,
        cuda_stress_sleep_s=0.01,
    )

    config = diag.pressure_config_from_args(args, shard_id=2)

    assert config.backend == "egl"
    assert config.shard_id == 2
    assert config.gpu_pool == (0, 2, 5)
    assert config.steps == 9
    assert config.action_mode == "random"
    assert config.action_scale == pytest.approx(0.25)
    assert config.cuda_stress_mb == 256
    assert config.cuda_stress_matmul_size == 384
    assert config.cuda_stress_sleep_s == pytest.approx(0.01)


def test_pressure_worker_runs_cuda_stress_context_around_env_steps(monkeypatch) -> None:
    events: list[str] = []
    fake_env = _FakeEnv()

    class _Stress:
        def __enter__(self):
            events.append("stress-start")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("stress-stop")
            return False

    def fake_build(config: diag.PressureWorkerConfig):
        events.append("build")
        return fake_env

    monkeypatch.setattr(diag, "apply_libero_render_regime", lambda *args: None)
    monkeypatch.setattr(diag, "_build_online_env", fake_build)
    monkeypatch.setattr(diag, "_egl_device_count", lambda: None)
    monkeypatch.setattr(diag, "_render_env_snapshot", lambda: {})
    monkeypatch.setattr(diag, "_cuda_stress_context", lambda config: _Stress())

    result = diag.run_pressure_worker(
        diag.PressureWorkerConfig(
            backend="egl",
            shard_id=0,
            gpu_pool=(0,),
            steps=2,
            cuda_stress_mb=64,
        )
    )

    assert result["ok"] is True
    assert events == ["build", "stress-start", "stress-stop"]
    assert result["cuda_stress"] == {
        "enabled": True,
        "mb": 64,
        "matmul_size": 1024,
        "sleep_s": 0.0,
    }


def test_ray_runtime_env_pins_each_egl_worker_to_one_gpu() -> None:
    args = argparse.Namespace(backend="egl", gpu_pool="0,2,5")

    runtime_env = diag.ray_runtime_env_for_worker(args, shard_id=2)

    assert runtime_env == {
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "5",
            "MUJOCO_EGL_DEVICE_ID": "5",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
    }
