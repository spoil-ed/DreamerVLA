"""Static no-GPU coverage for routing LIBERO env construction through one helper."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

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


def test_manual_real_env_rejects_egl_when_cfg_pool_absent(monkeypatch) -> None:
    for key in (
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    worker = RealEnvWorker(
        env_cfg={"target": "unused", "render_backend": "egl"},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_id=0,
    )

    monkeypatch.setattr(trajectory_env_worker, "_build_env_from_cfg", lambda cfg: _Env())

    with pytest.raises(ValueError, match="render_backend=egl requires ngpu>=1"):
        worker.init()


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


def test_eval_loop_applies_libero_helper_from_configured_pool(monkeypatch) -> None:
    from dreamervla.runners import pretokenize_vla_runner

    calls: list[tuple[str, int, list[int]]] = []

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        calls.append((backend, int(shard_id), list(gpu_pool)))

    monkeypatch.setattr(
        pretokenize_vla_runner,
        "apply_libero_render_regime",
        fake_apply,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "eval": {
                "render_backend": "egl",
                "render_gpu_pool": [4, 6],
                "render_shard_id": 3,
            }
        }
    )

    pretokenize_vla_runner._apply_libero_eval_render_regime(cfg, cfg.eval)

    assert calls == [("egl", 3, [4, 6])]


def test_train_launcher_does_not_select_libero_render_backend(monkeypatch) -> None:
    from dreamervla.launchers.train import _build_env

    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        monkeypatch.delenv(key, raising=False)

    env = _build_env({"data_root": "/tmp/dvla-data", "env": {}})

    assert "MUJOCO_GL" not in env
    assert "PYOPENGL_PLATFORM" not in env


def test_sync_cotrain_env_kwargs_carry_libero_render_regime() -> None:
    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    runner = OnlineCotrainRunner.__new__(OnlineCotrainRunner)
    runner.distributed = SimpleNamespace(rank=0)
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "env": {"task_suite_name": "libero_goal", "episode_horizon": 20},
            "online_rollout": {
                "render_backend": "egl",
                "render_devices": [2, 5],
            },
        }
    )

    kwargs = runner._env_cfg_kwargs(cfg)

    assert kwargs["_libero_render_backend"] == "egl"
    assert kwargs["_libero_render_gpu_pool"] == [2, 5]
    assert kwargs["_libero_render_shard_id"] == 0


def test_libero_utils_module_import_does_not_import_libero_backend() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockLibero(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("libero"):
                    raise RuntimeError(f"unexpected LIBERO import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockLibero())
        import dreamervla.envs.libero.utils as libero_env
        assert libero_env.TASK_MAX_STEPS["libero_goal"] == 300
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
