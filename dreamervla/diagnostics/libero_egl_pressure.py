"""Ray pressure diagnostic for LIBERO EGL rendering.

This module intentionally avoids importing LIBERO / robosuite at module import
time. Each worker first applies the render regime, then builds the LIBERO env.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dreamervla.runtime.render_device import (
    cuda_visible_devices_from_env,
    parse_device_ids,
)
from dreamervla.utils.egl_device import apply_libero_render_regime


@dataclass(frozen=True)
class PressureWorkerConfig:
    backend: str = "egl"
    shard_id: int = 0
    gpu_pool: tuple[int, ...] = ()
    steps: int = 512
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    seed: int = 0
    action_mode: str = "zeros"
    action_scale: float = 1.0
    image_size: int = 64
    resolution: int = 256
    warmup_steps: int = 10
    cuda_stress_mb: int = 0
    cuda_stress_matmul_size: int = 1024
    cuda_stress_sleep_s: float = 0.0


def pressure_config_from_args(
    args: argparse.Namespace,
    *,
    shard_id: int,
) -> PressureWorkerConfig:
    """Build one worker config from parsed CLI args."""

    gpu_pool = tuple(_gpu_pool_from_arg(args.gpu_pool))
    return PressureWorkerConfig(
        backend=str(args.backend).strip().lower(),
        shard_id=int(shard_id),
        gpu_pool=gpu_pool,
        steps=int(args.steps),
        task_suite_name=str(args.task_suite_name),
        task_id=int(args.task_id),
        seed=int(args.seed),
        action_mode=str(args.action_mode).strip().lower(),
        action_scale=float(args.action_scale),
        image_size=int(args.image_size),
        resolution=int(args.resolution),
        warmup_steps=int(args.warmup_steps),
        cuda_stress_mb=int(getattr(args, "cuda_stress_mb", 0)),
        cuda_stress_matmul_size=int(getattr(args, "cuda_stress_matmul_size", 1024)),
        cuda_stress_sleep_s=float(getattr(args, "cuda_stress_sleep_s", 0.0)),
    )


def ray_runtime_env_for_worker(
    args: argparse.Namespace,
    *,
    shard_id: int,
) -> dict[str, dict[str, str]]:
    """Return RLinf-style worker-local GPU render env for one Ray actor."""

    if str(args.backend).strip().lower() != "egl":
        return {"env_vars": {}}
    devices = _gpu_pool_from_arg(args.gpu_pool)
    if not devices:
        return {"env_vars": {}}
    device = str(int(devices[int(shard_id) % len(devices)]))
    return {
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": device,
            "MUJOCO_EGL_DEVICE_ID": device,
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
    }


def run_pressure_worker(config: PressureWorkerConfig) -> dict[str, Any]:
    """Run one LIBERO env worker and return a JSON-serializable diagnostic."""

    started = time.time()
    steps_completed = 0
    episodes = 0
    env: Any | None = None
    result: dict[str, Any] = {
        "ok": False,
        "config": asdict(config),
        "pid": os.getpid(),
        "started_at": started,
        "render_env_before": _render_env_snapshot(),
        "egl_device_count": None,
        "cuda_stress": _cuda_stress_summary(config),
        "steps_completed": 0,
        "episodes": 0,
    }

    try:
        apply_libero_render_regime(
            config.backend,
            int(config.shard_id),
            list(config.gpu_pool),
        )
        result["render_env_after_apply"] = _render_env_snapshot()
        result["egl_device_count"] = _egl_device_count()
        env = _build_online_env(config)
        env.reset(task_id=int(config.task_id), episode_id=0)
        rng = np.random.default_rng(int(config.seed) + int(config.shard_id))
        last_info: dict[str, Any] = {}
        with _cuda_stress_context(config):
            for step in range(int(config.steps)):
                action = _make_action(config, rng, step)
                _obs, reward, terminated, truncated, info = env.step(action)
                steps_completed += 1
                last_info = dict(info or {})
                if bool(terminated or truncated):
                    episodes += 1
                    env.reset(task_id=int(config.task_id), episode_id=episodes)
        result.update(
            {
                "ok": True,
                "steps_completed": int(steps_completed),
                "episodes": int(episodes),
                "last_reward": float(reward) if steps_completed else 0.0,
                "last_success": bool(last_info.get("success", False)),
            }
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must report failures
        result.update(
            {
                "ok": False,
                "steps_completed": int(steps_completed),
                "episodes": int(episodes),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "render_env_at_error": _render_env_snapshot(),
            }
        )
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        result["duration_s"] = float(time.time() - started)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _run(args)
    payload = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)
    return 0 if all(item.get("ok") for item in output["workers"]) else 1


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_workers) <= 1:
        workers = [
            run_pressure_worker(pressure_config_from_args(args, shard_id=0)),
        ]
    else:
        workers = _run_with_ray(args)
    return {
        "workers": workers,
        "summary": {
            "ok": all(item.get("ok") for item in workers),
            "num_workers": len(workers),
            "steps_completed": sum(int(item.get("steps_completed", 0)) for item in workers),
        },
    }


def _run_with_ray(args: argparse.Namespace) -> list[dict[str, Any]]:
    import ray

    if not ray.is_initialized():
        ray.init(
            namespace="DreamerVLA",
            ignore_reinit_error=True,
            include_dashboard=False,
            num_gpus=int(args.ray_num_gpus),
            log_to_driver=True,
        )

    remote_worker = ray.remote(num_gpus=0)(run_pressure_worker)
    refs = [
        remote_worker.options(
            runtime_env=ray_runtime_env_for_worker(args, shard_id=rank)
        ).remote(pressure_config_from_args(args, shard_id=rank))
        for rank in range(int(args.num_workers))
    ]
    return list(ray.get(refs))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--gpu-pool", default="")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--ray-num-gpus", type=int, default=0)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--task-suite-name", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-mode", choices=("zeros", "random"), default="zeros")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--cuda-stress-mb", type=int, default=0)
    parser.add_argument("--cuda-stress-matmul-size", type=int, default=1024)
    parser.add_argument("--cuda-stress-sleep-s", type=float, default=0.0)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def _gpu_pool_from_arg(value: Any) -> list[int]:
    devices = parse_device_ids(value)
    if devices:
        return devices
    return cuda_visible_devices_from_env()


def _build_online_env(config: PressureWorkerConfig) -> Any:
    from dreamervla.envs.libero.libero_env import DreamerVLAOnlineTrainEnv

    return DreamerVLAOnlineTrainEnv(
        {
            "task_suite_name": config.task_suite_name,
            "task_id": int(config.task_id),
            "image_size": int(config.image_size),
            "resolution": int(config.resolution),
            "warmup_steps": int(config.warmup_steps),
            "seed": int(config.seed) + int(config.shard_id),
            "action_input": "normalized",
            "clip_actions": True,
            "reward_mode": "sparse_success",
            "history_length": 1,
            "vla_rotate_180": True,
            "pixel_rotate_180": False,
            "prompt_style": "vla_policy",
            "include_state": False,
            "obs_hidden_source": "hidden_token",
            "action_head_type": "oft_discrete_token",
            "full_record": False,
            "validate_canonical": True,
        }
    )


def _make_action(
    config: PressureWorkerConfig,
    rng: np.random.Generator,
    step: int,
) -> np.ndarray:
    del step
    if config.action_mode == "zeros":
        return np.zeros((7,), dtype=np.float32)
    if config.action_mode == "random":
        scale = min(max(float(config.action_scale), 0.0), 1.0)
        return rng.uniform(-scale, scale, size=(7,)).astype(np.float32)
    raise ValueError(f"unsupported action_mode={config.action_mode!r}")


def _render_env_snapshot() -> dict[str, str | None]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    )
    return {key: os.environ.get(key) for key in keys}


def _egl_device_count() -> int | None:
    try:
        from mujoco.egl import egl_ext as egl

        return len(egl.eglQueryDevicesEXT())
    except Exception:  # noqa: BLE001 - absence is diagnostic data, not fatal
        return None


def _cuda_stress_summary(config: PressureWorkerConfig) -> dict[str, Any]:
    mb = max(0, int(config.cuda_stress_mb))
    return {
        "enabled": mb > 0,
        "mb": mb,
        "matmul_size": max(1, int(config.cuda_stress_matmul_size)),
        "sleep_s": max(0.0, float(config.cuda_stress_sleep_s)),
    }


def _cuda_stress_context(config: PressureWorkerConfig):
    if int(config.cuda_stress_mb) <= 0:
        return contextlib.nullcontext()
    return _CudaStressContext(config)


class _CudaStressContext:
    """Background CUDA compute load for render/compute contention diagnostics."""

    def __init__(self, config: PressureWorkerConfig) -> None:
        self._config = config
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def __enter__(self) -> _CudaStressContext:
        self._thread = threading.Thread(
            target=self._run,
            name=f"libero-egl-cuda-stress-{int(self._config.shard_id)}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            self._stop.set()
            raise RuntimeError("CUDA stress worker did not start within 30s")
        if self._error is not None:
            raise RuntimeError("CUDA stress worker failed to start") from self._error
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        if exc_type is None and self._error is not None:
            raise RuntimeError("CUDA stress worker failed") from self._error
        return False

    def _run(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA stress requested but torch.cuda is unavailable")
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
            dtype = torch.float16
            matmul_size = max(1, int(self._config.cuda_stress_matmul_size))
            reserve_elems = max(1, int(self._config.cuda_stress_mb) * 1024 * 1024 // 4)
            reserve = torch.empty(reserve_elems, device=device, dtype=torch.float32)
            reserve.fill_(1.0)
            a = torch.randn((matmul_size, matmul_size), device=device, dtype=dtype)
            b = torch.randn((matmul_size, matmul_size), device=device, dtype=dtype)
            sleep_s = max(0.0, float(self._config.cuda_stress_sleep_s))
            c = None
            self._ready.set()
            while not self._stop.is_set():
                c = a @ b
                reserve[0] = reserve[0] + c.flatten()[0].float()
                torch.cuda.synchronize(device)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            del a, b, reserve
            if c is not None:
                del c
            torch.cuda.empty_cache()
        except BaseException as exc:  # noqa: BLE001 - surfaced to diagnostic result
            self._error = exc
            self._ready.set()


if __name__ == "__main__":
    raise SystemExit(main())
