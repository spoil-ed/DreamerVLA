"""Spawn-subprocess LIBERO env for EGL-isolated evaluation.

Ported from RLinf (Apache-2.0): ``rlinf/envs/libero/venv.py`` (``_worker``,
``ReconfigureSubprocEnvWorker``) and ``rlinf/envs/venv/venv.py``
(``CloudpickleWrapper``, ``SubprocEnvWorker`` parent-side protocol), reduced to
the single-env case the eval loop needs and adapted to build the env (and set
its EGL device) inside the child via a cloudpickled ``env_fn``.

Why: the in-process eval rendered a LIBERO ``OffScreenRenderEnv`` in the same
process as the torch policy; mujoco EGL ``mjr_readPixels`` then aborts the
NVIDIA EGL driver after a few hundred renders. RLinf renders LIBERO in a spawned
child (``get_context("spawn")`` — fork would inherit CUDA/GL state and break
EGL), isolated from torch, and supports ``reconfigure`` (close + rebuild the env)
to reset the GL context between episodes. This mirrors that.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from typing import Any

import cloudpickle


class CloudpickleWrapper:
    """cloudpickle a callable so it survives ``spawn`` pickling (from RLinf)."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        self.data = cloudpickle.loads(data)


def _worker(p, env_fn_wrapper: CloudpickleWrapper) -> None:
    # env_fn runs in THIS child: it sets the EGL device then builds the env,
    # before any render, so mujoco EGL is isolated from the parent's torch CUDA.
    try:
        env = env_fn_wrapper.data()
        p.send(("ready", None))
    except Exception as exc:  # noqa: BLE001 - surface child init failure to parent
        p.send(("error", repr(exc)))
        p.close()
        return
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:
                p.close()
                break
            if cmd == "step":
                p.send(("ok", env.step(data)))
            elif cmd == "reset":
                p.send(("ok", env.reset(**(data or {}))))
            elif cmd == "set_init_state":
                p.send(("ok", env.set_init_state(data)))
            elif cmd == "get_sim_state":
                p.send(("ok", env.get_sim_state()))
            elif cmd == "reconfigure":
                env.close()
                env = env_fn_wrapper.data()
                p.send(("ok", None))
            elif cmd == "close":
                env.close()
                p.send(("ok", None))
                p.close()
                break
            else:
                p.send(("error", f"unknown cmd {cmd!r}"))
    except KeyboardInterrupt:
        p.close()


class EvalSubprocEnv:
    """Single LIBERO env in a spawned child; interface mirrors the in-proc env.

    ``env_fn`` is a picklable callable that (in the child) sets the render regime
    and returns a built ``OffScreenRenderEnv``.
    """

    def __init__(self, env_fn: Callable[[], Any], *, task_description: str) -> None:
        self.task_description = task_description
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, child_remote = ctx.Pipe()
        self.process = ctx.Process(
            target=_worker,
            args=(child_remote, CloudpickleWrapper(env_fn)),
            daemon=True,
        )
        self.process.start()
        child_remote.close()
        status, msg = self.parent_remote.recv()
        if status != "ready":
            raise RuntimeError(f"EvalSubprocEnv child failed to start: {msg}")

    def _rpc(self, cmd: str, data: Any = None) -> Any:
        self.parent_remote.send((cmd, data))
        status, val = self.parent_remote.recv()
        if status != "ok":
            raise RuntimeError(f"EvalSubprocEnv {cmd} failed: {val}")
        return val

    def reset(self, **kwargs: Any) -> Any:
        return self._rpc("reset", kwargs)

    def set_init_state(self, state: Any) -> Any:
        return self._rpc("set_init_state", state)

    def step(self, action: Any) -> Any:
        return self._rpc("step", action)

    def reconfigure(self) -> None:
        self._rpc("reconfigure")

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self._rpc("close")
            except (EOFError, OSError, BrokenPipeError):
                pass
        try:
            self.parent_remote.close()
        except OSError:
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()


def make_libero_env_fn(
    *,
    bddl_file_name: str,
    resolution: int,
    seed: int,
    render_backend: str,
    render_shard_id: int,
    render_gpu_pool: list[int],
) -> Callable[[], Any]:
    """Build a picklable env_fn that, in the child, applies the render regime
    (same shared helper collect/cotrain use) then returns an OffScreenRenderEnv."""

    def _env_fn() -> Any:
        from dreamervla.utils.egl_device import apply_libero_render_regime

        apply_libero_render_regime(
            str(render_backend), int(render_shard_id), list(render_gpu_pool)
        )
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            camera_heights=int(resolution),
            camera_widths=int(resolution),
        )
        env.seed(int(seed))
        return env

    return _env_fn


__all__ = ["EvalSubprocEnv", "make_libero_env_fn"]
