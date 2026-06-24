"""DreamerVLA online-rollout **EGL** vec env, built on RLinf's vendored classes.

This is approach 1 of the two cotrain render backends: run each online-rollout env
in its own ``multiprocessing.spawn`` subprocess through RLinf's exact
``SubprocVectorEnv`` machinery (``dreamervla/envs/rlinf_venv.py``, vendored verbatim),
with RLinf's exact per-child EGL device regime.  Approach 2 (osmesa) stays on the
proven ``dreamervla.runners.vec_rollout_env.VecRolloutEnv``.

The structure mirrors RLinf's per-env-family adapter ``rlinf/envs/libero/venv.py``:
a spawn ``SubprocEnvWorker`` subclass + a domain ``_worker`` command loop, on top of
the vendored ``BaseVectorEnv``.  Only two things are DreamerVLA-specific and live here
(not in the vendored ``venv.py``):

1. the command protocol the child serves — DreamerVLA's online train env
   (``set_task`` / ``reset(task_id, episode_id)`` / ``step`` -> ``full_record``), i.e.
   the same protocol as ``VecRolloutEnv`` so this class is drop-in for the egl path;
2. the per-child EGL device regime, set in the child **before** the env (robosuite/
   mujoco) is imported, EXACTLY as RLinf's scheduler does for its env workers
   (``rlinf/scheduler/hardware/accelerators/nvidia_gpu.py`` lines 107-114):
   ``CUDA_VISIBLE_DEVICES`` + ``MUJOCO_EGL_DEVICE_ID`` (+ ``MUJOCO_GL=egl`` /
   ``PYOPENGL_PLATFORM=egl`` / ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES``),
   the device chosen per child from ``egl_device_pool`` (round-robin), so env load is
   spread across GPUs instead of stacked on one.

Env contract the child's env object must provide (same as ``VecRolloutEnv``):
    set_task(task_id) ; task_description ; reset(task_id=, episode_id=) ;
    step(action) -> (obs, reward, terminated, truncated, info) ; full_record() -> dict ;
    usable as a context manager (the default factory calls ``__enter__``).
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from dreamervla.envs.rlinf_venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    SubprocEnvWorker,
)


def _apply_egl_device_regime(egl_device_id: int | None) -> None:
    """Set MUJOCO/EGL/CUDA env vars EXACTLY as RLinf does (nvidia_gpu.py:107-114).

    Must run in the child BEFORE robosuite/mujoco import so the egl platform and
    device are locked consistently. ``MUJOCO_EGL_DEVICE_ID`` is kept inside
    ``CUDA_VISIBLE_DEVICES`` so robosuite's ``binding_utils`` consistency assert
    (binding_utils.py:29-35) is satisfied rather than bypassed.
    """
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if egl_device_id is not None:
        device = str(int(egl_device_id))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = device
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"


class _EglEnvFn:
    """Picklable zero-arg env builder: apply the egl regime + extra env vars in the
    child, then build the DreamerVLA env via ``factory(cfg_kwargs)``.

    Mirrors RLinf's per-env ``env_fn`` closures (``rlinf/envs/libero/venv.py``
    ``get_env_fns``), which likewise set per-env os.environ before constructing the
    env. Cloudpickle (used by ``CloudpickleWrapper``) pickles this for spawn.
    """

    def __init__(
        self,
        factory: Callable[[dict[str, Any]], Any],
        cfg_kwargs: dict[str, Any],
        env_vars: dict[str, str],
        egl_device_id: int | None,
    ) -> None:
        self.factory = factory
        self.cfg_kwargs = cfg_kwargs
        self.env_vars = env_vars
        self.egl_device_id = egl_device_id

    def __call__(self) -> Any:
        _apply_egl_device_regime(self.egl_device_id)
        for key, value in self.env_vars.items():
            os.environ[key] = value
        return self.factory(self.cfg_kwargs)


def _worker(parent: Any, p: Any, env_fn_wrapper: CloudpickleWrapper, obs_bufs: Any = None) -> None:
    """Spawn-child loop serving the DreamerVLA online-rollout protocol.

    Same skeleton as RLinf's ``rlinf/envs/libero/venv.py`` ``_worker`` (close the
    parent end, build the env from the cloudpickled ``env_fn``, then serve commands
    over the pipe), with DreamerVLA's command set and an explicit ``ready``/``error``
    init handshake so the parent can surface a child that fails to build.
    """
    del obs_bufs  # no shared-memory buffer on this path (pipe-first, like VecRolloutEnv)
    parent.close()
    try:
        env = env_fn_wrapper.data()
        p.send(("ready", None))
    except Exception as exc:  # noqa: BLE001 — surface init failure to the parent
        p.send(("error", repr(exc)))
        p.close()
        return
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the parent end closed
                break
            if cmd == "close":
                break
            elif cmd == "set_task":
                env.set_task(data)
                p.send(("ok", env.task_description))
            elif cmd == "reset":
                task_id, episode_id = data
                env.reset(task_id=task_id, episode_id=episode_id)
                p.send(("ok", env.full_record()))
            elif cmd == "step":
                _obs, reward, terminated, truncated, info = env.step(data)
                p.send(
                    ("ok", (float(reward), bool(terminated), bool(truncated), info, env.full_record()))
                )
            elif cmd == "task_description":
                p.send(("ok", env.task_description))
            else:
                p.send(("error", f"unknown cmd {cmd!r}"))
    except Exception as exc:  # noqa: BLE001
        try:
            p.send(("error", repr(exc)))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            if hasattr(env, "__exit__"):
                env.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        p.close()


class _EglSubprocEnvWorker(SubprocEnvWorker):
    """Spawn-context ``SubprocEnvWorker`` pointing at this module's ``_worker``.

    Verbatim the RLinf ``ReconfigureSubprocEnvWorker`` pattern
    (``rlinf/envs/libero/venv.py``): force ``get_context("spawn")`` (so the egl GL
    context initializes in a fresh interpreter), no shared-memory obs buffer.
    """

    def __init__(self, env_fn: Callable[[], Any], share_memory: bool = False) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = share_memory
        self.buffer = None
        args = (self.parent_remote, self.child_remote, CloudpickleWrapper(env_fn), self.buffer)
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)


def _default_factory(cfg_kwargs: dict[str, Any]) -> Any:
    """Build + enter the DreamerVLA online train env (same env as the osmesa path)."""
    from dreamervla.runners.vec_rollout_env import default_env_factory

    return default_env_factory(cfg_kwargs)


class OnlineEglVecEnv(BaseVectorEnv):
    """K online-rollout envs in K spawn subprocesses, on RLinf's ``BaseVectorEnv``.

    Drop-in for ``VecRolloutEnv`` on the egl path: identical public API
    (``num_envs``, ``reset`` / ``step`` / ``set_task`` / ``close``, context manager),
    but each env runs through RLinf's vendored ``BaseVectorEnv`` + spawn
    ``SubprocEnvWorker`` with RLinf's per-child egl device regime.

    Args:
        num_envs: number of parallel env subprocesses (K).
        cfg_kwargs: kwargs forwarded to the env factory in each child.
        egl_device_pool: physical GPU ids; child ``i`` renders on ``pool[i % len]``
            (round-robin spread, matching RLinf's placement). ``None``/empty leaves
            the egl device unset (picks device 0) — pass a pool for multi-GPU spread.
        env_vars: extra env vars set in each child before the env build (after the
            egl regime), e.g. ``LIBERO_CONFIG_PATH``; spawn does not inherit the
            parent's runtime env edits.
        factory: picklable ``cfg_kwargs -> env`` (default builds a
            ``DreamerVLAOnlineTrainEnv``). Override in tests for a fake env.
        start_timeout_s: seconds to wait for each child to report ``ready``.
    """

    def __init__(
        self,
        num_envs: int,
        cfg_kwargs: dict[str, Any],
        egl_device_pool: Sequence[int] | None = None,
        env_vars: dict[str, str] | None = None,
        factory: Callable[[dict[str, Any]], Any] = _default_factory,
        start_timeout_s: float = 900.0,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self._closed = False
        pool = [int(d) for d in (egl_device_pool or [])]
        extra_env_vars = dict(env_vars or {})
        env_fns = [
            _EglEnvFn(
                factory=factory,
                cfg_kwargs=cfg_kwargs,
                env_vars=extra_env_vars,
                egl_device_id=(pool[i % len(pool)] if pool else None),
            )
            for i in range(num_envs)
        ]
        # Build the worker pool through RLinf's BaseVectorEnv (spawn workers).
        BaseVectorEnv.__init__(
            self, env_fns, lambda fn: _EglSubprocEnvWorker(fn, share_memory=False)
        )
        # VecRolloutEnv-compatible alias (BaseVectorEnv stores it as ``env_num``).
        self.num_envs = self.env_num
        # recv each child's ready/init-failure handshake.
        for i, worker in enumerate(self.workers):
            conn = worker.parent_remote
            if not conn.poll(start_timeout_s):
                self.close()
                raise RuntimeError(f"egl env {i} timed out during init ({start_timeout_s}s)")
            status, msg = conn.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"egl env {i} init failed: {msg}")

    # ── core barrier (send-all-then-recv-all, like RLinf BaseVectorEnv.step) ──────
    def _broadcast(
        self,
        cmd: str,
        payloads: Sequence[Any],
        env_ids: Iterable[int] | None = None,
    ) -> list[Any]:
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if len(payloads) != len(ids):
            raise ValueError(f"{cmd}: got {len(payloads)} payloads for {len(ids)} envs")
        for eid, payload in zip(ids, payloads, strict=True):
            self.workers[eid].parent_remote.send((cmd, payload))
        results: list[Any] = []
        for eid in ids:
            status, val = self.workers[eid].parent_remote.recv()
            if status == "error":
                raise RuntimeError(f"egl env {eid}: {val}")
            results.append(val)
        return results

    # ── public API (mirrors VecRolloutEnv) ───────────────────────────────────────
    def reset(  # type: ignore[override]
        self,
        task_ids: Sequence[int],
        episode_ids: Sequence[int],
        env_ids: Iterable[int] | None = None,
    ) -> list[dict[str, Any]]:
        payloads = list(zip(task_ids, episode_ids, strict=True))
        return self._broadcast("reset", payloads, env_ids)

    def step(  # type: ignore[override]
        self,
        actions: Sequence[Any],
        env_ids: Iterable[int] | None = None,
    ) -> list[tuple[float, bool, bool, dict, dict]]:
        return self._broadcast("step", list(actions), env_ids)

    def set_task(
        self,
        task_ids: Sequence[int],
        env_ids: Iterable[int] | None = None,
    ) -> list[str]:
        return self._broadcast("set_task", list(task_ids), env_ids)

    def close(self) -> None:  # type: ignore[override]
        if self._closed:
            return
        self._closed = True
        for worker in self.workers:
            try:
                worker.parent_remote.send(("close", None))
            except Exception:  # noqa: BLE001 — pipe may already be broken
                pass
        for worker in self.workers:
            proc = worker.process
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        self.is_closed = True

    def __enter__(self) -> OnlineEglVecEnv:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False
