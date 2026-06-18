"""SubprocVecEnv-style parallel env wrapper for within-rank rollout collection.

K LIBERO envs run in K spawned child processes; the parent drives them with a
**send-all-then-recv-all** protocol so the envs step in true parallel (CPU-bound
LIBERO/mujoco), while VLA inference batches across the K observations on the GPU
(see ``dreamervla.runners.rollout_hidden_extractor.batched_forward``).

This mirrors RLinf's ``SubprocVectorEnv`` (``rlinf/envs/venv/venv.py``): scatter one
action per env, then gather one result per env.  The key difference from the deleted
Layer-2 attempt is exactly this barrier — the old code did ``send; recv`` per handle in
a loop, serializing the envs and giving no speedup.

IPC: each step pipes back the env's ``full_record()`` (two 256x256x3 frames + proprio +
sim state, ~0.4 MB/env/step).  Pipe-first per the migration spec.

TODO(perf, deferred): swap the pipe for shared-memory obs buffers (RLinf ShmemVectorEnv /
``_setup_buf`` / ShArray) if a throughput profile shows IPC is a real fraction of step time.
Complication for us: ``states``/``init_state`` have a scene-dependent length S (libero_goal
79, object 110, ...) and our continuous loop crosses tasks, so a fixed shared buffer needs
max-S padding (or a hybrid: shared-mem for the fixed-shape 256x256 images, pipe for the
variable-S fields).  See migration spec §6.

Env contract (the child's env object must provide):
    set_task(task_id) ; task_description ; reset(episode_id=, task_id=) ;
    step(action) -> (obs, reward, terminated, truncated, info) ; full_record() -> dict ;
    and be usable as a context manager (the default factory calls ``__enter__``).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any


def default_env_factory(cfg_kwargs: dict[str, Any]) -> Any:
    """Build and enter a ``DreamerVLAOnlineTrainEnv`` from config kwargs (in the child)."""
    from dreamervla.envs.train_env import (
        DreamerVLAOnlineTrainEnv,
        DreamerVLAOnlineTrainEnvConfig,
    )

    env = DreamerVLAOnlineTrainEnv(DreamerVLAOnlineTrainEnvConfig(**cfg_kwargs))
    return env.__enter__()


def _worker(
    conn: Any,
    factory: Callable[[dict[str, Any]], Any],
    cfg_kwargs: dict[str, Any],
    env_vars: dict[str, str],
) -> None:
    """Child-process loop: build the env, then serve parent commands over the pipe."""
    for k, v in env_vars.items():
        os.environ[k] = v

    try:
        env = factory(cfg_kwargs)
        conn.send(("ready", None))
    except Exception as exc:  # noqa: BLE001 — surface init failure to the parent
        conn.send(("error", repr(exc)))
        conn.close()
        return

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            elif cmd == "set_task":
                env.set_task(payload)
                conn.send(("ok", env.task_description))
            elif cmd == "reset":
                task_id, episode_id = payload
                env.reset(episode_id=episode_id, task_id=task_id)
                conn.send(("ok", env.full_record()))
            elif cmd == "step":
                _obs, reward, terminated, truncated, info = env.step(payload)
                conn.send(
                    ("ok", (float(reward), bool(terminated), bool(truncated), info, env.full_record()))
                )
            elif cmd == "task_description":
                conn.send(("ok", env.task_description))
            else:
                conn.send(("error", f"unknown cmd {cmd!r}"))
    except EOFError:
        pass
    finally:
        try:
            env.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        conn.close()


class VecRolloutEnv:
    """K env subprocesses driven with a send-all-then-recv-all barrier.

    Args:
        num_envs: number of parallel env subprocesses (K).
        cfg_kwargs: kwargs forwarded to the env factory in each child.
        env_vars: env vars to set in each child before building the env (e.g. MUJOCO_GL,
            LIBERO_CONFIG_PATH) — spawn does not inherit the parent's runtime env edits.
        factory: picklable, module-level callable ``cfg_kwargs -> env`` (default builds a
            DreamerVLAOnlineTrainEnv).  Override in tests for a fake env.
        start_timeout_s: seconds to wait for each child to report ``ready``.
    """

    def __init__(
        self,
        num_envs: int,
        cfg_kwargs: dict[str, Any],
        env_vars: dict[str, str] | None = None,
        factory: Callable[[dict[str, Any]], Any] = default_env_factory,
        start_timeout_s: float = 180.0,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self._conns: list[Any] = []
        self._procs: list[Any] = []
        env_vars = env_vars or {}

        # send-all: start every child
        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child_conn, factory, cfg_kwargs, env_vars),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent keeps only its end
            self._conns.append(parent_conn)
            self._procs.append(proc)

        # recv-all: wait for every child's ready/error
        for i, conn in enumerate(self._conns):
            if not conn.poll(start_timeout_s):
                self.close()
                raise RuntimeError(f"env {i} timed out during init ({start_timeout_s}s)")
            status, msg = conn.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"env {i} init failed: {msg}")

    # ── core barrier ─────────────────────────────────────────────────────────
    def _broadcast(
        self,
        cmd: str,
        payloads: Sequence[Any],
        env_ids: Iterable[int] | None = None,
    ) -> list[Any]:
        """Send ``cmd`` to the addressed envs (all by default), then gather results."""
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if len(payloads) != len(ids):
            raise ValueError(f"{cmd}: got {len(payloads)} payloads for {len(ids)} envs")
        for eid, payload in zip(ids, payloads, strict=True):
            self._conns[eid].send((cmd, payload))
        results: list[Any] = []
        for eid in ids:
            status, val = self._conns[eid].recv()
            if status == "error":
                raise RuntimeError(f"env {eid}: {val}")
            results.append(val)
        return results

    # ── public API ───────────────────────────────────────────────────────────
    def reset(
        self,
        task_ids: Sequence[int],
        episode_ids: Sequence[int],
        env_ids: Iterable[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Reset the addressed envs; returns their post-reset ``full_record`` dicts."""
        payloads = list(zip(task_ids, episode_ids, strict=True))
        return self._broadcast("reset", payloads, env_ids)

    def step(
        self,
        actions: Sequence[Any],
        env_ids: Iterable[int] | None = None,
    ) -> list[tuple[float, bool, bool, dict, dict]]:
        """Step the addressed envs (all by default) in parallel.

        Returns per-env ``(reward, terminated, truncated, info, full_record)`` in the
        order of ``env_ids``.
        """
        return self._broadcast("step", list(actions), env_ids)

    def set_task(
        self,
        task_ids: Sequence[int],
        env_ids: Iterable[int] | None = None,
    ) -> list[str]:
        """Set the task on the addressed envs; returns their task descriptions."""
        return self._broadcast("set_task", list(task_ids), env_ids)

    def close(self) -> None:
        """Tell every child to exit and join; terminate any stragglers."""
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except Exception:  # noqa: BLE001 — pipe may already be broken
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    def __enter__(self) -> VecRolloutEnv:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False
