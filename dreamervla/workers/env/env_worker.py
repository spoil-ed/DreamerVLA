"""Ray EnvWorker for single-env online rollout collection.

When egl GPU rendering is requested (``env_cfg["egl_device_pool"]`` is set), the env runs in a
clean ``multiprocessing.spawn`` subprocess so its egl GL context initializes in a FRESH
interpreter — no Ray/torch/CUDA pollution. This mirrors RLinf's per-env spawn venv and avoids
the robosuite ``read_pixels`` SIGABRT that hits in-Ray-actor egl at multi-worker concurrency.
The osmesa / synthetic-test path (no egl pool) stays in-process and unchanged.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np
import ray

from dreamervla.scheduler.worker import Worker


def _build_env_from_cfg(env_cfg: dict[str, Any]) -> Any:
    target = env_cfg.get("target") or env_cfg.get("_target_") or env_cfg.get("class_path")
    if not target:
        raise ValueError("env_cfg must include target/_target_/class_path")
    kwargs = dict(env_cfg.get("kwargs", {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    env_cls = getattr(module, class_name)
    if hasattr(env_cls, "from_config") and env_cfg.get("use_from_config", False):
        return env_cls.from_config(kwargs)
    return env_cls(**kwargs)


def _env_subprocess_main(conn, env_cfg, task_id, record_builder, egl_device_id):  # noqa: ANN001
    """Spawn-child loop: build one env in a fresh interpreter (clean egl context) and serve
    set_task / current_obs / step / close over the pipe. Transitions are built HERE (they need
    the env object); the parent EnvWorker accumulates them and pushes to the Ray replay.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if egl_device_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device_id))
        # RLinf-faithful device alignment: make CUDA visible on the SAME physical GPU
        # the egl context renders to. RLinf's env procs set CUDA_VISIBLE_DEVICES +
        # MUJOCO_EGL_DEVICE_ID together (nvidia_gpu.py); our CPU-placed EnvWorkers
        # blank CUDA while egl renders on a physical GPU, and that CUDA-absent /
        # egl-present mismatch is the likely robosuite read_pixels instability under
        # sustained concurrency. Aligning them here mirrors RLinf without GPU-placing
        # the Ray actor. Set before the env build (no CUDA/torch init has happened yet).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(egl_device_id))
        os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    try:
        env = _build_env_from_cfg(env_cfg)
        cur_task = int(task_id)
        episode_id = 0
        if hasattr(env, "set_task"):
            env.set_task(cur_task)
        cur_obs, _ = env.reset(task_id=cur_task, episode_id=episode_id)
        conn.send(("ready", cur_obs))
    except Exception as exc:  # noqa: BLE001 — surface init failure to the parent
        conn.send(("error", repr(exc)))
        conn.close()
        return
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            if cmd == "current_obs":
                conn.send(("ok", cur_obs))
            elif cmd == "set_task":
                cur_task = int(payload)
                if hasattr(env, "set_task"):
                    env.set_task(cur_task)
                episode_id = 0
                cur_obs, _ = env.reset(task_id=cur_task, episode_id=episode_id)
                conn.send(("ok", cur_obs))
            elif cmd == "step":
                action, obs_embedding = payload
                obs = cur_obs
                record_obs = obs
                if record_builder is not None and hasattr(env, "full_record"):
                    record_obs = dict(obs)
                    record_obs["_full_record"] = env.full_record()
                next_obs, reward, terminated, truncated, info = env.step(action)
                if record_builder is not None:
                    transition = record_builder(
                        env, record_obs, action, reward, terminated, truncated, info, obs_embedding
                    )
                else:
                    transition = env.make_transition(
                        obs, action, reward, terminated, truncated, info
                    )
                    transition["obs_embedding"] = np.asarray(obs_embedding, dtype=np.float32)
                done = bool(terminated or truncated)
                if done:
                    episode_id += 1
                    cur_obs, reset_info = env.reset(task_id=cur_task, episode_id=episode_id)
                    merged = dict(info or {})
                    merged["reset_info"] = reset_info
                    conn.send(("step", (transition, cur_obs, True, merged)))
                else:
                    cur_obs = next_obs
                    conn.send(("step", (transition, cur_obs, False, dict(info or {}))))
            else:
                conn.send(("error", f"unknown cmd {cmd!r}"))
    except EOFError:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", repr(exc)))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:  # noqa: BLE001
            pass
        conn.close()


class EnvWorker(Worker):
    """Hold one env instance, collect episodes, and push completed episodes.

    egl mode runs the env in a spawn subprocess (clean egl context); osmesa/synthetic mode
    runs it in-process. The external API (init/current_obs/set_task/step/close) is identical.
    """

    def __init__(
        self,
        env_cfg: dict[str, Any],
        task_id: int,
        replay: Any,
        record_builder: Any | None = None,
    ) -> None:
        super().__init__()
        self.env_cfg = dict(env_cfg)
        self.task_id = int(task_id)
        self.replay = replay
        self._record_builder = record_builder
        self.env: Any | None = None
        self.obs: dict[str, Any] | None = None
        self.episode: list[dict[str, Any]] = []
        self.episode_id = 0
        self._proc: Any | None = None
        self._conn: Any | None = None
        # egl spawn-child crash resilience (Phase 1b): remember the egl device so a
        # dead child can be respawned, and cap respawns so a persistently-crashing
        # env fails loudly instead of thrashing.
        self._egl_device_id: int | None = None
        self._respawn_count = 0
        self._max_respawns = int(self.env_cfg.get("egl_max_respawns", 5))

    @property
    def _spawned(self) -> bool:
        return self._proc is not None

    def init(self) -> None:
        pool = self.env_cfg.get("egl_device_pool")
        if pool:
            self._init_spawn(int(pool[int(self.local_rank) % len(pool)]))
        else:
            self._init_inproc()

    def _init_inproc(self) -> None:
        self.env = self._build_env(self.env_cfg)
        if hasattr(self.env, "set_task"):
            self.env.set_task(self.task_id)
        self.obs, _ = self._reset_env()
        self.episode = []
        self.episode_id = 0

    def _init_spawn(self, egl_device_id: int) -> None:
        # WorkerGroup.launch fires every worker's init.remote() at once, so all env workers
        # spawn their child simultaneously. Each child cold-starts a FRESH interpreter and builds
        # LIBERO/robosuite/egl — a thundering herd of cold starts that thrashes CPU/disk and can
        # blow a tight init timeout. Stagger the spawns by rank (later ranks ride the warmed page
        # cache) and use a generous, configurable init timeout.
        self._egl_device_id = int(egl_device_id)
        stagger_s = float(self.env_cfg.get("egl_spawn_stagger_s", 3.0)) * int(self.local_rank)
        if stagger_s > 0:
            time.sleep(stagger_s)
        init_timeout_s = float(self.env_cfg.get("egl_spawn_init_timeout_s", 900.0))
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_env_subprocess_main,
            args=(child_conn, self.env_cfg, self.task_id, self._record_builder, egl_device_id),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        if not parent_conn.poll(init_timeout_s):
            proc.terminate()
            raise RuntimeError(
                f"EnvWorker spawn subprocess timed out during init "
                f"(rank={self.local_rank}, timeout={init_timeout_s:.0f}s)"
            )
        status, payload = parent_conn.recv()
        if status != "ready":
            proc.terminate()
            raise RuntimeError(f"EnvWorker spawn subprocess init failed: {payload}")
        self._proc = proc
        self._conn = parent_conn
        self.obs = payload
        self.episode = []
        self.episode_id = 0

    def _rpc(self, cmd: str, payload: Any = None) -> Any:
        self._conn.send((cmd, payload))
        status, val = self._conn.recv()
        if status == "error":
            raise RuntimeError(f"EnvWorker subprocess error: {val}")
        return val

    def current_obs(self) -> dict[str, Any]:
        if self.obs is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        return self.obs

    def set_task(self, task_id: int) -> dict[str, Any]:
        self.task_id = int(task_id)
        if self._spawned:
            self.obs = self._rpc("set_task", self.task_id)
            self.episode = []
            self.episode_id = 0
            return self.obs
        env = self._env()
        if hasattr(env, "set_task"):
            env.set_task(self.task_id)
        self.episode = []
        self.episode_id = 0
        self.obs, _ = self._reset_env()
        return self.obs

    def step(
        self,
        action: Any,
        obs_embedding: Any,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        if self._spawned:
            return self._step_spawn(action, obs_embedding)
        return self._step_inproc(action, obs_embedding)

    def _step_spawn(
        self, action: Any, obs_embedding: Any
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        try:
            transition, obs, done, info = self._rpc("step", (action, obs_embedding))
        except (EOFError, OSError):
            # The spawn child died (silent native crash under sustained concurrent
            # egl). Drop the partial episode and respawn a clean child instead of
            # killing the whole job.
            return self._recover_from_child_death()
        self.episode.append(transition)
        self.obs = obs
        if done:
            ray.get(self.replay.add_episode.remote(list(self.episode)))
            self.episode = []
            self.episode_id += 1
            return self.obs, True, dict(info or {})
        return self.obs, False, dict(info or {})

    def _recover_from_child_death(
        self,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Respawn a dead egl child and return an episode boundary.

        The crashed child's env state is gone, so the in-flight episode is discarded
        (it never reaches the replay) and a fresh child is spawned with a clean reset.
        Bounded by ``egl_max_respawns`` so a persistently-crashing env fails loudly
        instead of thrashing forever.
        """
        self._respawn_count += 1
        if self._respawn_count > self._max_respawns:
            raise RuntimeError(
                f"EnvWorker egl child died {self._respawn_count} times "
                f"(rank={self.local_rank}); exceeded egl_max_respawns={self._max_respawns}"
            )
        print(
            f"[EnvWorker] egl spawn child died (rank={self.local_rank}); "
            f"respawn {self._respawn_count}/{self._max_respawns}, dropping partial episode",
            flush=True,
        )
        self.episode = []
        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        self._proc = None
        self._conn = None
        self._init_spawn(self._egl_device_id)
        self.episode_id += 1
        return self.obs, True, {"env_crash_recovered": True}

    def _step_inproc(
        self, action: Any, obs_embedding: Any
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        env = self._env()
        obs = self.current_obs()
        record_obs = obs
        if self._record_builder is not None and hasattr(env, "full_record"):
            record_obs = dict(obs)
            record_obs["_full_record"] = env.full_record()
        next_obs, reward, terminated, truncated, info = env.step(action)
        if self._record_builder is not None:
            transition = self._record_builder(
                env,
                record_obs,
                action,
                reward,
                terminated,
                truncated,
                info,
                obs_embedding,
            )
        else:
            transition = env.make_transition(obs, action, reward, terminated, truncated, info)
            transition["obs_embedding"] = np.asarray(obs_embedding, dtype=np.float32)
        self.episode.append(transition)

        done = bool(terminated or truncated)
        if done:
            ray.get(self.replay.add_episode.remote(list(self.episode)))
            self.episode = []
            self.episode_id += 1
            self.obs, reset_info = self._reset_env()
            merged_info = dict(info or {})
            merged_info["reset_info"] = reset_info
            return self.obs, True, merged_info

        self.obs = next_obs
        return self.obs, False, dict(info or {})

    def close(self) -> None:
        if self._spawned:
            try:
                self._conn.send(("close", None))
            except Exception:  # noqa: BLE001
                pass
            self._proc.join(timeout=10.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None
            self._conn = None
            return
        env = self.env
        if env is not None and hasattr(env, "close"):
            env.close()
        self.env = None

    def _reset_env(self) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self._env()
        return env.reset(task_id=self.task_id, episode_id=self.episode_id)

    def _env(self) -> Any:
        if self.env is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        return self.env

    @staticmethod
    def _build_env(env_cfg: dict[str, Any]) -> Any:
        return _build_env_from_cfg(env_cfg)
