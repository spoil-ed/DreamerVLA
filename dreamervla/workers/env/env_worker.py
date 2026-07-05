"""Ray EnvWorker for online rollout collection.

The Ray EGL path follows RLinf's worker-level resource regime: WorkerGroup binds
each EnvWorker's ``CUDA_VISIBLE_DEVICES`` and ``MUJOCO_EGL_DEVICE_ID`` from
``cluster.component_placement.env``, then each EnvWorker may host multiple
spawned LIBERO env slots via ``env.num_envs_per_worker``. The children inherit
that already-bound regime; they do not pick independent render devices.

``env_cfg["egl_device_pool"]`` remains only as a legacy compatibility path for
old callers. Native child death is fatal by default; set
``env_cfg["egl_max_respawns"]`` to opt into dropping the partial episode and
respawning the affected slot.
"""

from __future__ import annotations

import importlib
import inspect
import multiprocessing as mp
import time
from typing import Any

import numpy as np
import ray

from dreamervla.scheduler.worker import Worker
from dreamervla.utils.egl_device import (
    apply_libero_render_regime,
    log_egl_device_diagnostics_from_env,
)


def _libero_render_backend(env_cfg: dict[str, Any], default: str = "osmesa") -> str:
    backend = str(env_cfg.get("render_backend", default)).strip().lower()
    if backend not in {"egl", "osmesa"}:
        raise ValueError(f"render_backend must be 'egl' or 'osmesa', got {backend!r}")
    return backend


def _libero_render_gpu_pool(env_cfg: dict[str, Any]) -> list[int]:
    from dreamervla.runners.render_device_config import parse_device_ids

    for key in ("gpu_pool", "render_devices", "egl_device_pool"):
        devices = parse_device_ids(env_cfg.get(key))
        if devices:
            return devices
    return []


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


def _record_builder_accepts_lang(record_builder: Any) -> bool:
    try:
        sig = inspect.signature(record_builder)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 9


def _call_record_builder(
    record_builder: Any,
    env: Any,
    obs: dict[str, Any],
    action: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    obs_embedding: Any,
    lang_emb: Any | None,
) -> dict[str, Any]:
    args = (env, obs, action, reward, terminated, truncated, info, obs_embedding)
    if _record_builder_accepts_lang(record_builder):
        return record_builder(*args, lang_emb)
    return record_builder(*args)


def _stamp_step_metadata(
    transition: dict[str, Any],
    step_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if not step_metadata:
        return transition
    metadata = dict(transition.get("episode_metadata") or {})
    metadata.update(
        {
            str(key): value
            for key, value in dict(step_metadata).items()
            if value is not None
        }
    )
    transition["episode_metadata"] = metadata
    return transition


def _env_subprocess_main(  # noqa: ANN001
    conn,
    env_cfg,
    task_id,
    record_builder,
    egl_device_id,
    start_episode_id=0,
):
    """Spawn-child loop: build one env in a fresh interpreter (clean egl context) and serve
    set_task / current_obs / step / close over the pipe. Transitions are built HERE (they need
    the env object); the parent EnvWorker accumulates them and pushes to the Ray replay.
    """
    render_backend = _libero_render_backend(env_cfg)
    render_pool = _libero_render_gpu_pool(env_cfg)
    if not render_pool and egl_device_id is not None:
        render_pool = [int(egl_device_id)]
    apply_libero_render_regime(
        render_backend,
        int(env_cfg.get("_render_shard_id", 0)),
        render_pool,
    )
    try:
        env = _build_env_from_cfg(env_cfg)
        cur_task = int(task_id)
        episode_id = int(start_episode_id)
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
                # payload is (task_id, start_episode_id) — resume/diversity continues the
                # init_state from a caller-chosen episode_id instead of always 0. An int
                # payload (legacy) starts at 0.
                if isinstance(payload, tuple):
                    cur_task, episode_id = int(payload[0]), int(payload[1])
                else:
                    cur_task, episode_id = int(payload), 0
                if hasattr(env, "set_task"):
                    env.set_task(cur_task)
                cur_obs, _ = env.reset(task_id=cur_task, episode_id=episode_id)
                conn.send(("ok", cur_obs))
            elif cmd == "step":
                if isinstance(payload, tuple) and len(payload) == 4:
                    action, obs_embedding, lang_emb, step_metadata = payload
                elif isinstance(payload, tuple) and len(payload) == 3:
                    action, obs_embedding, lang_emb = payload
                    step_metadata = None
                else:
                    action, obs_embedding = payload
                    lang_emb = None
                    step_metadata = None
                obs = cur_obs
                record_obs = obs
                if record_builder is not None and hasattr(env, "full_record"):
                    record_obs = dict(obs)
                    record_obs["_full_record"] = env.full_record()
                next_obs, reward, terminated, truncated, info = env.step(action)
                info = dict(info or {})
                info["episode_id"] = episode_id  # current episode (before the done-increment)
                if record_builder is not None:
                    transition = _call_record_builder(
                        record_builder,
                        env,
                        record_obs,
                        action,
                        reward,
                        terminated,
                        truncated,
                        info,
                        obs_embedding,
                        lang_emb,
                    )
                else:
                    transition = env.make_transition(
                        obs, action, reward, terminated, truncated, info
                    )
                    if "proprio" not in transition and "state" in transition:
                        transition["proprio"] = np.asarray(
                            transition["state"], dtype=np.float32
                        ).reshape(-1)
                    if obs_embedding is not None:
                        transition["obs_embedding"] = np.asarray(
                            obs_embedding, dtype=np.float32
                        )
                    if lang_emb is not None:
                        transition["lang_emb"] = np.asarray(lang_emb, dtype=np.float32)
                _stamp_step_metadata(transition, step_metadata)
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
            elif cmd == "load_world_model_state":
                state_dict, version = payload
                if not hasattr(env, "load_world_model_state"):
                    conn.send(("error", "active env does not support world model state sync"))
                else:
                    env.load_world_model_state(state_dict, int(version))
                    conn.send(("ok", None))
            elif cmd == "load_classifier_state":
                state_dict, version = payload
                if not hasattr(env, "load_classifier_state"):
                    conn.send(("error", "active env does not support classifier state sync"))
                else:
                    env.load_classifier_state(state_dict, int(version))
                    conn.send(("ok", None))
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
        dump: Any | None = None,
        record_builder: Any | None = None,
    ) -> None:
        super().__init__()
        self.env_cfg = dict(env_cfg)
        self.task_id = int(task_id)
        self.num_envs = max(1, int(self.env_cfg.get("num_envs_per_worker", 1)))
        self.replay = replay
        self.dump = dump
        self._record_builder = record_builder
        self.env: Any | None = None
        self.obs: dict[str, Any] | None = None
        self.episode: list[dict[str, Any]] = []
        self.episode_id = 0
        self._proc: Any | None = None
        self._conn: Any | None = None
        self._egl_device_id: int | None = None
        self._procs: list[Any | None] = [None for _ in range(self.num_envs)]
        self._conns: list[Any | None] = [None for _ in range(self.num_envs)]
        self._obs_by_slot: list[dict[str, Any] | None] = [None for _ in range(self.num_envs)]
        self._episodes_by_slot: list[list[dict[str, Any]]] = [
            [] for _ in range(self.num_envs)
        ]
        self._episode_ids_by_slot: list[int] = [0 for _ in range(self.num_envs)]
        self._task_ids_by_slot: list[int] = [self.task_id for _ in range(self.num_envs)]
        self._egl_respawns_by_slot: list[int] = [0 for _ in range(self.num_envs)]
        self._egl_diagnostics_logged = False

    @property
    def _spawned(self) -> bool:
        return self._proc is not None

    def init(self) -> None:
        pool = self.env_cfg.get("egl_device_pool")
        if pool:
            for slot_id in range(self.num_envs):
                self._init_spawn(int(pool[int(self.local_rank) % len(pool)]), slot_id)
        elif str(self.env_cfg.get("render_backend", "")).lower() == "egl":
            self._log_worker_egl_diagnostics()
            for slot_id in range(self.num_envs):
                self._init_spawn(None, slot_id)
        elif self.num_envs != 1:
            # Parallel osmesa (RLinf-aligned): render each env in its own spawn subprocess.
            # osmesa is CPU software rendering (no GPU/driver contention), so num_envs>1 scales
            # via worker subprocesses instead of egl within-worker batching.
            for slot_id in range(self.num_envs):
                self._init_spawn(None, slot_id)
        else:
            self._init_inproc()

    def _log_worker_egl_diagnostics(self) -> None:
        if self._egl_diagnostics_logged:
            return
        log_egl_device_diagnostics_from_env(logger_name=__name__)
        self._egl_diagnostics_logged = True

    def _init_inproc(self) -> None:
        apply_libero_render_regime(
            _libero_render_backend(self.env_cfg),
            int(self.local_rank),
            _libero_render_gpu_pool(self.env_cfg),
        )
        self.env = self._build_env(self.env_cfg)
        if hasattr(self.env, "set_task"):
            self.env.set_task(self.task_id)
        self.obs, _ = self._reset_env()
        self.episode = []
        self.episode_id = 0

    def _init_spawn(
        self,
        egl_device_id: int | None,
        slot_id: int = 0,
        *,
        task_id: int | None = None,
        start_episode_id: int = 0,
    ) -> None:
        # WorkerGroup.launch fires every worker's init.remote() at once, so all env workers
        # spawn their child simultaneously. Each child cold-starts a fresh interpreter and builds
        # LIBERO/robosuite/EGL; stagger startup to reduce CPU/disk thundering herd during init.
        self._egl_device_id = None if egl_device_id is None else int(egl_device_id)
        child_env_cfg = dict(self.env_cfg)
        child_env_cfg.setdefault("render_backend", "osmesa")
        child_env_cfg["_render_shard_id"] = int(self.local_rank)
        if egl_device_id is not None and not _libero_render_gpu_pool(child_env_cfg):
            child_env_cfg["render_devices"] = [int(egl_device_id)]
        stagger_s = float(self.env_cfg.get("egl_spawn_stagger_s", 3.0)) * int(self.local_rank)
        if stagger_s > 0:
            time.sleep(stagger_s)
        init_timeout_s = float(self.env_cfg.get("egl_spawn_init_timeout_s", 900.0))
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_env_subprocess_main,
            args=(
                child_conn,
                child_env_cfg,
                self.task_id if task_id is None else int(task_id),
                self._record_builder,
                egl_device_id,
                int(start_episode_id),
            ),
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
        self._set_spawn_slot(
            int(slot_id),
            proc,
            parent_conn,
            payload,
            task_id=self.task_id if task_id is None else int(task_id),
            episode_id=int(start_episode_id),
        )

    def _set_spawn_slot(
        self,
        slot_id: int,
        proc: Any,
        conn: Any,
        obs: dict[str, Any],
        *,
        task_id: int | None = None,
        episode_id: int = 0,
    ) -> None:
        self._procs[slot_id] = proc
        self._conns[slot_id] = conn
        self._obs_by_slot[slot_id] = obs
        self._episodes_by_slot[slot_id] = []
        self._episode_ids_by_slot[slot_id] = int(episode_id)
        if task_id is not None:
            self._task_ids_by_slot[slot_id] = int(task_id)
        if slot_id == 0:
            self._proc = proc
            self._conn = conn
            self.obs = obs
            self.episode = self._episodes_by_slot[0]
            self.episode_id = self._episode_ids_by_slot[0]
            self.task_id = self._task_ids_by_slot[0]

    def _rpc(self, cmd: str, payload: Any = None, slot_id: int = 0) -> Any:
        conn = self._conns[int(slot_id)] if self.num_envs > 1 else self._conn
        if conn is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        conn.send((cmd, payload))
        status, val = conn.recv()
        if status == "error":
            raise RuntimeError(f"EnvWorker subprocess error: {val}")
        return val

    def current_obs(self) -> dict[str, Any] | list[dict[str, Any]]:
        if self.num_envs > 1:
            if any(obs is None for obs in self._obs_by_slot):
                raise RuntimeError("EnvWorker.init() has not been called")
            return [dict(obs) for obs in self._obs_by_slot if obs is not None]
        if self.obs is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        return self.obs

    def set_task(self, task_id: int, start_episode_id: int = 0) -> dict[str, Any]:
        return self.set_task_slot(0, task_id, start_episode_id)

    def set_task_slot(
        self,
        slot_id: int,
        task_id: int,
        start_episode_id: int = 0,
    ) -> dict[str, Any]:
        """Switch task and reset, starting from ``start_episode_id``.

        episode_id is the init_state selector (env reset uses
        init_state = episode_id % num_init_states), so a non-zero start lets the Ray
        scheduler give each episode a DISTINCT init_state and continue past what is
        already collected on resume (default 0 = legacy behaviour).
        """
        slot_id = int(slot_id)
        self.task_id = int(task_id)
        self._task_ids_by_slot[slot_id] = int(task_id)
        self._episode_ids_by_slot[slot_id] = int(start_episode_id)
        if self._spawned:
            obs = self._rpc(
                "set_task",
                (self.task_id, self._episode_ids_by_slot[slot_id]),
                slot_id=slot_id,
            )
            self._obs_by_slot[slot_id] = obs
            self._episodes_by_slot[slot_id] = []
            if slot_id == 0:
                self.obs = obs
                self.episode = self._episodes_by_slot[0]
                self.episode_id = self._episode_ids_by_slot[0]
            return obs
        env = self._env()
        if hasattr(env, "set_task"):
            env.set_task(self.task_id)
        self.episode = []
        self.episode_id = int(start_episode_id)
        self.obs, _ = self._reset_env()
        return self.obs

    def step(
        self,
        action: Any,
        obs_embedding: Any = None,
        lang_emb: Any | None = None,
        step_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        return self.step_slot(0, action, obs_embedding, lang_emb, step_metadata)

    def step_slot(
        self,
        slot_id: int,
        action: Any,
        obs_embedding: Any = None,
        lang_emb: Any | None = None,
        step_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        if self._spawned:
            return self._step_spawn_slot(
                int(slot_id), action, obs_embedding, lang_emb, step_metadata
            )
        if int(slot_id) != 0:
            raise ValueError("in-process EnvWorker only supports slot 0")
        return self._step_inproc(action, obs_embedding, lang_emb, step_metadata)

    def load_world_model_state(self, state_dict: dict[str, Any], version: int) -> None:
        if self._spawned:
            for slot_id in range(self.num_envs):
                self._rpc("load_world_model_state", (state_dict, int(version)), slot_id=slot_id)
            return
        env = self._env()
        if not hasattr(env, "load_world_model_state"):
            raise RuntimeError("active env does not support world model state sync")
        env.load_world_model_state(state_dict, int(version))

    def load_classifier_state(self, state_dict: dict[str, Any], version: int) -> None:
        if self._spawned:
            for slot_id in range(self.num_envs):
                self._rpc("load_classifier_state", (state_dict, int(version)), slot_id=slot_id)
            return
        env = self._env()
        if not hasattr(env, "load_classifier_state"):
            raise RuntimeError("active env does not support classifier state sync")
        env.load_classifier_state(state_dict, int(version))

    def _step_spawn_slot(
        self,
        slot_id: int,
        action: Any,
        obs_embedding: Any,
        lang_emb: Any | None,
        step_metadata: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        try:
            transition, obs, done, info = self._rpc(
                "step",
                (action, obs_embedding, lang_emb, step_metadata),
                slot_id=slot_id,
            )
        except (EOFError, OSError) as exc:
            recovered = self._recover_spawn_slot_after_child_death(slot_id)
            if recovered is not None:
                obs, respawn_count = recovered
                return (
                    obs,
                    True,
                    {
                        "success": False,
                        "env_crash": True,
                        "respawned": True,
                        "respawn_count": int(respawn_count),
                    },
                )
            raise RuntimeError(
                f"EnvWorker egl child died (rank={self.local_rank}, slot={slot_id}); "
                "failing the rollout instead of hiding the native crash"
            ) from exc
        self._episodes_by_slot[slot_id].append(transition)
        self._obs_by_slot[slot_id] = obs
        if slot_id == 0:
            self.episode = self._episodes_by_slot[0]
            self.obs = obs
        if done:
            self._add_episode_to_replay(slot_id)
            self._episodes_by_slot[slot_id] = []
            self._episode_ids_by_slot[slot_id] += 1
            if slot_id == 0:
                self.episode = self._episodes_by_slot[0]
                self.episode_id = self._episode_ids_by_slot[0]
            return obs, True, dict(info or {})
        return obs, False, dict(info or {})

    def _recover_spawn_slot_after_child_death(
        self,
        slot_id: int,
    ) -> tuple[dict[str, Any], int] | None:
        max_respawns = int(self.env_cfg.get("egl_max_respawns", 0) or 0)
        if max_respawns <= 0:
            return None
        slot_id = int(slot_id)
        respawns = int(self._egl_respawns_by_slot[slot_id])
        if respawns >= max_respawns:
            return None

        self._close_spawn_slot(slot_id)
        self._episodes_by_slot[slot_id] = []
        start_episode_id = int(self._episode_ids_by_slot[slot_id]) + 1
        task_id = int(self._task_ids_by_slot[slot_id])
        self._egl_respawns_by_slot[slot_id] = respawns + 1
        self._init_spawn(
            self._egl_device_id,
            slot_id,
            task_id=task_id,
            start_episode_id=start_episode_id,
        )
        obs = self._obs_by_slot[slot_id]
        if obs is None:
            raise RuntimeError(
                f"EnvWorker egl respawn did not produce an observation "
                f"(rank={self.local_rank}, slot={slot_id})"
            )
        if slot_id == 0:
            self.episode = self._episodes_by_slot[0]
            self.episode_id = self._episode_ids_by_slot[0]
            self.obs = obs
        return dict(obs), int(self._egl_respawns_by_slot[slot_id])

    def _close_spawn_slot(self, slot_id: int) -> None:
        slot_id = int(slot_id)
        conn = self._conns[slot_id] if self.num_envs > 1 else self._conn
        proc = self._procs[slot_id] if self.num_envs > 1 else self._proc
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        if proc is not None:
            try:
                if hasattr(proc, "is_alive") and proc.is_alive():
                    proc.terminate()
                elif hasattr(proc, "terminate"):
                    proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            try:
                if hasattr(proc, "join"):
                    proc.join(timeout=10.0)
            except Exception:  # noqa: BLE001
                pass
        self._procs[slot_id] = None
        self._conns[slot_id] = None
        self._obs_by_slot[slot_id] = None
        if slot_id == 0:
            self._proc = None
            self._conn = None
            self.obs = None

    def _step_inproc(
        self,
        action: Any,
        obs_embedding: Any,
        lang_emb: Any | None,
        step_metadata: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        env = self._env()
        obs = self.current_obs()
        record_obs = obs
        if self._record_builder is not None and hasattr(env, "full_record"):
            record_obs = dict(obs)
            record_obs["_full_record"] = env.full_record()
        next_obs, reward, terminated, truncated, info = env.step(action)
        info = dict(info or {})
        info["episode_id"] = self.episode_id  # current episode (before the done-increment)
        if self._record_builder is not None:
            transition = _call_record_builder(
                self._record_builder,
                env,
                record_obs,
                action,
                reward,
                terminated,
                truncated,
                info,
                obs_embedding,
                lang_emb,
            )
        elif hasattr(env, "make_transition"):
            transition = env.make_transition(obs, action, reward, terminated, truncated, info)
            if "proprio" not in transition and "state" in transition:
                transition["proprio"] = np.asarray(
                    transition["state"], dtype=np.float32
                ).reshape(-1)
            if obs_embedding is not None:
                transition["obs_embedding"] = np.asarray(obs_embedding, dtype=np.float32)
            if lang_emb is not None:
                transition["lang_emb"] = np.asarray(lang_emb, dtype=np.float32)
        else:
            transition = self._make_generic_transition(
                obs,
                next_obs,
                action,
                reward,
                terminated,
                truncated,
                info,
                obs_embedding,
                lang_emb,
            )
        _stamp_step_metadata(transition, step_metadata)
        self.episode.append(transition)

        done = bool(terminated or truncated)
        if done:
            self._add_episode_to_replay()
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
            for conn in self._conns:
                if conn is None:
                    continue
                try:
                    conn.send(("close", None))
                except Exception:  # noqa: BLE001
                    pass
            for proc in self._procs:
                if proc is None:
                    continue
                proc.join(timeout=10.0)
                if proc.is_alive():
                    proc.terminate()
            self._proc = None
            self._conn = None
            self._procs = [None for _ in range(self.num_envs)]
            self._conns = [None for _ in range(self.num_envs)]
            return
        env = self.env
        if env is not None and hasattr(env, "close"):
            env.close()
        self.env = None

    def _reset_env(self) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self._env()
        return env.reset(task_id=self.task_id, episode_id=self.episode_id)

    def _add_episode_to_replay(self, slot_id: int = 0) -> None:
        episode = self._episodes_by_slot[int(slot_id)] if self._spawned else self.episode
        self._push_episode(self.replay, episode)
        self._push_episode(self.dump, episode)

    @staticmethod
    def _push_episode(target: Any | None, episode: list[dict[str, Any]]) -> None:
        if target is None:
            return
        add_episode = target.add_episode
        remote = getattr(add_episode, "remote", None)
        if remote is not None:
            ray.get(remote(list(episode)))
        else:
            add_episode(list(episode))

    @staticmethod
    def _make_generic_transition(
        obs: dict[str, Any],
        next_obs: dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        obs_embedding: Any,
        lang_emb: Any | None = None,
    ) -> dict[str, Any]:
        transition = {
            "obs": obs,
            "next_obs": next_obs,
            "actions": np.asarray(action, dtype=np.float32),
            "rewards": float(reward),
            "dones": bool(terminated or truncated),
            "is_terminal": bool(terminated),
            "is_last": bool(terminated or truncated),
            "is_first": bool(obs.get("is_first", False)) if isinstance(obs, dict) else False,
            "info": dict(info or {}),
            "obs_embedding": (
                None
                if obs_embedding is None
                else np.asarray(obs_embedding, dtype=np.float32)
            ),
        }
        if lang_emb is not None:
            transition["lang_emb"] = np.asarray(lang_emb, dtype=np.float32)
        if isinstance(obs, dict) and "state" in obs:
            transition["proprio"] = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        return transition

    def _env(self) -> Any:
        if self.env is None:
            raise RuntimeError("EnvWorker.init() has not been called")
        return self.env

    @staticmethod
    def _build_env(env_cfg: dict[str, Any]) -> Any:
        return _build_env_from_cfg(env_cfg)
