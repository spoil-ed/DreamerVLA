"""``VecRolloutEnv`` child-contract wrapper over a single raw LIBERO eval env.

``LiberoEvalEnv`` presents the single-env child contract that
``dreamervla.runners.vec_rollout_env.VecRolloutEnv`` drives (``set_task`` /
``task_description`` / ``reset(episode_id=, task_id=)`` / ``step`` /
``full_record`` / context manager) over one raw LIBERO ``OffScreenRenderEnv``.
The parallel eval path runs K of these in K spawned subprocesses so eval mirrors
collect/cotrain.

Correctness contract: a slot must start and step byte-identically to the current
sequential eval (``dreamervla/runners/pretokenize_vla_runner.py``). Concretely
``reset(episode_id=e)`` applies ``set_init_state(init_states[e])`` and runs the
``num_steps_wait`` dummy-action warmup *inside reset* (matching the sequential
per-episode warmup loop), and ``full_record`` reproduces the sequential per-step
inputs via the shared ``build_libero_eval_record`` helper.

LIBERO/robosuite imports stay lazy (inside methods / the module-level factory)
so CPU-only tests can inject a fake raw env without importing LIBERO.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from dreamervla.envs.libero_env import (
    build_libero_eval_record,
    get_libero_dummy_action,
    get_libero_env,
)
from dreamervla.utils.egl_device import apply_libero_render_regime

_LIBERO_RENDER_KEYS = (
    "_libero_render_backend",
    "_libero_render_gpu_pool",
    "_libero_render_shard_id",
)


class LiberoEvalEnv:
    """Single raw LIBERO eval env behind the ``VecRolloutEnv`` child contract.

    Args:
        task_suite_name: LIBERO benchmark suite name (e.g. ``libero_goal``).
        resolution: camera render resolution passed to ``get_libero_env``.
        seed: env seed passed to ``get_libero_env``.
        num_steps_wait: dummy-action warmup steps run inside ``reset``.
        max_steps: episode horizon (kept for parity / callers; the rollout core
            enforces the horizon).
        make_env: test seam ``task_id -> (raw_env, task_description)``. When
            ``None`` the raw LIBERO env is built from the benchmark suite.
        init_states: test seam ``task_id -> list[init_state]``. When ``None`` the
            init states come from the benchmark suite.
        reconfigure_per_episode: when True, every ``reset`` closes the current raw
            env and rebuilds a byte-fresh one (via the same construction path
            ``set_task`` uses) before applying ``set_init_state`` + the warmup, so
            an episode's outcome cannot carry over from prior episodes in this
            subprocess. When False (default), keep the reuse behavior.
    """

    def __init__(
        self,
        task_suite_name: str,
        resolution: int,
        seed: int,
        num_steps_wait: int,
        max_steps: int,
        *,
        make_env: Callable[[int], tuple[Any, str]] | None = None,
        init_states: Mapping[int, Sequence[Any]] | None = None,
        reconfigure_per_episode: bool = False,
    ) -> None:
        self._task_suite_name = str(task_suite_name)
        self._resolution = int(resolution)
        self._seed = int(seed)
        self._num_steps_wait = int(num_steps_wait)
        self._max_steps = int(max_steps)
        self._make_env = make_env
        self._init_states = init_states
        self._reconfigure_per_episode = bool(reconfigure_per_episode)

        self._task_suite: Any = None
        self._env: Any = None
        self._desc: str = ""
        self._task_id: int | None = None
        self._cur_init_states: Sequence[Any] | None = None
        self._obs: Any = None

    def _get_task_suite(self) -> Any:
        if self._task_suite is None:
            from dreamervla.envs.libero_env import _libero_benchmark_dict

            self._task_suite = _libero_benchmark_dict()[self._task_suite_name]()
        return self._task_suite

    def _build_raw_env(self, task_id: int) -> tuple[Any, str]:
        """Construct the raw LIBERO env + task text for ``task_id``.

        The single construction path shared by ``set_task`` and the per-episode
        reconfigure rebuild.
        """
        if self._make_env is not None:
            return self._make_env(task_id)
        task = self._get_task_suite().get_task(task_id)
        return get_libero_env(task, resolution=self._resolution, seed=self._seed)

    def set_task(self, task_id: int) -> None:
        """Build the raw env for ``task_id`` and cache its init-states + text."""
        task_id = int(task_id)
        self._env, self._desc = self._build_raw_env(task_id)
        if self._init_states is not None:
            self._cur_init_states = self._init_states[task_id]
        else:
            self._cur_init_states = self._get_task_suite().get_task_init_states(task_id)
        self._task_id = task_id

    @property
    def task_description(self) -> str:
        return self._desc

    def reset(
        self, episode_id: int, task_id: int | None = None
    ) -> tuple[Any, dict[str, int]]:
        """Reset to ``init_states[episode_id]`` then run the warmup inside reset."""
        if task_id is not None and int(task_id) != self._task_id:
            self.set_task(task_id)
        if self._env is None or self._cur_init_states is None:
            raise RuntimeError("reset called before set_task")
        if self._reconfigure_per_episode:
            # Full per-episode rebuild: close the current raw env and reconstruct
            # a byte-fresh one (same path as set_task) so no mujoco/osmesa state
            # carries over from previously-run episodes in this subprocess.
            self.close()
            self._env, self._desc = self._build_raw_env(int(self._task_id))
        self._env.reset()
        obs = self._env.set_init_state(self._cur_init_states[int(episode_id)])
        for _ in range(self._num_steps_wait):
            obs, _reward, _done, _info = self._env.step(get_libero_dummy_action())
        self._obs = obs
        return obs, {"init_state_index": int(episode_id)}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        self._obs = obs
        return obs, reward, bool(done), False, {"success": bool(done), **(info or {})}

    def full_record(self) -> dict[str, np.ndarray]:
        if self._obs is None:
            raise RuntimeError("full_record called before reset")
        record = build_libero_eval_record(self._obs, self._resolution)
        # Carry the raw LIBERO obs so the parallel path can reproduce the
        # sequential OFT base eval, whose action generation reads the raw obs
        # (agentview/wrist images + proprio) rather than the built PIL history.
        record["raw_obs"] = self._obs
        return record

    def close(self) -> None:
        env = getattr(self, "_env", None)
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass
        self._env = None

    def __enter__(self) -> LiberoEvalEnv:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def make_libero_eval_env(cfg_kwargs: dict[str, Any]) -> LiberoEvalEnv:
    """Module-level picklable factory for ``VecRolloutEnv``.

    Applies the (osmesa) render regime before building the env, mirroring
    ``vec_rollout_env.default_env_factory``, then constructs a ``LiberoEvalEnv``
    from the suite/resolution/seed/warmup/max-steps kwargs.
    """
    kwargs = dict(cfg_kwargs)
    render_backend = kwargs.get("_libero_render_backend")
    if render_backend is not None:
        apply_libero_render_regime(
            str(render_backend),
            int(kwargs.get("_libero_render_shard_id", 0)),
            list(kwargs.get("_libero_render_gpu_pool", [])),
        )
    for key in _LIBERO_RENDER_KEYS:
        kwargs.pop(key, None)

    env = LiberoEvalEnv(
        task_suite_name=str(kwargs["task_suite_name"]),
        resolution=int(kwargs.get("resolution", 256)),
        seed=int(kwargs.get("seed", 0)),
        num_steps_wait=int(kwargs.get("num_steps_wait", 10)),
        max_steps=int(kwargs["max_steps"]),
        reconfigure_per_episode=bool(kwargs.get("reconfigure_per_episode", False)),
    )
    return env.__enter__()
