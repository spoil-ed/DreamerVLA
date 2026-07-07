"""Tests for OnlineEglVecEnv — approach 1 (EGL) on RLinf's vendored SubprocVectorEnv.

These exercise the RLinf-vendored ``BaseVectorEnv`` + spawn ``SubprocEnvWorker``
machinery and DreamerVLA's per-child EGL device regime with a trivial in-process
fake env (module-level, picklable for spawn) — no LIBERO/mujoco/GPU. The egl-vs-osmesa
crash/throughput behaviour is a GPU property and is not asserted here.

The fake env snapshots ``os.environ`` at build time so the parent can verify the egl
device regime (CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID / MUJOCO_GL) was applied per
child BEFORE the env was built — the whole point of the spawn isolation.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from dreamervla.envs.libero.venv import OnlineEglVecEnv

# ── module-level fakes (must be importable for spawn pickling) ────────────────

_SNAPSHOT_KEYS = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "MUJOCO_EGL_DEVICE_ID",
    "CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "EXTRA_FLAG",
)


class _FakeEnv:
    """Deterministic stand-in for DreamerVLAOnlineTrainEnv; snapshots env at build."""

    def __init__(self, cfg_kwargs: dict):
        self._cfg = cfg_kwargs
        self.t = 0
        self.task_id = 0
        self.task_description = "task0"
        # captured AFTER the egl regime + env_vars were applied in the child
        self._env_snapshot = {k: os.environ.get(k) for k in _SNAPSHOT_KEYS}

    def set_task(self, task_id: int) -> None:
        self.task_id = int(task_id)
        self.task_description = f"task{int(task_id)}"

    def reset(self, episode_id=None, task_id=None):
        self.t = 0
        if task_id is not None:
            self.task_id = int(task_id)
            self.task_description = f"task{int(task_id)}"
        return None, {"episode_id": episode_id}

    def step(self, action):
        self.t += 1
        terminated = self.t >= 3
        info = {"action_sum": float(np.sum(action))}
        return None, float(self.t), terminated, False, info

    def full_record(self) -> dict:
        return {"t": self.t, "task_id": self.task_id, "env": self._env_snapshot}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_env_factory(cfg_kwargs: dict) -> _FakeEnv:
    return _FakeEnv(cfg_kwargs).__enter__()


def _raising_factory(cfg_kwargs: dict):
    raise ValueError("boom-during-init")


# ── tests ────────────────────────────────────────────────────────────────────


def test_rejects_zero_envs():
    with pytest.raises(ValueError, match="num_envs"):
        OnlineEglVecEnv(num_envs=0, cfg_kwargs={}, factory=_fake_env_factory)


def test_barrier_scatter_gather_matches_vecrolloutenv_api():
    """reset/step/set_task scatter per env and gather in env order (VecRolloutEnv API)."""
    vec = OnlineEglVecEnv(
        num_envs=3,
        cfg_kwargs={},
        egl_device_pool=[0],
        factory=_fake_env_factory,
        start_timeout_s=120.0,
    )
    try:
        assert vec.num_envs == 3
        recs = vec.reset(task_ids=[0, 1, 2], episode_ids=[0, 0, 0])
        assert [r["task_id"] for r in recs] == [0, 1, 2]
        assert [r["t"] for r in recs] == [0, 0, 0]

        results = vec.step([np.array([1.0]), np.array([2.0]), np.array([3.0])])
        assert len(results) == 3
        assert [r[3]["action_sum"] for r in results] == [1.0, 2.0, 3.0]  # scatter
        assert [r[4]["t"] for r in results] == [1, 1, 1]  # gather, one step each
        assert [r[0] for r in results] == [1.0, 1.0, 1.0]  # rewards
        assert [r[1] for r in results] == [False, False, False]  # not terminated

        descs = vec.set_task(task_ids=[7, 8, 9])
        assert descs == ["task7", "task8", "task9"]
    finally:
        vec.close()


def test_per_child_egl_device_regime_is_rlinf_faithful():
    """Each child gets MUJOCO_GL=egl + the configured EGL index in CVD/EGL id."""
    vec = OnlineEglVecEnv(
        num_envs=3,
        cfg_kwargs={},
        egl_device_pool=[0],
        env_vars={"EXTRA_FLAG": "on"},
        factory=_fake_env_factory,
        start_timeout_s=120.0,
    )
    try:
        recs = vec.reset(task_ids=[0, 0, 0], episode_ids=[0, 0, 0])
        envs = [r["env"] for r in recs]
        # consistent CVD == MUJOCO_EGL_DEVICE_ID for robosuite's import-time check
        assert [e["MUJOCO_EGL_DEVICE_ID"] for e in envs] == ["0", "0", "0"]
        assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["0", "0", "0"]
        # egl backend locked + extra env vars forwarded, on every child
        for e in envs:
            assert e["MUJOCO_GL"] == "egl"
            assert e["PYOPENGL_PLATFORM"] == "egl"
            assert e["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
            assert e["EXTRA_FLAG"] == "on"
    finally:
        vec.close()


def test_subset_step_only_addressed_envs():
    vec = OnlineEglVecEnv(
        num_envs=3, cfg_kwargs={}, egl_device_pool=[0], factory=_fake_env_factory, start_timeout_s=120.0
    )
    try:
        vec.reset(task_ids=[0, 0, 0], episode_ids=[0, 0, 0])
        results = vec.step([np.zeros(1), np.zeros(1)], env_ids=[0, 2])
        assert [r[4]["t"] for r in results] == [1, 1]
        results2 = vec.step([np.zeros(1), np.zeros(1), np.zeros(1)])
        assert [r[4]["t"] for r in results2] == [2, 1, 2]  # env1 untouched before
    finally:
        vec.close()


def test_child_init_failure_surfaces():
    with pytest.raises(RuntimeError, match="boom-during-init|init failed"):
        OnlineEglVecEnv(
            num_envs=2,
            cfg_kwargs={},
            egl_device_pool=[0],
            factory=_raising_factory,
            start_timeout_s=120.0,
        )
