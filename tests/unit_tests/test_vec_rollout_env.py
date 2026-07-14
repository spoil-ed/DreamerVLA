"""Tests for VecRolloutEnv — SubprocVecEnv-style parallel env wrapper.

The within-rank parallelism (migration §5.2) needs K LIBERO envs stepping in true
parallel: send all K actions, THEN recv all K results (the deleted Layer-2 serialized
per handle, which is why it gave no speedup).  These tests exercise the scatter/gather
and subset-reset orchestration with a trivial in-process fake env (picklable,
module-level) so they run fast and deterministically without LIBERO/mujoco.

The spawn × real-LIBERO compatibility (a §9 risk) is validated separately by a manual
check / the Step-5 integration smoke, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from dreamervla.runtime import vec_rollout_env as vec_mod
from dreamervla.runtime.vec_rollout_env import VecRolloutEnv, default_env_factory

# ── module-level fakes (must be importable for spawn pickling) ────────────────


class _FakeEnv:
    """Deterministic stand-in for DreamerVLAOnlineTrainEnv (no LIBERO)."""

    def __init__(self, cfg_kwargs: dict):
        self._cfg = cfg_kwargs
        self.t = 0
        self.task_id = 0
        self.num_tasks = 3
        self.task_description = "task0"

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
        return {"t": self.t, "task_id": self.task_id}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_env_factory(cfg_kwargs: dict) -> _FakeEnv:
    env = _FakeEnv(cfg_kwargs)
    return env.__enter__()


def _raising_factory(cfg_kwargs: dict):
    raise ValueError("boom-during-init")


class _Pipe:
    def __init__(self) -> None:
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(value)

    def recv(self):
        return ("close", None)

    def close(self) -> None:
        self.closed = True


# ── tests ────────────────────────────────────────────────────────────────────


def test_default_env_factory_instantiates_hydra_target():
    env = default_env_factory({"_target_": "types.SimpleNamespace", "marker": 3})

    assert env.marker == 3


def test_default_env_factory_applies_libero_render_helper_before_inproc_build(monkeypatch):
    events = []

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        events.append(("helper", backend, int(shard_id), list(gpu_pool)))

    monkeypatch.setattr(vec_mod, "apply_libero_render_regime", fake_apply)

    env = default_env_factory(
        {
            "_target_": "types.SimpleNamespace",
            "_libero_render_backend": "osmesa",
            "_libero_render_gpu_pool": [1],
            "_libero_render_shard_id": 4,
            "marker": 3,
        }
    )

    assert events == [("helper", "osmesa", 4, [1])]
    assert env.marker == 3


def test_worker_applies_libero_render_helper_before_factory(monkeypatch):
    events = []

    def fake_apply(backend: str, shard_id: int, gpu_pool: list[int]) -> None:
        events.append(("helper", backend, int(shard_id), list(gpu_pool)))

    def factory(cfg_kwargs: dict):
        events.append(("factory", dict(cfg_kwargs)))
        return _fake_env_factory({})

    monkeypatch.setattr(vec_mod, "apply_libero_render_regime", fake_apply)
    conn = _Pipe()

    vec_mod._worker(
        conn,
        factory,
        {
            "_libero_render_backend": "egl",
            "_libero_render_gpu_pool": [4, 6],
            "_libero_render_shard_id": 10,
        },
        {},
        worker_index=2,
    )

    assert events[:2] == [
        ("helper", "egl", 12, [4, 6]),
        ("factory", {}),
    ]
    assert conn.sent[0][0] == "ready"


def test_parallel_step_scatters_actions_and_gathers_records():
    """step(actions) sends env i action i and returns results in env order."""
    ve = VecRolloutEnv(num_envs=3, cfg_kwargs={}, factory=_fake_env_factory)
    try:
        recs = ve.reset(task_ids=[0, 1, 2], episode_ids=[0, 0, 0])
        assert [r["task_id"] for r in recs] == [0, 1, 2]
        assert [r["t"] for r in recs] == [0, 0, 0]

        results = ve.step([np.array([1.0]), np.array([2.0]), np.array([3.0])])
        assert len(results) == 3
        # scatter: env i received action i
        assert [r[3]["action_sum"] for r in results] == [1.0, 2.0, 3.0]
        # gather: each env stepped exactly once, in order
        assert [r[4]["t"] for r in results] == [1, 1, 1]
        assert [r[0] for r in results] == [1.0, 1.0, 1.0]  # rewards
        assert [r[1] for r in results] == [False, False, False]  # not terminated at t=1
    finally:
        ve.close()


def test_subset_reset_leaves_other_envs_untouched():
    """reset(env_ids=[1]) resets only env 1; envs 0,2 keep their in-progress state."""
    ve = VecRolloutEnv(num_envs=3, cfg_kwargs={}, factory=_fake_env_factory)
    try:
        ve.reset(task_ids=[0, 0, 0], episode_ids=[0, 0, 0])
        ve.step([np.zeros(1), np.zeros(1), np.zeros(1)])  # all -> t=1

        recs = ve.reset(task_ids=[5], episode_ids=[0], env_ids=[1])
        assert len(recs) == 1
        assert recs[0]["t"] == 0 and recs[0]["task_id"] == 5

        results = ve.step([np.zeros(1), np.zeros(1), np.zeros(1)])
        # env0: 1->2, env1: 0->1 (was reset), env2: 1->2
        assert [r[4]["t"] for r in results] == [2, 1, 2]
    finally:
        ve.close()


def test_step_subset_only_steps_addressed_envs():
    """step(actions, env_ids=[0,2]) advances only those envs; the rest stay put.

    The continuous-stepping loop needs this to drain a finite work-list: once a slot
    runs out of work it goes idle and must not be stepped.
    """
    ve = VecRolloutEnv(num_envs=3, cfg_kwargs={}, factory=_fake_env_factory)
    try:
        ve.reset(task_ids=[0, 0, 0], episode_ids=[0, 0, 0])
        results = ve.step([np.zeros(1), np.zeros(1)], env_ids=[0, 2])
        assert len(results) == 2
        assert [r[4]["t"] for r in results] == [1, 1]
        # env1 untouched (t=0); stepping all now -> env1 reaches 1, envs 0,2 reach 2
        results2 = ve.step([np.zeros(1), np.zeros(1), np.zeros(1)])
        assert [r[4]["t"] for r in results2] == [2, 1, 2]
    finally:
        ve.close()


def test_set_task_returns_descriptions():
    ve = VecRolloutEnv(num_envs=2, cfg_kwargs={}, factory=_fake_env_factory)
    try:
        descs = ve.set_task(task_ids=[7, 9])
        assert descs == ["task7", "task9"]
    finally:
        ve.close()


def test_terminated_flag_propagates():
    """Env terminates at t>=3 — the flag must come back through the pipe."""
    ve = VecRolloutEnv(num_envs=1, cfg_kwargs={}, factory=_fake_env_factory)
    try:
        ve.reset(task_ids=[0], episode_ids=[0])
        terms = [ve.step([np.zeros(1)])[0][1] for _ in range(3)]
        assert terms == [False, False, True]
    finally:
        ve.close()


def test_init_failure_in_child_raises():
    """A factory that raises during child init must surface as RuntimeError."""
    with pytest.raises(RuntimeError, match="boom-during-init|init failed"):
        VecRolloutEnv(num_envs=2, cfg_kwargs={}, factory=_raising_factory)
