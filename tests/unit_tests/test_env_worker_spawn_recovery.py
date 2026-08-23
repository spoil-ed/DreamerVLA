"""EnvWorker EGL child death is fatal unless respawn is explicitly enabled."""

import numpy as np
import pytest

from dreamervla.workers.env.env_worker import EnvWorker


class _DeadConn:
    """A pipe whose child has died: send is a no-op, recv reports EOF."""

    def send(self, *_):
        pass

    def recv(self):
        raise EOFError


class _Proc:
    def __init__(self):
        self.terminated = False

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        self.join_timeout = timeout


def _worker():
    w = EnvWorker(
        env_cfg={"render_backend": "egl"},
        task_id=0,
        replay=None,
    )
    w.local_rank = 0
    w._egl_device_id = None
    w._proc = _Proc()
    w._conn = _DeadConn()
    w.obs = {"x": np.zeros(1)}
    w.episode = [{"dummy": 1}]
    return w


def test_step_raises_on_child_death():
    w = _worker()

    with pytest.raises(RuntimeError, match="egl child died"):
        w.step(action=np.zeros(7), obs_embedding=np.zeros(4))

    assert w.episode == [{"dummy": 1}]


def test_step_respawns_child_when_enabled(monkeypatch):
    w = _worker()
    w.env_cfg["egl_max_respawns"] = 1
    old_proc = w._proc
    calls = []

    def fake_init_spawn(egl_device_id, slot_id=0, *, task_id=None, start_episode_id=0):
        calls.append((egl_device_id, slot_id, task_id, start_episode_id))
        w._set_spawn_slot(
            slot_id,
            _Proc(),
            object(),
            {"x": np.ones(1), "step": 0, "task_id": task_id},
            task_id=task_id,
            episode_id=start_episode_id,
        )

    monkeypatch.setattr(w, "_init_spawn", fake_init_spawn)

    obs, done, info = w.step(action=np.zeros(7), obs_embedding=np.zeros(4))

    assert done is True
    assert obs["step"] == 0
    assert info["env_crash"] is True
    assert info["respawned"] is True
    assert info["respawn_count"] == 1
    assert calls == [(None, 0, 0, 1)]
    assert w.episode == []
    assert w.episode_id == 1
    assert old_proc.terminated is True
