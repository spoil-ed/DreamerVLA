"""EnvWorker resilience: a dead egl spawn child is recovered, not fatal.

When a spawn env-child crashes mid-rollout (silent native SIGSEGV under sustained
concurrent egl), the parent EnvWorker sees EOFError/OSError on the pipe. It must
drop the partial episode, respawn a clean child, and return an episode-boundary
``done`` so the rollout continues — bounded by ``egl_max_respawns``.
"""

import numpy as np

from dreamervla.workers.env.env_worker import EnvWorker


class _DeadConn:
    """A pipe whose child has died: send is a no-op, recv reports EOF."""

    def send(self, *_):
        pass

    def recv(self):
        raise EOFError


class _Proc:
    def is_alive(self):
        return False

    def terminate(self):
        pass


def _worker():
    w = EnvWorker(
        env_cfg={"egl_device_pool": [0], "egl_max_respawns": 2},
        task_id=0,
        replay=None,
    )
    w.local_rank = 0
    w._egl_device_id = 0
    w._proc = _Proc()
    w._conn = _DeadConn()
    w.obs = {"x": np.zeros(1)}
    w.episode = [{"dummy": 1}]
    return w


def test_step_recovers_on_child_death(monkeypatch):
    w = _worker()
    spawned = {"n": 0}

    def _fake_init_spawn(egl_device_id):
        spawned["n"] += 1
        w._proc = _Proc()  # a live child handle so _spawned stays True
        w.obs = {"fresh": np.ones(1)}

    monkeypatch.setattr(w, "_init_spawn", _fake_init_spawn)

    obs, done, info = w.step(action=np.zeros(7), obs_embedding=np.zeros(4))
    assert done is True
    assert info.get("env_crash_recovered") is True
    assert spawned["n"] == 1  # respawned once
    assert w.episode == []  # partial episode dropped
    assert obs["fresh"].tolist() == [1.0]  # fresh reset obs returned


def test_respawn_cap_eventually_raises(monkeypatch):
    w = _worker()

    def _fake_init_spawn(egl_device_id):
        # The respawned child is immediately "dead" again, so the next step recrashes.
        w._proc = _Proc()
        w._conn = _DeadConn()

    monkeypatch.setattr(w, "_init_spawn", _fake_init_spawn)

    # egl_max_respawns=2 → two recoveries ok, the third raises.
    w.step(action=np.zeros(7), obs_embedding=np.zeros(4))  # respawn 1
    w.step(action=np.zeros(7), obs_embedding=np.zeros(4))  # respawn 2
    try:
        w.step(action=np.zeros(7), obs_embedding=np.zeros(4))  # respawn 3 -> raise
    except RuntimeError as exc:
        assert "egl child died" in str(exc)
    else:
        raise AssertionError("expected RuntimeError after exceeding egl_max_respawns")
