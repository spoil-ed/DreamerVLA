"""EnvWorker EGL child death is fatal, not hidden as a rollout boundary."""

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
    def is_alive(self):
        return False

    def terminate(self):
        pass


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
