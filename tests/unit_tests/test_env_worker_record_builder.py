from __future__ import annotations

import numpy as np


def test_env_worker_uses_injected_record_builder() -> None:
    from dreamervla.workers.env.env_worker import EnvWorker

    captured = {}

    def fake_builder(env, obs, action, reward, terminated, truncated, info, obs_embedding):
        captured["called"] = True
        return {"marker": 1, "obs_embedding": np.asarray(obs_embedding, np.float16)}

    cfg = {
        "target": "dreamervla.workers.env._test_envs:DumpCounterEnv",
        "kwargs": {"horizon": 2, "image_shape": (4, 4, 3), "embedding_dim": 4},
    }

    class _Sink:
        def __init__(self) -> None:
            self.eps = []

        def add_episode(self, ep):
            self.eps.append(ep)
            return None

    worker = EnvWorker(cfg, task_id=0, replay=_Sink(), record_builder=fake_builder)
    worker.init()
    worker.step(np.zeros(7, np.float32), np.zeros(4, np.float32))
    assert captured.get("called") is True


def test_env_worker_passes_pre_step_full_record_to_record_builder() -> None:
    from dreamervla.workers.env.env_worker import EnvWorker

    captured = {}

    def fake_builder(env, obs, action, reward, terminated, truncated, info, obs_embedding):
        captured["record_step"] = int(obs["_full_record"]["step"])
        return {"obs_embedding": np.asarray(obs_embedding, np.float16)}

    cfg = {
        "target": "dreamervla.workers.env._test_envs:DumpCounterEnv",
        "kwargs": {"horizon": 2, "image_shape": (4, 4, 3), "embedding_dim": 4},
    }

    class _Sink:
        def add_episode(self, ep):
            return None

    worker = EnvWorker(cfg, task_id=0, replay=_Sink(), record_builder=fake_builder)
    worker.init()
    worker.step(np.zeros(7, np.float32), np.zeros(4, np.float32))
    assert captured["record_step"] == 0
