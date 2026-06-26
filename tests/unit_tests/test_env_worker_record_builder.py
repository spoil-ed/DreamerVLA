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


def test_env_worker_pushes_completed_episode_to_remote_replay(monkeypatch) -> None:
    from dreamervla.workers.env import env_worker as env_worker_mod
    from dreamervla.workers.env.env_worker import EnvWorker

    class _RemoteAddEpisode:
        def __init__(self, sink: _Sink) -> None:
            self._sink = sink

        def remote(self, episode):
            self._sink.eps.append(list(episode))
            return {"ok": True}

    class _Sink:
        def __init__(self) -> None:
            self.eps = []
            self.add_episode = _RemoteAddEpisode(self)

    def fake_get(ref):
        return ref

    monkeypatch.setattr(env_worker_mod.ray, "get", fake_get)

    cfg = {
        "target": "dreamervla.workers.env._test_envs:DumpCounterEnv",
        "kwargs": {"horizon": 2, "image_shape": (4, 4, 3), "embedding_dim": 4},
    }
    sink = _Sink()
    worker = EnvWorker(cfg, task_id=0, replay=sink)
    worker.init()

    _obs, done, _info = worker.step(
        np.zeros(7, np.float32),
        np.zeros(4, np.float16),
    )
    assert done is False
    _obs, done, info = worker.step(
        np.ones(7, np.float32),
        np.ones(4, np.float16),
    )

    assert done is True
    assert "reset_info" in info
    assert len(sink.eps) == 1
    assert len(sink.eps[0]) == 2
    assert np.asarray(sink.eps[0][-1]["obs_embedding"]).dtype == np.float32
    assert worker.episode == []
    assert worker.episode_id == 1


def test_env_worker_copies_state_to_proprio_for_replay() -> None:
    from dreamervla.workers.env.env_worker import EnvWorker

    cfg = {
        "target": "dreamervla.workers.env._test_envs:CounterEnv",
        "kwargs": {"horizon": 2, "image_shape": (4, 4, 3), "embedding_dim": 4},
    }

    class _Sink:
        def add_episode(self, ep):
            return None

    worker = EnvWorker(cfg, task_id=0, replay=_Sink())
    worker.init()
    worker.step(
        np.zeros(7, np.float32),
        np.zeros(4, np.float32),
        np.arange(6, dtype=np.float32),
    )

    transition = worker.episode[0]
    np.testing.assert_array_equal(transition["proprio"], transition["state"])
    np.testing.assert_array_equal(transition["lang_emb"], np.arange(6, dtype=np.float32))


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


def test_env_worker_passes_language_embedding_to_record_builder() -> None:
    from dreamervla.workers.env.env_worker import EnvWorker

    captured = {}

    def fake_builder(
        env, obs, action, reward, terminated, truncated, info, obs_embedding, lang_emb
    ):
        captured["lang_emb"] = np.asarray(lang_emb, dtype=np.float32)
        return {
            "obs_embedding": np.asarray(obs_embedding, np.float16),
            "lang_emb": np.asarray(lang_emb, np.float32),
        }

    cfg = {
        "target": "dreamervla.workers.env._test_envs:DumpCounterEnv",
        "kwargs": {"horizon": 2, "image_shape": (4, 4, 3), "embedding_dim": 4},
    }

    class _Sink:
        def add_episode(self, ep):
            return None

    worker = EnvWorker(cfg, task_id=0, replay=_Sink(), record_builder=fake_builder)
    worker.init()
    worker.step(
        np.zeros(7, np.float32),
        np.zeros(4, np.float32),
        np.arange(6, dtype=np.float32),
    )

    assert np.array_equal(captured["lang_emb"], np.arange(6, dtype=np.float32))
