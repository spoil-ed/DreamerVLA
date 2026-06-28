from __future__ import annotations


class _Replay:
    def __init__(self) -> None:
        self.episodes = []

    def add_episode(self, episode):
        self.episodes.append(list(episode))


class _RemoteReplay:
    def __init__(self) -> None:
        self.inner = _Replay()
        self.add_episode = type(
            "_RemoteAddEpisode",
            (),
            {"remote": lambda _self, episode: self.inner.add_episode(episode)},
        )()


def test_env_worker_forwards_world_model_and_classifier_sync():
    from dreamervla.workers.env.env_worker import EnvWorker

    class _Env:
        def __init__(self):
            self.wm_version = 0
            self.classifier_version = 0

        def load_world_model_state(self, state_dict, version):
            self.wm_version = int(version)

        def load_classifier_state(self, state_dict, version):
            self.classifier_version = int(version)

    worker = EnvWorker(env_cfg={"target": "unused"}, task_id=0, replay=None)
    worker.env = _Env()

    worker.load_world_model_state({}, version=5)
    worker.load_classifier_state({}, version=7)

    assert worker.env.wm_version == 5
    assert worker.env.classifier_version == 7


def test_env_worker_step_accepts_missing_obs_embedding_for_generic_env(monkeypatch):
    from dreamervla.workers.env.env_worker import EnvWorker

    class _Env:
        def reset(self, *, task_id=0, episode_id=0):
            return {"latent": [0.0], "task_id": task_id}, {}

        def step(self, action):
            return {"latent": [1.0]}, 1.0, True, False, {"success": True}

    monkeypatch.setattr(
        "dreamervla.workers.env.env_worker.ray.get",
        lambda value: value,
    )
    replay = _RemoteReplay()
    worker = EnvWorker(env_cfg={"target": "unused"}, task_id=0, replay=replay)
    worker.env = _Env()
    worker.obs = {"latent": [0.0], "task_id": 0}

    next_obs, done, info = worker.step([0.0])

    assert done is True
    assert info["success"] is True
    assert "latent" in next_obs
    assert replay.inner.episodes[0][0]["obs_embedding"] is None


def test_env_worker_egl_backend_spawns_with_inherited_worker_regime(monkeypatch):
    from dreamervla.workers.env.env_worker import EnvWorker

    calls = []
    worker = EnvWorker(
        env_cfg={"target": "unused", "render_backend": "egl"},
        task_id=0,
        replay=None,
    )

    monkeypatch.setattr(
        worker,
        "_init_spawn",
        lambda egl_device_id, slot_id=0: calls.append(("spawn", egl_device_id, slot_id)),
    )
    monkeypatch.setattr(worker, "_init_inproc", lambda: calls.append(("inproc", None)))

    worker.init()

    assert calls == [("spawn", None, 0)]


def test_env_worker_egl_backend_logs_worker_level_diagnostics(monkeypatch):
    from dreamervla.workers.env import env_worker as module

    calls = []
    worker = module.EnvWorker(
        env_cfg={"target": "unused", "render_backend": "egl"},
        task_id=0,
        replay=None,
    )

    monkeypatch.setattr(
        worker,
        "_init_spawn",
        lambda egl_device_id, slot_id=0: calls.append(("spawn", egl_device_id, slot_id)),
    )
    monkeypatch.setattr(
        module,
        "log_egl_device_diagnostics_from_env",
        lambda logger_name: calls.append(("diagnostics", logger_name)),
    )

    worker.init()

    assert calls == [("diagnostics", module.__name__), ("spawn", None, 0)]


def test_env_worker_egl_initializes_each_worker_slot(monkeypatch):
    from dreamervla.workers.env.env_worker import EnvWorker

    calls = []
    worker = EnvWorker(
        env_cfg={
            "target": "unused",
            "render_backend": "egl",
            "num_envs_per_worker": 3,
        },
        task_id=0,
        replay=None,
    )

    monkeypatch.setattr(
        worker,
        "_init_spawn",
        lambda egl_device_id, slot_id=0: calls.append((egl_device_id, slot_id)),
    )

    worker.init()

    assert calls == [(None, 0), (None, 1), (None, 2)]
