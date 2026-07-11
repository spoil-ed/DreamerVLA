from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class _Cluster:
    num_gpus: int


class _Ready:
    def __init__(self, result) -> None:
        self.result = result

    def wait(self):
        return self.result

    def done(self) -> bool:
        return True


class _Replay:
    def __init__(self) -> None:
        self.ready_calls = 0

    def ready(self, **kwargs) -> _Ready:
        del kwargs
        self.ready_calls += 1
        return _Ready([True])

    def size(self) -> _Ready:
        return _Ready([2])


class _CheckpointReplay(_Replay):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_state: dict | None = None
        self.policy_versions: list[int] = []

    def state_dict(self) -> _Ready:
        return _Ready(
            [
                {
                    "episodes_by_task": {0: [{"episode_id": 3, "length": 12}]},
                    "num_transitions": 12,
                    "current_policy_version": 7,
                }
            ]
        )

    def load_state_dict(self, state: dict) -> _Ready:
        self.loaded_state = dict(state)
        return _Ready([None])

    def set_policy_version(self, version: int) -> _Ready:
        self.policy_versions.append(int(version))
        return _Ready([None])


class _EnvGroup:
    def current_obs(self) -> _Ready:
        return _Ready([{"step": 0, "env_id": 0, "is_first": False}])

    def execute_on(self, rank: int):
        assert rank == 0
        return self

    def step(self, action, hidden, lang_emb=None, step_metadata=None) -> _Ready:
        del action, hidden, lang_emb, step_metadata
        return _Ready([({"step": 1, "env_id": 0}, False, {})])


class _Infer:
    def forward_batch(self, obs_batch, env_ids) -> _Ready:
        assert len(obs_batch) == 1
        assert env_ids == [0]
        return _Ready(
            [
                {
                    "actions": [[0.0] * 7],
                    "obs_embedding": [[1.0, 1.0, 1.0, 1.0]],
                    "timing": {
                        "encode_s": 0.1,
                        "world_model_s": 0.2,
                        "policy_s": 0.3,
                    },
                }
            ]
        )

    def reset_states(self, done_envs) -> _Ready:
        del done_envs
        return _Ready([None])

    def pull_weights(self, store_name: str, what: str, local_version: int) -> _Ready:
        assert store_name
        assert what == "policy"
        assert local_version == 0
        return _Ready([1])


class _Learner:
    def __init__(self) -> None:
        self.update_phases: list[tuple[str, int]] = []

    def update(self, phase: str, num_steps: int) -> _Ready:
        self.update_phases.append((phase, int(num_steps)))
        return _Ready([{"rl/actor_loss": 0.25, "wm/loss": 0.5}])

    def sync_weights(self, what: str, version: int) -> _Ready:
        assert what == "policy"
        assert version == 1
        return _Ready([None])

    def state_dicts(self):
        import torch

        return {"policy": {"weight": torch.ones(1)}}


class _MultiUpdateLearner:
    def __init__(self) -> None:
        self.update_phases: list[tuple[str, int]] = []
        self.synced_versions: list[int] = []

    def update(self, phase: str, num_steps: int) -> _Ready:
        self.update_phases.append((phase, int(num_steps)))
        value = float(len(self.update_phases))
        return _Ready([{"rl/actor_loss": value}])

    def sync_weights(self, what: str, version: int) -> _Ready:
        assert what == "policy"
        self.synced_versions.append(int(version))
        return _Ready([None])

    def state_dicts(self):
        import torch

        return {"policy": {"weight": torch.tensor([float(len(self.update_phases))])}}


class _NeverReadyReplay:
    def ready(self, **kwargs) -> _Ready:
        del kwargs
        return _Ready([False])

    def size(self) -> _Ready:
        return _Ready([0])


class _CountingEnvGroup:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self.step_ranks: list[int] = []
        self._rank = 0

    def current_obs(self) -> _Ready:
        return _Ready(
            [
                {"step": len(self.step_ranks), "env_id": env_id, "is_first": False}
                for env_id in range(self.num_envs)
            ]
        )

    def execute_on(self, rank: int):
        self._rank = int(rank)
        return self

    def step(self, action, hidden, lang_emb=None, step_metadata=None) -> _Ready:
        del action, hidden, lang_emb, step_metadata
        self.step_ranks.append(self._rank)
        return _Ready([({"step": len(self.step_ranks), "env_id": self._rank}, False, {})])


class _DoneEnvGroup:
    def current_obs(self) -> _Ready:
        return _Ready([{"step": 0, "env_id": 0, "is_first": True}])

    def execute_on(self, rank: int):
        assert rank == 0
        return self

    def step(self, action, hidden, lang_emb=None, step_metadata=None) -> _Ready:
        del action, hidden, lang_emb, step_metadata
        return _Ready([({"step": 1, "env_id": 0}, True, {"success": True})])


class _TaskSwitchEnvGroup:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self.task_by_env = {env_id: env_id for env_id in range(self.num_envs)}
        self.set_task_calls: list[tuple[int, int, int]] = []
        self.step_ranks: list[int] = []
        self._rank = 0

    def current_obs(self) -> _Ready:
        return _Ready(
            [
                {
                    "step": len(self.step_ranks),
                    "env_id": env_id,
                    "task_id": self.task_by_env[env_id],
                    "is_first": False,
                }
                for env_id in range(self.num_envs)
            ]
        )

    def execute_on(self, rank: int):
        self._rank = int(rank)
        return self

    def set_task(self, task_id: int, start_episode_id: int = 0) -> _Ready:
        self.task_by_env[self._rank] = int(task_id)
        self.set_task_calls.append((self._rank, int(task_id), int(start_episode_id)))
        return _Ready(
            [
                {
                    "step": 0,
                    "env_id": self._rank,
                    "task_id": int(task_id),
                    "is_first": True,
                }
            ]
        )

    def step(self, action, hidden, lang_emb=None, step_metadata=None) -> _Ready:
        del action, hidden, lang_emb, step_metadata
        self.step_ranks.append(self._rank)
        return _Ready(
            [
                (
                    {
                        "step": len(self.step_ranks),
                        "env_id": self._rank,
                        "task_id": self.task_by_env[self._rank],
                    },
                    True,
                    {"success": False, "task_id": self.task_by_env[self._rank]},
                )
            ]
        )


class _BatchInfer:
    def forward_batch(self, obs_batch, env_ids) -> _Ready:
        assert len(obs_batch) == len(env_ids)
        return _Ready(
            [
                {
                    "actions": [[0.0] * 7 for _ in env_ids],
                    "obs_embedding": [[1.0, 1.0, 1.0, 1.0] for _ in env_ids],
                    "timing": {},
                }
            ]
        )

    def reset_states(self, done_envs) -> _Ready:
        del done_envs
        return _Ready([None])

    def pull_weights(self, store_name: str, what: str, local_version: int) -> _Ready:
        del store_name, what, local_version
        return _Ready([None])


class _UnexpectedLearner:
    def update(self, phase: str, num_steps: int) -> _Ready:
        raise AssertionError(f"learner should not update: {phase=} {num_steps=}")

    def sync_weights(self, what: str, version: int) -> _Ready:
        raise AssertionError(f"learner should not sync: {what=} {version=}")


class _MetricLoggerSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, int | None]] = []

    def log(self, metrics, *, step=None, **kwargs) -> None:
        del kwargs
        self.calls.append((dict(metrics), step))


class _BoundaryReplay:
    def __init__(self) -> None:
        self.transitions: list[dict] = []

    @property
    def last_transition(self) -> dict:
        return self.transitions[-1]

    def add_transition(self, transition: dict) -> None:
        self.transitions.append(dict(transition))


class _BoundaryPolicyWorker:
    def __init__(self) -> None:
        self.forward_batch_calls = 0
        self.pull_calls: list[int] = []

    def forward_batch(self, obs_batch, env_ids):
        self.forward_batch_calls += 1
        return _Ready([{"actions": [[0.0] * 7 for _ in env_ids], "obs_embedding": []}])

    def pull_weights(self, store_name: str, what: str, local_version: int):
        del store_name, local_version
        assert what == "policy"
        self.pull_calls.append(1)
        return _Ready([1])


class _BoundaryEnvWorker:
    def __init__(self) -> None:
        self.wm_sync_calls: list[int] = []
        self.classifier_sync_calls: list[int] = []

    def load_world_model_state(self, state_dict, version: int):
        del state_dict
        self.wm_sync_calls.append(int(version))
        return _Ready([None])

    def load_classifier_state(self, state_dict, version: int):
        del state_dict
        self.classifier_sync_calls.append(int(version))
        return _Ready([None])


class _BoundaryEnvGroup:
    def __init__(self, num_env_workers: int = 1) -> None:
        self.workers = [_BoundaryEnvWorker() for _ in range(int(num_env_workers))]
        self.step_calls_by_worker = [0 for _ in self.workers]
        self._rank = 0

    def execute_on(self, rank: int):
        self._rank = int(rank)
        return self

    def step(self, action, hidden=None, lang_emb=None, step_metadata=None):
        del action, hidden, lang_emb, step_metadata
        self.step_calls_by_worker[self._rank] += 1
        return _Ready([({"env_id": self._rank}, False, {})])

    def load_world_model_state(self, state_dict, version: int):
        for worker in self.workers:
            worker.load_world_model_state(state_dict, version)
        return _Ready([None for _ in self.workers])

    def load_classifier_state(self, state_dict, version: int):
        for worker in self.workers:
            worker.load_classifier_state(state_dict, version)
        return _Ready([None for _ in self.workers])

    @property
    def wm_sync_calls(self) -> list[int]:
        return self.workers[0].wm_sync_calls

    @property
    def classifier_sync_calls(self) -> list[int]:
        return self.workers[0].classifier_sync_calls


class _BoundaryLearner:
    def __init__(self) -> None:
        self.synced: list[tuple[str, int]] = []
        self.optimizer_step_calls = 0

    def sync_weights(self, what: str, version: int):
        self.synced.append((str(what), int(version)))
        return _Ready([None])

    def state_dicts(self):
        return _Ready(
            [
                {
                    "policy": {},
                    "world_model": {},
                    "classifier": {},
                }
            ]
        )


def test_ray_runner_uses_cotrain_phase_for_dreamervla_learner_mode() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    from dreamervla.utils.metric_logger import NullMetricLogger

    cfg = OmegaConf.create(
        {
            "rollout": {"steps": 1, "min_replay_episodes": 1},
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    runner._metric_logger = NullMetricLogger()
    learner = _Learner()

    history = runner._run_loop(
        {
            "envs": _EnvGroup(),
            "infer": _Infer(),
            "replay": _Replay(),
            "learner": learner,
            "store_name": "test_store",
            "num_envs": 1,
        }
    )

    assert learner.update_phases == [("cotrain", 1)]
    assert history["train/learner_updates"] == 1
    assert history["train/ppo_updates"] == 1
    assert history["train/rl_loss"] == 0.25
    assert history["rl/actor_loss"] == 0.25
    assert history["wm/loss"] == 0.5
    assert history["time/infer_encode_s"] == 0.1
    assert history["time/infer_world_model_s"] == 0.2
    assert history["time/infer_policy_s"] == 0.3


def test_ray_runner_final_metrics_distinguish_env_workers_from_slots(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners import online_cotrain_ray_runner as module

    class _ClusterStub:
        def __init__(self) -> None:
            self.shutdown_called = False

        def require_single_node(self) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_called = True

    cluster = _ClusterStub()
    monkeypatch.setattr(module, "Cluster", lambda _cfg: cluster)
    runner = module.OnlineCotrainRayRunner.__new__(module.OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({})
    runner._build_components = lambda _cluster: {
        "num_envs": 4,
        "num_env_workers": 1,
        "envs_per_worker": 4,
    }
    runner._run_loop = lambda _groups: {"rollout/steps": 160}

    metrics = runner.run()

    assert metrics["env/num_env_workers"] == 1
    assert metrics["env/num_logical_envs"] == 4
    assert metrics["env/envs_per_worker"] == 4
    assert cluster.shutdown_called is True


def test_runner_syncs_snapshots_only_at_rollout_boundary() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({})
    runner._policy_version = 0
    runner._wm_version = 0
    runner._classifier_version = 0
    runner._fake_replay = _BoundaryReplay()
    policy_worker = _BoundaryPolicyWorker()
    env_group = _BoundaryEnvGroup()
    learner = _BoundaryLearner()
    groups = {
        "infer": policy_worker,
        "envs": env_group,
        "learner": learner,
        "store_name": "boundary_store",
    }

    runner._begin_rollout_round()
    runner._record_transition({"obs": 1, "action": 2})

    assert policy_worker.pull_calls == []
    assert env_group.wm_sync_calls == []
    assert runner._fake_replay.last_transition["policy_version"] == 0
    assert runner._fake_replay.last_transition["wm_version"] == 0
    assert runner._fake_replay.last_transition["classifier_version"] == 0

    runner._mark_learner_update_result(
        {
            "policy": {"updated": True},
            "world_model": {"updated": True},
            "classifier": {"updated": True},
        }
    )
    runner._sync_after_rollout_boundary(groups)

    assert runner._policy_version == 1
    assert runner._wm_version == 1
    assert runner._classifier_version == 1
    assert learner.synced == [("policy", 1), ("world_model", 1), ("classifier", 1)]
    assert policy_worker.pull_calls == [1]
    assert env_group.wm_sync_calls == [1]
    assert env_group.classifier_sync_calls == [1]


def test_runner_dispatches_rollout_work_to_worker_groups() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({})
    runner._fake_policy_worker = _BoundaryPolicyWorker()
    runner._fake_env_group = _BoundaryEnvGroup(num_env_workers=3)
    runner._fake_learner = _BoundaryLearner()

    runner._dispatch_rollout_round(
        obs_batch=[{"env_id": 0}, {"env_id": 1}, {"env_id": 2}],
        env_ids=[0, 1, 2],
    )

    assert runner._fake_policy_worker.forward_batch_calls == 1
    assert runner._fake_env_group.step_calls_by_worker == [1, 1, 1]
    assert runner._fake_learner.optimizer_step_calls == 0


def test_runner_dispatches_language_sidecars_to_env_workers() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    class LangPolicyWorker:
        def forward_batch(self, obs_batch, env_ids):
            del obs_batch
            return _Ready(
                [
                    {
                        "actions": [[float(env_id)] * 7 for env_id in env_ids],
                        "obs_embedding": [[float(env_id), 1.0] for env_id in env_ids],
                        "lang_emb": [[float(env_id), 2.0, 3.0] for env_id in env_ids],
                    }
                ]
            )

    class LangEnvGroup:
        def __init__(self) -> None:
            self._rank = 0
            self.calls: list[tuple[int, list[float], list[float]]] = []

        def execute_on(self, rank: int):
            self._rank = int(rank)
            return self

        def step(self, action, hidden=None, lang_emb=None):
            del action
            self.calls.append((self._rank, list(hidden), list(lang_emb)))
            return _Ready([({"env_id": self._rank}, False, {})])

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({})
    runner._fake_policy_worker = LangPolicyWorker()
    runner._fake_env_group = LangEnvGroup()
    runner._fake_learner = _BoundaryLearner()

    runner._dispatch_rollout_round(
        obs_batch=[{"env_id": 0}, {"env_id": 1}],
        env_ids=[0, 1],
    )

    assert runner._fake_env_group.calls == [
        (0, [0.0, 1.0], [0.0, 2.0, 3.0]),
        (1, [1.0, 1.0], [1.0, 2.0, 3.0]),
    ]


def test_ray_runner_rollout_steps_count_real_env_steps() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    from dreamervla.utils.metric_logger import NullMetricLogger

    cfg = OmegaConf.create(
        {
            "rollout": {"steps": 3, "min_replay_episodes": 1},
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    runner._metric_logger = NullMetricLogger()
    envs = _CountingEnvGroup(num_envs=2)

    history = runner._run_loop(
        {
            "envs": envs,
            "infer": _BatchInfer(),
            "replay": _NeverReadyReplay(),
            "learner": _UnexpectedLearner(),
            "store_name": "test_store",
            "num_envs": 2,
        }
    )

    assert envs.step_ranks == [0, 1, 0]
    assert history["rollout/steps"] == 3
    assert history["time/rollout_env_ready_batches"] == 3


def test_ray_runner_prints_episode_success_rate(capsys) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    from dreamervla.utils.metric_logger import NullMetricLogger

    cfg = OmegaConf.create(
        {
            "rollout": {"steps": 1, "min_replay_episodes": 1},
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    runner._metric_logger = NullMetricLogger()

    history = runner._run_loop(
        {
            "envs": _DoneEnvGroup(),
            "infer": _BatchInfer(),
            "replay": _NeverReadyReplay(),
            "learner": _UnexpectedLearner(),
            "store_name": "test_store",
            "num_envs": 1,
        }
    )

    out = capsys.readouterr().out
    assert "[rollout] episode=1 success=1 avg_success_rate=1.000" in out
    assert history["rollout/episodes"] == 1
    assert history["rollout/success_rate"] == 1.0
    assert history["rollout/success_rate_valid"] == 1.0
    assert history["rollout/recent_success_rate"] == 1.0
    assert history["rollout/recent_success_rate_valid"] == 1.0
    assert "rollout/current_success_rate" not in history
    assert "rollout/avg_success_rate" not in history


def test_ray_runner_episode_success_print_does_not_log_metrics(capsys) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({})
    spy = _MetricLoggerSpy()
    runner._metric_logger = spy

    runner._record_rollout_episode(episode=1, success=True, successes=1)

    assert "[rollout] episode=1 success=1 avg_success_rate=1.000" in capsys.readouterr().out
    assert spy.calls == []


def test_ray_runner_passes_replay_ready_gates(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    del monkeypatch
    captured = {}

    class ReadyResult:
        refs = ["ready"]

        def wait(self):
            return [False]

    class EmptyEnvGroup:
        def current_obs(self):
            return _Ready([])

    class Replay:
        def ready(self, **kwargs):
            captured.update(kwargs)
            return ReadyResult()

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "rollout": {
                "steps": 0,
                "min_replay_episodes": 2,
                "min_replay_transitions": 24,
                "min_sampleable_windows": 12,
                "require_classifier_evidence": True,
            },
            "ray_data": {"task_ids": [0, 1]},
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )
    runner.console_progress = lambda *_args, **_kwargs: None

    runner._run_loop_overlap(
        {
            "envs": EmptyEnvGroup(),
            "infer": object(),
            "replay": Replay(),
            "learner": object(),
            "store_name": "test_store",
            "num_envs": 0,
        }
    )

    assert captured["min_episodes_per_task"] == 2
    assert captured["min_transitions"] == 24
    assert captured["task_ids"] == (0, 1)
    assert captured["min_sampleable_windows"] == 12
    assert captured["require_classifier_evidence"] is True


def test_ray_runner_accepts_declared_oft_fixed_base_mode() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({"ray_rollout": {"mode": "oft_fixed_base"}})

    assert runner._ray_rollout_mode() == "oft_fixed_base"


def test_ray_runner_rejects_unknown_rollout_mode() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({"ray_rollout": {"mode": "mixed"}})

    with pytest.raises(ValueError, match="ray_rollout.mode"):
        runner._ray_rollout_mode()


def test_ray_runner_saves_learner_checkpoint(tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    cfg = OmegaConf.create(
        {
            "checkpoint": {
                "every_updates": 1,
                "save_final": True,
                "format_str": "ray_step={env_step:07d}-global={global_step:07d}.ckpt",
                "latest_name": "ray_latest.ckpt",
            },
            "training": {"out_dir": str(tmp_path)},
        }
    )
    runner.cfg = cfg
    runner.config = cfg
    runner._output_dir = None
    groups = {"learner": _Learner()}

    path = runner._maybe_save_ray_checkpoint(
        groups,
        env_steps=123,
        learner_updates=1,
        policy_version=1,
        metrics={"rl/actor_loss": 0.25},
    )

    assert path == tmp_path / "checkpoints" / "ray_step=0000123-global=0000001.ckpt"
    latest = tmp_path / "checkpoints" / "ray_latest.ckpt"
    assert path.is_file()
    assert latest.is_file()
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1
    assert payload["ray"] == {"global_step": 1, "env_step": 123}
    assert torch.equal(payload["state_dicts"]["policy"]["weight"], torch.ones(1))
    assert payload["metrics"]["rl/actor_loss"] == 0.25


def test_ray_runner_uses_learner_global_step_for_progress_and_checkpoint(tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    from dreamervla.utils.metric_logger import NullMetricLogger

    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path), "max_steps": 1},
            "checkpoint": {
                "save_interval": 1,
                "save_final": False,
                "filename": "learner.ckpt",
                "latest_name": "latest.ckpt",
            },
            "rollout": {"steps": 4, "min_replay_episodes": 1},
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    runner.config = cfg
    runner._output_dir = None
    runner.global_step = 0
    runner._metric_logger = NullMetricLogger()
    progress_calls: list[tuple[int, int, str, str | None, str | None]] = []
    runner.console_progress = lambda current, total, label, **kwargs: progress_calls.append(
        (int(current), int(total), str(label), kwargs.get("unit"), kwargs.get("status"))
    )
    runner.console_banner = lambda *_args, **_kwargs: None
    runner.console_metrics = lambda *_args, **_kwargs: None
    runner.console_metric_table = lambda *_args, **_kwargs: None

    learner = _MultiUpdateLearner()
    history = runner._run_loop(
        {
            "envs": _EnvGroup(),
            "infer": _BatchInfer(),
            "replay": _Replay(),
            "learner": learner,
            "store_name": "test_store",
            "num_envs": 1,
        }
    )

    ckpt_path = tmp_path / "checkpoints" / "global_step_1" / "learner.ckpt"
    latest_path = tmp_path / "checkpoints" / "latest.ckpt"
    assert history["train/learner_updates"] == 1
    assert runner.global_step == 1
    assert learner.update_phases == [("cotrain", 1)]
    assert progress_calls[-1][:4] == (1, 1, "train", "step")
    assert progress_calls[-1][4] is not None
    assert "global_step=1" in progress_calls[-1][4]
    assert "learner_step=" not in progress_calls[-1][4]
    assert "train_step=" in progress_calls[-1][4]
    assert "env_step=1/4" in progress_calls[-1][4]
    assert ckpt_path.is_file()
    assert latest_path.is_file()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1
    assert payload["ray"] == {"global_step": 1, "env_step": 1}
    assert torch.equal(payload["state_dicts"]["policy"]["weight"], torch.tensor([1.0]))


def test_ray_runner_progress_status_summarizes_rollout_and_training_state() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({})
    progress_calls: list[tuple[int, int, str, str | None, str | None]] = []
    runner.console_progress = lambda current, total, label, **kwargs: progress_calls.append(
        (int(current), int(total), str(label), kwargs.get("unit"), kwargs.get("status"))
    )

    runner._console_cotrain_progress(
        3,
        10,
        57,
        200,
        episode_count=5,
        episode_successes=2,
        active_task_by_env={0: 0, 1: 3, 2: 9},
        episode_steps_by_env={0: 12, 1: 4, 2: 0},
        last_loss=0.1234,
        last_metrics={"cls/acc": 0.75},
    )

    current, total, label, unit, status = progress_calls[-1]
    assert (current, total, label, unit) == (3, 10, "train", "step")
    assert status is not None
    assert "phase=rollout" in status
    assert "global_step=3" in status
    assert "learner_step=" not in status
    assert "train_step=0/0" in status
    assert "wm_step=0/0" in status
    assert "cls_step=0/0" in status
    assert "vlarl_step=0/0" in status
    assert "env_step=57/200" in status
    assert "rollout_step=t0:s12,t3:s4,t9:s0" in status
    assert "eps=5" in status
    assert "succ=0.400" in status
    assert "loss=0.123" in status
    assert "cls_acc=0.750" in status


def test_ray_runner_progress_status_reads_internal_train_step(tmp_path) -> None:
    import json

    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    progress_path = tmp_path / "learner_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "active": True,
                "train_step": 2,
                "total_train_steps": 3,
                "wm_step": 1,
                "wm_total_steps": 1,
                "cls_step": 1,
                "cls_total_steps": 1,
                "vlarl_step": 0,
                "vlarl_total_steps": 1,
            }
        ),
        encoding="utf-8",
    )

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {"learner": {"train_cfg": {"progress_path": str(progress_path)}}}
    )
    progress_calls: list[str | None] = []
    runner.console_progress = lambda *_args, **kwargs: progress_calls.append(
        kwargs.get("status")
    )

    runner._console_cotrain_progress(
        41,
        960,
        36192,
        200000,
        phase="overlap",
    )

    status = progress_calls[-1]
    assert status is not None
    assert "phase=overlap" in status
    assert "global_step=41" in status
    assert "train_step=2/3" in status
    assert "wm_step=1/1" in status
    assert "cls_step=1/1" in status
    assert "vlarl_step=0/1" in status
    assert "learner_step=" not in status


def test_ray_runner_metric_table_maps_cotrain_metrics() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({})
    table_calls: list[dict] = []
    runner.console_success_rate = lambda: 0.25
    runner.console_metric_table = lambda **kwargs: table_calls.append(kwargs)

    runner._console_cotrain_metric_table(
        global_step=7,
        target_global_steps=1000,
        start_global_step=0,
        train_start_t=0.0,
        env_steps=172,
        target_env_steps=200000,
        infer_batches=43,
        episode_count=8,
        episode_successes=2,
        last_episode_success=1,
        metrics={
            "rl/returns_mean": 0.5,
            "rl/actor_loss": 0.12,
            "rl/kl": 0.01,
            "cls/acc": 0.75,
            "cls/f1": 0.8,
            "wm/loss": 0.33,
            "wm/recon_loss": 0.44,
            "train/rl_loss": 0.45,
        },
        timing={
            "time/infer_wait_s": 2.0,
            "time/env_step_wait_s": 3.0,
            "time/learner_wait_s": 1.2,
            "time/weight_sync_wait_s": 0.4,
            "time/infer_encode_s": 9.9,
        },
    )

    call = table_calls[-1]
    assert call["step"] == 6
    assert call["total_steps"] == 1000
    metrics = call["metrics"]
    assert metrics["rollout/env_steps"] == 172.0
    assert metrics["rollout/returns_mean"] == 0.5
    assert metrics["train/actor/actor_loss"] == 0.12
    assert metrics["train/classifier/acc"] == 0.75
    assert metrics["train/world_model/loss"] == 0.33
    assert metrics["train/rl_loss"] == 0.45
    assert metrics["env/success_once"] == 0.25
    assert metrics["time/infer_wait_s"] == 2.0
    assert metrics["time/env_step_wait_s"] == 3.0
    assert metrics["time/learner_wait_s"] == 1.2
    assert metrics["time/weight_sync_wait_s"] == 0.4
    assert "time/infer_encode_s" not in metrics
    assert "train/actor/kl" not in metrics
    assert "train/classifier/f1" not in metrics
    assert "train/world_model/recon_loss" not in metrics


def test_ray_runner_rotates_task_pool_on_episode_done() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    from dreamervla.utils.metric_logger import NullMetricLogger

    cfg = OmegaConf.create(
        {
            "training": {"max_steps": 1},
            "rollout": {"steps": 6, "min_replay_episodes": 1},
            "ray_data": {"task_ids": [0, 1, 2, 3, 4, 5]},
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    runner._metric_logger = NullMetricLogger()
    runner.console_progress = lambda *_args, **_kwargs: None
    runner.console_banner = lambda *_args, **_kwargs: None
    runner.console_metrics = lambda *_args, **_kwargs: None
    runner.console_metric_table = lambda *_args, **_kwargs: None
    envs = _TaskSwitchEnvGroup(num_envs=4)

    history = runner._run_loop(
        {
            "envs": envs,
            "infer": _BatchInfer(),
            "replay": _NeverReadyReplay(),
            "learner": _UnexpectedLearner(),
            "store_name": "test_store",
            "num_envs": 4,
        }
    )

    assert history["rollout/steps"] == 6
    assert envs.set_task_calls[:2] == [(0, 4, 0), (1, 5, 0)]
    switched_tasks = [task_id for _env_id, task_id, _episode_id in envs.set_task_calls]
    assert switched_tasks[:6] == [4, 5, 0, 1, 2, 3]


def test_ray_runner_reserves_episode_ids_from_rollout_dump_resume() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({"ray_data": {"task_ids": [0, 1, 2, 3, 4, 5]}})
    runner._rollout_episode_resume_counts_value = {0: 3, 1: 2, 4: 7}

    assignments, counts = runner._rollout_initial_task_assignments(num_envs=4)

    assert assignments == [(0, 0, 3), (1, 1, 2), (2, 2, 0), (3, 3, 0)]
    assert counts[0] == 4
    assert counts[1] == 3
    assert counts[2] == 1
    assert counts[3] == 1
    assert counts[4] == 7

    state = runner._make_rollout_task_state(num_envs=4)
    envs = _TaskSwitchEnvGroup(num_envs=4)
    runner._maybe_rotate_rollout_task(
        envs,
        env_id=0,
        next_obs={"task_id": 0},
        task_state=state,
    )

    assert envs.set_task_calls[-1] == (0, 4, 7)
    assert state["task_episode_counts"][4] == 8


@pytest.mark.parametrize(
    "path",
    [
        "configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml",
    ],
)
def test_ray_lumos_configs_use_non_degenerate_grpo_groups(path: str) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    group_size = int(
        OmegaConf.select(
            cfg, "learner.train_cfg.algorithm_cfg.ppo_rollouts_per_start"
        )
    )
    filters_zero_variance = bool(
        OmegaConf.select(
            cfg,
            "learner.train_cfg.algorithm_cfg.lumos.filter_zero_variance_groups",
            default=True,
        )
    )

    assert not filters_zero_variance or group_size > 1


@pytest.mark.parametrize(
    "path",
    [
        "configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml",
    ],
)
def test_ray_lumos_configs_expose_rlinf_rollout_scale(path: str) -> None:
    from omegaconf import OmegaConf

    from dreamervla.config_resolvers import register_dreamervla_resolvers

    register_dreamervla_resolvers()
    cfg = OmegaConf.load(path)

    assert int(OmegaConf.select(cfg, "algorithm.group_size")) == 8
    assert int(OmegaConf.select(cfg, "algorithm.rollout_epoch")) == 16
    assert int(OmegaConf.select(cfg, "env.train.total_num_envs")) == 64
    assert int(OmegaConf.select(cfg, "env.train.max_episode_steps")) == 512
    assert int(OmegaConf.select(cfg, "env.train.max_steps_per_rollout_epoch")) == 512

    algorithm_cfg = cfg.learner.train_cfg.algorithm_cfg
    assert int(cfg.learner.train_cfg.batch_size) == 2
    assert int(algorithm_cfg.rollout_epoch) == 16
    assert int(algorithm_cfg.ppo_rollouts_per_start) == 8
    assert int(algorithm_cfg.lumos.episode_max_steps) == 512
    assert int(algorithm_cfg.lumos.ppo_rollouts_per_start_min) == 8
    assert int(algorithm_cfg.lumos.ppo_rollouts_per_start_max) == 8

    imagined_per_epoch = (
        int(cfg.learner.train_cfg.batch_size)
        * int(algorithm_cfg.imag_last)
        * int(algorithm_cfg.ppo_rollouts_per_start)
    )
    assert imagined_per_epoch == int(cfg.env.train.total_num_envs)
    assert imagined_per_epoch * int(algorithm_cfg.rollout_epoch) == 1024


@pytest.mark.parametrize(
    ("path", "subdir"),
    [
        (
            "configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml",
            "online_cotrain_input_token_embedding",
        ),
    ],
)
def test_ray_cotrain_configs_dump_one_hdf5_per_episode(
    path: str,
    subdir: str,
) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    dump = cfg.rollout.dump

    assert dump.enabled is True
    assert int(dump.demos_per_shard) == 1
    assert str(dump.shard_prefix) == "cotrain_episode"
    assert int(dump.preprocess_config.sidecar_schema_version) == 1
    assert list(dump.preprocess_config.required_demo_datasets) == ["obs_embedding"]
    unresolved = OmegaConf.to_yaml(dump, resolve=False)
    assert "/collected_rollouts/" in unresolved
    assert "OpenVLA_Onetraj_LIBERO" in unresolved
    assert f"/{subdir}/reward" in unresolved
    assert f"/{subdir}/hidden" in unresolved


def test_rollout_dump_cfg_rejects_incomplete_explicit_sidecar_metadata() -> None:
    from pathlib import Path

    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "rollout": {
                "dump": {
                    "enabled": True,
                    "preprocess_config": {
                        "action_head_type": "oft_discrete_token",
                        "obs_hidden_source": "input_token_embedding",
                    },
                }
            }
        }
    )
    runner.get_run_dir = lambda: Path("/tmp/dvla-test")

    with pytest.raises(ValueError, match="only supported observation contract"):
        runner._rollout_dump_cfg(oft_plan=None)


@pytest.mark.parametrize(
    "path",
    [
        "configs/dreamervla/openvla_onetraj_libero_cotrain_ray_base.yaml",
    ],
)
def test_ray_real_configs_use_component_placement_for_egl_mainline(path: str) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    from dreamervla.scheduler.placement import ComponentPlacement

    cfg = OmegaConf.load(path)
    cfg.render_backend = "egl"

    placement = ComponentPlacement(cfg)
    assert placement.has_component("env")
    assert placement.has_component("rollout")
    assert placement.has_component("actor")

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    placements = runner._env_placement().get_placement(_Cluster(num_gpus=2))

    assert placements
    assert all(p.visible_accelerators == ["0"] for p in placements)
    assert OmegaConf.select(cfg, "env.cfg.egl_device_pool", default=None) is None


def test_ray_runner_rejects_env_placement_count_mismatch() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "render_backend": "egl",
            "cluster": {"component_placement": {"env": "0-5"}},
            "env": {"num_workers": 1, "envs_per_worker": 2},
        }
    )

    with pytest.raises(ValueError, match="env.num_workers"):
        runner._validate_env_placement(_Cluster(num_gpus=6), runner._env_placement())


@pytest.mark.parametrize(
    ("component", "placement_cfg"),
    [
        ("inference", {"cluster": {"component_placement": {"rollout": "0-5"}}}),
        ("learner", {"cluster": {"component_placement": {"actor": "0-5"}}}),
    ],
)
def test_ray_runner_rejects_unsupported_multiworker_compute_placement(
    component: str,
    placement_cfg: dict,
) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(placement_cfg)
    strategy = (
        runner._inference_placement()
        if component == "inference"
        else runner._learner_placement()
    )

    with pytest.raises(ValueError, match=f"{component}.*single worker"):
        runner._validate_single_worker_placement(
            component,
            _Cluster(num_gpus=6),
            strategy,
        )


def test_ray_runner_builds_packed_multigpu_learner_placement() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    cfg = OmegaConf.create(
        {
            "learner": {
                "num_workers": 2,
                "placement": {
                    "strategy": "packed",
                    "start_gpu": 0,
                    "end_gpu": 1,
                    "num_gpus_per_worker": 1,
                },
                "train_cfg": {
                    "mode": "dreamervla_cotrain",
                    "device": "auto",
                },
            }
        }
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg

    strategy = runner._learner_placement()
    placements = strategy.get_placement(_Cluster(num_gpus=2))
    train_cfg = runner._learner_train_cfg("weight_store", placement_has_gpu=True)

    assert [placement.visible_accelerators for placement in placements] == [["0"], ["1"]]
    assert [placement.local_world_size for placement in placements] == [2, 2]
    assert train_cfg["device"] == "cuda:0"
    assert train_cfg["syncer"]["store_name"] == "weight_store"


def test_ray_runner_rejects_multinode_cluster_before_launch(monkeypatch) -> None:
    from omegaconf import OmegaConf

    import dreamervla.runners.online_cotrain_ray_runner as runner_module

    class _Cluster:
        def __init__(self, cfg) -> None:
            del cfg
            self.shutdown_called = False

        def require_single_node(self) -> None:
            raise RuntimeError("DreamerVLA Ray backend is single-node only")

        def shutdown(self) -> None:
            self.shutdown_called = True

    cluster = _Cluster(None)
    monkeypatch.setattr(runner_module, "Cluster", lambda cfg: cluster)

    runner = runner_module.OnlineCotrainRayRunner.__new__(
        runner_module.OnlineCotrainRayRunner
    )
    runner.cfg = OmegaConf.create({})

    with pytest.raises(RuntimeError, match="single-node"):
        runner.run()
    assert cluster.shutdown_called is True


def test_ray_runner_top_level_render_backend_is_canonical(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {"render_backend": "osmesa", "env": {"render_backend": "egl"}}
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    assert runner._egl_device_pool() == []


def test_ray_runner_egl_requires_explicit_render_devices(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "render_backend": "egl",
            "env": {"num_workers": 4},
            "inference": {"placement": {"strategy": "packed", "gpu_id": 0}},
            "learner": {
                "placement": {
                    "strategy": "packed",
                    "start_gpu": 1,
                    "end_gpu": 1,
                }
            },
        }
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    with pytest.raises(ValueError, match="render_devices.*osmesa"):
        runner._rollout_env_cfg(use_oft_collect_path=False)


def test_ray_runner_egl_rejects_render_compute_overlap() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "render_backend": "egl",
            "render_devices": [1, 2],
            "env": {"num_workers": 4},
            "inference": {"placement": {"strategy": "packed", "gpu_id": 0}},
            "learner": {
                "placement": {
                    "strategy": "packed",
                    "start_gpu": 1,
                    "end_gpu": 1,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="not overlap.*render_backend=osmesa"):
        runner._rollout_env_cfg(use_oft_collect_path=False)


def test_ray_runner_egl_cfg_uses_explicit_disjoint_render_devices_for_placement() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "render_backend": "egl",
            "render_devices": [2, 3],
            "env": {"num_workers": 4},
            "inference": {"placement": {"strategy": "packed", "gpu_id": 0}},
            "learner": {
                "placement": {
                    "strategy": "packed",
                    "start_gpu": 1,
                    "end_gpu": 1,
                }
            },
        }
    )

    env_cfg = runner._rollout_env_cfg(use_oft_collect_path=False)
    placements = runner._env_placement().get_placement(_Cluster(num_gpus=4))

    assert env_cfg["render_backend"] == "egl"
    assert "egl_device_pool" not in env_cfg
    assert [p.visible_accelerators for p in placements] == [["2"], ["2"], ["3"], ["3"]]


def test_ray_runner_egl_places_env_workers_on_render_gpu() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "render_backend": "egl",
            "render_devices": [2],
            "env": {"num_workers": 4},
            "inference": {"placement": {"strategy": "packed", "gpu_id": 0}},
            "learner": {
                "placement": {
                    "strategy": "packed",
                    "start_gpu": 1,
                    "end_gpu": 1,
                }
            },
        }
    )

    placements = runner._env_placement().get_placement(_Cluster(num_gpus=3))

    assert [p.visible_accelerators for p in placements] == [["2"]] * 4
    assert [p.device for p in placements] == ["cuda:2"] * 4


def test_ray_runner_env_placement_defaults_to_osmesa_node_workers() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({"env": {"num_workers": 2}})

    placements = runner._env_placement().get_placement(_Cluster(num_gpus=2))

    assert [p.visible_accelerators for p in placements] == [[], []]
    assert [p.device for p in placements] == ["cpu", "cpu"]


def test_ray_runner_egl_env_cfg_uses_worker_level_regime() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "render_backend": "egl",
            "render_devices": [2],
            "env": {"num_workers": 4},
            "inference": {"placement": {"strategy": "packed", "gpu_id": 0}},
            "learner": {
                "placement": {
                    "strategy": "packed",
                    "start_gpu": 1,
                    "end_gpu": 1,
                }
            },
        }
    )

    env_cfg = runner._rollout_env_cfg(use_oft_collect_path=False)

    assert env_cfg["render_backend"] == "egl"
    assert "egl_device_pool" not in env_cfg


def test_ray_runner_envs_per_worker_defines_logical_env_count() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "render_backend": "egl",
            "cluster": {"component_placement": {"env": 0}},
            "env": {"num_workers": 2, "envs_per_worker": 3},
        }
    )

    env_cfg = runner._rollout_env_cfg(use_oft_collect_path=False)

    assert runner._env_worker_count() == 2
    assert runner._envs_per_worker() == 3
    assert runner._logical_env_count() == 6
    assert env_cfg["num_envs_per_worker"] == 3


def test_ray_runner_osmesa_ignores_envs_per_worker_slots() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {"render_backend": "osmesa", "env": {"num_workers": 2, "envs_per_worker": 3}}
    )

    env_cfg = runner._rollout_env_cfg(use_oft_collect_path=False)

    assert runner._envs_per_worker() == 3
    assert runner._effective_envs_per_worker() == 1
    assert runner._logical_env_count() == 2
    assert env_cfg["num_envs_per_worker"] == 1


def test_ray_runner_maps_logical_env_ids_to_worker_slots() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {"render_backend": "egl", "env": {"num_workers": 2, "envs_per_worker": 3}}
    )

    assert runner._env_worker_rank_slot(0) == (0, 0)
    assert runner._env_worker_rank_slot(2) == (0, 2)
    assert runner._env_worker_rank_slot(3) == (1, 0)
    assert runner._env_worker_rank_slot(5) == (1, 2)


def test_ray_runner_env_step_dispatches_to_worker_slot() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    class _SlotEnvs:
        def __init__(self) -> None:
            self.rank = None
            self.calls = []

        def execute_on(self, rank):
            self.rank = int(rank)
            return self

        def step_slot(self, slot, action, hidden, lang_emb=None):
            self.calls.append((self.rank, int(slot), action, hidden, lang_emb))
            return _Ready([({"env": self.rank, "slot": int(slot)}, False, {})])

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {"render_backend": "egl", "env": {"num_workers": 2, "envs_per_worker": 3}}
    )
    envs = _SlotEnvs()

    result = runner._env_step(envs, env_id=5, action="a", hidden="h", lang_emb="l")

    assert result.wait() == [({"env": 1, "slot": 2}, False, {})]
    assert envs.calls == [(1, 2, "a", "h", "l")]


def test_ray_runner_flattens_worker_slot_observations() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {"render_backend": "egl", "env": {"num_workers": 2, "envs_per_worker": 2}}
    )

    flattened = runner._flatten_env_obs(
        [
            [{"env_id": 0}, {"env_id": 1}],
            [{"env_id": 2}, {"env_id": 3}],
        ]
    )

    assert flattened == [{"env_id": 0}, {"env_id": 1}, {"env_id": 2}, {"env_id": 3}]


def test_ray_oft_rollout_env_cfg_reuses_collect_plan(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner
    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "ray_rollout": {"mode": "oft_fixed_base"},
            "render_backend": "osmesa",
            "env": {
                "cfg": {
                    "target": "hand.authored.Env",
                    "kwargs": {"task_ids": [0, 1, 2], "task_id": 0},
                }
            },
        }
    )

    monkeypatch.setattr(
        ColdStartRayCollectRunner,
        "build_oft_worker_plan",
        lambda _self: {
            "env": {
                "target": "collect.path.Env",
                "use_from_config": True,
                "kwargs": {
                    "task_ids": [1],
                    "task_id": 1,
                    "full_record": True,
                    "init_state_sampling": "sequential",
                    "validate_canonical": False,
                },
            }
        },
    )

    env_cfg = runner._rollout_env_cfg()

    assert env_cfg["target"] == "collect.path.Env"
    assert env_cfg["kwargs"]["task_ids"] == [1]
    assert env_cfg["kwargs"]["full_record"] is True
    assert "egl_device_pool" not in env_cfg


def test_ray_runner_loads_init_ckpt_by_component_name(tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    ckpt = tmp_path / "warmup.ckpt"
    torch.save(
        {
            "state_dicts": {
                "policy": {"weight": torch.ones(1)},
                "world_model": {"bias": torch.zeros(1)},
                "classifier": {"head": torch.full((1,), 2.0)},
            }
        },
        ckpt,
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "learner": {
                "init_ckpt": {
                    "path": str(ckpt),
                    "components": ["policy", "classifier"],
                }
            }
        }
    )

    state = runner._load_init_ckpt("learner.init_ckpt")

    assert set(state) == {"policy", "classifier"}
    assert torch.equal(state["policy"]["weight"], torch.ones(1))
    assert torch.equal(state["classifier"]["head"], torch.full((1,), 2.0))


def test_ray_checkpoint_persists_only_canonical_loop_state(tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path), "checkpoint_every": 1},
            "checkpoint": {"save_interval": 1, "latest_name": "latest.ckpt"},
        }
    )
    runner._output_dir = str(tmp_path)
    runner.global_step = 0
    runner._policy_version = 7
    runner._wm_version = 5
    runner._classifier_version = 6
    replay = _CheckpointReplay()

    path = runner._maybe_save_ray_checkpoint(
        {"learner": _BoundaryLearner(), "replay": replay},
        env_steps=123,
        learner_updates=10,
        policy_version=7,
        metrics={
            "rollout/episodes": 8,
            "rollout/successes": 3,
            "env/last_success": 1,
        },
        loop_state={
            "global_step": 10,
            "env_steps": 123,
            "policy_version": 7,
            "wm_version": 5,
            "classifier_version": 6,
            "episode_count": 8,
            "episode_successes": 3,
            "last_episode_success": 1,
            "task_episode_counts": {0: 2, 1: 5},
        },
    )

    assert path is not None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "replay" not in payload
    assert payload["ray"] == {"global_step": 10, "env_step": 123}
    assert (tmp_path / "checkpoints" / "latest.ckpt").is_file()


def test_ray_runner_restores_replay_and_loop_state_from_init_ckpt(tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    ckpt = tmp_path / "latest.ckpt"
    replay_state = {"episodes_by_task": {0: [{"episode_id": 3, "length": 12}]}}
    torch.save(
        {
            "state_dicts": {
                "policy": {},
                "world_model": {},
                "classifier": {},
            },
            "replay": replay_state,
            "ray": {
                "global_step": 40,
                "env_step": 33578,
                "policy_version": 40,
                "wm_version": 39,
                "classifier_version": 38,
                "episode_count": 218,
                "episode_successes": 125,
                "last_episode_success": 1,
                "task_episode_counts": {0: 9, "1": 4},
            },
        },
        ckpt,
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "learner": {
                "init_ckpt": {
                    "path": str(ckpt),
                    "components": ["policy", "world_model", "classifier"],
                }
            }
        }
    )
    runner.global_step = 0
    replay = _CheckpointReplay()

    state = runner._restore_ray_resume_state({"replay": replay})

    assert state == {"global_step": 40, "env_step": 33578}
    assert runner.global_step == 40
    assert runner._policy_version == 40
    assert runner._wm_version == 0
    assert runner._classifier_version == 0
    assert replay.loaded_state is None
    assert replay.policy_versions == [40]


def test_ray_runner_restores_loop_state_without_replay_from_init_ckpt(tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    ckpt = tmp_path / "latest.ckpt"
    torch.save(
        {
            "global_step": 40,
            "state_dicts": {"policy": {}, "world_model": {}, "classifier": {}},
            "ray": {
                "global_step": 40,
                "env_step": 33578,
                "update_step": 40,
                "policy_version": 40,
            },
        },
        ckpt,
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {"learner": {"init_ckpt": {"path": str(ckpt), "components": ["policy"]}}}
    )
    runner.global_step = 0
    replay = _CheckpointReplay()

    state = runner._restore_ray_resume_state({"replay": replay})

    assert state["global_step"] == 40
    assert state["env_step"] == 33578
    assert runner.global_step == 40
    assert runner._policy_version == 40
    assert replay.loaded_state is None
    assert replay.policy_versions == [40]


def test_ray_runner_synthesizes_loop_state_from_top_level_checkpoint(tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    ckpt = tmp_path / "latest.ckpt"
    torch.save(
        {
            "global_step": 40,
            "state_dicts": {"policy": {}},
            "metrics": {
                "rollout/steps": 33578,
                "rollout/episodes": 218,
                "rl/classifier_updates": 39,
                "wm/loss": 1.5,
            },
        },
        ckpt,
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({"learner": {"init_ckpt": {"path": str(ckpt)}}})
    runner.global_step = 0
    replay = _CheckpointReplay()

    state = runner._restore_ray_resume_state({"replay": replay})

    assert state == {"global_step": 40, "env_step": 33578}
    assert runner.global_step == 40
    assert runner._wm_version == 0
    assert runner._classifier_version == 0
    assert replay.loaded_state is None
