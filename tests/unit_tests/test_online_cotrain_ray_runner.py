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


class _EnvGroup:
    def current_obs(self) -> _Ready:
        return _Ready([{"step": 0, "env_id": 0, "is_first": False}])

    def execute_on(self, rank: int):
        assert rank == 0
        return self

    def step(self, action, hidden, lang_emb=None) -> _Ready:
        del action, hidden, lang_emb
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

    def step(self, action, hidden, lang_emb=None) -> _Ready:
        del action, hidden, lang_emb
        self.step_ranks.append(self._rank)
        return _Ready([({"step": len(self.step_ranks), "env_id": self._rank}, False, {})])


class _DoneEnvGroup:
    def current_obs(self) -> _Ready:
        return _Ready([{"step": 0, "env_id": 0, "is_first": True}])

    def execute_on(self, rank: int):
        assert rank == 0
        return self

    def step(self, action, hidden, lang_emb=None) -> _Ready:
        del action, hidden, lang_emb
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

    def step(self, action, hidden, lang_emb=None) -> _Ready:
        del action, hidden, lang_emb
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

    def step(self, action, hidden=None, lang_emb=None):
        del action, hidden, lang_emb
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
                "format_str": "ray_step={env_step:07d}-updates={update_step:07d}.ckpt",
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

    assert path == tmp_path / "checkpoints" / "ray_step=0000123-updates=0000001.ckpt"
    latest = tmp_path / "checkpoints" / "ray_latest.ckpt"
    assert path.is_file()
    assert latest.is_file()
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1
    assert payload["ray"] == {
        "global_step": 1,
        "env_step": 123,
        "update_step": 1,
        "policy_version": 1,
    }
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
    assert "env_steps=1/4" in progress_calls[-1][4]
    assert ckpt_path.is_file()
    assert latest_path.is_file()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1
    assert payload["ray"] == {
        "global_step": 1,
        "env_step": 1,
        "update_step": 1,
        "policy_version": 1,
    }
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
    assert "env_steps=57/200" in status
    assert "collect=t0:s12,t3:s4,t9:s0" in status
    assert "eps=5" in status
    assert "succ=0.400" in status
    assert "loss=0.123" in status
    assert "cls_acc=0.750" in status


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


@pytest.mark.parametrize(
    "path",
    [
        "configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml",
        "configs/dreamervla/ray_online_cotrain_oft_backbone_latent.yaml",
        "configs/dreamervla/ray_online_cotrain_rynn_action_hidden.yaml",
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


def test_online_cotrain_ray_oft_experiment_accepts_render_backend_override() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_ray_oft_action_hidden",
                "render_backend=osmesa",
            ],
        )

    assert cfg.render_backend == "osmesa"


def test_online_cotrain_ray_oft_backbone_latent_uses_input_token_contract() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=online_cotrain_ray_oft_backbone_latent"],
        )

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    plan = runner._oft_worker_plan()

    task_spec = cfg.task.openvla_oft.input_tokens
    assert cfg.latent_type == "backbone_latent"
    assert cfg.ray_components.world_model.kwargs.latent_stage == "query_before"
    assert cfg.ray_components.world_model.kwargs.obs_dim == task_spec.wm_obs_dim
    assert cfg.ray_components.world_model.kwargs.token_count == task_spec.token_count
    assert cfg.ray_components.world_model.kwargs.token_dim == task_spec.token_dim
    assert cfg.ray_data.sequence_length == 12
    assert cfg.ray_components.world_model.kwargs.model_dim == 4138
    assert cfg.ray_components.world_model.kwargs.proprio_dim == 0
    assert cfg.ray_components.world_model.kwargs.proprio_emb_dim == 0
    assert cfg.ray_components.world_model.kwargs.num_proprio_repeat == 1
    assert cfg.ray_components.world_model.kwargs.lang_dim == 4096
    assert cfg.ray_components.world_model.kwargs.lang_emb_dim == 32
    assert cfg.ray_components.world_model.kwargs.num_lang_repeat == 1
    assert cfg.ray_components.world_model.kwargs.action_emb_dim == 10
    assert cfg.ray_components.world_model.kwargs.model_dim == (
        cfg.ray_components.world_model.kwargs.token_dim
        + cfg.ray_components.world_model.kwargs.lang_emb_dim
        + cfg.ray_components.world_model.kwargs.action_emb_dim
    )
    assert cfg.ray_components.world_model.kwargs.cosine_loss_scale == 0.0
    assert cfg.ray_components.world_model.kwargs.chunk_rollout_chunks == 1
    assert cfg.ray_components.world_model.kwargs.chunk_rollout_loss_scale == 0.0
    assert (
        cfg.ray_components.policy.target
        == "dreamervla.models.actor.LatentToOpenVLAHiddenStateActor"
    )
    assert cfg.ray_components.policy.kwargs.source_token_count == task_spec.token_count
    assert cfg.ray_components.policy.kwargs.hidden_state_dim == task_spec.hidden_state_dim
    assert "action_hidden_dim" not in cfg.ray_components.policy.kwargs
    assert cfg.ray_components.classifier.kwargs.token_count == task_spec.token_count
    assert cfg.env.cfg.kwargs.obs_hidden_source == "input_token_embedding"
    assert plan["collect"]["expected_obs_hidden_source"] == "input_token_embedding"
    assert plan["collect"]["token_count"] == task_spec.token_count
    assert plan["collect"]["hidden_dim"] == task_spec.wm_obs_dim
    assert plan["env"]["kwargs"]["obs_hidden_source"] == "input_token_embedding"
    assert (
        plan["inference"]["decoder"]["kwargs"]["obs_hidden_source"]
        == "input_token_embedding"
    )
    assert plan["dump"]["preprocess_config"]["obs_hidden_source"] == "input_token_embedding"
    assert plan["dump"]["preprocess_config"]["token_count"] == task_spec.token_count
    assert plan["dump"]["preprocess_config"]["hidden_dim"] == task_spec.wm_obs_dim


def test_online_cotrain_ray_oft_alias_uses_backbone_latent_route() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=online_cotrain_ray_oft"])

    assert cfg.latent_type == "backbone_latent"
    assert cfg.env.cfg.kwargs.obs_hidden_source == "input_token_embedding"
    assert cfg.ray_components.world_model.kwargs.latent_stage == "query_before"


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


def test_online_cotrain_ray_oft_experiment_composes_real_components() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=online_cotrain_ray_oft"])

    assert cfg._target_.endswith("OnlineCotrainRayRunner")
    assert cfg.learner.train_cfg.mode == "dreamervla_cotrain"
    assert cfg.ray_components.policy.target == cfg.learner.model_cfg.policy.target
    assert cfg.ray_components.world_model.target == cfg.learner.model_cfg.world_model.target
    assert cfg.ray_components.classifier.target == cfg.learner.model_cfg.classifier.target
    assert cfg.ray_data.sequence_length == cfg.replay.cfg.sequence_length
    task_spec = cfg.task.openvla_oft.input_tokens
    assert (
        cfg.learner.model_cfg.policy.target
        == "dreamervla.models.actor.LatentToOpenVLAHiddenStateActor"
    )
    assert (
        cfg.learner.model_cfg.world_model.target
        == "dreamervla.models.world_model.dino_wm_chunk.ChunkAwareDinoWMWorldModel"
    )
    assert (
        cfg.learner.model_cfg.classifier.target
        == "dreamervla.models.reward.latent_success_classifier.LatentSuccessClassifier"
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
    plan = runner._oft_worker_plan()
    assert cfg.ray_rollout.mode == "oft_fixed_base"
    assert plan["collect"]["model_path"] == cfg.task.openvla_oft.ckpt_path
    assert plan["inference"]["decoder"]["target"].endswith("OFTRolloutBundle")
    assert (
        cfg.ray_components.policy.kwargs.time_horizon
        == cfg.learner.model_cfg.policy.kwargs.time_horizon
    )
    assert cfg.replay.cfg.sequence_length >= cfg.learner.model_cfg.classifier.kwargs.window

    assert plan["collect"]["action_dim"] == cfg.task.action_dim
    assert plan["inference"]["action_steps"] == task_spec.chunk_size
    assert task_spec.wm_obs_dim == task_spec.token_count * task_spec.token_dim
    assert cfg.ray_components.world_model.kwargs.obs_dim == task_spec.wm_obs_dim
    assert cfg.ray_components.world_model.kwargs.token_count == task_spec.token_count
    assert cfg.ray_components.world_model.kwargs.token_dim == task_spec.token_dim
    assert cfg.ray_components.world_model.kwargs.chunk_size == task_spec.chunk_size
    assert cfg.ray_components.policy.kwargs.hidden_state_dim == task_spec.hidden_state_dim
    assert "action_hidden_dim" not in cfg.ray_components.policy.kwargs
    assert cfg.ray_components.policy.kwargs.time_horizon == task_spec.chunk_size
    assert cfg.ray_components.classifier.kwargs.latent_dim == task_spec.token_dim
    assert cfg.learner.model_cfg.world_model.kwargs.obs_dim == task_spec.wm_obs_dim

    unresolved_wm = OmegaConf.to_yaml(cfg.ray_components.world_model.kwargs, resolve=False)
    unresolved_policy = OmegaConf.to_yaml(cfg.ray_components.policy.kwargs, resolve=False)
    unresolved_classifier = OmegaConf.to_yaml(cfg.ray_components.classifier.kwargs, resolve=False)
    assert "${task.openvla_oft.input_tokens.wm_obs_dim}" in unresolved_wm
    assert "${task.openvla_oft.input_tokens.token_count}" in unresolved_wm
    assert "${task.openvla_oft.input_tokens.token_dim}" in unresolved_wm
    assert "${task.openvla_oft.input_tokens.hidden_state_dim}" in unresolved_policy
    assert "${task.openvla_oft.input_tokens.token_dim}" in unresolved_classifier


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
