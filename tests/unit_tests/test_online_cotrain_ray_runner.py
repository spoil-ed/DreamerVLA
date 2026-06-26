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

    def step(self, action, hidden) -> _Ready:
        del action, hidden
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

    def step(self, action, hidden) -> _Ready:
        del action, hidden
        self.step_ranks.append(self._rank)
        return _Ready([({"step": len(self.step_ranks), "env_id": self._rank}, False, {})])


class _DoneEnvGroup:
    def current_obs(self) -> _Ready:
        return _Ready([{"step": 0, "env_id": 0, "is_first": True}])

    def execute_on(self, rank: int):
        assert rank == 0
        return self

    def step(self, action, hidden) -> _Ready:
        del action, hidden
        return _Ready([({"step": 1, "env_id": 0}, True, {"success": True})])


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


@pytest.mark.parametrize(
    "path",
    [
        "configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml",
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


def test_online_cotrain_ray_oft_experiment_composes_real_components() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=online_cotrain_ray_oft"])

    assert cfg._target_.endswith("OnlineCotrainRayRunner")
    assert cfg.learner.train_cfg.mode == "dreamervla_cotrain"
    assert cfg.ray_components.policy.target == cfg.learner.model_cfg.policy.target
    assert cfg.ray_components.world_model.target == cfg.learner.model_cfg.world_model.target
    assert cfg.ray_components.classifier.target == cfg.learner.model_cfg.classifier.target
    assert cfg.ray_data.sequence_length == cfg.replay.cfg.sequence_length
    assert cfg.learner.model_cfg.policy.target == "dreamervla.models.actor.RynnVLAActionHiddenActor"
    assert (
        cfg.learner.model_cfg.world_model.target
        == "dreamervla.models.world_model.dino_wm_chunk.ChunkAwareDinoWMWorldModel"
    )
    assert (
        cfg.learner.model_cfg.classifier.target
        == "dreamervla.models.reward.latent_success_classifier.LatentSuccessClassifier"
    )
    assert cfg.inference.cfg.encoder.target == "dreamervla.models.encoder.RynnVLAEncoder"
    assert cfg.inference.cfg.policy.kwargs.time_horizon == cfg.learner.model_cfg.policy.kwargs.time_horizon
    assert cfg.replay.cfg.sequence_length >= cfg.learner.model_cfg.classifier.kwargs.window

    task_spec = cfg.task.legacy_action_hidden
    assert task_spec.token_count == task_spec.chunk_size * cfg.task.action_dim
    assert task_spec.wm_obs_dim == task_spec.token_count * task_spec.token_dim
    assert cfg.ray_components.world_model.kwargs.obs_dim == task_spec.wm_obs_dim
    assert cfg.ray_components.world_model.kwargs.token_count == task_spec.token_count
    assert cfg.ray_components.world_model.kwargs.token_dim == task_spec.token_dim
    assert cfg.ray_components.world_model.kwargs.chunk_size == task_spec.chunk_size
    assert cfg.ray_components.policy.kwargs.action_hidden_dim == task_spec.token_dim
    assert cfg.ray_components.policy.kwargs.time_horizon == task_spec.chunk_size
    assert cfg.ray_components.classifier.kwargs.latent_dim == task_spec.wm_obs_dim
    assert cfg.learner.model_cfg.world_model.kwargs.obs_dim == task_spec.wm_obs_dim
    assert cfg.inference.cfg.world_model.kwargs.obs_dim == task_spec.wm_obs_dim

    unresolved_wm = OmegaConf.to_yaml(cfg.ray_components.world_model.kwargs, resolve=False)
    unresolved_policy = OmegaConf.to_yaml(cfg.ray_components.policy.kwargs, resolve=False)
    unresolved_classifier = OmegaConf.to_yaml(cfg.ray_components.classifier.kwargs, resolve=False)
    assert "${task.legacy_action_hidden.wm_obs_dim}" in unresolved_wm
    assert "${task.legacy_action_hidden.token_count}" in unresolved_wm
    assert "${task.legacy_action_hidden.token_dim}" in unresolved_wm
    assert "${task.legacy_action_hidden.token_dim}" in unresolved_policy
    assert "${task.legacy_action_hidden.wm_obs_dim}" in unresolved_classifier


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
