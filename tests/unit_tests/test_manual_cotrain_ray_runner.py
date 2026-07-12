from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import dreamervla.runners as runners
import dreamervla.runners.manual_cotrain_ray_runner as manual_runner
from dreamervla.runners.manual_cotrain_ray_runner import (
    _ManualCotrainEnvProgressMonitor,
    _ManualCotrainProgressSnapshot,
    _read_manual_cotrain_progress_snapshot,
    _split_actor_keyed_shard_counts,
    _split_actor_shard_counts,
    _sum_metric_lists,
    _wait_env_metrics_with_rollout_guard,
)
from dreamervla.workers.cotrain.messages import StopMsg


class _Ready:
    def __init__(self, value, events=None, wait_event: str | None = None):
        self.value = value
        self.events = events
        self.wait_event = wait_event

    def wait(self):
        if self.events is not None and self.wait_event is not None:
            self.events.append(self.wait_event)
        return self.value

    def done(self):
        return True


def _cfg(
    ngpu: int = 2,
    *,
    out_dir: str = "/tmp/dvla-manual-cotrain-test",
    checkpoint_every: int = 0,
    save_replay_state: bool = False,
    publish_learner_weights: bool = False,
    real_env_workers: int | None = None,
    real_rollout_target_trajectories: int | None = None,
    wm_envs_per_worker: int | None = None,
    wm_rollout_multiplier: int | None = None,
    wm_rollout_target_trajectories: int | None = None,
    real_env_enabled: bool = True,
    learner_updates_enabled: bool = True,
):
    manual_cotrain = {
        "ngpu": ngpu,
        "global_steps": 1,
        "learner_update_step": 1,
        "checkpoint_every": checkpoint_every,
        "save_replay_state": bool(save_replay_state),
        "rollout_epoch": 1,
        "max_steps_per_rollout_epoch": 2,
        "num_action_chunks": 2,
        "envs_per_worker": 1,
        "sync_every": 1,
        "publish_learner_weights": publish_learner_weights,
        "real_env_enabled": bool(real_env_enabled),
        "learner_updates_enabled": bool(learner_updates_enabled),
    }
    if real_env_workers is not None:
        manual_cotrain["real_env_workers"] = int(real_env_workers)
    if real_rollout_target_trajectories is not None:
        manual_cotrain["real_rollout_target_trajectories"] = int(
            real_rollout_target_trajectories
        )
    if wm_rollout_multiplier is not None:
        manual_cotrain["wm_rollout_multiplier"] = int(wm_rollout_multiplier)
    if wm_envs_per_worker is not None:
        manual_cotrain["wm_envs_per_worker"] = int(wm_envs_per_worker)
    if wm_rollout_target_trajectories is not None:
        manual_cotrain["wm_rollout_target_trajectories"] = int(
            wm_rollout_target_trajectories
        )
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.ManualCotrainRayRunner",
            "seed": 7,
            "training": {"out_dir": out_dir, "seed": 7},
            "logger": {"logger_backends": []},
            "cluster": {"num_nodes": 1, "num_gpus": ngpu},
            "manual_cotrain": manual_cotrain,
            "actor": {"train_cfg": {"algorithm_cfg": {"group_size": 2}}},
        }
    )


def test_runner_plans_manual_notes_groups() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=5))
    plan = runner._placement_plan()

    assert [spec.role for spec in plan.env_specs] == [
        "real_env",
        "wm_env",
        "wm_env",
        "wm_env",
        "wm_env",
    ]
    assert len(plan.actor_specs) == 5
    assert len(plan.rollout_specs) == 5
    assert plan.learner_spec.gpu_ids == [0]


def test_runner_can_assign_multiple_real_env_workers_without_changing_budget() -> None:
    cfg = _cfg(
        ngpu=7,
        real_env_workers=4,
        real_rollout_target_trajectories=32,
        wm_envs_per_worker=16,
        wm_rollout_target_trajectories=256,
    )
    cfg.manual_cotrain.envs_per_worker = 8
    runner = runners.ManualCotrainRayRunner(cfg)
    plan = runner._placement_plan()

    assert [spec.role for spec in plan.env_specs] == [
        "real_env",
        "real_env",
        "real_env",
        "real_env",
        "wm_env",
        "wm_env",
        "wm_env",
    ]
    assert runner._real_rollout_epochs_by_worker(4) == [1, 1, 1, 1]
    assert (
        sum(runner._real_rollout_epochs_by_worker(4)) * runner._envs_per_worker()
        == 32
    )
    assert runner._wm_rollout_epochs_by_worker(3) == [6, 5, 5]
    assert (
        sum(runner._wm_rollout_epochs_by_worker(3)) * runner._wm_envs_per_worker()
        == 256
    )


def test_runner_uses_cluster_component_placement_for_manual_topology() -> None:
    cfg = _cfg(
        ngpu=7,
        real_env_workers=4,
        real_rollout_target_trajectories=32,
        wm_envs_per_worker=16,
        wm_rollout_target_trajectories=256,
    )
    cfg.cluster.component_placement = {
        "env": "0-4",
        "rollout": "5:0-4",
        "actor": "6",
    }
    runner = runners.ManualCotrainRayRunner(cfg)
    plan = runner._placement_plan()

    assert [spec.role for spec in plan.env_specs] == [
        "real_env",
        "real_env",
        "real_env",
        "real_env",
        "wm_env",
    ]
    assert [spec.gpu_ids for spec in plan.env_specs] == [[0], [1], [2], [3], [4]]
    assert [spec.gpu_ids for spec in plan.rollout_specs] == [[5]] * 5
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[6]]
    assert plan.learner_spec.gpu_ids == [6]


def test_runner_resource_map_preserves_shared_component_workers() -> None:
    cfg = _cfg(ngpu=2, real_env_workers=4)
    cfg.cluster.component_placement = {
        "env": "0:0-3,1:4",
        "rollout": "1:0-4",
        "actor": "1",
    }
    runner = runners.ManualCotrainRayRunner(cfg)
    plan = runner._placement_plan()
    real_specs = [spec for spec in plan.env_specs if spec.role == "real_env"]

    placement = runner._resource_map_for_specs(real_specs)
    resolved = placement.get_placement(type("ClusterStub", (), {"num_gpus": 2})())

    assert [item.visible_accelerators for item in resolved] == [
        ["0"],
        ["0"],
        ["0"],
        ["0"],
    ]


def test_frozen_runner_plan_omits_real_env_and_learner_and_uses_all_actor_gpus() -> None:
    cfg = _cfg(
        ngpu=8,
        real_env_workers=0,
        real_env_enabled=False,
        learner_updates_enabled=False,
    )
    runner = runners.ManualCotrainRayRunner(cfg)

    plan = runner._placement_plan()

    assert plan.real_env_ranks == []
    assert plan.wm_env_ranks == list(range(8))
    assert [spec.gpu_ids for spec in plan.actor_specs] == [[gpu] for gpu in range(8)]
    assert plan.learner_spec is None
    assert runner._target_group_names() == ["ActorGroup", "RolloutGroup", "WMEnvGroup"]


def test_runner_loop_order_names_actor_before_learner_update() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=2))
    order = runner._global_step_operation_names()

    assert order[:4] == [
        "set_global_step",
        "actor_to_rollout_sync",
        "env_interact_and_rollout_generate",
        "actor_recv_trajectories",
    ]
    assert "actor_run_training" in order
    assert "learner_update_wm_classifier" in order


def test_runner_builds_group_names_from_target_topology() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=3))
    names = runner._target_group_names()

    assert names == ["LearnerGroup", "ActorGroup", "RolloutGroup", "EnvGroup"]


def test_runner_launches_ray_actors_with_formal_worker_names(monkeypatch) -> None:
    launched: list[tuple[str, str | None]] = []

    class _CaptureWorkerGroup:
        def __init__(self, worker_cls, *args, **kwargs):
            del args, kwargs
            self.worker_cls = worker_cls
            self.workers = [object()]

        def launch(self, cluster, placement, name=None, env_vars=None):
            del cluster, placement, env_vars
            launched.append((self.worker_cls.__name__, name))
            return self

        def execute_on(self, *ranks):
            del ranks
            return self

        def configure_rollout_epoch(self, rollout_epoch):
            return _Ready({"env/rollout_epoch": float(rollout_epoch)})

    class _FakeChannel:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(manual_runner, "WorkerGroup", _CaptureWorkerGroup)
    monkeypatch.setattr(
        manual_runner.Channel,
        "create",
        staticmethod(lambda name: _FakeChannel(str(name))),
    )
    cfg = _cfg(ngpu=2)
    cfg.replay = {"cfg": {"target": "test:Replay"}}
    cfg.env = {
        "real": {"cfg": {"target": "test:RealEnv"}},
        "wm": {"cfg": {"target": "test:WMEnv"}},
    }
    cfg.rollout = {
        "policy_cfg": {"target": "test:Policy"},
        "train_cfg": {"device": "cpu"},
    }
    cfg.actor.policy_cfg = {"target": "test:Policy"}
    cfg.actor.train_cfg.device = "cpu"
    cfg.learner = {
        "model_cfg": {
            "world_model": {"target": "test:WorldModel"},
            "classifier": {"target": "test:Classifier"},
        },
        "train_cfg": {"mode": "wm_classifier_only", "device": "cpu"},
    }
    runner = manual_runner.ManualCotrainRayRunner(cfg)

    runner._build_groups(cluster=object())

    assert launched == [
        ("ReplayWorker", "ReplayWorker"),
        ("RealEnvWorker", "RealEnvWorker"),
        ("WMEnvWorker", "WMEnvWorker"),
        ("MultiStepRolloutWorker", "MultiStepRolloutWorker"),
        ("EmbodiedFSDPActor", "EmbodiedFSDPActor"),
        ("LearnerWorker", "LearnerWorker"),
    ]
    assert all(name is None or not name.startswith("Manual") for _, name in launched)


def test_frozen_runner_launches_no_real_env_or_learner(monkeypatch) -> None:
    launched: list[tuple[str, str | None]] = []
    sequence: list[str] = []
    seed_calls: list[dict] = []
    component_loads: list[tuple[str, dict, int]] = []

    class _CaptureWorkerGroup:
        def __init__(self, worker_cls, *args, **kwargs):
            del args, kwargs
            self.worker_cls = worker_cls
            self.workers = [object()]

        def launch(self, cluster, placement, name=None, env_vars=None):
            del cluster, placement, env_vars
            launched.append((self.worker_cls.__name__, name))
            sequence.append(f"launch:{self.worker_cls.__name__}")
            return self

        def execute_on(self, *ranks):
            del ranks
            return self

        def configure_rollout_epoch(self, rollout_epoch):
            return _Ready({"env/rollout_epoch": float(rollout_epoch)})

        def seed_from_offline(self, seed_cfg):
            seed_calls.append(dict(seed_cfg))
            sequence.append("seed:ReplayWorker")
            return _Ready(
                [
                    {
                        "replay_buffer/seeded_episodes": 430.0,
                        "replay_buffer/seeded_task_count": 10.0,
                    }
                ]
            )

        def load_component_states(self, states, version):
            component_loads.append(
                (self.worker_cls.__name__, dict(states), int(version))
            )
            sequence.append(f"load:{self.worker_cls.__name__}")
            return _Ready([{"sync/load_component_states_s": 0.1}])

        def component_state_hashes(self):
            sequence.append(f"hash:{self.worker_cls.__name__}")
            return _Ready(
                [
                    {
                        "world_model": "wm-hash",
                        "classifier": "cls-hash",
                    }
                ]
            )

        def state_dict(self):
            sequence.append(f"state:{self.worker_cls.__name__}")
            return _Ready([{"policy.weight": torch.ones(1)}])

    class _FakeChannel:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(manual_runner, "WorkerGroup", _CaptureWorkerGroup)
    monkeypatch.setattr(
        manual_runner.Channel,
        "create",
        staticmethod(lambda name: _FakeChannel(str(name))),
    )
    cfg = _cfg(
        ngpu=8,
        real_env_workers=0,
        real_env_enabled=False,
        learner_updates_enabled=False,
    )
    cfg.replay = {
        "cfg": {"target": "test:Replay"},
        "seed": {
            "data_dir": "/official/reward",
            "hidden_dir": "/official/hidden",
            "task_ids": list(range(10)),
        },
    }
    cfg.env = {"wm": {"cfg": {"target": "test:WMEnv"}}}
    cfg.rollout = {
        "policy_cfg": {"target": "test:Policy"},
        "train_cfg": {"device": "cpu"},
    }
    cfg.actor.policy_cfg = {"target": "test:Policy"}
    cfg.actor.train_cfg.device = "cpu"
    runner = manual_runner.ManualCotrainRayRunner(cfg)
    monkeypatch.setattr(
        runner,
        "_load_frozen_component_states",
        lambda: {
            "component_states": {
                "world_model": {"weight": torch.tensor([1.0])},
                "classifier": {"weight": torch.tensor([2.0])},
                "classifier_threshold": 0.6,
            },
            "frozen_state_hashes": {"world_model": "wm-hash", "classifier": "cls-hash"},
            "source_checkpoints": {"world_model": "/wm.ckpt", "classifier": "/cls.ckpt"},
        },
        raising=False,
    )

    groups = runner._build_groups(cluster=object())

    assert launched == [
        ("ReplayWorker", "ReplayWorker"),
        ("WMEnvWorker", "WMEnvWorker"),
        ("MultiStepRolloutWorker", "MultiStepRolloutWorker"),
        ("EmbodiedFSDPActor", "EmbodiedFSDPActor"),
    ]
    assert groups["RealEnvGroup"] is None
    assert groups["LearnerGroup"] is None
    assert seed_calls == [
        {
            "data_dir": "/official/reward",
            "hidden_dir": "/official/hidden",
            "task_ids": list(range(10)),
        }
    ]
    assert sequence.index("seed:ReplayWorker") < sequence.index("launch:WMEnvWorker")
    assert sequence.index("launch:WMEnvWorker") < sequence.index("load:WMEnvWorker")
    assert sequence.index("load:WMEnvWorker") < sequence.index("hash:WMEnvWorker")
    assert sequence.index("load:WMEnvWorker") < sequence.index(
        "launch:MultiStepRolloutWorker"
    )
    assert len(component_loads) == 1
    assert component_loads[0][0] == "WMEnvWorker"
    assert component_loads[0][1]["classifier_threshold"] == 0.6
    assert component_loads[0][2] == 0
    assert groups["replay_seed_metrics"] == {
        "replay_buffer/seeded_episodes": 430.0,
        "replay_buffer/seeded_task_count": 10.0,
    }

    assert groups["frozen_state_hashes"] == {
        "world_model": "wm-hash",
        "classifier": "cls-hash",
    }
    assert groups["frozen_hashes_verified"] is True


def test_frozen_runner_loads_strict_component_checkpoints(tmp_path) -> None:
    world_model_cfg = {"_target_": "test.WorldModel", "hidden_dim": 4}
    classifier_cfg = {"_target_": "test.Classifier", "hidden_dim": 4}
    wm_path = tmp_path / "wm.ckpt"
    classifier_path = tmp_path / "classifier.ckpt"
    torch.save(
        {
            "world_model": {"weight": torch.tensor([1.0])},
            "config": {"world_model": world_model_cfg},
        },
        wm_path,
    )
    torch.save(
        {
            "model": {"weight": torch.tensor([2.0])},
            "config": {"classifier": classifier_cfg},
            "threshold": 0.65,
        },
        classifier_path,
    )
    cfg = _cfg(
        ngpu=8,
        real_env_workers=0,
        real_env_enabled=False,
        learner_updates_enabled=False,
    )
    cfg.init = {
        "world_model_state_ckpt": str(wm_path),
        "classifier_state_ckpt": str(classifier_path),
    }
    cfg.world_model = world_model_cfg
    cfg.classifier = classifier_cfg
    cfg.algorithm = {"lumos": {"classifier_threshold": None}}
    runner = manual_runner.ManualCotrainRayRunner(cfg)

    loaded = runner._load_frozen_component_states()

    assert loaded["component_states"]["classifier_threshold"] == 0.65
    assert loaded["component_states"]["world_model"]["weight"].item() == 1.0
    assert loaded["component_states"]["classifier"]["weight"].item() == 2.0
    assert set(loaded["frozen_state_hashes"]) == {"world_model", "classifier"}
    assert loaded["source_checkpoints"] == {
        "world_model": str(wm_path.resolve()),
        "classifier": str(classifier_path.resolve()),
    }


def test_frozen_runner_rejects_wm_env_component_hash_drift() -> None:
    class _HashGroup:
        @staticmethod
        def component_state_hashes():
            return _Ready(
                [
                    {"world_model": "wm-hash", "classifier": "changed"},
                    {"world_model": "wm-hash", "classifier": "cls-hash"},
                ]
            )

    runner = manual_runner.ManualCotrainRayRunner(
        _cfg(
            ngpu=2,
            real_env_workers=0,
            real_env_enabled=False,
            learner_updates_enabled=False,
        )
    )
    runner._frozen_state_hashes = {
        "world_model": "wm-hash",
        "classifier": "cls-hash",
    }

    with pytest.raises(RuntimeError, match="frozen WM/classifier state drift"):
        runner._assert_frozen_component_hashes({"WMEnvGroup": _HashGroup()})


def test_runner_launches_zero_gpu_tiny_groups_on_node_placement(monkeypatch) -> None:
    launched: list[tuple[str, str | None]] = []

    class _CaptureWorkerGroup:
        def __init__(self, worker_cls, *args, **kwargs):
            del args, kwargs
            self.worker_cls = worker_cls
            self.workers = [object()]

        def launch(self, cluster, placement, name=None, env_vars=None):
            del cluster, placement, env_vars
            launched.append((self.worker_cls.__name__, name))
            return self

        def execute_on(self, *ranks):
            del ranks
            return self

        def configure_rollout_epoch(self, rollout_epoch):
            return _Ready({"env/rollout_epoch": float(rollout_epoch)})

    class _FakeChannel:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(manual_runner, "WorkerGroup", _CaptureWorkerGroup)
    monkeypatch.setattr(
        manual_runner.Channel,
        "create",
        staticmethod(lambda name: _FakeChannel(str(name))),
    )
    cfg = _cfg(ngpu=0)
    cfg.replay = {"cfg": {"target": "test:Replay"}}
    cfg.env = {
        "real": {"cfg": {"target": "test:RealEnv"}},
    }
    cfg.rollout = {
        "policy_cfg": {"target": "test:Policy"},
        "train_cfg": {"device": "cpu"},
    }
    cfg.actor.policy_cfg = {"target": "test:Policy"}
    cfg.actor.train_cfg.device = "cpu"
    cfg.learner = {
        "model_cfg": {
            "world_model": {"target": "test:WorldModel"},
            "classifier": {"target": "test:Classifier"},
        },
        "train_cfg": {"mode": "wm_classifier_only", "device": "cpu"},
    }
    runner = manual_runner.ManualCotrainRayRunner(cfg)

    groups = runner._build_groups(cluster=object())

    assert launched == [
        ("ReplayWorker", "ReplayWorker"),
        ("RealEnvWorker", "RealEnvWorker"),
        ("MultiStepRolloutWorker", "MultiStepRolloutWorker"),
        ("EmbodiedFSDPActor", "EmbodiedFSDPActor"),
        ("LearnerWorker", "LearnerWorker"),
    ]
    assert groups["WMEnvGroup"] is None
    assert groups["placement"].ngpu == 0


def test_runner_scales_wm_rollout_budget_independently_from_real_env() -> None:
    runner = runners.ManualCotrainRayRunner(
        _cfg(ngpu=6, wm_rollout_multiplier=4)
    )

    assert runner._max_steps_per_rollout_epoch() == 2
    assert runner._wm_max_steps_per_rollout_epoch() == 8


def test_runner_uses_independent_wm_env_slots_when_configured() -> None:
    cfg = _cfg(
        ngpu=4,
        wm_envs_per_worker=8,
        wm_rollout_target_trajectories=1024,
    )
    cfg.manual_cotrain.envs_per_worker = 2
    cfg.manual_cotrain.real_rollout_epoch = 4
    runner = runners.ManualCotrainRayRunner(cfg)

    assert runner._envs_per_worker() == 2
    assert runner._wm_envs_per_worker() == 8
    assert runner._wm_rollout_epochs_by_worker(3) == [43, 43, 42]


def test_runner_rollout_slots_cover_largest_env_batch() -> None:
    cfg = _cfg(ngpu=4, wm_envs_per_worker=8)
    cfg.manual_cotrain.envs_per_worker = 2
    runner = runners.ManualCotrainRayRunner(cfg)

    assert runner._rollout_num_slots() == 8


def test_runner_distributes_wm_target_trajectories_across_wm_workers() -> None:
    cfg = _cfg(ngpu=4, wm_rollout_target_trajectories=1024)
    cfg.manual_cotrain.envs_per_worker = 2
    cfg.manual_cotrain.wm_rollout_epoch = 999
    runner = runners.ManualCotrainRayRunner(cfg)

    assert runner._wm_rollout_epochs_by_worker(3) == [171, 171, 170]

    class _Group:
        def __init__(self, count: int) -> None:
            self.workers = [object() for _ in range(count)]

    expected = runner._configured_expected_trajectory_shards(
        {
            "RealEnvGroup": _Group(1),
            "WMEnvGroup": _Group(3),
        }
    )

    assert expected == 512


def test_runner_uses_wm_rollout_epoch_fallback_without_target() -> None:
    cfg = _cfg(ngpu=4)
    cfg.manual_cotrain.envs_per_worker = 2
    cfg.manual_cotrain.wm_rollout_epoch = 7
    runner = runners.ManualCotrainRayRunner(cfg)

    assert runner._wm_rollout_epochs_by_worker(3) == [7, 7, 7]


@pytest.mark.parametrize(
    ("field", "accessor"),
    [
        ("global_steps", "_global_steps"),
        ("sync_every", "_sync_every"),
        ("learner_update_step", "_learner_update_step"),
        ("rollout_epoch", "_rollout_epoch"),
        ("real_env_workers", "_real_env_workers"),
        ("real_rollout_epoch", "_real_rollout_epoch"),
        ("real_rollout_target_trajectories", "_real_rollout_target_trajectories"),
        ("wm_rollout_epoch", "_wm_rollout_epoch"),
        ("wm_rollout_target_trajectories", "_wm_rollout_target_trajectories"),
        ("wm_rollout_lease_epochs", "_wm_rollout_lease_epochs"),
        ("max_steps_per_rollout_epoch", "_max_steps_per_rollout_epoch"),
        ("wm_rollout_multiplier", "_wm_rollout_multiplier"),
        ("num_action_chunks", "_num_action_chunks"),
        ("envs_per_worker", "_envs_per_worker"),
    ],
)
def test_manual_runner_rejects_non_positive_loop_controls_without_coercion(
    field: str,
    accessor: str,
) -> None:
    cfg = _cfg()
    cfg.manual_cotrain[field] = 0
    runner = runners.ManualCotrainRayRunner(cfg)

    with pytest.raises(ValueError, match=f"manual_cotrain.{field}"):
        getattr(runner, accessor)()


def test_split_actor_shard_counts_preserves_group_size_groups() -> None:
    assert _split_actor_shard_counts(1344, actor_ranks=5, group_size=8) == [
        272,
        272,
        272,
        264,
        264,
    ]


def test_split_actor_keyed_shard_counts_uses_only_wm_trajectories() -> None:
    counts = _split_actor_keyed_shard_counts(
        real_shards=0,
        wm_shards=64,
        wm_shard_batch_size=16,
        actor_ranks=3,
        group_size=8,
    )

    assert counts == [
        [("wm_env", 22)],
        [("wm_env", 21)],
        [("wm_env", 21)],
    ]
    trajectory_counts = [
        sum(
            count if key == "real_env" else count * 16
            for key, count in rank_counts
        )
        for rank_counts in counts
    ]
    assert trajectory_counts == [352, 336, 336]
    assert sum(trajectory_counts) == 1024
    assert all(count % 8 == 0 for count in trajectory_counts)


def test_split_actor_keyed_shard_counts_groups_small_wm_batches() -> None:
    counts = _split_actor_keyed_shard_counts(
        real_shards=0,
        wm_shards=48,
        wm_shard_batch_size=2,
        actor_ranks=3,
        group_size=8,
    )

    assert counts == [
        [("wm_env", 16)],
        [("wm_env", 16)],
        [("wm_env", 16)],
    ]
    trajectory_counts = [
        sum(count * 2 for key, count in rank_counts if key == "wm_env")
        for rank_counts in counts
    ]
    assert trajectory_counts == [32, 32, 32]
    assert all(count % 8 == 0 for count in trajectory_counts)


def test_split_actor_keyed_shard_counts_rejects_real_actor_shards() -> None:
    with pytest.raises(ValueError, match="real_env.*must not enter ActorGroup"):
        _split_actor_keyed_shard_counts(
            real_shards=8,
            wm_shards=64,
            wm_shard_batch_size=16,
            actor_ranks=3,
            group_size=8,
        )


def test_runner_starts_multigpu_actor_receivers_with_keyed_trajectory_counts() -> None:
    cfg = _cfg(ngpu=4, wm_envs_per_worker=16, wm_rollout_target_trajectories=1024)
    cfg.manual_cotrain.envs_per_worker = 2
    cfg.manual_cotrain.real_rollout_epoch = 1
    cfg.actor.train_cfg.algorithm_cfg.group_size = 8
    runner = runners.ManualCotrainRayRunner(cfg)
    actor = _FakeActorGroup()
    actor.workers = ["actor0", "actor1", "actor2"]

    class _Group:
        def __init__(self, count: int) -> None:
            self.workers = [object() for _ in range(count)]

    groups = {
        "ActorGroup": actor,
        "RealEnvGroup": _Group(1),
        "WMEnvGroup": _Group(3),
    }

    started = runner._start_actor_trajectory_receivers(
        groups,
        expected_shards=64,
        actor_channel_name="actor",
    )

    assert started is not None
    assert actor.recv_calls == [
        ((0,), "actor", 22, [("wm_env", 22)]),
        ((1,), "actor", 21, [("wm_env", 21)]),
        ((2,), "actor", 21, [("wm_env", 21)]),
    ]


def test_sum_metric_lists_derives_batch_size_distribution_metrics() -> None:
    metrics = _sum_metric_lists(
        [
            [
                {
                    "env/wm_env/model_forwards": 10.0,
                    "env/wm_env/batch_size_sum": 80.0,
                    "env/wm_env/batch_size_avg": 8.0,
                    "env/wm_env/batch_size_min": 8.0,
                    "env/wm_env/batch_size_max": 8.0,
                },
                {
                    "env/wm_env/model_forwards": 5.0,
                    "env/wm_env/batch_size_sum": 20.0,
                    "env/wm_env/batch_size_avg": 4.0,
                    "env/wm_env/batch_size_min": 2.0,
                    "env/wm_env/batch_size_max": 6.0,
                },
            ]
        ]
    )

    assert metrics["env/wm_env/model_forwards"] == 15.0
    assert metrics["env/wm_env/batch_size_sum"] == 100.0
    assert metrics["env/wm_env/batch_size_avg"] == 100.0 / 15.0
    assert metrics["env/wm_env/batch_size_min"] == 2.0
    assert metrics["env/wm_env/batch_size_max"] == 8.0


def test_sum_metric_lists_derives_wm_score_distribution_metrics() -> None:
    metrics = _sum_metric_lists(
        [
            {
                "env/wm_env/score_sum": 2.0,
                "env/wm_env/score_count": 4.0,
                "env/wm_env/score_mean": 0.5,
                "env/wm_env/score_p50": 0.4,
                "env/wm_env/score_p90": 0.8,
                "env/wm_env/score_max": 0.9,
            },
            {
                "env/wm_env/score_sum": 1.8,
                "env/wm_env/score_count": 2.0,
                "env/wm_env/score_mean": 0.9,
                "env/wm_env/score_p50": 0.9,
                "env/wm_env/score_p90": 0.95,
                "env/wm_env/score_max": 1.0,
            },
        ]
    )

    assert metrics["env/wm_env/score_sum"] == pytest.approx(3.8)
    assert metrics["env/wm_env/score_count"] == 6.0
    assert metrics["env/wm_env/score_mean"] == pytest.approx(3.8 / 6.0)
    assert metrics["env/wm_env/score_p50"] == pytest.approx(
        (0.4 * 4.0 + 0.9 * 2.0) / 6.0
    )
    assert metrics["env/wm_env/score_p90"] == pytest.approx(
        (0.8 * 4.0 + 0.95 * 2.0) / 6.0
    )
    assert metrics["env/wm_env/score_max"] == 1.0


def test_sum_metric_lists_derives_wm_classifier_success_rates() -> None:
    metrics = _sum_metric_lists(
        [
            {
                "env/wm_env/classifier_success_chunks": 2.0,
                "env/wm_env/classifier_total_chunks": 4.0,
                "env/wm_env/classifier_success_rate": 0.5,
                "env/wm_env/classifier_success_trajectories": 1.0,
                "env/wm_env/classifier_total_trajectories": 2.0,
                "env/wm_env/classifier_trajectory_success_rate": 0.5,
            },
            {
                "env/wm_env/classifier_success_chunks": 1.0,
                "env/wm_env/classifier_total_chunks": 2.0,
                "env/wm_env/classifier_success_rate": 0.5,
                "env/wm_env/classifier_success_trajectories": 2.0,
                "env/wm_env/classifier_total_trajectories": 2.0,
                "env/wm_env/classifier_trajectory_success_rate": 1.0,
            },
        ]
    )

    assert metrics["env/wm_env/classifier_success_chunks"] == 3.0
    assert metrics["env/wm_env/classifier_total_chunks"] == 6.0
    assert metrics["env/wm_env/classifier_success_rate"] == 0.5
    assert metrics["env/wm_env/classifier_success_trajectories"] == 3.0
    assert metrics["env/wm_env/classifier_total_trajectories"] == 4.0
    assert metrics["env/wm_env/classifier_trajectory_success_rate"] == 0.75


def test_train_metric_aliases_include_update_counts_and_classifier_success() -> None:
    metrics = manual_runner._with_train_learner_aliases(
        {
            "learner/updates": 3.0,
            "cls/updated": 1.0,
            "cls/updates": 7.0,
            "cls/acc": 0.75,
            "cls/f1": 0.5,
            "env/wm_env/classifier_success_rate": 0.25,
            "env/wm_env/classifier_trajectory_success_rate": 0.5,
        }
    )

    assert metrics["train/learner_updates"] == 3.0
    assert metrics["train/classifier_updated"] == 1.0
    assert metrics["train/classifier_updates"] == 7.0
    assert metrics["train/classifier_acc"] == 0.75
    assert metrics["train/classifier_f1"] == 0.5
    assert metrics["train/wm_env_classifier_success_rate"] == 0.25
    assert metrics["train/wm_env_classifier_trajectory_success_rate"] == 0.5


def test_manual_runner_reports_real_env_success_rate_metrics() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg())
    metrics = runner._real_env_success_rate_metrics(
        {
            "env/real_env/episodes_completed": 3.0,
            "env/real_env/episodes_successful": 2.0,
        },
        global_step=1,
    )

    assert metrics["rollout/episodes"] == 3.0
    assert metrics["rollout/successes"] == 2.0
    assert metrics["rollout/success_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["rollout/success_rate_valid"] == 1.0
    assert "eval/success_rate" not in metrics


def test_manual_runner_keeps_online_real_env_metrics_out_of_eval_namespace() -> None:
    cfg = _cfg()
    cfg.manual_cotrain.eval_interval_global_steps = 1
    runner = runners.ManualCotrainRayRunner(cfg)

    metrics = runner._real_env_success_rate_metrics(
        {
            "env/real_env/episodes_completed": 3.0,
            "env/real_env/episodes_successful": 2.0,
        },
        global_step=1,
    )

    assert metrics["rollout/success_rate"] == pytest.approx(2.0 / 3.0)
    assert "eval/success_rate" not in metrics
    assert "eval/episodes" not in metrics


def test_manual_runner_does_not_mirror_real_rollout_to_eval_on_interval() -> None:
    cfg = _cfg()
    cfg.manual_cotrain.eval_interval_global_steps = 10
    runner = runners.ManualCotrainRayRunner(cfg)

    metrics = runner._real_env_success_rate_metrics(
        {
            "env/real_env/episodes_completed": 1.0,
            "env/real_env/episodes_successful": 1.0,
        },
        global_step=10,
    )

    assert metrics["rollout/success_rate"] == 1.0
    assert not any(key.startswith("eval/") for key in metrics)


def test_manual_runner_does_not_mirror_real_rollout_to_eval_in_debug() -> None:
    cfg = _cfg()
    cfg.training.debug = True
    cfg.manual_cotrain.eval_interval_global_steps = 10
    cfg.manual_cotrain.debug_eval_interval_global_steps = 1
    runner = runners.ManualCotrainRayRunner(cfg)

    metrics = runner._real_env_success_rate_metrics(
        {
            "env/real_env/episodes_completed": 1.0,
            "env/real_env/episodes_successful": 0.0,
        },
        global_step=1,
    )

    assert metrics["rollout/success_rate"] == 0.0
    assert not any(key.startswith("eval/") for key in metrics)


def _compose_train_config(*overrides: str):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamervla.config_resolvers import register_dreamervla_resolvers

    register_dreamervla_resolvers()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="train",
            overrides=list(overrides),
        )


def _load_tiny_manual_config():
    from pathlib import Path

    from dreamervla.config_resolvers import register_dreamervla_resolvers

    register_dreamervla_resolvers()
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "manual_cotrain_ray_tiny.yaml"
    )
    cfg = OmegaConf.load(fixture)
    OmegaConf.resolve(cfg)
    return cfg


def test_manual_cotrain_oft_backbone_experiment_composes() -> None:
    cfg = _compose_train_config(
        "experiment=openvla_onetraj_libero_cotrain_ray",
        "task=openvla_onetraj_coldstart_libero",
    )

    assert cfg._target_ == "dreamervla.runners.ManualCotrainRayRunner"
    assert cfg.learner.train_cfg.mode == "wm_classifier_only"
    assert cfg.algorithm.group_size == 8
    assert cfg.algorithm.rollout_epoch == 16
    assert cfg.manual_cotrain.real_rollout_epoch == 1
    assert cfg.manual_cotrain.wm_rollout_epoch == 16
    assert cfg.manual_cotrain.real_rollout_target_trajectories == 32
    assert cfg.manual_cotrain.wm_rollout_target_trajectories == 1024
    assert cfg.manual_cotrain.wm_rollout_lease_epochs == 1
    assert cfg.manual_cotrain.max_steps_per_rollout_epoch == 512
    assert cfg.manual_cotrain.wm_rollout_multiplier == 1
    assert cfg.manual_cotrain.eval_interval_global_steps == 0
    assert cfg.manual_cotrain.debug_eval_interval_global_steps == 0
    assert list(cfg.runner.logger.logger_backends) == ["tensorboard", "wandb"]
    assert cfg.runner.logger.wandb_mode == "offline"
    assert cfg.runner.logger.wandb_proxy is None
    assert cfg.ray_components.world_model.kwargs.attn_impl == "sdpa"
    assert cfg.env.wm.cfg.kwargs.inference_dtype == "bf16"
    assert cfg.env.wm.cfg.kwargs.observation_format == "tensor"
    assert cfg.actor.train_cfg.algorithm_cfg.clip_ratio_low == 0.2
    assert cfg.actor.train_cfg.algorithm_cfg.clip_ratio_high == 0.28
    assert cfg.actor.train_cfg.algorithm_cfg.filter_rewards is False
    assert cfg.actor.train_cfg.algorithm_cfg.rewards_lower_bound == 0.5
    assert cfg.actor.train_cfg.algorithm_cfg.rewards_upper_bound == 4.5
    assert cfg.actor.train_cfg.algorithm_cfg.reward_type == "action_level"
    assert cfg.actor.train_cfg.algorithm_cfg.logprob_type == "token_level"
    assert cfg.actor.train_cfg.algorithm_cfg.entropy_type == "token_level"
    assert cfg.actor.train_cfg.algorithm_cfg.loss_type == "actor"
    assert cfg.actor.train_cfg.algorithm_cfg.loss_agg_func == "token-mean"
    assert cfg.algorithm.kl_beta == 0.0
    assert cfg.actor.train_cfg.lr == 5.0e-7
    assert cfg.actor.train_cfg.optimizers.policy.lr == 5.0e-7
    assert cfg.actor.train_cfg.algorithm_cfg.kl_beta == 0.0
    assert "kl_coef" not in cfg.actor.train_cfg.algorithm_cfg
    assert cfg.actor.train_cfg.algorithm_cfg.clip_log_ratio is None
    assert cfg.actor.train_cfg.algorithm_cfg.loss_normalization == "token_mean"
    assert cfg.actor.train_cfg.micro_batch_size == 32
    assert cfg.actor.train_cfg.global_batch_size == 16384
    assert cfg.rollout.train_cfg.logprob_type == "token_level"


def test_manual_cotrain_oft_rollout_carries_checkpoint_num_images() -> None:
    cfg = _compose_train_config(
        "experiment=openvla_onetraj_libero_cotrain_ray",
        "task=openvla_onetraj_coldstart_libero",
    )
    OmegaConf.resolve(cfg)

    assert cfg.task.openvla_oft.num_images_in_input == 1
    assert cfg.rollout.encoder_cfg.kwargs.policy_cfg.num_images_in_input == 1


def test_manual_cotrain_oft_wm_env_num_envs_tracks_wm_envs_per_worker() -> None:
    cfg = _compose_train_config(
        "experiment=openvla_onetraj_libero_cotrain_ray",
        "task=openvla_onetraj_coldstart_libero",
    )

    assert cfg.manual_cotrain.wm_envs_per_worker == 16
    assert cfg.env.wm.cfg.kwargs.num_envs == cfg.manual_cotrain.wm_envs_per_worker
    assert cfg.env.wm.cfg.kwargs.device == "cuda"


def test_manual_cotrain_actor_rollout_budget_uses_wm_only() -> None:
    cfg = _compose_train_config(
        "experiment=openvla_onetraj_libero_cotrain_ray",
        "task=openvla_onetraj_coldstart_libero",
    )

    assert cfg.manual_cotrain.real_rollout_epoch == 1
    assert cfg.manual_cotrain.real_rollout_target_trajectories == 32
    assert cfg.manual_cotrain.wm_rollout_target_trajectories == 1024
    assert cfg.manual_cotrain.wm_rollout_target_trajectories % cfg.algorithm.group_size == 0
    real_trajectories = (
        cfg.manual_cotrain.envs_per_worker * cfg.manual_cotrain.real_rollout_epoch
    )
    assert real_trajectories == 8


def test_manual_cotrain_oft_real_rollout_uses_oft_encoder_and_action_postprocess() -> None:
    cfg = _compose_train_config(
        "experiment=openvla_onetraj_libero_cotrain_ray",
        "task=openvla_onetraj_coldstart_libero",
    )

    assert cfg.rollout.encoder_cfg.target.endswith("oft_rollout:OFTRolloutBundle")
    assert cfg.rollout.encoder_cfg.kwargs.unnorm_key == cfg.task.openvla_oft.dataset_statistics_key
    assert cfg.rollout.encoder_cfg.kwargs.image_keys == cfg.task.image_keys
    assert cfg.rollout.encoder_cfg.kwargs.history == cfg.task.openvla_oft.hidden_token.expected_history
    assert (
        cfg.rollout.encoder_cfg.kwargs.obs_hidden_source
        == cfg.task.openvla_oft.hidden_token.expected_obs_hidden_source
    )
    assert cfg.env.real.cfg.action_postprocess == "openvla_oft"
    assert cfg.env.real.cfg.render_backend == cfg.render_backend
    assert cfg.env.wm.cfg.action_postprocess == "openvla_oft"
    assert "egl_step_timeout_s" not in cfg.env.real.cfg


def test_manual_runner_injects_top_level_render_backend_into_real_env_cfg() -> None:
    cfg = _cfg(ngpu=2)
    cfg.render_backend = "egl"
    cfg.env = {
        "real": {
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:CounterEnv",
                "kwargs": {"horizon": 1},
                "action_postprocess": "openvla_oft",
            }
        }
    }
    runner = manual_runner.ManualCotrainRayRunner(cfg)

    real_env_cfg = runner._real_env_cfg()

    assert real_env_cfg["render_backend"] == "egl"
    assert real_env_cfg["num_envs_per_worker"] == 1
    assert real_env_cfg["action_postprocess"] == "openvla_oft"
    assert "spawn_env_slots" not in real_env_cfg


def test_manual_runner_injects_real_env_egl_gpu_pool_from_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3,4")
    cfg = _cfg(ngpu=4, real_env_workers=3)
    cfg.render_backend = "egl"
    cfg.env = {
        "real": {
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:CounterEnv",
                "kwargs": {"horizon": 1},
            }
        }
    }
    runner = manual_runner.ManualCotrainRayRunner(cfg)

    real_env_cfg = runner._real_env_cfg()

    assert real_env_cfg["gpu_pool"] == [0, 1, 3]
    assert "spawn_env_slots" not in real_env_cfg


def test_manual_runner_real_render_backend_overrides_real_env_cfg_backend() -> None:
    cfg = _cfg(ngpu=2)
    cfg.render_backend = "egl"
    cfg.manual_cotrain.real_render_backend = "osmesa"
    cfg.env = {
        "real": {
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:CounterEnv",
                "kwargs": {"horizon": 1},
                "render_backend": "egl",
            }
        }
    }
    runner = manual_runner.ManualCotrainRayRunner(cfg)

    real_env_cfg = runner._real_env_cfg()

    assert real_env_cfg["render_backend"] == "osmesa"


def test_manual_runner_preserves_explicit_real_env_spawn_slots_config() -> None:
    cfg = _cfg(ngpu=2)
    cfg.env = {
        "real": {
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:CounterEnv",
                "kwargs": {"horizon": 1},
                "spawn_env_slots": True,
            }
        }
    }
    runner = manual_runner.ManualCotrainRayRunner(cfg)

    real_env_cfg = runner._real_env_cfg()

    assert real_env_cfg["spawn_env_slots"] is True


def test_manual_cotrain_tiny_wm_env_num_envs_tracks_envs_per_worker_and_disables_loggers() -> None:
    cfg = _load_tiny_manual_config()

    assert cfg.manual_cotrain.real_rollout_epoch == cfg.manual_cotrain.rollout_epoch
    assert cfg.manual_cotrain.wm_rollout_epoch == cfg.manual_cotrain.rollout_epoch
    assert cfg.manual_cotrain.wm_rollout_target_trajectories is None
    assert cfg.manual_cotrain.wm_rollout_lease_epochs == 1
    assert cfg.manual_cotrain.wm_envs_per_worker == cfg.manual_cotrain.envs_per_worker
    assert cfg.env.real.cfg.emit_actor_trajectories is True
    assert cfg.env.wm.cfg.kwargs.num_envs == cfg.manual_cotrain.wm_envs_per_worker
    assert list(cfg.runner.logger.logger_backends) == []


def test_manual_runner_loads_component_init_ckpt(tmp_path) -> None:
    ckpt = tmp_path / "warmup.ckpt"
    torch.save(
        {
            "state_dicts": {
                "world_model": {"wm.weight": torch.ones(1)},
                "classifier": {"cls.weight": torch.full((1,), 2.0)},
                "policy": {"unused": torch.full((1,), 3.0)},
            },
            "classifier_threshold": 0.125,
        },
        ckpt,
    )
    cfg = _cfg(ngpu=0)
    cfg.learner = {
        "init_ckpt": {
            "path": str(ckpt),
            "components": ["world_model", "classifier"],
        }
    }

    runner = manual_runner.ManualCotrainRayRunner(cfg)
    loaded = runner._load_init_ckpt("learner.init_ckpt")

    assert sorted(loaded) == ["classifier", "classifier_threshold", "world_model"]
    assert torch.equal(loaded["world_model"]["wm.weight"], torch.ones(1))
    assert torch.equal(
        loaded["classifier"]["cls.weight"],
        torch.full((1,), 2.0),
    )
    assert loaded["classifier_threshold"] == 0.125


class _FakeActorGroup:
    def __init__(self, events: list[str] | None = None) -> None:
        self.workers = ["actor0", "actor1"]
        self.events = events
        self.selected: tuple[int, ...] | None = None
        self.sync_calls: list[tuple[tuple[int, ...] | None, str, int]] = []
        self.state_dict_calls: list[tuple[int, ...] | None] = []
        self.recv_calls: list[
            tuple[tuple[int, ...] | None, str, int, list[tuple[str, int]] | None]
        ] = []
        self.loaded_shards: list[object] = []
        self.loaded_by_rank: list[tuple[tuple[int, ...] | None, list[object]]] = []
        self.load_calls = 0

    def execute_on(self, *ranks: int):
        self.selected = tuple(int(rank) for rank in ranks)
        return self

    def set_global_step(self, global_step: int):
        self.global_step = int(global_step)
        return _Ready([None for _ in self.workers])

    def sync_model_to_rollout(self, key: str, version: int):
        self.sync_calls.append((self.selected, str(key), int(version)))
        self.selected = None
        return _Ready(
            [
                {
                    "sync/policy_version": float(version),
                    "sync/policy_export_s": 0.1,
                    "sync/policy_push_s": 0.2,
                }
            ]
        )

    def load_trajectory_shards(self, shards: list[object]):
        if self.events is not None:
            self.events.append("actor_load")
        self.load_calls += 1
        loaded = list(shards)
        self.loaded_by_rank.append((self.selected, loaded))
        self.loaded_shards.extend(loaded)
        self.selected = None
        return _Ready([None for _ in self.workers])

    def recv_rollout_trajectories(
        self,
        actor_channel_name: str,
        expected_shards: int | None = None,
        keyed_counts: list[tuple[str, int]] | None = None,
    ):
        if self.events is not None:
            self.events.append("actor_recv_start")
        count = (
            sum(int(value) for _key, value in keyed_counts)
            if keyed_counts is not None
            else 1 if expected_shards is None else int(expected_shards)
        )
        self.recv_calls.append(
            (
                self.selected,
                str(actor_channel_name),
                count,
                list(keyed_counts) if keyed_counts is not None else None,
            )
        )
        rank = -1 if not self.selected else int(self.selected[0])
        self.loaded_shards.extend([f"rank{rank}-shard{i}" for i in range(count)])
        self.selected = None
        return _Ready(
            [
                {
                    "actor/received_shards": float(count),
                    "actor/channel_get_batch_s": 0.01,
                    "actor/load_trajectory_shards_s": 0.02,
                }
            ],
            self.events,
            "actor_recv_wait",
        )

    def compute_advantages_and_returns(self):
        return _Ready([{"actor/trajectory_count": float(len(self.loaded_shards))}])

    def run_training(self):
        if self.events is not None:
            self.events.append("actor_run_training")
        return _Ready([{"actor/ppo_updates": 1.0}])

    def state_dict(self):
        self.state_dict_calls.append(self.selected)
        self.selected = None
        return _Ready([{"policy.weight": torch.ones(1)}])

    def optimizer_state_dict(self):
        return _Ready(
            [
                {
                    "state": {0: {"step": torch.tensor(1.0)}},
                    "param_groups": [{"params": [0]}],
                }
            ]
        )


class _FakeRolloutGroup:
    def __init__(self, events: list[str] | None = None) -> None:
        self.workers = ["rollout0", "rollout1"]
        self.events = events
        self.pulled: list[tuple[str, int | None]] = []
        self.generate_call: tuple[str, str, int] | None = None

    def set_global_step(self, global_step: int):
        self.global_step = int(global_step)
        return _Ready([None for _ in self.workers])

    def sync_model_from_actor(self, key: str, local_version: int | None = None):
        self.pulled.append((str(key), None if local_version is None else int(local_version)))
        return _Ready(
            [
                {"sync/rollout_policy_pull_s": 0.3},
                {"sync/rollout_policy_pull_s": 0.4},
            ]
        )

    def generate(
        self,
        env_channel_name: str,
        rollout_channel_name: str,
        num_slots: int = 1,
    ):
        self.generate_call = (
            str(env_channel_name),
            str(rollout_channel_name),
            int(num_slots),
        )
        return _Ready(
            [
                {"rollout/generated": 2.0},
                {"rollout/generated": 3.0},
            ],
            self.events,
            "rollout_wait",
        )


class _FakeEnvGroup:
    def __init__(self, metrics, events: list[str] | None = None):
        self.metrics = metrics
        self.events = events
        self.prefetch_calls = 0
        self.global_steps: list[int] = []
        self.wm_versions: list[int] = []
        self.classifier_versions: list[int] = []
        self.world_model_states: list[dict] = []
        self.classifier_states: list[dict] = []
        self.classifier_thresholds: list[float] = []
        self.component_state_versions: list[int] = []
        self.component_state_keys: list[list[str]] = []
        self.progress_configs: list[tuple[str | None, float]] = []

    def set_global_step(self, global_step: int):
        self.global_steps.append(int(global_step))
        return _Ready([None])

    def configure_progress(self, progress_dir: str | None, min_interval_s: float = 5.0):
        self.progress_configs.append(
            (None if progress_dir is None else str(progress_dir), float(min_interval_s))
        )
        return _Ready([{"env/progress_configured": 1.0}])

    def interact(
        self,
        env_channel_name: str,
        rollout_channel_name: str,
        actor_channel_name: str,
    ):
        del env_channel_name, rollout_channel_name, actor_channel_name
        if self.events is not None:
            self.events.append("env_interact_start")
        return _Ready(self.metrics, self.events, "env_wait")

    def prefetch_bootstrap(self):
        self.prefetch_calls += 1
        if self.events is not None:
            self.events.append("env_prefetch_bootstrap")
        return _Ready([{"env/real_env/prefetched_bootstrap_slots": 1.0}])

    def load_world_model_state(self, state_dict, version: int):
        self.world_model_states.append(dict(state_dict))
        self.wm_versions.append(int(version))
        return _Ready([None])

    def load_classifier_state(self, state_dict, version: int):
        self.classifier_states.append(dict(state_dict))
        self.classifier_versions.append(int(version))
        return _Ready([None])

    def load_component_states(self, state_dicts, version: int):
        states = {}
        for key, value in dict(state_dicts).items():
            if str(key) == "classifier_threshold":
                states[str(key)] = float(value)
            else:
                states[str(key)] = dict(value)
        self.component_state_versions.append(int(version))
        self.component_state_keys.append(sorted(states))
        if "world_model" in states:
            self.world_model_states.append(states["world_model"])
            self.wm_versions.append(int(version))
        if "classifier" in states:
            self.classifier_states.append(states["classifier"])
            self.classifier_versions.append(int(version))
        if "classifier_threshold" in states:
            self.classifier_thresholds.append(float(states["classifier_threshold"]))
        return _Ready([{"sync/load_component_states_s": 0.5}])


class _FakeLearnerGroup:
    def __init__(self, update_metrics: dict[str, float] | None = None) -> None:
        self.synced: list[tuple[str, int]] = []
        self.update_metrics = dict(update_metrics or {"learner/updates": 1.0})

    def update(self, phase: str, num_steps: int):
        self.update_call = (str(phase), int(num_steps))
        return _Ready([dict(self.update_metrics)])

    def sync_weights(self, what: str, version: int):
        self.synced.append((str(what), int(version)))
        return _Ready([None])

    def state_dicts(self):
        return _Ready(
            [
                {
                    "world_model": {"wm": 1},
                    "classifier": {"cls": 2},
                    "classifier_threshold": 0.95,
                    "policy": {"unused": 3},
                    "sync/state_dicts_s": 0.6,
                }
            ]
        )


class _FakeChannel:
    def __init__(
        self,
        items: list[object] | None = None,
        *,
        qsize_sequence: list[int] | None = None,
    ) -> None:
        self.items = list(items or [])
        self.qsize_sequence = list(qsize_sequence or [])
        self.puts: list[tuple[str, object]] = []
        self.gets: list[str] = []
        self.get_batch_calls: list[tuple[int, str]] = []
        self.qsize_calls: list[str | None] = []

    def get(self, *, key: str = "default"):
        self.gets.append(str(key))
        return self.items.pop(0)

    def get_batch(self, n: int, *, key: str = "default"):
        self.get_batch_calls.append((int(n), str(key)))
        out = self.items[: int(n)]
        del self.items[: int(n)]
        return out

    def put(self, value, *, key: str = "default"):
        self.puts.append((str(key), value))

    def qsize(self, *, key: str | None = None):
        self.qsize_calls.append(None if key is None else str(key))
        if self.qsize_sequence:
            return int(self.qsize_sequence.pop(0))
        return len(self.items)


def test_actor_shard_role_counts_use_real_env_key_without_wm_group() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=0))

    role_counts = runner._actor_shard_role_counts_from_metrics(
        {"WMEnvGroup": None},
        {
            "env/trajectory_shards": 2.0,
            "env/real_env/trajectory_shards": 2.0,
        },
        expected_shards=2,
    )

    assert role_counts == [("real_env", 2)]


class _FakeReplayGroup:
    def __init__(self) -> None:
        self.policy_versions: list[int] = []
        self.loaded_state: dict | None = None
        self.loaded_sampling_state: dict | None = None
        self.state_dict_calls = 0
        self.sampling_state_dict_calls = 0

    def set_policy_version(self, version: int):
        self.policy_versions.append(int(version))
        return _Ready([None])

    def size(self):
        return _Ready([3])

    def num_transitions(self):
        return _Ready([7])

    def state_dict(self):
        self.state_dict_calls += 1
        return _Ready([{"episodes": ["current"], "num_transitions": 7}])

    def sampling_state_dict(self):
        self.sampling_state_dict_calls += 1
        return _Ready(
            [{"task_sample_cursor": 11, "initial_condition_cursor": 13}]
        )

    def load_state_dict(self, state):
        self.loaded_state = dict(state)
        return _Ready([None])

    def load_sampling_state_dict(self, state):
        self.loaded_sampling_state = dict(state)
        return _Ready([None])


class _NeverReady:
    def done(self):
        return False

    def wait(self):
        raise AssertionError("env wait should not be reached")


class _EarlyFailedRollout:
    def ready(self):
        return ["rank0-ref"]

    def wait_refs(self, refs):
        assert refs == ["rank0-ref"]
        raise RuntimeError("rank0 failed key=0:0")


class _RunningRollout:
    def ready(self):
        return []


class _DelayedReady:
    def __init__(self, value, *, done_after_polls: int = 0):
        self.value = value
        self.done_after_polls = int(done_after_polls)
        self.polls = 0
        self.waited = False

    def done(self):
        self.polls += 1
        return self.polls > self.done_after_polls

    def wait(self):
        self.waited = True
        return self.value


class _RecordingCentralProgress:
    def __init__(self) -> None:
        self.snapshots: list[_ManualCotrainProgressSnapshot] = []

    def records(self):
        return []

    def report_snapshot(self, snapshot, *, force: bool = False):
        del force
        self.snapshots.append(snapshot)
        return snapshot

    def report(self, *, force: bool = False):
        del force
        snapshot = _ManualCotrainProgressSnapshot(
            done=0,
            total=0,
            status=None,
            worker_count=0,
            finished_count=0,
        )
        self.snapshots.append(snapshot)
        return snapshot


class _DynamicFakeWMEnvGroup:
    def __init__(self, *, worker_count: int, slow_rank: int = 1) -> None:
        self.workers = [object() for _ in range(worker_count)]
        self.selected: tuple[int, ...] | None = None
        self.configure_calls: list[tuple[int, int]] = []
        self.refresh_calls: list[int] = []
        self.interact_calls: list[int] = []
        self.slow_rank = int(slow_rank)

    def execute_on(self, *ranks: int):
        self.selected = tuple(int(rank) for rank in ranks)
        return self

    def configure_rollout_epoch(self, rollout_epoch: int):
        if self.selected is None or len(self.selected) != 1:
            raise AssertionError("configure_rollout_epoch must target one rank")
        self.configure_calls.append((int(self.selected[0]), int(rollout_epoch)))
        return _Ready([{"env/rollout_epoch": float(rollout_epoch)}])

    def refresh_wm_initial_conditions(self):
        if self.selected is None or len(self.selected) != 1:
            raise AssertionError("refresh must target one rank")
        self.refresh_calls.append(int(self.selected[0]))
        return _Ready([{"env/wm_env/initial_conditions_refreshed": 1.0}])

    def interact(
        self,
        env_channel_name: str,
        rollout_channel_name: str,
        actor_channel_name: str,
    ):
        del env_channel_name, rollout_channel_name, actor_channel_name
        if self.selected is None or len(self.selected) != 1:
            raise AssertionError("interact must target one rank")
        rank = int(self.selected[0])
        self.interact_calls.append(rank)
        self.selected = None
        return _DelayedReady(
            {
                "env/trajectory_shards": 2.0,
                "env/steps": 4.0,
                "env/wm_env/classifier_success_chunks": 2.0,
                "env/wm_env/classifier_total_chunks": 4.0,
                "env/wm_env/classifier_success_trajectories": 1.0,
                "env/wm_env/classifier_total_trajectories": 2.0,
            },
            done_after_polls=6 if rank == self.slow_rank else 0,
        )


def test_dynamic_wm_leases_let_fast_worker_consume_more_imagine_budget() -> None:
    cfg = _cfg(ngpu=3, wm_envs_per_worker=2, wm_rollout_target_trajectories=12)
    cfg.manual_cotrain.wm_rollout_lease_epochs = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    wm_env = _DynamicFakeWMEnvGroup(worker_count=2, slow_rank=1)

    metrics = runner._wait_env_metrics_with_dynamic_wm_leases(
        real_env_results=[_Ready({"env/trajectory_shards": 1.0, "env/steps": 2.0})],
        wm_env=wm_env,
        rollout_result=_RunningRollout(),
        env_channel_name="env",
        rollout_channel_name="rollout",
        actor_channel_name="actor",
        timeout_s=10.0,
        poll_s=0.0,
        progress=None,
    )

    assert len(wm_env.interact_calls) == 6
    assert wm_env.interact_calls.count(0) > wm_env.interact_calls.count(1)
    assert all(epoch == 1 for _rank, epoch in wm_env.configure_calls)
    assert metrics["env/trajectory_shards"] == 13.0
    assert metrics["env/steps"] == 26.0


def test_dynamic_wm_leases_refresh_grouped_initial_condition_before_every_lease() -> None:
    cfg = _cfg(ngpu=3, wm_envs_per_worker=2, wm_rollout_target_trajectories=12)
    cfg.manual_cotrain.wm_rollout_lease_epochs = 1
    cfg.manual_cotrain.refresh_wm_initial_conditions_per_lease = True
    runner = runners.ManualCotrainRayRunner(cfg)
    wm_env = _DynamicFakeWMEnvGroup(worker_count=2, slow_rank=1)

    runner._wait_env_metrics_with_dynamic_wm_leases(
        real_env_results=[],
        wm_env=wm_env,
        rollout_result=_RunningRollout(),
        env_channel_name="env",
        rollout_channel_name="rollout",
        actor_channel_name="actor",
        timeout_s=10.0,
        poll_s=0.0,
        progress=None,
    )

    assert wm_env.refresh_calls == wm_env.interact_calls


def test_dynamic_wm_progress_is_reported_from_central_pool() -> None:
    cfg = _cfg(ngpu=3, wm_envs_per_worker=2, wm_rollout_target_trajectories=12)
    cfg.manual_cotrain.wm_rollout_lease_epochs = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    wm_env = _DynamicFakeWMEnvGroup(worker_count=2, slow_rank=1)
    progress = _RecordingCentralProgress()

    runner._wait_env_metrics_with_dynamic_wm_leases(
        real_env_results=[_Ready({"env/trajectory_shards": 1.0, "env/steps": 2.0})],
        wm_env=wm_env,
        rollout_result=_RunningRollout(),
        env_channel_name="env",
        rollout_channel_name="rollout",
        actor_channel_name="actor",
        timeout_s=10.0,
        poll_s=0.0,
        progress=progress,  # type: ignore[arg-type]
    )

    reported = [snapshot for snapshot in progress.snapshots if snapshot.total > 0]
    assert reported
    dones = [snapshot.done for snapshot in reported]
    assert dones == sorted(dones)
    assert reported[-1].done == reported[-1].total
    assert any(snapshot.status and "wm=" in snapshot.status for snapshot in reported)
    assert all(
        snapshot.status is None
        or "leases(d/r/q/t)=" in snapshot.status
        for snapshot in reported
    )
    assert all(
        snapshot.status is None
        or "remaining_wm_epochs=" not in snapshot.status
        for snapshot in reported
    )
    assert all(
        snapshot.status is None or "wm_env#1=" not in snapshot.status
        for snapshot in reported
    )


class _MultiRealProgress:
    def __init__(self) -> None:
        self.snapshots: list[_ManualCotrainProgressSnapshot] = []

    def records(self):
        return [
            {
                "role": "real_env",
                "rank": 0,
                "env_rank": 0,
                "global_step": 1,
                "done": 2,
                "total": 4,
            },
            {
                "role": "real_env",
                "rank": 1,
                "env_rank": 1,
                "global_step": 1,
                "done": 3,
                "total": 4,
            },
        ]

    def report_snapshot(self, snapshot, *, force: bool = False):
        del force
        self.snapshots.append(snapshot)
        return snapshot


def test_dynamic_wm_progress_sums_multiple_real_workers() -> None:
    cfg = _cfg(
        ngpu=4,
        real_env_workers=2,
        real_rollout_target_trajectories=8,
        wm_envs_per_worker=2,
        wm_rollout_target_trajectories=4,
    )
    cfg.manual_cotrain.envs_per_worker = 2
    cfg.manual_cotrain.wm_rollout_lease_epochs = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    wm_env = _DynamicFakeWMEnvGroup(worker_count=2, slow_rank=1)
    progress = _MultiRealProgress()

    with pytest.raises(TimeoutError):
        runner._wait_env_metrics_with_dynamic_wm_leases(
            real_env_results=[_NeverReady()],
            wm_env=wm_env,
            rollout_result=_RunningRollout(),
            env_channel_name="env",
            rollout_channel_name="rollout",
            actor_channel_name="actor",
            timeout_s=0.05,
            poll_s=0.0,
            progress=progress,  # type: ignore[arg-type]
        )

    reported = [snapshot for snapshot in progress.snapshots if snapshot.status]
    assert reported
    assert "real_env=5/8" in str(reported[-1].status)
    assert reported[-1].done >= 5


class _ResetClassifierProgress:
    def __init__(self) -> None:
        self.snapshots: list[_ManualCotrainProgressSnapshot] = []

    def records(self):
        return [
            {
                "role": "real_env",
                "rank": 0,
                "env_rank": 0,
                "global_step": 1,
                "done": 0,
                "total": 1,
            },
            {
                "role": "wm_env",
                "rank": 0,
                "env_rank": 1,
                "global_step": 1,
                "done": 0,
                "total": 2,
                "classifier_success_chunks": 0,
                "classifier_total_chunks": 0,
                "classifier_success_trajectories": 0,
                "classifier_total_trajectories": 0,
            },
        ]

    def report_snapshot(self, snapshot, *, force: bool = False):
        del force
        self.snapshots.append(snapshot)
        return snapshot


def test_dynamic_wm_progress_keeps_classifier_rate_after_lease_file_reset() -> None:
    cfg = _cfg(ngpu=2, wm_envs_per_worker=2, wm_rollout_target_trajectories=4)
    cfg.manual_cotrain.wm_rollout_lease_epochs = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    wm_env = _DynamicFakeWMEnvGroup(worker_count=1, slow_rank=0)
    progress = _ResetClassifierProgress()

    with pytest.raises(TimeoutError):
        runner._wait_env_metrics_with_dynamic_wm_leases(
            real_env_results=[_NeverReady()],
            wm_env=wm_env,
            rollout_result=_RunningRollout(),
            env_channel_name="env",
            rollout_channel_name="rollout",
            actor_channel_name="actor",
            timeout_s=0.05,
            poll_s=0.0,
            progress=progress,  # type: ignore[arg-type]
        )

    reported = [snapshot for snapshot in progress.snapshots if snapshot.status]
    assert reported
    assert any("cls_sr=0.500" in str(snapshot.status) for snapshot in reported)
    assert any("cls_traj_sr=0.500" in str(snapshot.status) for snapshot in reported)


def test_wait_env_metrics_surfaces_rollout_failure_before_env_wait() -> None:
    with pytest.raises(RuntimeError, match="rank0 failed key=0:0"):
        _wait_env_metrics_with_rollout_guard(
            [_NeverReady()],
            _EarlyFailedRollout(),
            timeout_s=10.0,
            poll_s=0.0,
        )


def test_wait_env_metrics_timeout_points_to_handshake_trace() -> None:
    with pytest.raises(TimeoutError, match="DVLA_COTRAIN_HANDSHAKE_TRACE=1"):
        _wait_env_metrics_with_rollout_guard(
            [_NeverReady()],
            _RunningRollout(),
            timeout_s=0.001,
            poll_s=0.0,
        )


def test_manual_cotrain_progress_snapshot_sums_worker_files(tmp_path) -> None:
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    (progress_dir / "real_env_0.json").write_text(
        json.dumps(
            {
                "role": "real_env",
                "rank": 0,
                "env_rank": 0,
                "global_step": 4,
                "done": 1,
                "total": 2,
                "finished": False,
            }
        ),
        encoding="utf-8",
    )
    (progress_dir / "wm_env_1.json").write_text(
        json.dumps(
            {
                "role": "wm_env",
                "rank": 0,
                "env_rank": 1,
                "global_step": 4,
                "done": 3,
                "total": 5,
                "finished": True,
            }
        ),
        encoding="utf-8",
    )

    snapshot = _read_manual_cotrain_progress_snapshot(progress_dir)

    assert snapshot.done == 4
    assert snapshot.total == 7
    assert snapshot.status is not None
    assert "global_step=4" in snapshot.status
    assert "real_env#0=1/2" in snapshot.status
    assert "wm_env#1=3/5" in snapshot.status
    assert "finished=1/2" in snapshot.status


def test_manual_cotrain_progress_snapshot_reports_classifier_success_rate(tmp_path) -> None:
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    (progress_dir / "wm_env_1.json").write_text(
        json.dumps(
            {
                "role": "wm_env",
                "rank": 0,
                "env_rank": 1,
                "global_step": 4,
                "done": 3,
                "total": 5,
                "finished": False,
                "classifier_success_chunks": 2,
                "classifier_total_chunks": 4,
                "classifier_success_trajectories": 1,
                "classifier_total_trajectories": 2,
            }
        ),
        encoding="utf-8",
    )

    snapshot = _read_manual_cotrain_progress_snapshot(progress_dir)

    assert snapshot.status is not None
    assert "cls_sr=0.500" in snapshot.status
    assert "cls_traj_sr=0.500" in snapshot.status


def test_wait_env_metrics_timeout_includes_manual_progress(tmp_path) -> None:
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    (progress_dir / "wm_env_1.json").write_text(
        json.dumps(
            {
                "role": "wm_env",
                "rank": 0,
                "env_rank": 1,
                "global_step": 2,
                "done": 3,
                "total": 8,
                "finished": False,
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[int, int, str, str | None, str | None]] = []
    monitor = _ManualCotrainEnvProgressMonitor(
        progress_dir,
        lambda current, total, desc, **kwargs: calls.append(
            (current, total, desc, kwargs.get("unit"), kwargs.get("status"))
        ),
    )

    with pytest.raises(TimeoutError, match="wm_env#1=3/8"):
        _wait_env_metrics_with_rollout_guard(
            [_NeverReady()],
            _RunningRollout(),
            timeout_s=0.001,
            poll_s=0.0,
            progress=monitor,
        )

    assert calls
    assert calls[-1][:4] == (3, 8, "manual-cotrain-env", "chunk")
    assert calls[-1][4] is not None
    assert "global_step=2" in calls[-1][4]


def test_prepare_manual_cotrain_progress_dir_clears_stale_json(tmp_path) -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(out_dir=str(tmp_path / "run")))
    progress_dir = runner._manual_cotrain_progress_dir(global_step=3)
    progress_dir.mkdir(parents=True)
    stale = progress_dir / "real_env_0.json"
    stale.write_text("{}", encoding="utf-8")
    keep = progress_dir / "notes.txt"
    keep.write_text("keep", encoding="utf-8")

    prepared = runner._prepare_manual_cotrain_progress_dir(global_step=3)

    assert prepared == progress_dir
    assert not stale.exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_run_global_step_syncs_actor_policy_and_wm_env_states(monkeypatch) -> None:
    traces: list[str] = []
    monkeypatch.setattr(manual_runner, "_hs_trace", traces.append)
    cfg = _cfg(ngpu=2)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    cfg.manual_cotrain.wm_rollout_epoch = 2
    runner = runners.ManualCotrainRayRunner(cfg)
    events: list[str] = []
    actor = _FakeActorGroup(events)
    rollout = _FakeRolloutGroup(events)
    learner = _FakeLearnerGroup()
    wm_env = _FakeEnvGroup(
        [
            {
                "env/trajectory_shards": 2.0,
                "env/wm_env/trajectory_shards": 2.0,
                "env/steps": 4.0,
            }
        ],
        events,
    )
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": rollout,
        "LearnerGroup": learner,
        "RealEnvGroup": _FakeEnvGroup(
            {"env/trajectory_shards": 1.0, "env/steps": 2.0},
            events,
        ),
        "WMEnvGroup": wm_env,
        "ReplayGroup": _FakeReplayGroup(),
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm0", "wm1"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    metrics = runner._run_global_step(groups, global_step=1)

    assert actor.sync_calls == [(None, "policy", 1)]
    assert rollout.pulled == [("policy", None)]
    assert rollout.generate_call == ("env", "rollout", 1)
    assert events.index("env_wait") < events.index("actor_recv_wait")
    assert groups["RealEnvGroup"].progress_configs
    assert wm_env.progress_configs
    assert groups["RealEnvGroup"].progress_configs[0][0] == wm_env.progress_configs[0][0]
    assert groups["RealEnvGroup"].progress_configs[0][0].endswith(
        "manual_cotrain_progress/global_step_00000001"
    )
    assert [key for key, _ in groups["env_channel"].puts] == ["0", "1"]
    assert all(isinstance(value, StopMsg) for _, value in groups["env_channel"].puts)
    assert "[global_step=1] EnvGroup.interact start" in traces
    assert "[global_step=1] RolloutGroup.generate start" in traces
    assert "[global_step=1] EnvGroup.interact done" in traces
    assert "[env rank=0] send StopMsg key=0" in traces
    assert "[env rank=1] send StopMsg key=1" in traces
    assert actor.recv_calls == [
        ((0,), "actor", 1, [("wm_env", 1)]),
        ((1,), "actor", 1, [("wm_env", 1)]),
    ]
    assert actor.load_calls == 0
    assert actor.loaded_by_rank == []
    assert len(actor.loaded_shards) == 2
    assert groups["actor_channel"].qsize_calls == ["wm_env"]
    assert groups["actor_channel"].get_batch_calls == []
    assert groups["actor_channel"].gets == []
    assert learner.synced == []
    assert wm_env.component_state_versions == [1]
    assert wm_env.component_state_keys == [
        ["classifier", "classifier_threshold", "world_model"]
    ]
    assert wm_env.wm_versions == [1]
    assert wm_env.classifier_versions == [1]
    assert wm_env.world_model_states == [{"wm": 1}]
    assert wm_env.classifier_states == [{"cls": 2}]
    assert wm_env.classifier_thresholds == [0.95]
    assert metrics["env/trajectory_shards"] == 3.0
    assert metrics["env/steps"] == 6.0
    assert metrics["actor/received_shards"] == 2.0
    assert metrics["rollout/generated"] == 5.0
    assert metrics["sync/policy_version"] == 1.0
    assert metrics["sync/policy_export_s"] == 0.1
    assert metrics["sync/policy_push_s"] == 0.2
    assert metrics["sync/rollout_policy_pull_s"] == 0.4
    assert metrics["sync/learner_state_dicts_s"] >= 0.0
    assert metrics["sync/wm_env_load_component_states_s"] >= 0.0
    assert metrics["sync/load_component_states_s"] == 0.5
    assert metrics["replay_buffer/size"] == 3.0
    assert metrics["replay_buffer/transitions"] == 7.0
    assert metrics["learner/updates"] == 1.0
    assert metrics["train/learner_updates"] == 1.0
    assert "time/manual_cotrain/set_global_step_s" in metrics
    assert "time/manual_cotrain/actor_to_rollout_sync_s" in metrics
    assert "time/manual_cotrain/env_interact_and_rollout_generate_s" in metrics
    assert "time/manual_cotrain/actor_recv_rollout_trajectories_s" in metrics
    assert "time/manual_cotrain/actor_run_training_s" in metrics
    assert "time/manual_cotrain/learner_update_wm_classifier_s" in metrics
    assert "time/manual_cotrain/learner_to_wm_env_sync_s" in metrics
    assert metrics["time/manual_cotrain/global_step_s"] >= 0.0


def test_frozen_run_global_step_trains_policy_without_real_or_learner_calls() -> None:
    cfg = _cfg(
        ngpu=2,
        real_env_workers=0,
        real_env_enabled=False,
        learner_updates_enabled=False,
    )
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    cfg.manual_cotrain.wm_rollout_epoch = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    actor = _FakeActorGroup()
    rollout = _FakeRolloutGroup()
    wm_env = _FakeEnvGroup(
        [
            {
                "env/trajectory_shards": 2.0,
                "env/wm_env/trajectory_shards": 2.0,
                "env/steps": 4.0,
            }
        ]
    )
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": rollout,
        "LearnerGroup": None,
        "RealEnvGroup": None,
        "WMEnvGroup": wm_env,
        "ReplayGroup": _FakeReplayGroup(),
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm0", "wm1"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    metrics = runner._run_global_step(groups, global_step=1)

    assert actor.sync_calls == [(None, "policy", 1)]
    assert rollout.pulled == [("policy", None)]
    assert wm_env.global_steps == [1]
    assert wm_env.component_state_versions == []
    assert "learner/updates" not in metrics
    assert metrics["rollout/episodes"] == 0.0
    assert metrics["env/wm_env/trajectory_shards"] == 2.0
    assert metrics["actor/ppo_updates"] > 0.0


def test_run_global_step_can_update_only_world_model_for_fixed_classifier() -> None:
    cfg = _cfg(ngpu=2)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    cfg.manual_cotrain.wm_rollout_epoch = 1
    cfg.manual_cotrain.learner_update_phase = "wm"
    runner = runners.ManualCotrainRayRunner(cfg)
    learner = _FakeLearnerGroup()
    groups = {
        "ActorGroup": _FakeActorGroup(),
        "RolloutGroup": _FakeRolloutGroup(),
        "LearnerGroup": learner,
        "RealEnvGroup": _FakeEnvGroup({"env/trajectory_shards": 1.0, "env/steps": 2.0}),
        "WMEnvGroup": _FakeEnvGroup(
            {
                "env/trajectory_shards": 1.0,
                "env/wm_env/trajectory_shards": 1.0,
                "env/steps": 2.0,
            }
        ),
        "ReplayGroup": _FakeReplayGroup(),
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm0"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    runner._run_global_step(groups, global_step=1)

    assert learner.update_call == ("wm", 1)


def test_run_global_step_does_not_sync_wm_to_env_when_classifier_update_skips() -> None:
    cfg = _cfg(ngpu=2)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    cfg.manual_cotrain.wm_rollout_epoch = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    wm_env = _FakeEnvGroup(
        [
            {
                "env/trajectory_shards": 1.0,
                "env/wm_env/trajectory_shards": 1.0,
                "env/steps": 2.0,
            }
        ]
    )
    groups = {
        "ActorGroup": _FakeActorGroup(),
        "RolloutGroup": _FakeRolloutGroup(),
        "LearnerGroup": _FakeLearnerGroup(
            {
                "learner/updates": 1.0,
                "cls/updated": 0.0,
                "cls/skipped_single_class_batch": 1.0,
                "wm/loss": 10.0,
            }
        ),
        "RealEnvGroup": _FakeEnvGroup(
            {"env/trajectory_shards": 1.0, "env/steps": 2.0}
        ),
        "WMEnvGroup": wm_env,
        "ReplayGroup": _FakeReplayGroup(),
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm0"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    metrics = runner._run_global_step(groups, global_step=1)

    assert wm_env.component_state_versions == [1]
    assert wm_env.component_state_keys == [["classifier", "classifier_threshold"]]
    assert wm_env.wm_versions == []
    assert wm_env.classifier_versions == [1]
    assert wm_env.world_model_states == []
    assert wm_env.classifier_states == [{"cls": 2}]
    assert wm_env.classifier_thresholds == [0.95]
    assert metrics["sync/wm_env_world_model_skipped_classifier_not_updated"] == 1.0


def _overlap_step_groups(events: list[str]):
    real_env = _FakeEnvGroup(
        {"env/trajectory_shards": 1.0, "env/steps": 2.0},
        events,
    )
    wm_env = _FakeEnvGroup(
        [
            {
                "env/trajectory_shards": 2.0,
                "env/wm_env/trajectory_shards": 2.0,
                "env/steps": 4.0,
            }
        ],
        events,
    )
    groups = {
        "ActorGroup": _FakeActorGroup(events),
        "RolloutGroup": _FakeRolloutGroup(events),
        "LearnerGroup": _FakeLearnerGroup(),
        "RealEnvGroup": real_env,
        "WMEnvGroup": wm_env,
        "ReplayGroup": _FakeReplayGroup(),
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm0", "wm1"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }
    return groups, real_env


def test_run_global_step_overlaps_env_bootstrap_after_training_when_enabled() -> None:
    cfg = _cfg(ngpu=2)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    cfg.manual_cotrain.overlap_env_bootstrap = True
    runner = runners.ManualCotrainRayRunner(cfg)
    events: list[str] = []
    groups, real_env = _overlap_step_groups(events)

    runner._run_global_step(groups, global_step=1)

    assert real_env.prefetch_calls == 1
    assert "env_prefetch_bootstrap" in events
    assert events.index("actor_run_training") < events.index("env_prefetch_bootstrap")


def test_run_global_step_skips_env_bootstrap_overlap_under_actor_offload() -> None:
    cfg = _cfg(ngpu=2)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    cfg.manual_cotrain.overlap_env_bootstrap = True
    cfg.actor.train_cfg.fsdp = {"cpu_offload": True}
    runner = runners.ManualCotrainRayRunner(cfg)
    events: list[str] = []
    groups, real_env = _overlap_step_groups(events)

    runner._run_global_step(groups, global_step=1)

    assert real_env.prefetch_calls == 0
    assert "env_prefetch_bootstrap" not in events


def test_run_global_step_does_not_overlap_env_bootstrap_by_default() -> None:
    cfg = _cfg(ngpu=2)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    events: list[str] = []
    groups, real_env = _overlap_step_groups(events)

    runner._run_global_step(groups, global_step=1)

    assert real_env.prefetch_calls == 0
    assert "env_prefetch_bootstrap" not in events


def test_run_global_step_dynamic_wm_progress_uses_central_snapshots(monkeypatch) -> None:
    cfg = _cfg(ngpu=2, wm_envs_per_worker=1, wm_rollout_target_trajectories=4)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 1
    runner = runners.ManualCotrainRayRunner(cfg)
    progress_calls: list[tuple[str, int | bool, str | None]] = []

    class _RecordingProgressMonitor:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def records(self):
            return []

        def report(self, *, force: bool = False):
            progress_calls.append(("file", bool(force), None))
            return _ManualCotrainProgressSnapshot(
                done=0,
                total=4,
                status="file-progress",
                worker_count=1,
                finished_count=0,
            )

        def report_snapshot(self, snapshot, *, force: bool = False):
            del force
            progress_calls.append(("central", int(snapshot.total), snapshot.status))
            return snapshot

    monkeypatch.setattr(
        manual_runner,
        "_ManualCotrainEnvProgressMonitor",
        _RecordingProgressMonitor,
    )

    def fake_dynamic_wait(self, **kwargs):
        del self
        progress = kwargs["progress"]
        progress.report_snapshot(
            _ManualCotrainProgressSnapshot(
                done=0,
                total=5,
                status="real_env=0/1 wm_pool=0/4",
                worker_count=2,
                finished_count=0,
            ),
            force=True,
        )
        progress.report_snapshot(
            _ManualCotrainProgressSnapshot(
                done=5,
                total=5,
                status="real_env=1/1 wm_pool=4/4",
                worker_count=2,
                finished_count=2,
            ),
            force=True,
        )
        return {
            "env/trajectory_shards": 4.0,
            "env/wm_env/trajectory_shards": 4.0,
            "env/steps": 8.0,
        }

    monkeypatch.setattr(
        runner._runner.__class__,
        "_wait_env_metrics_with_dynamic_wm_leases",
        fake_dynamic_wait,
    )

    groups = {
        "ActorGroup": _FakeActorGroup(),
        "RolloutGroup": _FakeRolloutGroup(),
        "LearnerGroup": _FakeLearnerGroup(),
        "RealEnvGroup": _FakeEnvGroup({"env/trajectory_shards": 1.0}),
        "WMEnvGroup": _FakeEnvGroup({"env/wm_env/trajectory_shards": 4.0}),
        "ReplayGroup": _FakeReplayGroup(),
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm0", "wm1", "wm2", "wm3"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    metrics = runner._run_global_step(groups, global_step=1)

    assert metrics["actor/received_shards"] == 4.0
    assert [call[0] for call in progress_calls] == ["central", "central"]
    assert all(call[1] == 5 for call in progress_calls)


def test_manual_cotrain_global_progress_reports_completed_policy_update() -> None:
    runner = manual_runner.ManualCotrainRayRunner(_cfg(ngpu=2))
    calls: list[tuple[int, int, str, str, str | None, bool]] = []
    runner.console_progress = (
        lambda current, total, desc, unit="it", status=None, force=False: calls.append(
            (int(current), int(total), str(desc), str(unit), status, bool(force))
        )
    )

    runner._report_global_step_progress(
        global_step=3,
        total_steps=20,
        metrics={
            "actor/ppo_updates": 4.0,
            "actor/ppo_optimizer_steps": 4.0,
            "actor/global_ppo_samples": 65536.0,
            "actor/global_loss_mask_sum": 64000.0,
            "actor/global_batch_size": 16384.0,
            "actor/micro_batch_size": 32.0,
            "actor/loss": 0.125,
            "actor/approx_kl": 0.00125,
            "actor/clip_fraction": 0.075,
            "actor/lr": 5.0e-7,
            "time/manual_cotrain/env_interact_and_rollout_generate_s": 10.5,
            "time/manual_cotrain/actor_run_training_s": 2.25,
            "time/manual_cotrain/global_step_s": 13.75,
        },
    )

    assert calls == [
        (
            3,
            20,
            "manual-cotrain",
            "step",
            "ppo_steps=4 samples=64000/65536 batch=16384 micro=32 "
            "loss=0.125 approx_kl=0.00125 clip_frac=0.075 lr=5e-07 "
            "imagine=10.5s actor=2.2s step=13.8s",
            True,
        )
    ]


def test_actor_metric_aggregation_preserves_count_semantics() -> None:
    metrics = manual_runner._aggregate_actor_metric_lists(
        [
            [
                {
                    "actor/trajectory_count": 128.0,
                    "actor/reward_filtered_rollouts": 8.0,
                    "actor/ppo_optimizer_steps": 4.0,
                    "actor/global_ppo_samples": 65536.0,
                    "actor/loss": 0.1,
                },
                {
                    "actor/trajectory_count": 128.0,
                    "actor/reward_filtered_rollouts": 16.0,
                    "actor/ppo_optimizer_steps": 4.0,
                    "actor/global_ppo_samples": 65536.0,
                    "actor/loss": 0.3,
                },
            ]
        ]
    )

    assert metrics["actor/trajectory_count"] == 256.0
    assert metrics["actor/reward_filtered_rollouts"] == 24.0
    assert metrics["actor/ppo_optimizer_steps"] == 4.0
    assert metrics["actor/global_ppo_samples"] == 65536.0
    assert metrics["actor/loss"] == pytest.approx(0.2)


def test_receive_actor_trajectories_loads_actual_wm_shards_by_actor_rank() -> None:
    cfg = _cfg(ngpu=3, wm_envs_per_worker=1)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 2
    runner = runners.ManualCotrainRayRunner(cfg)
    actor = _FakeActorGroup()
    actor.workers = ["actor0", "actor1"]
    stage_times: dict[str, float] = {}
    groups = {
        "ActorGroup": actor,
        "WMEnvGroup": _FakeEnvGroup({}),
        "actor_channel": _FakeChannel(["wm0", "wm1", "wm2", "wm3"]),
    }

    metrics = runner._receive_actor_trajectories(
        groups,
        expected_shards=4,
        env_metrics={"env/wm_env/trajectory_shards": 4.0},
        actor_channel_name="actor",
        stage_times=stage_times,
        started_receivers=None,
    )

    assert actor.recv_calls == [
        ((0,), "actor", 2, [("wm_env", 2)]),
        ((1,), "actor", 2, [("wm_env", 2)]),
    ]
    assert groups["actor_channel"].qsize_calls == ["wm_env"]
    assert groups["actor_channel"].get_batch_calls == []
    assert actor.loaded_by_rank == []
    assert metrics["actor/received_shards"] == 4.0
    assert "time/manual_cotrain/actor_channel_get_batch_s" in stage_times
    assert "time/manual_cotrain/actor_load_trajectory_shards_s" in stage_times


def test_receive_actor_trajectories_dispatches_receive_to_actor_ranks() -> None:
    cfg = _cfg(ngpu=3, wm_envs_per_worker=1)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 2
    runner = runners.ManualCotrainRayRunner(cfg)
    actor = _FakeActorGroup()
    actor.workers = ["actor0", "actor1"]
    actor_channel = _FakeChannel(["wm0", "wm1", "wm2", "wm3"])
    stage_times: dict[str, float] = {}

    metrics = runner._receive_actor_trajectories(
        {
            "ActorGroup": actor,
            "WMEnvGroup": _FakeEnvGroup({}),
            "actor_channel": actor_channel,
        },
        expected_shards=4,
        env_metrics={"env/wm_env/trajectory_shards": 4.0},
        actor_channel_name="actor",
        stage_times=stage_times,
        started_receivers=None,
    )

    assert actor.recv_calls == [
        ((0,), "actor", 2, [("wm_env", 2)]),
        ((1,), "actor", 2, [("wm_env", 2)]),
    ]
    assert actor_channel.qsize_calls == ["wm_env"]
    assert actor_channel.get_batch_calls == []
    assert metrics["actor/received_shards"] == 4.0


def test_receive_actor_trajectories_waits_for_actor_channel_ready() -> None:
    cfg = _cfg(ngpu=3, wm_envs_per_worker=1)
    cfg.actor.train_cfg.algorithm_cfg.group_size = 2
    runner = runners.ManualCotrainRayRunner(cfg)
    actor = _FakeActorGroup()
    actor.workers = ["actor0", "actor1"]
    actor_channel = _FakeChannel(
        ["wm0", "wm1", "wm2", "wm3"],
        qsize_sequence=[3, 4],
    )
    stage_times: dict[str, float] = {}

    metrics = runner._receive_actor_trajectories(
        {
            "ActorGroup": actor,
            "WMEnvGroup": _FakeEnvGroup({}),
            "actor_channel": actor_channel,
        },
        expected_shards=4,
        env_metrics={"env/wm_env/trajectory_shards": 4.0},
        actor_channel_name="actor",
        stage_times=stage_times,
        started_receivers=None,
    )

    assert metrics["actor/received_shards"] == 4.0
    assert actor_channel.qsize_calls == ["wm_env", "wm_env"]
    assert actor_channel.get_batch_calls == []
    assert actor.recv_calls == [
        ((0,), "actor", 2, [("wm_env", 2)]),
        ((1,), "actor", 2, [("wm_env", 2)]),
    ]
    assert "time/manual_cotrain/actor_channel_wait_ready_s" in stage_times


def test_run_global_step_can_publish_learner_weights_when_configured() -> None:
    runner = runners.ManualCotrainRayRunner(
        _cfg(ngpu=2, publish_learner_weights=True)
    )
    learner = _FakeLearnerGroup()
    groups = {
        "ActorGroup": _FakeActorGroup(),
        "RolloutGroup": _FakeRolloutGroup(),
        "LearnerGroup": learner,
        "RealEnvGroup": _FakeEnvGroup(
            {"env/trajectory_shards": 1.0, "env/steps": 2.0}
        ),
        "WMEnvGroup": _FakeEnvGroup(
            {
                "env/trajectory_shards": 1.0,
                "env/wm_env/trajectory_shards": 1.0,
                "env/steps": 2.0,
            }
        ),
        "ReplayGroup": None,
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm-shard"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    runner._run_global_step(groups, global_step=1)

    assert learner.synced == [("world_model", 1), ("classifier", 1)]


def test_run_global_step_writes_manual_checkpoint_when_enabled(tmp_path) -> None:
    runner = runners.ManualCotrainRayRunner(
        _cfg(ngpu=2, out_dir=str(tmp_path), checkpoint_every=1)
    )
    actor = _FakeActorGroup()
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": _FakeRolloutGroup(),
        "LearnerGroup": _FakeLearnerGroup(),
        "RealEnvGroup": _FakeEnvGroup({"env/trajectory_shards": 1.0, "env/steps": 2.0}),
        "WMEnvGroup": _FakeEnvGroup(
            {
                "env/trajectory_shards": 1.0,
                "env/wm_env/trajectory_shards": 1.0,
                "env/steps": 2.0,
            }
        ),
        "ReplayGroup": None,
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["wm-shard"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    runner._run_global_step(groups, global_step=1)

    ckpt = (
        tmp_path
        / "checkpoints"
        / "manual_cotrain_step_1"
        / "manual_cotrain.ckpt"
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    manifest_path = ckpt.parent / "manual_cotrain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_manifest_path = (
        tmp_path
        / "checkpoints"
        / "global_step_1"
        / "manual_cotrain_manifest.json"
    )
    canonical_manifest = json.loads(
        canonical_manifest_path.read_text(encoding="utf-8")
    )
    assert actor.state_dict_calls == [None]
    assert payload["global_step"] == 1
    assert payload["cfg"]["_target_"] == "dreamervla.runners.ManualCotrainRayRunner"
    assert payload["cfg"]["manual_cotrain"]["checkpoint_every"] == 1
    assert sorted(payload["state_dicts"]) == ["classifier", "policy", "world_model"]
    assert manifest["schema_version"] == 1
    assert manifest["global_step"] == 1
    assert manifest["components"]["policy"]["path"] == "manual_cotrain.ckpt"
    assert manifest["components"]["world_model"]["path"] == "manual_cotrain.ckpt"
    assert manifest["components"]["classifier"]["path"] == "manual_cotrain.ckpt"
    assert manifest["versions"]["global_step"] == 1
    assert manifest["versions"]["policy_version"] == 1
    assert manifest["versions"]["world_model_version"] == 1
    assert manifest["versions"]["classifier_version"] == 1
    assert manifest["versions"]["actor_policy_version"] == 1
    assert manifest["versions"]["rollout_policy_version"] == 1
    assert manifest["versions"]["wm_version"] == 1
    assert manifest["run"] == {
        "root": "../..",
        "resolved_config": "../../resolved_config.yaml",
        "run_manifest": "../../run_manifest.json",
    }
    assert canonical_manifest["global_step"] == 1
    assert (
        canonical_manifest["components"]["policy"]["path"]
        == "../manual_cotrain_step_1/manual_cotrain.ckpt"
    )
    assert canonical_manifest["run"] == manifest["run"]
    assert not (canonical_manifest_path.parent / "manual_cotrain.ckpt").exists()
    alias_payload = manual_runner._load_manual_resume_payload(
        str(canonical_manifest_path),
        required=True,
    )
    assert alias_payload is not None
    assert alias_payload["global_step"] == 1
    canonical_dir_payload = manual_runner._load_manual_resume_payload(
        str(canonical_manifest_path.parent),
        required=True,
    )
    compat_dir_payload = manual_runner._load_manual_resume_payload(
        str(ckpt.parent),
        required=True,
    )
    assert canonical_dir_payload is not None
    assert compat_dir_payload is not None
    assert canonical_dir_payload["global_step"] == 1
    assert compat_dir_payload["global_step"] == 1
    assert torch.equal(
        payload["state_dicts"]["policy"]["policy.weight"],
        torch.ones(1),
    )


def test_manual_checkpoint_skips_replay_state_by_default(tmp_path) -> None:
    runner = runners.ManualCotrainRayRunner(
        _cfg(ngpu=2, out_dir=str(tmp_path), checkpoint_every=1)
    )
    replay = _FakeReplayGroup()
    groups = {
        "ActorGroup": _FakeActorGroup(),
        "LearnerGroup": _FakeLearnerGroup(),
        "ReplayGroup": replay,
    }

    runner._maybe_save_manual_checkpoint(
        groups,
        global_step=1,
        metrics={"global_step": 1.0, "replay_buffer/size": 3.0},
    )

    ckpt = (
        tmp_path
        / "checkpoints"
        / "manual_cotrain_step_1"
        / "manual_cotrain.ckpt"
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    manifest = json.loads(
        (ckpt.parent / "manual_cotrain_manifest.json").read_text(encoding="utf-8")
    )
    assert replay.state_dict_calls == 0
    assert replay.sampling_state_dict_calls == 1
    assert payload["replay"] is None
    assert payload["replay_sampling_state"] == {
        "task_sample_cursor": 11,
        "initial_condition_cursor": 13,
    }
    assert manifest["replay"]["present"] is False


def test_manual_checkpoint_can_include_replay_state_when_configured(tmp_path) -> None:
    runner = runners.ManualCotrainRayRunner(
        _cfg(
            ngpu=2,
            out_dir=str(tmp_path),
            checkpoint_every=1,
            save_replay_state=True,
        )
    )
    replay = _FakeReplayGroup()
    groups = {
        "ActorGroup": _FakeActorGroup(),
        "LearnerGroup": _FakeLearnerGroup(),
        "ReplayGroup": replay,
    }

    runner._maybe_save_manual_checkpoint(
        groups,
        global_step=1,
        metrics={"global_step": 1.0, "replay_buffer/size": 3.0},
    )

    ckpt = (
        tmp_path
        / "checkpoints"
        / "manual_cotrain_step_1"
        / "manual_cotrain.ckpt"
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    manifest = json.loads(
        (ckpt.parent / "manual_cotrain_manifest.json").read_text(encoding="utf-8")
    )
    assert replay.state_dict_calls == 1
    assert payload["replay"] == {"episodes": ["current"], "num_transitions": 7}
    assert manifest["replay"]["present"] is True


def test_frozen_checkpoint_contains_policy_and_source_hashes_without_model_copies(
    tmp_path,
) -> None:
    class _StableHashGroup:
        @staticmethod
        def component_state_hashes():
            return _Ready(
                [{"world_model": "wm-hash", "classifier": "cls-hash"}]
            )

    cfg = _cfg(
        ngpu=8,
        out_dir=str(tmp_path),
        checkpoint_every=1,
        save_replay_state=True,
        real_env_workers=0,
        real_env_enabled=False,
        learner_updates_enabled=False,
    )
    runner = manual_runner.ManualCotrainRayRunner(cfg)
    runner._frozen_state_hashes = {
        "world_model": "wm-hash",
        "classifier": "cls-hash",
    }
    runner._frozen_source_checkpoints = {
        "world_model": "/wm.ckpt",
        "classifier": "/classifier.ckpt",
    }
    runner._frozen_classifier_threshold = 0.65

    runner._maybe_save_manual_checkpoint(
        {
            "ActorGroup": _FakeActorGroup(),
            "LearnerGroup": None,
            "ReplayGroup": _FakeReplayGroup(),
            "WMEnvGroup": _StableHashGroup(),
        },
        global_step=1,
        metrics={"global_step": 1.0, "actor/ppo_updates": 1.0},
    )

    ckpt = (
        tmp_path
        / "checkpoints"
        / "manual_cotrain_step_1"
        / "manual_cotrain.ckpt"
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    manifest = json.loads(
        (ckpt.parent / "manual_cotrain_manifest.json").read_text(encoding="utf-8")
    )

    assert set(payload["state_dicts"]) == {"policy", "policy_optimizer"}
    assert payload["frozen_state_hashes"] == runner._frozen_state_hashes
    assert payload["source_checkpoints"] == runner._frozen_source_checkpoints
    assert payload["classifier_threshold"] == 0.65
    assert set(manifest["components"]) == {"policy"}


def test_frozen_finalize_writes_causal_summary_and_rechecks_hashes(tmp_path) -> None:
    class _StableHashGroup:
        @staticmethod
        def component_state_hashes():
            return _Ready(
                [{"world_model": "wm-hash", "classifier": "cls-hash"}]
            )

    cfg = _cfg(
        ngpu=8,
        out_dir=str(tmp_path),
        real_env_workers=0,
        real_env_enabled=False,
        learner_updates_enabled=False,
    )
    cfg.training.require_policy_update = True
    cfg.replay = {
        "seed": {
            "data_dir": "/official/reward",
            "hidden_dir": "/official/hidden",
        }
    }
    runner = manual_runner.ManualCotrainRayRunner(cfg)
    runner._frozen_state_hashes = {
        "world_model": "wm-hash",
        "classifier": "cls-hash",
    }
    runner._frozen_source_checkpoints = {
        "world_model": "/wm.ckpt",
        "classifier": "/classifier.ckpt",
    }
    runner._frozen_classifier_threshold = 0.65
    runner._policy_initial_hash = "initial-policy-hash"
    runner._applied_policy_steps = 3

    runner._finalize_frozen_policy_run(
        {
            "ActorGroup": _FakeActorGroup(),
            "ReplayGroup": None,
            "LearnerGroup": None,
            "WMEnvGroup": _StableHashGroup(),
        },
        global_step=3,
        metrics={"global_step": 3, "actor/ppo_updates": 1.0},
    )

    summary = json.loads(
        (tmp_path / "frozen_rl_summary.json").read_text(encoding="utf-8")
    )
    assert summary["execution"] == "ray_manual_policy_only"
    assert summary["ngpu"] == 8
    assert summary["frozen_hashes_before"] == summary["frozen_hashes_after"]
    assert summary["policy_hash_before"] == "initial-policy-hash"
    assert summary["policy_changed"] is True
    assert summary["applied_policy_steps"] == 3
    assert Path(summary["final_checkpoint"]).is_file()


def test_frozen_finalize_reuses_checkpoint_already_written_at_final_step(
    tmp_path,
) -> None:
    class _StableHashGroup:
        @staticmethod
        def component_state_hashes():
            return _Ready(
                [{"world_model": "wm-hash", "classifier": "cls-hash"}]
            )

    cfg = _cfg(
        ngpu=8,
        out_dir=str(tmp_path),
        checkpoint_every=1,
        real_env_workers=0,
        real_env_enabled=False,
        learner_updates_enabled=False,
    )
    runner = manual_runner.ManualCotrainRayRunner(cfg)
    runner._frozen_state_hashes = {
        "world_model": "wm-hash",
        "classifier": "cls-hash",
    }
    runner._frozen_source_checkpoints = {
        "world_model": "/wm.ckpt",
        "classifier": "/classifier.ckpt",
    }
    runner._frozen_classifier_threshold = 0.65
    runner._policy_initial_hash = "different-initial-hash"
    runner._applied_policy_steps = 1
    actor = _FakeActorGroup()
    groups = {
        "ActorGroup": actor,
        "ReplayGroup": None,
        "LearnerGroup": None,
        "WMEnvGroup": _StableHashGroup(),
    }
    runner._maybe_save_manual_checkpoint(
        groups,
        global_step=1,
        metrics={"global_step": 1.0},
    )

    runner._finalize_frozen_policy_run(
        groups,
        global_step=1,
        metrics={"global_step": 1},
    )

    assert actor.state_dict_calls == [None]


def test_manual_runner_resume_restores_replay_and_continues_after_checkpoint_step(
    tmp_path,
    monkeypatch,
) -> None:
    import dreamervla.runners.manual_cotrain_ray_runner as manual_runner

    ckpt = tmp_path / "manual_cotrain.ckpt"
    replay_state = {"episodes": ["saved"], "num_transitions": 5}
    replay_sampling_state = {
        "task_sample_cursor": 17,
        "initial_condition_cursor": 19,
    }
    torch.save(
        {
            "global_step": 2,
            "state_dicts": {
                "policy": {"policy.weight": torch.ones(1)},
                "world_model": {"wm": torch.ones(1)},
                "classifier": {"cls": torch.ones(1)},
            },
            "replay": replay_state,
            "replay_sampling_state": replay_sampling_state,
        },
        ckpt,
    )

    class _FakeCluster:
        def __init__(self, cfg):
            del cfg

        def require_single_node(self):
            return None

        def shutdown(self):
            return None

    monkeypatch.setattr(manual_runner, "Cluster", _FakeCluster)
    cfg = _cfg(ngpu=2)
    cfg.manual_cotrain.global_steps = 4
    cfg.manual_cotrain.resume_ckpt = str(ckpt)
    runner = manual_runner.ManualCotrainRayRunner(cfg)
    replay = _FakeReplayGroup()
    runner._build_groups = lambda cluster: {"ReplayGroup": replay}
    seen_steps: list[int] = []

    def _fake_run_global_step(groups, global_step: int):
        assert groups["ReplayGroup"] is replay
        seen_steps.append(int(global_step))
        return {"global_step": float(global_step)}

    runner._run_global_step = _fake_run_global_step

    history = runner.run()

    assert seen_steps == [3, 4]
    assert replay.loaded_state == replay_state
    assert replay.loaded_sampling_state == replay_sampling_state
    assert history["global_step"] == 4


def test_manual_runner_resume_loads_checkpoint_components_into_wm_env() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=2))
    wm_env = _FakeEnvGroup({})
    payload = {
        "global_step": 3,
        "state_dicts": {
            "policy": {"policy.weight": torch.ones(1)},
            "world_model": {"wm": torch.ones(1)},
            "classifier": {"cls": torch.ones(1)},
        },
        "replay": None,
    }

    runner._restore_manual_resume_state({"WMEnvGroup": wm_env}, payload)

    assert wm_env.component_state_versions == [3]
    assert wm_env.component_state_keys == [["classifier", "world_model"]]
    assert "wm" in wm_env.world_model_states[0]
    assert "cls" in wm_env.classifier_states[0]


def test_frozen_runner_resume_restores_policy_optimizer_and_binds_sources() -> None:
    class _StableHashGroup:
        @staticmethod
        def component_state_hashes():
            return _Ready(
                [{"world_model": "wm-hash", "classifier": "cls-hash"}]
            )

    runner = manual_runner.ManualCotrainRayRunner(
        _cfg(
            ngpu=8,
            real_env_workers=0,
            real_env_enabled=False,
            learner_updates_enabled=False,
        )
    )
    runner._frozen_state_hashes = {
        "world_model": "wm-hash",
        "classifier": "cls-hash",
    }
    runner._frozen_source_checkpoints = {
        "world_model": "/wm.ckpt",
        "classifier": "/classifier.ckpt",
    }
    runner._frozen_classifier_threshold = 0.65
    payload = {
        "global_step": 500,
        "state_dicts": {
            "policy": {"policy.weight": torch.ones(1)},
            "policy_optimizer": {
                "state": {0: {"step": torch.tensor(500.0)}},
                "param_groups": [{"params": [0]}],
            },
        },
        "replay": None,
        "frozen_state_hashes": dict(runner._frozen_state_hashes),
        "source_checkpoints": dict(runner._frozen_source_checkpoints),
        "classifier_threshold": 0.65,
    }
    runner._pending_manual_resume_payload = payload

    init_state = runner._actor_init_checkpoint()
    runner._restore_manual_resume_state(
        {"WMEnvGroup": _StableHashGroup()},
        payload,
    )

    assert set(init_state) == {"policy", "policy_optimizer"}
    assert init_state["policy_optimizer"]["state"][0]["step"].item() == 500.0
