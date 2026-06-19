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

    def ready(self, min_episodes: int) -> _Ready:
        del min_episodes
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


def test_ray_runner_uses_cotrain_phase_for_dreamervla_learner_mode() -> None:
    from omegaconf import OmegaConf

    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    cfg = OmegaConf.create(
        {
            "rollout": {"steps": 1, "min_replay_episodes": 1},
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )
    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = cfg
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
