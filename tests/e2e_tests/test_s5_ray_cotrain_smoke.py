from __future__ import annotations

import ray


def test_ray_cotrain_runner_smoke_generates_data_and_runs_ppo() -> None:
    try:
        from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    except ModuleNotFoundError as exc:
        raise AssertionError("OnlineCotrainRayRunner module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()

    cfg = {
        "num_env_workers": 2,
        "rollout_steps": 9,
        "episode_horizon": 3,
        "sequence_length": 3,
        "replay_capacity": 100,
        "min_replay_episodes": 1,
        "ppo_batch_size": 2,
        "ppo_lr": 0.05,
        "weight_sync_every": 1,
    }
    runner = OnlineCotrainRayRunner(cfg)

    history = runner.run()

    assert history["rollout/episodes"] >= 2
    assert history["train/ppo_updates"] >= 1
    assert history["sync/policy_version"] >= 1
    assert history["time/overlap_events"] >= 1
    assert history["train/rl_loss"] >= 0.0
    assert history["env/num_env_workers"] == 2
    assert not ray.is_initialized()


def test_ray_cotrain_runner_accepts_nested_component_config() -> None:
    try:
        from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner
    except ModuleNotFoundError as exc:
        raise AssertionError("OnlineCotrainRayRunner module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()

    cfg = {
        "env": {
            "num_workers": 3,
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:CounterEnv",
                "kwargs": {"horizon": 2, "image_shape": (4, 4, 3), "embedding_dim": 4},
            },
        },
        "rollout": {"steps": 6, "min_replay_episodes": 1},
        "replay": {"cfg": {"capacity": 100, "sequence_length": 2, "task_ids": (0,), "rank": 0}},
        "policy": {
            "cfg": {
                "target": "dreamervla.workers.actor._test_models:TinySharedPolicy",
                "kwargs": {"hidden_dim": 4, "action_dim": 7},
            }
        },
        "learner": {
            "train_cfg": {
                "mode": "synthetic_ppo",
                "batch_size": 2,
                "lr": 0.05,
                "device": "cpu",
                "syncer": {"store_name": "ray_cotrain_nested_store"},
            }
        },
    }
    runner = OnlineCotrainRayRunner(cfg)

    history = runner.run()

    assert history["env/num_env_workers"] == 3
    assert history["rollout/episodes"] >= 3
    assert history["train/ppo_updates"] >= 1
    assert not ray.is_initialized()
