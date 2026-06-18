from __future__ import annotations

import json

import h5py
import ray


def test_ray_coldstart_runner_writes_reward_and_sidecar(tmp_path) -> None:
    try:
        from dreamervla.runners.cold_start_ray_collect_runner import (
            ColdStartRayCollectRunner,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("ColdStartRayCollectRunner module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    cfg = {
        "env": {
            "num_workers": 2,
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:DumpCounterEnv",
                "kwargs": {"horizon": 3, "image_shape": (4, 4, 3), "embedding_dim": 4},
            },
        },
        "rollout": {"target_episodes": 4, "max_steps": 12},
        "dump": {
            "reward_dir": str(reward_dir),
            "hidden_dir": str(hidden_dir),
            "shard_name": "ray_shard_000.hdf5",
            "preprocess_config": {
                "action_head_type": "oft_discrete_token",
                "history": 1,
                "include_state": False,
                "hidden_key": "obs_embedding",
            },
            "data_attrs": {"task_suite_name": "synthetic_libero", "env_name": "dump_counter"},
        },
        "policy": {
            "cfg": {
                "target": "dreamervla.workers.actor._test_models:TinySharedPolicy",
                "kwargs": {"hidden_dim": 4, "action_dim": 7},
            }
        },
        "inference": {
            "cfg": {
                "encoder": {"target": "dreamervla.workers.inference._test_models:TinyEncoder"},
                "world_model": {
                    "target": "dreamervla.workers.inference._test_models:TinyWorldModel",
                    "kwargs": {"hidden_dim": 4, "action_dim": 7},
                },
                "device": "cpu",
            }
        },
    }

    history = ColdStartRayCollectRunner(cfg).run()

    assert history["rollout/episodes"] == 4
    assert history["env/num_env_workers"] == 2
    assert not ray.is_initialized()

    reward_path = reward_dir / "ray_shard_000.hdf5"
    hidden_path = hidden_dir / "ray_shard_000.hdf5"
    assert reward_path.is_file()
    assert hidden_path.is_file()
    assert json.loads((hidden_dir / "preprocess_config.json").read_text())["hidden_key"] == "obs_embedding"

    with h5py.File(reward_path, "r") as reward_f, h5py.File(hidden_path, "r") as hidden_f:
        assert reward_f["data"].attrs["num_demos"] == "4"
        assert reward_f["data"].attrs["task_suite_name"] == "synthetic_libero"
        demo = reward_f["data"]["demo_0"]
        assert demo.attrs["task_id"] == 0
        assert demo.attrs["episode_horizon"] == 3
        assert demo["actions"].shape == (3, 7)
        assert demo["obs"]["agentview_rgb"].shape == (3, 4, 4, 3)
        assert demo["dones"][-1] == 1
        assert demo["sparse_rewards"][-1] == 1
        assert hidden_f["data"]["demo_0"]["obs_embedding"].shape == (3, 4)


def test_ray_coldstart_synthetic_experiment_runs_through_train_entry(tmp_path) -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamervla.train import run

    if ray.is_initialized():
        ray.shutdown()

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts_ray_synthetic",
                f"training.out_dir={tmp_path / 'run'}",
                f"dump.reward_dir={tmp_path / 'reward'}",
                f"dump.hidden_dir={tmp_path / 'hidden'}",
            ],
        )

    run(cfg)

    assert (tmp_path / "run" / "resolved_config.yaml").is_file()
    assert (tmp_path / "run" / "run_manifest.json").is_file()
    assert (tmp_path / "reward" / "ray_shard_000.hdf5").is_file()
    assert (tmp_path / "hidden" / "ray_shard_000.hdf5").is_file()
    assert not ray.is_initialized()


def test_ray_coldstart_overlaps_env_and_inference(tmp_path) -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner

    if ray.is_initialized():
        ray.shutdown()

    cfg = {
        "env": {
            "num_workers": 2,
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:DumpCounterEnv",
                "kwargs": {"horizon": 3, "image_shape": (4, 4, 3), "embedding_dim": 4},
            },
        },
        "rollout": {"target_episodes": 4, "max_steps": 12, "overlap": True},
        "dump": {
            "reward_dir": str(tmp_path / "r"),
            "hidden_dir": str(tmp_path / "h"),
            "shard_name": "s.hdf5",
            "preprocess_config": {
                "action_head_type": "oft_discrete_token",
                "history": 1,
                "include_state": False,
                "hidden_key": "obs_embedding",
            },
            "data_attrs": {"task_suite_name": "synthetic", "env_name": "dc"},
        },
        "policy": {
            "cfg": {
                "target": "dreamervla.workers.actor._test_models:TinySharedPolicy",
                "kwargs": {"hidden_dim": 4, "action_dim": 7},
            }
        },
        "inference": {
            "cfg": {
                "encoder": {"target": "dreamervla.workers.inference._test_models:TinyEncoder"},
                "world_model": {
                    "target": "dreamervla.workers.inference._test_models:TinyWorldModel",
                    "kwargs": {"hidden_dim": 4, "action_dim": 7},
                },
                "device": "cpu",
            }
        },
    }
    history = ColdStartRayCollectRunner(cfg).run()
    assert history["rollout/episodes"] == 4
    assert history["time/overlap_events"] >= 1
    assert not ray.is_initialized()
