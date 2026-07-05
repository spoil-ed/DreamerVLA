from __future__ import annotations

import json

import h5py
import numpy as np
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
    assert history["time/driver_roundtrips"] > 0
    assert history["time/driver_step_waits"] == history["rollout/steps"]
    assert history["time/driver_step_calls"] > history["time/driver_step_waits"]
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
    assert history["time/overlap_events"] >= (
        history["rollout/steps"] - history["env/num_env_workers"]
    )
    assert history["time/infer_wait_s"] >= 0.0
    assert history["time/env_step_wait_s"] >= 0.0
    assert history["time/dump_wait_s"] >= 0.0
    assert history["time/env_ready_batches"] >= 1
    assert not ray.is_initialized()


def test_fake_coldstart_50pct_success_seeds_cotrain_warmup(tmp_path, monkeypatch) -> None:
    import h5py
    import torch

    from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner
    from dreamervla.runners.offline_seed import seed_replay_from_offline
    from dreamervla.runners.online_replay import OnlineReplay

    if ray.is_initialized():
        ray.shutdown()

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    cfg = {
        "env": {
            "num_workers": 2,
            "cfg": {
                "target": "dreamervla.workers.env._test_envs:AlternatingSuccessDumpEnv",
                "kwargs": {"horizon": 4, "image_shape": (4, 4, 3), "embedding_dim": 4},
            },
        },
        "rollout": {"target_episodes": 4, "max_steps": 16},
        "dump": {
            "reward_dir": str(reward_dir),
            "hidden_dir": str(hidden_dir),
            "shard_name": "fake_flow.hdf5",
            "preprocess_config": {
                "action_head_type": "oft_discrete_token",
                "history": 1,
                "include_state": False,
                "hidden_key": "obs_embedding",
            },
            "data_attrs": {"task_suite_name": "fake", "env_name": "alternating_success"},
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
    assert not ray.is_initialized()

    reward_path = reward_dir / "fake_flow.hdf5"
    with h5py.File(reward_path, "r") as handle:
        demos = [handle["data"][key] for key in sorted(handle["data"])]
        successes = [bool(np.asarray(demo["sparse_rewards"])[-1]) for demo in demos]
    assert successes.count(True) == 2
    assert successes.count(False) == 2
    assert sum(successes) / len(successes) == 0.5

    replay = OnlineReplay(capacity=10_000, sequence_length=3, task_ids=(0,), rank=0)
    added = seed_replay_from_offline(
        replay,
        data_dir=reward_dir,
        hidden_dir=hidden_dir,
        default_task_id=0,
    )
    stats = replay.task_stats((0,))["0"]
    assert added == 4
    assert stats["episodes"] == 4
    assert stats["successes"] == 2
    assert stats["failures"] == 2
    batch = replay.sample(4)
    assert batch["obs_embedding"].shape == (4, 3, 4)
    assert batch["current_actions"].shape[-1] == 7

    import dreamervla.runners.online_cotrain_pipeline_runner as pipeline

    calls = {"wm": 0, "cls": 0}

    def fake_wm_step(**kwargs):
        assert kwargs["batch"]["obs_embedding"].shape[-1] == 4
        calls["wm"] += 1
        return {"loss": 0.1}

    def fake_cls_step(**kwargs):
        assert kwargs["replay"] is replay
        calls["cls"] += 1
        return {"loss": 0.2, "acc": 0.5}

    monkeypatch.setattr(pipeline, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(pipeline, "online_classifier_update_step", fake_cls_step)

    runner = pipeline.OnlineCotrainPipelineRunner.__new__(pipeline.OnlineCotrainPipelineRunner)
    runner.device = torch.device("cpu")
    runner.global_step = 0
    runner._build_wm_pretrain_batch = lambda sampled: {
        "images": sampled["images"],
        "obs_embedding": sampled["obs_embedding"],
        "actions": sampled["actions"],
    }
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner.classifier = torch.nn.Module()
    runner.classifier_optimizer = object()
    runner._cls_window = 1

    runner._offline_warmup_wm(replay, steps=1, batch_size=2, optim_cfg=None)
    runner._offline_warmup_classifier(
        replay,
        steps=1,
        batch_size=2,
        early_neg_stride=1,
        grad_clip=1.0,
    )
    assert calls == {"wm": 1, "cls": 1}
