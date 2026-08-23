from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.runners import RolloutCollectionRunner


def test_collect_rollouts_experiment_composes_and_validates() -> None:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts",
                "task=openvla_onetraj_coldstart_libero",
            ],
        )
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)

    assert cfg._target_ == "dreamervla.runners.RolloutCollectionRunner"
    assert cfg.collect.backend == "ray"
    assert cfg.env.cfg.render_backend == "osmesa"
    assert cfg.task.openvla_oft.hidden_token.expected_history == 1
    assert cfg.task.openvla_oft.hidden_token.token_count == 256


def _fake_cfg():
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.RolloutCollectionRunner",
            "task": {
                "suite": "libero_goal",
                "action_dim": 7,
                "image_resolution": 256,
                "image_keys": ["agentview_rgb"],
                "openvla_oft": {
                    "ckpt_path": "data/checkpoints/openvla-oft",
                    "dataset_statistics_key": "libero_goal_no_noops",
                    "hdf5_reward_dir": "data/rollouts/reward",
                    "hidden_token_dir": "data/rollouts/hidden",
                    "hidden_token": {
                        "expected_action_head_type": "oft_discrete_token",
                        "expected_include_state": False,
                        "expected_obs_hidden_source": "hidden_token",
                        "expected_prompt_style": "vla_policy",
                        "expected_rotate_images_180": True,
                        "expected_history": 1,
                        "time_horizon": 8,
                        "token_dim": 4096,
                        "token_count": 256,
                        "wm_obs_dim": 1_048_576,
                        "patches_per_image": 256,
                        "chunk_size": 8,
                    },
                },
            },
            "collect": {
                "backend": "ray",
                "policy_mode": "discrete",
                "task_ids": "all",
                "episodes_per_task": 2,
                "episode_horizon": 64,
                "envs_per_gpu": 2,
                "memory_fraction": 0.7,
            },
        }
    )


def test_build_collect_cfg_maps_task_and_collect() -> None:
    cfg = RolloutCollectionRunner(_fake_cfg())._build_collect_cfg()

    assert cfg["model_path"].endswith("openvla-oft")
    assert cfg["unnorm_key"] == "libero_goal_no_noops"
    assert cfg["reward_dir"].endswith("rollouts/reward")
    assert cfg["hidden_dir"].endswith("rollouts/hidden")
    assert cfg["expected_obs_hidden_source"] == "hidden_token"
    assert cfg["token_count"] == 256
    assert cfg["hidden_dim"] == 1_048_576
    assert cfg["patches_per_image"] == 256
    assert cfg["num_images_in_input"] == 1
    assert cfg["resolution"] == 256
    assert cfg["task_suite_name"] == "libero_goal"
    assert cfg["envs_per_gpu"] == 2
    assert cfg["demos_per_shard"] == 0


def test_build_collect_cfg_forwards_ray_worker_controls() -> None:
    cfg = _fake_cfg()
    cfg.collect.demos_per_shard = 25
    cfg.collect.num_inference_workers = 2

    resolved = RolloutCollectionRunner(cfg)._build_collect_cfg()

    assert resolved["demos_per_shard"] == 25
    assert resolved["num_inference_workers"] == 2
