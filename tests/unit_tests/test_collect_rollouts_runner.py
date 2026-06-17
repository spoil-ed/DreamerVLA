from omegaconf import OmegaConf


def test_collect_rollouts_experiment_composes_and_validates():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamervla.config import validate_cfg

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts_onetraj",
                "task=OpenVLA_Onetraj_ColdStart_LIBERO",
            ],
        )
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)

    assert cfg._target_ == "dreamervla.runners.CollectRolloutsRunner"
    oft = cfg.task.openvla_oft
    assert str(oft.ckpt_path).endswith("Openvla-oft-SFT-libero-goal-traj1")
    assert oft.expected_action_head_type == "oft_discrete_token"
    assert oft.expected_include_state is False
    assert int(oft.expected_history) == 1
    assert int(oft.time_horizon) == 8
    assert str(oft.action_hidden_dir).endswith("_oft_legacy_action_hidden_vla_policy_h1")
    assert "OpenVLA_Onetraj_LIBERO_libero_goal" in str(oft.hdf5_reward_dir)
    assert cfg.collect.envs_per_gpu == 1


def _fake_cfg():
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.CollectRolloutsRunner",
            "task": {
                "suite": "libero_goal",
                "action_dim": 7,
                "image_resolution": 256,
                "image_keys": ["agentview_rgb", "eye_in_hand_rgb"],
                "openvla_oft": {
                    "ckpt_path": "data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1",
                    "dataset_statistics_key": "libero_goal_no_noops",
                    "hdf5_reward_dir": "data/processed_data/X/no_noops_t_256_remaining_reward",
                    "action_hidden_dir": "data/processed_data/X/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1",
                    "expected_action_head_type": "oft_discrete_token",
                    "expected_include_state": False,
                    "expected_obs_hidden_source": "action_query",
                    "expected_prompt_style": "vla_policy",
                    "expected_rotate_images_180": True,
                    "expected_history": 1,
                    "time_horizon": 8,
                    "token_dim": 4096,
                    "chunk_size": 8,
                },
            },
            "collect": {
                "policy_mode": "auto",
                "task_ids": "all",
                "episodes_per_task": 2,
                "episode_horizon": 64,
                "envs_per_gpu": 1,
            },
        }
    )


def test_build_collect_cfg_maps_task_and_collect():
    from dreamervla.runners import CollectRolloutsRunner

    runner = CollectRolloutsRunner(_fake_cfg())
    cc = runner._build_collect_cfg()

    assert cc["model_path"].endswith("Openvla-oft-SFT-libero-goal-traj1")
    assert cc["unnorm_key"] == "libero_goal_no_noops"
    assert cc["reward_dir"].endswith("no_noops_t_256_remaining_reward")
    assert cc["hidden_dir"].endswith("_oft_legacy_action_hidden_vla_policy_h1")
    assert cc["expected_history"] == 1
    assert cc["num_images_in_input"] == 2  # history(1) x views(2)
    assert cc["expected_action_head_type"] == "oft_discrete_token"
    assert cc["expected_include_state"] is False
    assert cc["time_horizon"] == 8
    assert cc["resolution"] == 256  # task.image_resolution, NOT image_size
    assert cc["task_suite_name"] == "libero_goal"
    assert cc["task_ids"] == "all"
    assert cc["envs_per_gpu"] == 1
    # every required key present
    from dreamervla.runners.collect_parallel_rollouts import _require_keys
    _require_keys(cc)
