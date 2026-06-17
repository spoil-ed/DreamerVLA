def test_collect_rollouts_experiment_composes_and_validates():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

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
