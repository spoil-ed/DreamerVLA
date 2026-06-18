from __future__ import annotations


def test_oft_collect_common_exposes_shared_helpers() -> None:
    from dreamervla.runners.oft_collect_common import (
        assert_policy_mode_matches,
        load_policy,
        make_preprocess_config,
        resolve_num_images_in_input,
    )

    for fn in (
        load_policy,
        make_preprocess_config,
        assert_policy_mode_matches,
        resolve_num_images_in_input,
    ):
        assert callable(fn)


def test_runner_builds_bundle_cfg_from_central_config(tmp_path) -> None:
    from dreamervla.runners.cold_start_ray_collect_runner import ColdStartRayCollectRunner

    cfg = {
        "mode": "oft",
        "collect": {
            "num_images_in_input": 1,
            "episode_horizon": 8,
            "envs_per_gpu": 2,
            "episodes_per_task": 2,
            "task_ids": [0],
            "policy_mode": "discrete",
        },
        "task": {
            "suite": "libero_goal",
            "action_dim": 7,
            "image_resolution": 256,
            "image_keys": ["agentview_rgb", "eye_in_hand_rgb"],
            "openvla_oft": {
                "ckpt_path": str(tmp_path / "ckpt"),
                "dataset_statistics_key": "libero_goal_no_noops",
                "hdf5_reward_dir": str(tmp_path / "reward"),
                "action_hidden_dir": str(tmp_path / "hidden"),
                "expected_action_head_type": "oft_discrete_token",
                "expected_include_state": False,
                "expected_obs_hidden_source": "action_query",
                "expected_prompt_style": "vla_policy",
                "expected_history": 1,
                "expected_rotate_images_180": True,
                "time_horizon": 8,
                "token_dim": 4096,
                "chunk_size": 8,
            },
        },
    }
    plan = ColdStartRayCollectRunner(cfg).build_oft_worker_plan()
    assert plan["inference"]["decoder"]["target"].endswith("oft_rollout:OFTRolloutBundle")
    assert plan["inference"]["decoder"]["kwargs"]["history"] == 1
    assert plan["inference"]["decoder"]["kwargs"]["image_keys"] == ["agentview_rgb"]
    env_kwargs = plan["env"]["kwargs"]
    assert env_kwargs["history_length"] == 1
    assert env_kwargs["include_state"] is False
    assert env_kwargs["action_head_type"] == "oft_discrete_token"
    assert env_kwargs["validate_canonical"] is False
    assert plan["dump"]["preprocess_config"]["hidden_key"] == "obs_embedding"
    assert plan["dump"]["preprocess_config"]["action_head_type"] == "oft_discrete_token"
    assert plan["dump"]["preprocess_config"]["num_images_in_input"] == 1


def test_collect_rollouts_ray_experiment_composes() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=collect_rollouts_ray"])

    assert cfg._target_.endswith("ColdStartRayCollectRunner")
    assert cfg.mode == "oft"
    assert cfg.collect.num_images_in_input == 1
