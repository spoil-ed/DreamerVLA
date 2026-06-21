from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

MATRIX = [
    ("openvla_onetraj_libero", "openvla_onetraj_coldstart_libero", "libero_goal"),
    (
        "openvla_onetraj_libero_object",
        "openvla_onetraj_coldstart_libero_object",
        "libero_object",
    ),
    (
        "openvla_onetraj_libero_spatial",
        "openvla_onetraj_coldstart_libero_spatial",
        "libero_spatial",
    ),
    ("openvla_onetraj_libero_10", "openvla_onetraj_coldstart_libero_10", "libero_10"),
]


def _compose(overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _assert_openvla_traj1_contract(cfg) -> None:
    oft = cfg.task.openvla_oft
    assert oft.dataset_statistics_key == f"{cfg.task.suite}_no_noops"
    assert oft.expected_action_head_type == "oft_discrete_token"
    assert oft.expected_obs_hidden_source == "action_query"
    assert oft.expected_include_state is False
    assert oft.expected_history == 1
    assert oft.num_images_in_input == 1
    assert oft.use_proprio is False
    assert oft.use_wrist_image is False
    assert oft.use_l1_regression is False
    assert oft.wm_obs_dim == oft.token_count * oft.token_dim
    assert oft.actor_target == "dreamervla.models.actor.OpenVLADiscreteTokenActor"
    assert oft.actor_head_type == "oft_discrete_token"


@pytest.mark.parametrize("offline_task,coldstart_task,suite", MATRIX)
def test_openvla_traj1_libero_tasks_define_vla_dataset_contract(
    offline_task,
    coldstart_task,
    suite,
) -> None:
    offline = _compose(["experiment=openvla_oft_hdf5_one_trajectory", f"task={offline_task}"])
    coldstart = _compose(["experiment=collect_rollouts_onetraj", f"task={coldstart_task}"])

    assert offline.task.suite == suite
    assert coldstart.task.suite == suite
    _assert_openvla_traj1_contract(offline)
    _assert_openvla_traj1_contract(coldstart)
    assert "/processed_data/" in str(offline.task.openvla_oft.hdf5_reward_dir)
    assert "/collected_rollouts/" in str(coldstart.task.openvla_oft.hdf5_reward_dir)
    assert offline.task.openvla_oft.action_hidden_dir.endswith("_h1")
    assert coldstart.task.openvla_oft.action_hidden_dir.endswith("_h1")


@pytest.mark.parametrize("offline_task,coldstart_task,_suite", MATRIX)
@pytest.mark.parametrize(
    "experiment",
    [
        "openvla_oft_hdf5_one_trajectory",
        "oft_discrete_token_world_model_dinowm_chunk",
        "dreamervla_oft_discrete_token_dino_wm_wmpo_outcome",
        "online_cotrain_pipeline_oft_action_hidden",
    ],
)
def test_openvla_traj1_libero_experiments_derive_interfaces_from_task(
    experiment,
    offline_task,
    coldstart_task,
    _suite,
) -> None:
    selected_task = (
        coldstart_task
        if experiment == "online_cotrain_pipeline_oft_action_hidden"
        else offline_task
    )
    task_override = f"task={selected_task}"
    cfg = _compose([f"experiment={experiment}", task_override])
    oft = cfg.task.openvla_oft
    _assert_openvla_traj1_contract(cfg)

    if hasattr(cfg, "policy"):
        if "model_path" in cfg.policy:
            assert cfg.policy.model_path == oft.ckpt_path
            assert cfg.policy.num_images_in_input == oft.num_images_in_input
            assert cfg.policy.use_proprio == oft.use_proprio
            assert cfg.policy.use_l1_regression == oft.use_l1_regression
        else:
            assert cfg.policy._target_ == oft.actor_target
            assert cfg.policy.action_hidden_dim == oft.token_dim
            assert cfg.policy.time_horizon == oft.chunk_size
            assert cfg.policy.head_type == oft.actor_head_type
            assert cfg.policy.adapter_hidden_dim == oft.actor_adapter_hidden_dim

    if hasattr(cfg, "world_model"):
        assert cfg.world_model._target_ == oft.wm_target
        assert cfg.world_model.obs_dim == oft.wm_obs_dim
        assert cfg.world_model.token_count == oft.token_count
        assert cfg.world_model.token_dim == oft.token_dim
        assert cfg.world_model.chunk_size == oft.chunk_size
        assert cfg.world_model.time_horizon == oft.time_horizon

    if hasattr(cfg, "dataset") and "expected_obs_hidden_source" in cfg.dataset:
        assert cfg.dataset.hidden_dir == oft.action_hidden_dir
        assert cfg.dataset.expected_obs_hidden_source == oft.expected_obs_hidden_source
        assert cfg.dataset.expected_action_head_type == oft.expected_action_head_type
        assert cfg.dataset.expected_include_state == oft.expected_include_state
        assert cfg.dataset.expected_history == oft.expected_history

    if hasattr(cfg, "algorithm"):
        assert cfg.algorithm.wmpo.chunk_size == oft.chunk_size
