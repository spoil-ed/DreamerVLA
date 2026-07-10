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


def _compose_unresolved(overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="train", overrides=overrides)


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
    assert oft.actor_target == "dreamervla.algorithms.actor.OpenVLADiscreteTokenActor"
    assert oft.actor_head_type == "oft_discrete_token"

    input_tokens = oft.input_tokens
    assert input_tokens.expected_action_head_type == oft.expected_action_head_type
    assert input_tokens.expected_obs_hidden_source == "input_token_embedding"
    assert input_tokens.expected_prompt_style == oft.expected_prompt_style
    assert input_tokens.expected_include_state == oft.expected_include_state
    assert input_tokens.expected_history == oft.expected_history
    assert input_tokens.num_images_in_input == oft.num_images_in_input
    assert input_tokens.patches_per_image == 256
    assert input_tokens.token_count == oft.num_images_in_input * input_tokens.patches_per_image
    assert input_tokens.wm_obs_dim == input_tokens.token_count * input_tokens.token_dim
    assert "num_images_in_input*patches_per_image" in input_tokens.latent_source
    assert list(input_tokens.proprio_keys) == ["ee_pos", "ee_ori", "gripper_states"]
    assert input_tokens.proprio_dim == 8
    assert input_tokens.proprio_emb_dim == 10
    assert input_tokens.lang_dim == 4096
    assert input_tokens.lang_emb_dim == 32
    assert input_tokens.action_emb_dim == 10
    assert input_tokens.model_dim == 4096 + 10 + 32 + 10


@pytest.mark.parametrize("offline_task,_coldstart_task,_suite", MATRIX)
def test_openvla_traj1_input_token_dims_are_resolver_expressions(
    offline_task,
    _coldstart_task,
    _suite,
) -> None:
    raw_cfg = _compose_unresolved([f"task={offline_task}"])
    raw = OmegaConf.to_container(
        raw_cfg.task.openvla_oft.input_tokens,
        resolve=False,
    )
    cfg = _compose([f"task={offline_task}"])

    assert raw["token_count"] == (
        "${dvla_mul:${task.openvla_oft.input_tokens.num_images_in_input},"
        "${task.openvla_oft.input_tokens.patches_per_image}}"
    )
    assert raw["wm_obs_dim"] == (
        "${dvla_mul:${task.openvla_oft.input_tokens.token_count},"
        "${task.openvla_oft.input_tokens.token_dim}}"
    )
    assert cfg.task.openvla_oft.num_images_in_input == 1
    assert cfg.task.openvla_oft.input_tokens.token_count == 256
    assert cfg.task.openvla_oft.input_tokens.wm_obs_dim == 256 * 4096


def test_standard_h1_classifier_experiment_composes() -> None:
    cfg = _compose(["experiment=latent_classifier_openvla_onetraj_libero_goal_h1"])

    assert cfg._target_ == "dreamervla.runners.LatentClassifierRunner"
    assert cfg.task.hdf5_dir.endswith(
        "data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256"
    )
    assert cfg.task.openvla_oft.input_token_hidden_dir.endswith(
        "data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_oft_input_token_embedding_vla_policy_h1"
    )
    assert cfg.task.openvla_oft.action_hidden_dir.endswith(
        "data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1"
    )
    assert cfg.task.pretokenize_config_path.endswith(
        "data/configs/OpenVLA_Onetraj_LIBERO_libero_goal/"
        "his_1_third_view_wrist_w_state_1_256_pretokenize.yaml"
    )
    assert cfg.data.success_dir_raw.endswith(
        "data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256"
    )
    assert cfg.data.success_dir_hidden.endswith(
        "data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_oft_input_token_embedding_vla_policy_h1"
    )
    assert cfg.data.failure_dir_raw.endswith(
        "data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_failures"
    )
    assert cfg.data.failure_dir_hidden.endswith(
        "data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_failures_oft_input_token_embedding_vla_policy_h1"
    )
    assert cfg.data.lang_emb_dir == "__source_hidden__"
    assert cfg.data.chunk_subsample == cfg.task.openvla_oft.input_tokens.chunk_size == 8
    assert cfg.data.sampling_protocol == "wmpo"
    assert cfg.data.balance_batches is True
    assert cfg.training.loss_type == "bce"
    assert cfg.training.batch_size == 4
    assert cfg.training.episode_eval_enabled is True
    assert cfg.classifier.head_type == "spatial_tf"
    assert cfg.classifier.output_dim == 1
    assert cfg.classifier.token_count == 256
    assert cfg.classifier.latent_dim == 4138
    assert cfg.classifier.proprio_dim == 8
    assert cfg.classifier.lang_dim == 4096
    assert list(cfg.runner.logger.logger_backends) == ["tensorboard"]


def test_wmpo_token_h1_classifier_experiment_composes() -> None:
    cfg = _compose(["experiment=wmpo_token_classifier_openvla_onetraj_libero_goal_h1"])

    assert cfg._target_ == "dreamervla.runners.LatentClassifierRunner"
    assert cfg.training.episode_eval_enabled is True
    assert cfg.training.lr == 3.0e-5
    assert cfg.classifier.head_type == "spatial_tf"
    assert cfg.classifier.granularity == "chunk"
    assert cfg.classifier.output_dim == 1
    assert cfg.training.loss_type == "bce"
    assert cfg.data.sampling_protocol == "wmpo"
    assert cfg.data.balance_batches is True
    assert cfg.classifier.token_count == cfg.task.openvla_oft.input_tokens.token_count
    assert cfg.data.success_dir_hidden.endswith(
        "OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_oft_input_token_embedding_vla_policy_h1"
    )
    assert cfg.data.failure_dir_hidden.endswith(
        "OpenVLA_Onetraj_LIBERO_libero_goal/"
        "no_noops_t_256_failures_oft_input_token_embedding_vla_policy_h1"
    )


def test_openvla_onetraj_cotrain_noray_uses_wmpo_classifier_protocol() -> None:
    cfg = _compose(["experiment=openvla_onetraj_libero_cotrain_noray"])

    assert cfg._target_ == "dreamervla.runners.OnlineCotrainPipelineRunner"
    assert cfg.classifier.output_dim == 1
    assert cfg.online_rollout.classifier_loss_type == "bce"
    assert cfg.online_rollout.classifier_sampling_protocol == "wmpo"
    assert cfg.online_rollout.classifier_balance_batches is True
    assert cfg.algorithm.lumos.calibrate_threshold is True
    assert cfg.training.classifier_batch_size % 2 == 0


def test_openvla_onetraj_cotrain_ray_uses_wmpo_classifier_protocol() -> None:
    cfg = _compose(["experiment=openvla_onetraj_libero_cotrain_ray"])

    assert cfg._target_ == "dreamervla.runners.ManualCotrainRayRunner"
    assert cfg.ray_components.classifier.kwargs.output_dim == 1
    assert cfg.learner.train_cfg.classifier_loss_type == "bce"
    assert cfg.learner.train_cfg.classifier_sampling_protocol == "wmpo"
    assert cfg.learner.train_cfg.classifier_balance_batches is True
    assert cfg.learner.train_cfg.classifier_threshold is None
    assert cfg.learner.train_cfg.classifier_batch_size % 2 == 0


def test_openvla_onetraj_cotrain_ray_aligns_world_model_full_dataset_recipe() -> None:
    cfg = _compose(["experiment=openvla_onetraj_libero_cotrain_ray"])
    wm = cfg.ray_components.world_model.kwargs

    assert cfg.ray_data.replay_capacity == 160000
    assert cfg.ray_data.sequence_length == 36
    assert cfg.replay.cfg.capacity == 160000
    assert cfg.replay.cfg.sequence_length == 36
    assert cfg.learner.train_cfg.batch_size == 16
    assert cfg.learner.train_cfg.optimizers.world_model.lr == 3.0e-5
    assert wm.chunk_rollout_chunks == 4
    assert wm.chunk_rollout_loss_scale == 0.2
    assert wm.proprio_reconstruction_loss_scale == 0.0


@pytest.mark.parametrize("_offline_task,coldstart_task,suite", MATRIX)
def test_openvla_traj1_libero_tasks_define_vla_dataset_contract(
    _offline_task,
    coldstart_task,
    suite,
) -> None:
    coldstart = _compose(["experiment=collect_rollouts_onetraj", f"task={coldstart_task}"])

    assert coldstart.task.suite == suite
    _assert_openvla_traj1_contract(coldstart)
    assert "/collected_rollouts/" in str(coldstart.task.openvla_oft.hdf5_reward_dir)
    assert coldstart.task.openvla_oft.action_hidden_dir.endswith("_h1")
