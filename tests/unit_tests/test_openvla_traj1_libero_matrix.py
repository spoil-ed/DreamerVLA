from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg

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

MAINLINE_ROUTES = (
    ("collect_rollouts", "coldstart", "dreamervla.runners.RolloutCollectionRunner"),
    (
        "openvla_onetraj_libero_cotrain",
        "offline",
        "dreamervla.runners.CotrainRunner",
    ),
)


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
    assert ("input_" + "tokens") not in oft
    assert oft.num_images_in_input == 1
    assert oft.use_proprio is False
    assert oft.use_wrist_image is False
    assert oft.use_l1_regression is False
    assert (
        oft.actor_target
        == "dreamervla.algorithms.actor.LatentToOpenVLAHiddenStateActor"
    )
    assert oft.actor_head_type == "oft_discrete_token"

    hidden_token = oft.hidden_token
    assert hidden_token.expected_action_head_type == "oft_discrete_token"
    assert hidden_token.expected_obs_hidden_source == "hidden_token"
    assert hidden_token.expected_prompt_style == "vla_policy"
    assert hidden_token.expected_include_state is False
    assert hidden_token.expected_history == 1
    assert hidden_token.num_images_in_input == oft.num_images_in_input
    assert hidden_token.patches_per_image == 256
    assert hidden_token.token_count == oft.num_images_in_input * hidden_token.patches_per_image
    assert hidden_token.wm_obs_dim == hidden_token.token_count * hidden_token.token_dim
    assert "[256,4096]" in hidden_token.latent_source
    assert list(hidden_token.proprio_keys) == ["ee_pos", "ee_ori", "gripper_states"]
    assert hidden_token.proprio_dim == 8
    assert hidden_token.proprio_emb_dim == 10
    assert hidden_token.lang_dim == 4096
    assert hidden_token.lang_emb_dim == 32
    assert hidden_token.action_emb_dim == 10
    assert hidden_token.model_dim == 4096 + 10 + 32 + 10


@pytest.mark.parametrize("offline_task,coldstart_task,_suite", MATRIX)
@pytest.mark.parametrize("experiment,task_kind,target", MAINLINE_ROUTES)
def test_every_mainline_route_suite_composition_has_exact_hidden_token_contract(
    offline_task,
    coldstart_task,
    _suite,
    experiment,
    task_kind,
    target,
    tmp_path,
) -> None:
    task = coldstart_task if task_kind == "coldstart" else offline_task
    overrides = [f"experiment={experiment}", f"task={task}"]
    cfg = _compose(overrides)

    validate_cfg(cfg)
    assert cfg._target_ == target
    _assert_openvla_traj1_contract(cfg)

    if experiment == "collect_rollouts":
        assert cfg.collect.policy_mode == "discrete"
        assert cfg.collect.num_images_in_input == 1
    else:
        assert cfg.ray_components.world_model.kwargs.token_count == 256
        assert cfg.ray_components.world_model.kwargs.token_dim == 4096
        assert cfg.ray_components.world_model.kwargs.token_normalization == "layer_norm"
        assert cfg.ray_components.world_model.kwargs.token_norm_eps == 1.0e-6
        assert cfg.ray_components.world_model.kwargs.obs_dim == 1_048_576
        assert cfg.ray_components.classifier.kwargs.token_count == 256
        assert cfg.ray_components.classifier.kwargs.token_dim == 4096
        assert (
            cfg.ray_components.policy.target
            == "dreamervla.models.embodiment.OpenVLAOFTPolicy"
        )
        assert cfg.ray_components.policy.kwargs.model_path == cfg.task.openvla_oft.ckpt_path
        assert cfg.ray_components.policy.kwargs.num_images_in_input == 1
        assert cfg.ray_components.policy.kwargs.use_lora is False
        assert "source_token_count" not in cfg.ray_components.policy.kwargs
        assert cfg.rollout.encoder_cfg is None
        assert cfg.env.wm.cfg.kwargs.token_count == 256
        assert cfg.env.wm.cfg.kwargs.token_dim == 4096


@pytest.mark.parametrize("offline_task,_coldstart_task,_suite", MATRIX)
def test_openvla_traj1_hidden_token_dims_are_resolver_expressions(
    offline_task,
    _coldstart_task,
    _suite,
) -> None:
    raw_cfg = _compose_unresolved([f"task={offline_task}"])
    raw = OmegaConf.to_container(
        raw_cfg.task.openvla_oft.hidden_token,
        resolve=False,
    )
    cfg = _compose([f"task={offline_task}"])

    assert raw["token_count"] == (
        "${dvla_mul:${task.openvla_oft.hidden_token.num_images_in_input},"
        "${task.openvla_oft.hidden_token.patches_per_image}}"
    )
    assert raw["wm_obs_dim"] == (
        "${dvla_mul:${task.openvla_oft.hidden_token.token_count},"
        "${task.openvla_oft.hidden_token.token_dim}}"
    )
    assert cfg.task.openvla_oft.num_images_in_input == 1
    assert cfg.task.openvla_oft.hidden_token.token_count == 256
    assert cfg.task.openvla_oft.hidden_token.wm_obs_dim == 256 * 4096


def test_wmpo_token_h1_classifier_experiment_composes() -> None:
    cfg = _compose(["experiment=wmpo_token_classifier_openvla_onetraj_libero_goal_h1"])

    assert cfg._target_ == "dreamervla.runners.SuccessClassifierTrainingRunner"
    assert cfg.training.episode_eval_enabled is True
    assert cfg.training.lr == 3.0e-5
    assert cfg.classifier.head_type == "spatial_tf"
    assert (
        cfg.classifier._target_
        == "dreamervla.algorithms.critic.LatentSuccessClassifier"
    )
    assert (
        cfg.task.classifier.dataset.train._target_
        == "dreamervla.dataset.LumosAlignedLatentTrainDataset"
    )
    assert (
        cfg.task.classifier.dataset.validation._target_
        == "dreamervla.dataset.LumosAlignedLatentValDataset"
    )
    assert cfg.classifier.granularity == "chunk"
    assert cfg.classifier.output_dim == 1
    assert cfg.training.loss_type == "bce"
    assert cfg.data.sampling_protocol == "wmpo"
    assert cfg.data.balance_batches is True
    assert cfg.classifier.token_count == cfg.task.openvla_oft.hidden_token.token_count
    assert cfg.classifier.latent_dim == 4138
    assert cfg.classifier.proprio_dim == 8
    assert cfg.classifier.lang_dim == 4096
    assert cfg.data.success_dir_raw == cfg.task.collected_reward_dir
    assert cfg.data.success_dir_hidden == cfg.task.collected_hidden_token_dir
    assert cfg.data.failure_dir_raw is None
    assert cfg.data.failure_dir_hidden is None
    assert cfg.data.lang_emb_dir == "__source_hidden__"
    assert list(cfg.runner.logger.logger_backends) == ["tensorboard"]


def test_dreamer_classifier_model_and_dataset_are_hydra_selected() -> None:
    cfg = _compose_unresolved(
        ["experiment=wmpo_token_classifier_openvla_onetraj_libero_goal_h1"]
    )
    classifier_recipe = Path(__file__).resolve().parents[2] / "configs" / "classifier"

    assert {path.name for path in classifier_recipe.glob("*.yaml")} == {
        "dreamer-cls.yaml"
    }
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert raw["classifier"]["_target_"] == "${task.classifier.model._target_}"
    assert cfg.task.classifier.model._target_ == (
        "dreamervla.algorithms.critic.LatentSuccessClassifier"
    )
    assert cfg.task.classifier.dataset.train._target_ == (
        "dreamervla.dataset.LumosAlignedLatentTrainDataset"
    )
    assert cfg.task.classifier.dataset.validation._target_ == (
        "dreamervla.dataset.LumosAlignedLatentValDataset"
    )
    assert "classifier_target" not in cfg.task.openvla_oft


def test_openvla_onetraj_cotrain_uses_wmpo_classifier_protocol() -> None:
    cfg = _compose(["experiment=openvla_onetraj_libero_cotrain"])

    assert cfg._target_ == "dreamervla.runners.CotrainRunner"
    assert cfg.ray_components.classifier.kwargs.output_dim == 1
    assert cfg.learner.train_cfg.classifier_loss_type == "bce"
    assert cfg.learner.train_cfg.classifier_sampling_protocol == "wmpo"
    assert cfg.learner.train_cfg.classifier_balance_batches is True
    assert cfg.learner.train_cfg.classifier_threshold is None
    assert cfg.learner.train_cfg.classifier_batch_size % 2 == 0


def test_cotrain_components_are_selected_from_worldmodel_and_classifier_groups() -> None:
    cfg = _compose(["experiment=openvla_libero"])

    assert cfg.manual_cotrain.training_mode == "failure_imagined_rl"
    assert cfg.manual_cotrain.initial_condition_selector == "failed_episode_start"
    assert cfg.manual_cotrain.learner_updates_enabled is False
    assert cfg.manual_cotrain.staged_policy_update is False
    assert cfg.manual_cotrain.save_replay_state is True
    assert cfg.env.wm.cfg.initial_condition_selector == "failed_episode_start"
    assert cfg.env.wm.cfg.bootstrap_group_size == cfg.algorithm.group_size
    assert cfg.ray_components.world_model.target == cfg.world_model._target_
    assert cfg.ray_components.classifier.target == cfg.classifier._target_
    for key in (
        "token_count",
        "token_dim",
        "token_normalization",
        "model_dim",
        "num_hist",
        "depth",
    ):
        assert cfg.ray_components.world_model.kwargs[key] == cfg.world_model[key]
    for key in (
        "token_count",
        "token_dim",
        "head_type",
        "window",
        "hidden_dim",
        "num_layers",
    ):
        assert cfg.ray_components.classifier.kwargs[key] == cfg.classifier[key]


def test_wmcls_cotrain_offloads_rollout_vla_only() -> None:
    cfg = _compose(["experiment=openvla_libero"])

    assert cfg.rollout.train_cfg.enable_offload is True
    assert cfg.actor.train_cfg.fsdp.cpu_offload is False
    assert "wrap_policy" not in cfg.actor.train_cfg.fsdp


def test_openvla_onetraj_cotrain_aligns_world_model_full_dataset_recipe() -> None:
    cfg = _compose(["experiment=openvla_onetraj_libero_cotrain"])
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
    coldstart = _compose(["experiment=collect_rollouts", f"task={coldstart_task}"])

    assert coldstart.task.suite == suite
    _assert_openvla_traj1_contract(coldstart)
    assert "/collected_rollouts/" in str(coldstart.task.openvla_oft.hdf5_reward_dir)
    assert coldstart.task.openvla_oft.hidden_token_dir.endswith("_h1")
