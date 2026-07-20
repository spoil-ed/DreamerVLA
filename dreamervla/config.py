from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.algorithms.registry import get_actor_update_route
from dreamervla.algorithms.validation import (
    validate_ppo_hyperparameters,
    validate_tdmpc_hyperparameters,
)
from dreamervla.models.registry import validate_model_type
from dreamervla.preprocess.sidecar_schema import (
    HIDDEN_TOKEN_ACTION_HEAD,
    HIDDEN_TOKEN_SOURCE,
)
from dreamervla.utils.metric_logger import MetricLogger
from dreamervla.utils.paths import data_root
from dreamervla.workers.cotrain.config_placement import (
    build_manual_cotrain_placement_from_config,
)

_LIBERO_GOAL_OFFICIAL_SHARDS = (
    "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
    "put_the_bowl_on_the_stove_demo.hdf5",
    "put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5",
    "open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
    "put_the_bowl_on_top_of_the_cabinet_demo.hdf5",
    "push_the_plate_to_the_front_of_the_stove_demo.hdf5",
    "put_the_cream_cheese_in_the_bowl_demo.hdf5",
    "turn_on_the_stove_demo.hdf5",
    "put_the_bowl_on_the_plate_demo.hdf5",
    "put_the_wine_bottle_on_the_rack_demo.hdf5",
)


def validate_cfg(cfg: DictConfig, *, world_size: int | None = None) -> DictConfig:
    """Validate high-value Dreamer-VLA config invariants before runner setup.

    The validation is intentionally lightweight: relationship checks are always
    enabled, while filesystem existence checks are opt-in via
    ``validation.require_existing_paths=true`` so config composition remains
    usable on machines without the full dataset mounted.
    """
    _validate_logger_backends(cfg)
    _validate_algorithm_routes(cfg)
    _validate_algorithm_hyperparameters(cfg)
    _validate_training_batch(cfg, world_size=_resolve_world_size(world_size))
    _validate_precision_controls(cfg)
    _validate_resume_paths(cfg)
    _validate_removed_observation_routes(cfg)
    _validate_mainline_hidden_token_contract(cfg)
    _validate_pre_mainline_routes(cfg)
    _validate_sidecar_routes(cfg)
    _validate_chunk_horizon_consistency(cfg)
    _validate_latent_dimension_contracts(cfg)
    _validate_model_registry_refs(cfg)
    _validate_dino_token_training(cfg)
    _validate_epoch_checkpoint_cadence(cfg)
    _validate_world_model_training_pipeline(cfg)
    _validate_ray_manual_resources(cfg)
    _validate_fsdp_config(cfg)
    if bool(OmegaConf.select(cfg, "validation.require_existing_paths", default=False)):
        _validate_existing_paths(cfg)
    return cfg


def _validate_precision_controls(cfg: DictConfig) -> None:
    """Validate separately configured compute and master-parameter dtypes."""

    compute_values = {"fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}
    parameter_values = {"fp32", "float32", "bf16", "bfloat16"}
    for path in ("optim.precision", "learner.train_cfg.precision"):
        value = OmegaConf.select(cfg, path, default=None)
        if value is not None and str(value).strip().lower() not in compute_values:
            raise ValueError(f"{path} must be one of fp32, bf16, or fp16; got {value!r}")
    for path in ("optim.param_precision", "learner.train_cfg.param_precision"):
        value = OmegaConf.select(cfg, path, default=None)
        if value is not None and str(value).strip().lower() not in parameter_values:
            raise ValueError(f"{path} must be fp32 or bf16; got {value!r}")


def _validate_logger_backends(cfg: DictConfig) -> None:
    backends = _normalize_backends(
        OmegaConf.select(cfg, "runner.logger.logger_backends", default=None)
    )
    unsupported = [backend for backend in backends if backend not in MetricLogger.supported_logger]
    if unsupported:
        raise ValueError(
            "runner.logger.logger_backends contains unsupported backend(s): "
            f"{unsupported}. Supported backends: {MetricLogger.supported_logger}"
        )


def _validate_algorithm_routes(cfg: DictConfig) -> None:
    update_type = OmegaConf.select(cfg, "algorithm.update_type", default=None)
    if update_type in (None, "", "dreamer"):
        return
    get_actor_update_route(str(update_type))


def _validate_algorithm_hyperparameters(cfg: DictConfig) -> None:
    for prefix in (
        "algorithm",
        "actor.train_cfg.algorithm_cfg",
        "learner.train_cfg.algorithm_cfg",
    ):
        section = OmegaConf.select(cfg, prefix, default=None)
        if section is not None:
            validate_ppo_hyperparameters(section, prefix=prefix)

    tdmpc_prefix = "eval.tdmpc_mpc"
    tdmpc = OmegaConf.select(cfg, tdmpc_prefix, default=None)
    if tdmpc is not None and bool(OmegaConf.select(tdmpc, "enabled", default=False)):
        validate_tdmpc_hyperparameters(tdmpc, prefix=tdmpc_prefix)


def _validate_training_batch(cfg: DictConfig, *, world_size: int) -> None:
    batch_size = OmegaConf.select(cfg, "dataloader.batch_size", default=None)
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError(f"dataloader.batch_size must be > 0, got {batch_size!r}")

    grad_accum = int(OmegaConf.select(cfg, "training.gradient_accumulate_every", default=1) or 1)
    if grad_accum <= 0:
        raise ValueError(f"training.gradient_accumulate_every must be > 0, got {grad_accum!r}")

    global_batch_size = OmegaConf.select(cfg, "training.global_batch_size", default=None)
    if global_batch_size is None:
        return

    global_batch_size = int(global_batch_size)
    divisor = max(1, int(world_size)) * grad_accum
    if global_batch_size <= 0:
        raise ValueError(f"training.global_batch_size must be > 0, got {global_batch_size!r}")
    if global_batch_size % divisor != 0:
        raise ValueError(
            "training.global_batch_size must be divisible by "
            "world_size * training.gradient_accumulate_every "
            f"({global_batch_size} % {divisor} != 0)"
        )


def _validate_resume_paths(cfg: DictConfig) -> None:
    if not bool(OmegaConf.select(cfg, "training.resume", default=False)):
        return
    for key in ("training.resume_path", "training.resume_dir"):
        value = OmegaConf.select(cfg, key, default=None)
        if value in (None, "", "auto"):
            continue
        if not Path(str(value)).expanduser().exists():
            raise ValueError(f"{key} does not exist: {value}")


def _validate_removed_observation_routes(cfg: DictConfig) -> None:
    """Reject configuration surfaces that could revive the 56x1024 route."""

    removed_sections = (
        "task.hidden_token_dir",
        "task.hidden_token_tokens",
        "task.hidden_token_specs",
        "task.openvla_oft.action_hidden_dir",
        "task.openvla_oft.action_head_ckpt",
        "task.openvla_oft.proprio_projector_ckpt",
        "task.openvla_oft.component_ckpt_dir",
        "task.openvla_oft.resume_step",
        "encoder.action_head_ckpt",
        "encoder.proprio_projector_ckpt",
        "encoder.component_ckpt_dir",
        "encoder.resume_step",
        "latent_type",
    )
    missing = object()
    present = [
        key
        for key in removed_sections
        if OmegaConf.select(cfg, key, default=missing) is not missing
    ]
    if present:
        raise ValueError(
            "removed action-query/hidden-token configuration is not supported: "
            + ", ".join(present)
        )

    source_paths = (
        "dataset.expected_obs_hidden_source",
        "env.obs_hidden_source",
        "eval.obs_hidden_source",
        "collect.oft_latent_spec.expected_obs_hidden_source",
        "task.openvla_oft.expected_obs_hidden_source",
        "task.openvla_oft.hidden_token.expected_obs_hidden_source",
    )
    for key in source_paths:
        value = _select_str(cfg, key)
        if value == "action_query":
            raise ValueError(f"{key}={value!r} is removed; use {HIDDEN_TOKEN_SOURCE!r}")

    action_head_paths = (
        "dataset.expected_action_head_type",
        "env.action_head_type",
        "encoder.action_head_type",
        "task.openvla_oft.expected_action_head_type",
        "task.openvla_oft.hidden_token.expected_action_head_type",
    )
    for key in action_head_paths:
        if _select_str(cfg, key) == "action_query":
            raise ValueError(f"{key}='action_query' is removed; use {HIDDEN_TOKEN_ACTION_HEAD!r}")

    target_paths = (
        "encoder._target_",
        "policy._target_",
        "task.openvla_oft.actor_target",
        "ray_components.policy.target",
        "ray_components.policy._target_",
        "actor.policy_cfg.target",
        "actor.policy_cfg._target_",
        "rollout.policy_cfg.target",
        "rollout.policy_cfg._target_",
        "learner.model_cfg.policy.target",
        "learner.model_cfg.policy._target_",
    )
    removed_target_fragments = (
        "RynnVLA",
        "LatentToHiddenTokenActor",
        "OpenVLADiscreteTokenActor",
        "VLAActionHeadActor",
    )
    for key in target_paths:
        target = _select_str(cfg, key)
        if target and any(fragment in target for fragment in removed_target_fragments):
            raise ValueError(f"{key} points to removed observation interface: {target}")

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    dimension_fields = {
        "obs_dim",
        "latent_dim",
        "hidden_dim",
        "wm_obs_dim",
        "flat_dim",
    }
    token_count_fields = {"token_count", "source_token_count"}
    for path, value in _iter_config_nodes(resolved):
        field = path.rsplit(".", 1)[-1]
        if field in dimension_fields and _is_exact_int(value, 56 * 1024):
            raise ValueError(f"{path} exposes the removed 56x1024 observation interface")
        if field in token_count_fields and _is_exact_int(value, 56):
            raise ValueError(f"{path} exposes the removed 56-token observation interface")
        if field == "obs_embedding_shape" and _int_sequence(value) == [56, 1024]:
            raise ValueError(f"{path} exposes the removed 56x1024 observation interface")


def _validate_mainline_hidden_token_contract(cfg: DictConfig) -> None:
    """Validate task-derived OpenVLA token geometry across active components."""

    if OmegaConf.select(cfg, "task.openvla_oft", default=None) is None:
        return

    exact_values: dict[str, Any] = {
        "task.openvla_oft.expected_action_head_type": HIDDEN_TOKEN_ACTION_HEAD,
        "task.openvla_oft.expected_obs_hidden_source": HIDDEN_TOKEN_SOURCE,
        "task.openvla_oft.expected_history": 1,
        "task.openvla_oft.expected_include_state": False,
        "task.openvla_oft.num_images_in_input": 1,
        "task.openvla_oft.use_wrist_image": False,
        "task.openvla_oft.use_proprio": False,
        "task.openvla_oft.use_l1_regression": False,
        "task.openvla_oft.hidden_token.expected_action_head_type": HIDDEN_TOKEN_ACTION_HEAD,
        "task.openvla_oft.hidden_token.expected_obs_hidden_source": HIDDEN_TOKEN_SOURCE,
        "task.openvla_oft.hidden_token.expected_history": 1,
        "task.openvla_oft.hidden_token.expected_include_state": False,
        "task.openvla_oft.hidden_token.num_images_in_input": 1,
    }
    for key, expected in exact_values.items():
        got = OmegaConf.select(cfg, key, default=None)
        if got != expected:
            raise ValueError(f"{key} must be {expected!r}, got {got!r}")

    geometry_root = "task.openvla_oft.hidden_token"
    patches_per_image = _select_int(cfg, f"{geometry_root}.patches_per_image")
    token_count = _select_int(cfg, f"{geometry_root}.token_count")
    token_dim = _select_int(cfg, f"{geometry_root}.token_dim")
    wm_obs_dim = _select_int(cfg, f"{geometry_root}.wm_obs_dim")
    num_images = _select_int(cfg, f"{geometry_root}.num_images_in_input")
    geometry = {
        "patches_per_image": patches_per_image,
        "token_count": token_count,
        "token_dim": token_dim,
        "wm_obs_dim": wm_obs_dim,
        "num_images_in_input": num_images,
    }
    missing = [name for name, value in geometry.items() if value is None]
    if missing:
        raise ValueError(f"{geometry_root} is missing geometry fields: {missing!r}")
    non_positive = [name for name, value in geometry.items() if int(value) <= 0]
    if non_positive:
        raise ValueError(f"{geometry_root} geometry must be positive: {non_positive!r}")
    assert patches_per_image is not None
    assert token_count is not None
    assert token_dim is not None
    assert wm_obs_dim is not None
    assert num_images is not None
    expected_token_count = patches_per_image * num_images
    if token_count != expected_token_count:
        raise ValueError(
            f"{geometry_root}.token_count must equal patches_per_image * "
            f"num_images_in_input ({expected_token_count}), got {token_count}"
        )
    expected_obs_dim = token_count * token_dim
    if wm_obs_dim != expected_obs_dim:
        raise ValueError(
            f"{geometry_root}.wm_obs_dim must equal token_count * token_dim "
            f"({expected_obs_dim}), got {wm_obs_dim}"
        )
    task_num_images = _select_int(cfg, "task.openvla_oft.num_images_in_input")
    if task_num_images != num_images:
        raise ValueError(
            "task.openvla_oft.num_images_in_input must match hidden-token metadata: "
            f"task={task_num_images}, hidden_token={num_images}"
        )

    hidden_token_dir = _select_str(cfg, "task.openvla_oft.hidden_token_dir")
    if hidden_token_dir is None or "hidden_token" not in hidden_token_dir:
        raise ValueError("task.openvla_oft.hidden_token_dir must name the hidden_token sidecar")

    component_specs = (
        ("world_model", "obs_dim"),
        ("classifier", None),
        ("ray_components.world_model.kwargs", "obs_dim"),
        ("ray_components.classifier.kwargs", None),
        ("learner.model_cfg.world_model.kwargs", "obs_dim"),
        ("learner.model_cfg.classifier.kwargs", None),
        ("inference.cfg.world_model.kwargs", "obs_dim"),
        ("env.wm.cfg.kwargs", "latent_dim"),
    )
    for key, obs_dim_field in component_specs:
        if OmegaConf.select(cfg, key, default=None) is None:
            continue
        component_token_count = _select_int(cfg, f"{key}.token_count")
        component_token_dim = _select_int(cfg, f"{key}.token_dim")
        if component_token_count is not None and component_token_count != token_count:
            raise ValueError(
                f"{key}.token_count must match task metadata {token_count}, "
                f"got {component_token_count}"
            )
        if component_token_dim is not None and component_token_dim != token_dim:
            raise ValueError(
                f"{key}.token_dim must match task metadata {token_dim}, got {component_token_dim}"
            )
        if obs_dim_field is not None:
            obs_dim = _select_int(cfg, f"{key}.{obs_dim_field}")
            if obs_dim is not None and obs_dim != wm_obs_dim:
                raise ValueError(
                    f"{key}.{obs_dim_field} must match task metadata {wm_obs_dim}, got {obs_dim}"
                )

    for key in (
        "policy.source_token_count",
        "ray_components.policy.kwargs.source_token_count",
        "actor.policy_cfg.kwargs.source_token_count",
    ):
        value = _select_int(cfg, key)
        if value is not None and value != token_count:
            raise ValueError(f"{key} must match task metadata {token_count}, got {value}")
    for key in (
        "policy.source_token_dim",
        "ray_components.policy.kwargs.source_token_dim",
        "actor.policy_cfg.kwargs.source_token_dim",
    ):
        value = _select_int(cfg, key)
        if value is not None and value != token_dim:
            raise ValueError(f"{key} must match task metadata {token_dim}, got {value}")

    for key in (
        "collect.policy_mode",
        "rollout.encoder_cfg.kwargs.policy_cfg.policy_mode",
        "inference.cfg.policy.policy_mode",
    ):
        value = _select_str(cfg, key)
        if value is not None and value != "discrete":
            raise ValueError(f"{key} must be 'discrete', got {value!r}")


def _validate_sidecar_routes(cfg: DictConfig) -> None:
    dataset_hidden = _select_str(cfg, "dataset.hidden_dir")
    if dataset_hidden is None:
        return

    oft_hidden_token = _select_str(cfg, "task.openvla_oft.hidden_token_dir")
    if oft_hidden_token is not None and dataset_hidden != oft_hidden_token:
        raise ValueError(
            "dataset.hidden_dir must match task.openvla_oft.hidden_token_dir "
            f"for OpenVLA-OFT hidden-token routes: {dataset_hidden!r} != "
            f"{oft_hidden_token!r}"
        )


def _validate_pre_mainline_routes(cfg: DictConfig) -> None:
    """Keep independent WM/classifier stages on canonical official data."""

    stage = _select_str(cfg, "pre_mainline.stage")
    if stage is None:
        return
    if _select_str(cfg, "task.suite") != "libero_goal":
        raise ValueError(
            "independent component training currently supports only task.suite=libero_goal"
        )
    if _select_str(cfg, "pre_mainline.suite") != "libero_goal":
        raise ValueError(
            "independent component training requires pre_mainline=libero_goal_official"
        )
    artifact_name = _select_str(cfg, "task.artifact_name")
    if artifact_name != "OpenVLA_Onetraj_LIBERO_libero_goal":
        raise ValueError(
            "the pre-mainline route requires the canonical official LIBERO goal artifact"
        )
    official_task_ids = [
        int(value)
        for value in (
            OmegaConf.select(
                cfg,
                "pre_mainline.official_task_ids",
                default=[],
            )
            or []
        )
    ]
    official_filenames = tuple(
        str(value)
        for value in (
            OmegaConf.select(
                cfg,
                "pre_mainline.official_reward_filenames",
                default=[],
            )
            or []
        )
    )
    if official_task_ids != list(range(10)):
        raise ValueError("canonical official LIBERO metadata requires task IDs [0..9]")
    if official_filenames != _LIBERO_GOAL_OFFICIAL_SHARDS:
        raise ValueError("canonical official LIBERO metadata requires all ten reward shards")
    canonical_processed_root = data_root().expanduser().resolve() / "processed_data" / artifact_name
    canonical_paths = {
        "task.hdf5_reward_dir": canonical_processed_root / "no_noops_t_256_remaining_reward",
        "task.openvla_oft.hidden_token_dir": canonical_processed_root
        / "no_noops_t_256_oft_hidden_token_vla_policy_h1",
    }
    for key, expected_path in canonical_paths.items():
        actual = _select_str(cfg, key)
        if actual is None or Path(actual).expanduser().resolve() != expected_path.resolve():
            raise ValueError(
                f"{key} must use the canonical official LIBERO path: "
                f"{actual!r} != {str(expected_path.resolve())!r}"
            )
    if stage == "wm_upper_bound":
        path_pairs = (
            ("offline_warmup.data_dir", "task.hdf5_reward_dir"),
            ("offline_warmup.hidden_dir", "task.openvla_oft.hidden_token_dir"),
        )
        if _select_str(cfg, "_target_") != ("dreamervla.runners.WorldModelTrainingRunner"):
            raise ValueError("independent WM training must use WorldModelTrainingRunner")
        if int(_select_int(cfg, "training.wm_warmup_steps") or 0) <= 0:
            raise ValueError("pre-mainline WM upper bound requires wm_warmup_steps > 0")
        if bool(OmegaConf.select(cfg, "training.debug", default=False)):
            raise ValueError(
                "pre-mainline WM upper bound forbids training.debug because it "
                "rewrites classifier and online rollout budgets at runtime"
            )
        if _select_int(cfg, "training.classifier_warmup_steps") != 0:
            raise ValueError("pre-mainline WM upper bound requires classifier_warmup_steps=0")
        if _select_int(cfg, "online_rollout.total_env_steps") != 0:
            raise ValueError("pre-mainline WM upper bound cannot run online environment steps")
        required_task_ids = [
            int(value)
            for value in (
                OmegaConf.select(
                    cfg,
                    "offline_warmup.required_task_ids",
                    default=[],
                )
                or []
            )
        ]
        if required_task_ids != official_task_ids:
            raise ValueError("pre-mainline WM upper bound must require all ten official task IDs")
        _reject_official_complete_marker_requirement(
            cfg,
            "offline_warmup.require_reference_complete",
        )
    elif stage == "classifier_upper_bound":
        path_pairs = (
            ("data.success_dir_raw", "task.hdf5_reward_dir"),
            ("data.success_dir_hidden", "task.openvla_oft.hidden_token_dir"),
        )
        if _select_str(cfg, "_target_") != ("dreamervla.runners.SuccessClassifierTrainingRunner"):
            raise ValueError(
                "independent classifier training must use SuccessClassifierTrainingRunner"
            )
        if int(_select_int(cfg, "training.num_epochs") or 0) <= 0:
            raise ValueError("pre-mainline classifier upper bound requires num_epochs > 0")
        if bool(
            OmegaConf.select(
                cfg,
                "training.episode_eval_enabled",
                default=False,
            )
        ):
            raise ValueError(
                "pre-mainline classifier upper bound uses held-out window F1; "
                "episode evaluation is invalid without failure trajectories"
            )
        if (
            _select_str(cfg, "data.train_split") != "train"
            or _select_str(cfg, "data.val_split") != "val"
        ):
            raise ValueError(
                "pre-mainline classifier upper bound requires disjoint train/val splits"
            )
        val_fraction = float(OmegaConf.select(cfg, "data.val_fraction", default=0.0) or 0.0)
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("pre-mainline classifier data.val_fraction must be within (0,1)")
        if _select_str(cfg, "training.final_selection_metric") != "window_f1":
            raise ValueError("pre-mainline classifier upper bound must select held-out window F1")
        if not bool(
            OmegaConf.select(
                cfg,
                "data.require_sidecar_contract",
                default=False,
            )
        ):
            raise ValueError(
                "pre-mainline classifier requires complete official sidecar validation"
            )
        _reject_official_complete_marker_requirement(
            cfg,
            "data.require_reference_complete",
        )
        required_filenames = tuple(
            str(value)
            for value in (
                OmegaConf.select(
                    cfg,
                    "data.required_filenames",
                    default=[],
                )
                or []
            )
        )
        if required_filenames != official_filenames:
            raise ValueError("pre-mainline classifier must require all ten official reward shards")
        if any(
            OmegaConf.select(cfg, key, default=None) is not None
            for key in ("data.failure_dir_raw", "data.failure_dir_hidden")
        ):
            raise ValueError("pre-mainline classifier upper bound cannot add failure datasets")
    else:
        raise ValueError(f"unknown pre_mainline.stage: {stage!r}")

    for active_key, official_key in path_pairs:
        active = _select_str(cfg, active_key)
        official = _select_str(cfg, official_key)
        if active is None or official is None or active != official:
            raise ValueError(
                f"{active_key} must use official LIBERO data from {official_key}: "
                f"{active!r} != {official!r}"
            )


def _reject_official_complete_marker_requirement(
    cfg: DictConfig,
    config_path: str,
) -> None:
    if bool(OmegaConf.select(cfg, config_path, default=True)):
        raise ValueError("official LIBERO reward shards do not use rollout complete markers")


def _validate_chunk_horizon_consistency(cfg: DictConfig) -> None:
    _require_equal_if_present(
        cfg,
        "world_model.chunk_size",
        "algorithm.lumos.chunk_size",
        message="LUMOS chunk size must match the world-model chunk size.",
    )
    _require_equal_if_present(
        cfg,
        "world_model.chunk_size",
        "policy.time_horizon",
        message="Policy horizon must match the world-model chunk size.",
    )
    _require_equal_if_present(
        cfg,
        "world_model.time_horizon",
        "dataset.expected_time_horizon",
        message="Dataset expected horizon must match the world-model horizon.",
    )

    selected_horizon = _selected_sidecar_action_horizon_key(cfg)
    if selected_horizon is not None:
        _require_equal_if_present(
            cfg,
            "dataset.expected_time_horizon",
            selected_horizon,
            message="Dataset expected horizon must match the selected sidecar action horizon.",
        )
    _validate_chunk_wm_sequence_lengths(cfg)
    _validate_dino_token_wm_sequence_lengths(cfg)


def _validate_chunk_wm_sequence_lengths(cfg: DictConfig) -> None:
    for key in (
        "world_model",
        "ray_components.world_model.kwargs",
        "learner.model_cfg.world_model.kwargs",
        "inference.cfg.world_model.kwargs",
    ):
        _validate_chunk_wm_sequence_length_for_component(cfg, key)


def _validate_chunk_wm_sequence_length_for_component(
    cfg: DictConfig,
    key: str,
) -> None:
    target = _component_target(cfg, key)
    if target is None or not target.endswith("ChunkAwareWorldModel"):
        return

    num_hist = _select_int(cfg, f"{key}.num_hist")
    chunk_size = _select_int(cfg, f"{key}.chunk_size")
    rollout_chunks = _select_int(cfg, f"{key}.chunk_rollout_chunks")
    if num_hist is None or chunk_size is None or rollout_chunks is None:
        return

    expected = num_hist + rollout_chunks * chunk_size + 1
    for sequence_key in (
        "dataset.sequence_length",
        "online_rollout.sequence_length",
        "replay.cfg.sequence_length",
        "ray_data.sequence_length",
    ):
        value = _select_int(cfg, sequence_key)
        if value is None:
            continue
        if value != expected:
            raise ValueError(
                f"{sequence_key} must equal {key}.num_hist + "
                f"{key}.chunk_rollout_chunks * {key}.chunk_size + 1 "
                f"({value} != {num_hist} + {rollout_chunks} * {chunk_size} + 1 = "
                f"{expected})"
            )


def _validate_dino_token_wm_sequence_lengths(cfg: DictConfig) -> None:
    for key in (
        "world_model",
        "ray_components.world_model.kwargs",
        "learner.model_cfg.world_model.kwargs",
        "inference.cfg.world_model.kwargs",
    ):
        target = _component_target(cfg, key)
        if target is None or not target.endswith("DinoTokenWorldModel"):
            continue
        num_hist = _select_int(cfg, f"{key}.num_hist")
        num_pred = _select_int(cfg, f"{key}.num_pred")
        if num_hist is None or num_pred is None:
            continue
        expected = num_hist + num_pred
        for sequence_key in (
            "dataset.sequence_length",
            "online_rollout.sequence_length",
            "replay.cfg.sequence_length",
            "ray_data.sequence_length",
        ):
            value = _select_int(cfg, sequence_key)
            if value is None:
                continue
            if value != expected:
                raise ValueError(
                    f"{sequence_key} must equal {key}.num_hist + "
                    f"{key}.num_pred for shifted DINO token training "
                    f"({value} != {num_hist} + {num_pred} = {expected})"
                )


def _validate_latent_dimension_contracts(cfg: DictConfig) -> None:
    for key in ("task.openvla_oft.hidden_token",):
        _validate_latent_spec(cfg, key, obs_dim_field="wm_obs_dim")
    for key in ("task.openvla_oft.hidden_token",):
        _validate_latent_stage_value(cfg, key)
    _validate_oft_hidden_token_patch_contract(cfg)

    for key in (
        "world_model",
        "ray_components.world_model.kwargs",
        "learner.model_cfg.world_model.kwargs",
        "inference.cfg.world_model.kwargs",
    ):
        _validate_latent_spec(cfg, key, obs_dim_field="obs_dim")
        _validate_latent_stage_contract(cfg, key)
        _validate_chunk_wm_token_space(cfg, key)


def _validate_oft_hidden_token_patch_contract(cfg: DictConfig) -> None:
    key = "task.openvla_oft.hidden_token"
    if OmegaConf.select(cfg, key, default=None) is None:
        return

    token_count = _select_int(cfg, f"{key}.token_count")
    patches_per_image = _select_int(cfg, f"{key}.patches_per_image")
    num_images = _select_int(cfg, f"{key}.num_images_in_input")
    if num_images is None:
        num_images = _select_int(cfg, "task.openvla_oft.num_images_in_input")
    if token_count is None or patches_per_image is None or num_images is None:
        return

    expected = num_images * patches_per_image
    if token_count != expected:
        raise ValueError(
            f"{key}.token_count must equal num_images_in_input * patches_per_image "
            f"({token_count} != {num_images} * {patches_per_image} = {expected})"
        )


def _validate_latent_stage_value(cfg: DictConfig, key: str) -> None:
    stage = _select_str(cfg, f"{key}.latent_stage")
    if stage is not None and stage not in {"query_before", "query_after"}:
        raise ValueError(
            f"{key}.latent_stage must be 'query_before' or 'query_after', got {stage!r}"
        )


def _validate_latent_stage_contract(cfg: DictConfig, key: str) -> None:
    stage = _select_str(cfg, f"{key}.latent_stage")
    _validate_latent_stage_value(cfg, key)
    expected_stage = _matching_task_latent_stage(cfg, key)
    if expected_stage is None:
        return
    if stage is None:
        raise ValueError(
            f"{key}.latent_stage must be set and match the selected task sidecar "
            f"stage {expected_stage!r}"
        )
    if stage != expected_stage:
        raise ValueError(
            f"{key}.latent_stage must match selected task sidecar stage "
            f"({stage!r} != {expected_stage!r})"
        )


def _matching_task_latent_stage(cfg: DictConfig, key: str) -> str | None:
    obs_dim = _select_int(cfg, f"{key}.obs_dim")
    token_count = _select_int(cfg, f"{key}.token_count")
    token_dim = _select_int(cfg, f"{key}.token_dim")
    if obs_dim is None or token_count is None or token_dim is None:
        return None

    for spec_key in ("task.openvla_oft.hidden_token",):
        spec_obs_dim = _select_int(cfg, f"{spec_key}.wm_obs_dim")
        spec_token_count = _select_int(cfg, f"{spec_key}.token_count")
        spec_token_dim = _select_int(cfg, f"{spec_key}.token_dim")
        if (
            spec_obs_dim == obs_dim
            and spec_token_count == token_count
            and spec_token_dim == token_dim
        ):
            return _select_str(cfg, f"{spec_key}.latent_stage")
    return None


def _validate_chunk_wm_token_space(cfg: DictConfig, key: str) -> None:
    target = _component_target(cfg, key)
    if target is None or not target.endswith("ChunkAwareWorldModel"):
        return

    for required_key in ("depth", "heads", "dim_head", "mlp_dim"):
        if _select_int(cfg, f"{key}.{required_key}") is None:
            raise ValueError(
                f"{key}.{required_key} must be set in Hydra config for "
                "ChunkAwareWorldModel transformer sizing"
            )

    if (
        key == "world_model"
        and _looks_oft_hidden_token_cfg(cfg)
        and _select_str(cfg, "_target_") == "dreamervla.runners.LatentWMRunner"
    ):
        for required_key in (
            "proprio_dim",
            "proprio_emb_dim",
            "num_proprio_repeat",
            "lang_dim",
            "lang_emb_dim",
            "num_lang_repeat",
        ):
            if _select_int(cfg, f"{key}.{required_key}") is None:
                raise ValueError(
                    f"{key}.{required_key} must be set for OpenVLA-OFT "
                    "hidden-token query_before proprio/language conditioning"
                )
        for required_key in ("dataset.proprio_keys", "dataset.lang_emb_dir"):
            if OmegaConf.select(cfg, required_key, default=None) in (None, ""):
                raise ValueError(
                    f"{required_key} must be set for OpenVLA-OFT hidden-token "
                    "query_before proprio/language conditioning"
                )

    model_dim = _select_int(cfg, f"{key}.model_dim")
    token_dim = _select_int(cfg, f"{key}.token_dim")
    action_emb_dim = _select_int(cfg, f"{key}.action_emb_dim")
    num_action_repeat = _select_int(cfg, f"{key}.num_action_repeat")
    proprio_emb_dim = _select_int(cfg, f"{key}.proprio_emb_dim")
    num_proprio_repeat = _select_int(cfg, f"{key}.num_proprio_repeat")
    lang_emb_dim = _select_int(cfg, f"{key}.lang_emb_dim")
    num_lang_repeat = _select_int(cfg, f"{key}.num_lang_repeat")
    if model_dim is None or token_dim is None:
        return
    if action_emb_dim is None:
        raise ValueError(
            f"{key}.action_emb_dim must be set for ChunkAwareWorldModel WM concat conditioning"
        )
    if action_emb_dim < 1:
        raise ValueError(f"{key}.action_emb_dim must be > 0, got {action_emb_dim}")
    if num_action_repeat is None:
        num_action_repeat = 1
    if num_action_repeat < 1:
        raise ValueError(f"{key}.num_action_repeat must be > 0, got {num_action_repeat}")
    if proprio_emb_dim is None:
        proprio_emb_dim = 0
    if proprio_emb_dim < 0:
        raise ValueError(f"{key}.proprio_emb_dim must be >= 0, got {proprio_emb_dim}")
    if num_proprio_repeat is None:
        num_proprio_repeat = 1
    if num_proprio_repeat < 1:
        raise ValueError(f"{key}.num_proprio_repeat must be > 0, got {num_proprio_repeat}")
    if lang_emb_dim is None:
        lang_emb_dim = 0
    if lang_emb_dim < 0:
        raise ValueError(f"{key}.lang_emb_dim must be >= 0, got {lang_emb_dim}")
    if num_lang_repeat is None:
        num_lang_repeat = 1
    if num_lang_repeat < 1:
        raise ValueError(f"{key}.num_lang_repeat must be > 0, got {num_lang_repeat}")
    expected_model_dim = (
        token_dim
        + proprio_emb_dim * num_proprio_repeat
        + lang_emb_dim * num_lang_repeat
        + action_emb_dim * num_action_repeat
    )
    if model_dim == expected_model_dim:
        return
    raise ValueError(
        f"{key}.model_dim must equal {key}.token_dim + "
        f"{key}.proprio_emb_dim * {key}.num_proprio_repeat + "
        f"{key}.lang_emb_dim * {key}.num_lang_repeat + "
        f"{key}.action_emb_dim * {key}.num_action_repeat for "
        "ChunkAwareWorldModel WM concat conditioning "
        f"({model_dim} != {token_dim} + "
        f"{proprio_emb_dim} * {num_proprio_repeat} + "
        f"{lang_emb_dim} * {num_lang_repeat} + "
        f"{action_emb_dim} * {num_action_repeat})"
    )


def _component_target(cfg: DictConfig, key: str) -> str | None:
    target = _select_str(cfg, f"{key}._target_")
    if target is None and key.endswith(".kwargs"):
        parent_key = key.removesuffix(".kwargs")
        target = _select_str(cfg, f"{parent_key}.target")
    return target


def _validate_latent_spec(
    cfg: DictConfig,
    key: str,
    *,
    obs_dim_field: str,
    action_dim_key: str | None = None,
    check_action_token_count: bool = False,
) -> None:
    section = OmegaConf.select(cfg, key, default=None)
    if section is None:
        return

    obs_dim = _select_int(cfg, f"{key}.{obs_dim_field}")
    token_count = _select_int(cfg, f"{key}.token_count")
    token_dim = _select_int(cfg, f"{key}.token_dim")
    if obs_dim is not None and token_count is not None and token_dim is not None:
        expected = token_count * token_dim
        if obs_dim != expected:
            raise ValueError(
                f"{key}.{obs_dim_field} must equal token_count * token_dim "
                f"({obs_dim} != {token_count} * {token_dim} = {expected})"
            )

    chunk_size = _select_int(cfg, f"{key}.chunk_size")
    time_horizon = _select_int(cfg, f"{key}.time_horizon")
    if chunk_size is not None and time_horizon is not None and chunk_size != time_horizon:
        raise ValueError(
            f"{key}.chunk_size must match {key}.time_horizon ({chunk_size} != {time_horizon})"
        )

    if not check_action_token_count:
        return
    if action_dim_key is None:
        return
    action_dim = _select_int(cfg, action_dim_key)
    if token_count is None or chunk_size is None or action_dim is None:
        return
    expected_tokens = chunk_size * action_dim
    if token_count != expected_tokens:
        raise ValueError(
            f"{key}.token_count must equal chunk_size * {action_dim_key} "
            f"({token_count} != {chunk_size} * {action_dim} = {expected_tokens})"
        )


def _validate_dino_token_training(cfg: DictConfig) -> None:
    """Validate the dedicated token-sidecar DINO-WM reproduction contract."""

    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if target.rsplit(".", 1)[-1] != "DinoTokenWorldModelTrainingRunner":
        return
    model_target = _component_target(cfg, "world_model")
    if model_target is None or not model_target.endswith("DinoTokenWorldModel"):
        raise ValueError(
            "DinoTokenWorldModelTrainingRunner requires world_model=DinoTokenWorldModel"
        )
    precision = str(OmegaConf.select(cfg, "optim.precision", default="")).lower()
    if precision != "fp32":
        raise ValueError("DinoTokenWorldModelTrainingRunner requires optim.precision=fp32")
    param_precision = str(OmegaConf.select(cfg, "optim.param_precision", default="")).lower()
    if param_precision != "fp32":
        raise ValueError("DinoTokenWorldModelTrainingRunner requires optim.param_precision=fp32")
    frameskip = _select_int(cfg, "dino_wm.frameskip")
    action_dim = _select_int(cfg, "task.action_dim")
    model_action_dim = _select_int(cfg, "world_model.action_dim")
    if frameskip is None or frameskip < 1:
        raise ValueError("dino_wm.frameskip must be positive")
    if action_dim is None or model_action_dim != action_dim * frameskip:
        raise ValueError(
            "world_model.action_dim must equal task.action_dim * dino_wm.frameskip "
            f"({model_action_dim} != {action_dim} * {frameskip})"
        )
    model_hist = _select_int(cfg, "world_model.num_hist")
    model_pred = _select_int(cfg, "world_model.num_pred")
    for split in ("train", "valid"):
        prefix = f"dataset.{split}"
        for field, expected in (
            ("num_hist", model_hist),
            ("num_pred", model_pred),
            ("frameskip", frameskip),
        ):
            value = _select_int(cfg, f"{prefix}.{field}")
            if value != expected:
                raise ValueError(
                    f"{prefix}.{field} must match the DINO model/data contract "
                    f"({value} != {expected})"
                )
    for removed in ("token_normalization", "token_norm_eps"):
        if OmegaConf.select(cfg, f"world_model.{removed}", default=None) is not None:
            raise ValueError(
                f"world_model.{removed} must not be configured; normalized token "
                "output is intrinsic to the DINO token adapter"
            )


def _validate_world_model_training_pipeline(cfg: DictConfig) -> None:
    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if target.rsplit(".", 1)[-1] != "WorldModelTrainingRunner":
        return
    total_env_steps = int(OmegaConf.select(cfg, "online_rollout.total_env_steps", default=0) or 0)
    if total_env_steps != 0:
        raise ValueError(
            "WorldModelTrainingRunner only supports offline warmup; "
            "use CotrainRunner and the Ray cotrain experiment for online training"
        )
    data_dir = OmegaConf.select(cfg, "offline_warmup.data_dir", default=None)
    hidden_dir = OmegaConf.select(cfg, "offline_warmup.hidden_dir", default=None)
    if not data_dir or not os.path.isdir(str(data_dir)):
        raise ValueError(f"offline_warmup.data_dir does not exist: {data_dir!r}")
    if not hidden_dir or not os.path.isdir(str(hidden_dir)):
        raise ValueError(f"offline_warmup.hidden_dir does not exist: {hidden_dir!r}")
    for key in (
        "wm_warmup_steps",
        "classifier_warmup_steps",
        "warmup_replay_epochs",
        "warmup_replay_max_steps",
        "warmup_checkpoint_every_epochs",
    ):
        val = int(OmegaConf.select(cfg, f"training.{key}", default=0))
        if val < 0:
            raise ValueError(f"training.{key} must be >= 0, got {val}")
    log_every = int(OmegaConf.select(cfg, "training.replay_warmup_log_every", default=1))
    if log_every < 1:
        raise ValueError(f"training.replay_warmup_log_every must be >= 1, got {log_every}")


def _validate_epoch_checkpoint_cadence(cfg: DictConfig) -> None:
    target = str(OmegaConf.select(cfg, "_target_", default="") or "").rsplit(".", 1)[-1]
    if target == "WorldModelTrainingRunner":
        if OmegaConf.select(cfg, "training.warmup_checkpoint_every", default=None) is not None:
            raise ValueError(
                "training.warmup_checkpoint_every was removed; use "
                "training.warmup_checkpoint_every_epochs"
            )
        cadence = int(OmegaConf.select(cfg, "training.warmup_checkpoint_every_epochs", default=1))
        if cadence < 0:
            raise ValueError(f"training.warmup_checkpoint_every_epochs must be >= 0, got {cadence}")
        epochs = int(OmegaConf.select(cfg, "training.warmup_replay_epochs", default=0) or 0)
        if epochs > 1 and cadence == 0:
            raise ValueError(
                "training.warmup_checkpoint_every_epochs must be > 0 when "
                "training.warmup_replay_epochs > 1"
            )
    if target == "SuccessClassifierTrainingRunner":
        if OmegaConf.select(cfg, "training.ckpt_every", default=None) is not None:
            raise ValueError(
                "training.ckpt_every was removed; use training.checkpoint_every_epochs"
            )
        cadence = int(OmegaConf.select(cfg, "training.checkpoint_every_epochs", default=1))
        if cadence < 0:
            raise ValueError(f"training.checkpoint_every_epochs must be >= 0, got {cadence}")
        epochs = int(OmegaConf.select(cfg, "training.num_epochs", default=1) or 0)
        if epochs > 1 and cadence == 0:
            raise ValueError(
                "training.checkpoint_every_epochs must be > 0 when training.num_epochs > 1"
            )


def _validate_model_registry_refs(cfg: DictConfig) -> None:
    for key in (
        "model.model_type",
        "policy.model_type",
        "encoder.model_type",
        "world_model.model_type",
        "learner.model_cfg.policy.model_type",
        "learner.model_cfg.world_model.model_type",
        "learner.model_cfg.classifier.model_type",
    ):
        model_type = OmegaConf.select(cfg, key, default=None)
        if model_type is not None:
            validate_model_type(str(model_type))


def _validate_ray_manual_resources(cfg: DictConfig) -> None:
    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    is_ray_runner = target.endswith(
        (
            "CotrainRunner",
            "DreamerRunner",
            "RolloutCollectionRunner",
        )
    )
    if not is_ray_runner:
        return

    forbidden = [
        key
        for key in (
            "training.auto_vram_batch",
            "collect.auto_vram_envs",
        )
        if OmegaConf.select(cfg, key, default=None) is not None
    ]
    if forbidden:
        raise ValueError(
            "Ray backend follows RLinf-style manual resource tuning; "
            f"auto_vram knobs are not supported: {forbidden}"
        )

    _require_positive_if_present(cfg, "env.num_workers")
    _require_positive_if_present(cfg, "rollout.steps")
    _require_positive_if_present(cfg, "replay.cfg.sequence_length")
    _require_positive_if_present(cfg, "learner.train_cfg.batch_size")
    _require_positive_if_present(cfg, "learner.num_workers")
    _require_positive_if_present(cfg, "collect.envs_per_gpu")
    _require_non_negative_int_if_present(cfg, "manual_cotrain.ngpu")
    _require_non_negative_int_if_present(cfg, "manual_cotrain.task_id")
    _require_non_negative_if_present(cfg, "manual_cotrain.env_rollout_timeout_s")
    _require_non_negative_int_if_present(cfg, "manual_cotrain.checkpoint_every")
    if OmegaConf.select(cfg, "manual_cotrain.keep_last_checkpoints", default=None) is not None:
        raise ValueError(
            "manual_cotrain.keep_last_checkpoints was removed; configure checkpoint.topk.k"
        )
    _require_non_negative_int_if_present(cfg, "checkpoint.topk.k")
    topk_k = int(OmegaConf.select(cfg, "checkpoint.topk.k", default=0) or 0)
    if topk_k > 0:
        topk_mode = str(OmegaConf.select(cfg, "checkpoint.topk.mode", default=""))
        if topk_mode not in {"min", "max"}:
            raise ValueError("checkpoint.topk.mode must be one of: min, max")
        monitor_key = str(
            OmegaConf.select(cfg, "checkpoint.topk.monitor_key", default="") or ""
        ).strip()
        if not monitor_key:
            raise ValueError("checkpoint.topk.monitor_key must be non-empty when top-k is enabled")
    _require_positive_if_present(cfg, "manual_cotrain.global_steps")
    _require_positive_if_present(cfg, "manual_cotrain.learner_update_step")
    _require_positive_if_present(
        cfg,
        "manual_cotrain.learner_updates_per_global_step",
    )
    _require_positive_if_present(
        cfg,
        "manual_cotrain.learner_early_stop_patience",
    )
    _require_positive_if_present(cfg, "manual_cotrain.max_policy_kl")
    _require_positive_if_present(cfg, "manual_cotrain.sync_every")
    _require_positive_if_present(cfg, "manual_cotrain.rollout_epoch")
    _require_positive_if_present(cfg, "manual_cotrain.real_rollout_epoch")
    _require_positive_if_present(cfg, "manual_cotrain.wm_rollout_epoch")
    _require_positive_int_if_present(
        cfg,
        "manual_cotrain.wm_rollout_target_trajectories",
    )
    _require_positive_int_if_present(cfg, "manual_cotrain.wm_rollout_lease_epochs")
    _require_positive_if_present(cfg, "manual_cotrain.max_steps_per_rollout_epoch")
    _require_positive_if_present(
        cfg,
        "manual_cotrain.real_max_steps_per_rollout_epoch",
    )
    _require_positive_if_present(cfg, "manual_cotrain.wm_rollout_multiplier")
    _require_positive_if_present(cfg, "manual_cotrain.num_action_chunks")
    _require_positive_if_present(cfg, "manual_cotrain.envs_per_worker")
    _require_positive_if_present(cfg, "manual_cotrain.real_envs_per_worker")
    _require_positive_if_present(cfg, "manual_cotrain.wm_envs_per_worker")
    _validate_ray_single_node_placement(cfg)
    _validate_manual_cotrain_placement(cfg)

    max_steps = OmegaConf.select(
        cfg,
        "manual_cotrain.max_steps_per_rollout_epoch",
        default=None,
    )
    chunk = OmegaConf.select(cfg, "manual_cotrain.num_action_chunks", default=None)
    if max_steps is not None and chunk is not None:
        if int(max_steps) % int(chunk) != 0:
            raise ValueError(
                "manual_cotrain.max_steps_per_rollout_epoch must be divisible by "
                "manual_cotrain.num_action_chunks"
            )
    real_max_steps = OmegaConf.select(
        cfg,
        "manual_cotrain.real_max_steps_per_rollout_epoch",
        default=None,
    )
    if real_max_steps is not None and chunk is not None:
        if int(real_max_steps) % int(chunk) != 0:
            raise ValueError(
                "manual_cotrain.real_max_steps_per_rollout_epoch must be divisible "
                "by manual_cotrain.num_action_chunks"
            )
    _validate_manual_cotrain_group_geometry(cfg)
    _validate_manual_actor_ppo_batches(cfg)
    _validate_manual_cotrain_replay_window(cfg)
    _validate_manual_cotrain_classifier_window(cfg)


def _validate_manual_cotrain_placement(cfg: DictConfig) -> None:
    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith(("CotrainRunner", "DreamerRunner")):
        return
    if OmegaConf.select(cfg, "manual_cotrain", default=None) is None:
        return

    try:
        plan = build_manual_cotrain_placement_from_config(cfg)
    except Exception as exc:
        raise ValueError(f"manual cotrain placement is invalid: {exc}") from exc
    if not plan.real_env_ranks:
        raise ValueError("manual cotrain placement requires at least one RealEnvWorker")
    if not plan.wm_env_ranks:
        raise ValueError("manual cotrain placement requires at least one WMEnvWorker")

    training_mode = str(
        OmegaConf.select(
            cfg,
            "manual_cotrain.training_mode",
            default="staged_full_cotrain",
        )
    ).strip()
    allowed_training_modes = {"failure_imagined_rl", "staged_full_cotrain"}
    if training_mode not in allowed_training_modes:
        raise ValueError(
            "manual_cotrain.training_mode must be one of "
            f"{sorted(allowed_training_modes)}, got {training_mode!r}"
        )
    if training_mode == "failure_imagined_rl":
        if bool(
            OmegaConf.select(
                cfg,
                "manual_cotrain.learner_updates_enabled",
                default=True,
            )
        ):
            raise ValueError("failure_imagined_rl requires learner_updates_enabled=false")
        if bool(
            OmegaConf.select(
                cfg,
                "manual_cotrain.staged_policy_update",
                default=False,
            )
        ):
            raise ValueError("failure_imagined_rl requires staged_policy_update=false")
        selector = str(
            OmegaConf.select(
                cfg,
                "manual_cotrain.initial_condition_selector",
                default="",
            )
        ).strip()
        allowed_selectors = {"episode_start", "failed_episode_start"}
        if selector not in allowed_selectors:
            raise ValueError(
                "failure_imagined_rl requires initial_condition_selector to be one of "
                f"{sorted(allowed_selectors)}"
            )

    if bool(
        OmegaConf.select(
            cfg,
            "manual_cotrain.staged_policy_update",
            default=False,
        )
    ):
        if not bool(
            OmegaConf.select(
                cfg,
                "manual_cotrain.real_env_enabled",
                default=True,
            )
        ):
            raise ValueError("manual_cotrain.staged_policy_update requires real_env_enabled=true")
        if not bool(
            OmegaConf.select(
                cfg,
                "manual_cotrain.learner_updates_enabled",
                default=True,
            )
        ):
            raise ValueError(
                "manual_cotrain.staged_policy_update requires learner_updates_enabled=true"
            )
        if int(OmegaConf.select(cfg, "manual_cotrain.sync_every", default=1)) != 1:
            raise ValueError("manual_cotrain.staged_policy_update requires sync_every=1")

    precision = OmegaConf.select(cfg, "learner.train_cfg.precision", default=None)
    if precision is not None:
        normalized = str(precision).strip().lower()
        if normalized not in {"fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}:
            raise ValueError(
                f"learner.train_cfg.precision must be one of fp32, bf16, or fp16; got {precision!r}"
            )
    _warn_manual_cotrain_baseline_overrides(cfg)


def _warn_manual_cotrain_baseline_overrides(cfg: DictConfig) -> None:
    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith(("CotrainRunner", "DreamerRunner")):
        return
    baselines = {
        "real_rollout_target_trajectories": 32,
        "wm_rollout_target_trajectories": 1024,
        "max_steps_per_rollout_epoch": 512,
    }
    for field, baseline in baselines.items():
        path = f"manual_cotrain.{field}"
        value = OmegaConf.select(cfg, path, default=None)
        if value is None or int(value) == baseline:
            continue
        warnings.warn(
            f"{path} overrides the mainline baseline {baseline}; got {value!r}. "
            "This is allowed for smoke/tiny runs but should not become the "
            "OpenVLA one-trajectory cotrain default.",
            UserWarning,
            stacklevel=3,
        )


def _validate_manual_cotrain_group_geometry(cfg: DictConfig) -> None:
    """Validate manual-cotrain rollout slots against actor GRPO grouping."""

    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith(("CotrainRunner", "DreamerRunner")):
        return

    envs_per_worker_raw = OmegaConf.select(
        cfg,
        "manual_cotrain.envs_per_worker",
        default=None,
    )
    group_size_raw = OmegaConf.select(
        cfg,
        "actor.train_cfg.algorithm_cfg.group_size",
        default=OmegaConf.select(cfg, "algorithm.group_size", default=None),
    )
    if envs_per_worker_raw is None or group_size_raw is None:
        return

    ngpu = int(OmegaConf.select(cfg, "manual_cotrain.ngpu", default=1))
    real_enabled = bool(
        OmegaConf.select(
            cfg,
            "manual_cotrain.real_env_enabled",
            default=True,
        )
    )
    configured_real_workers = int(
        OmegaConf.select(
            cfg,
            "manual_cotrain.real_env_workers",
            default=1,
        )
    )
    if real_enabled:
        real_workers = max(0, configured_real_workers)
        if not target.endswith("DreamerRunner"):
            real_workers = min(real_workers, max(1, ngpu))
    else:
        real_workers = 0
    if target.endswith("DreamerRunner"):
        wm_workers = max(0, ngpu - (1 if real_workers and ngpu > 0 else 0))
    else:
        wm_workers = max(0, (max(1, ngpu) if ngpu == 0 else ngpu) - real_workers)
    envs_per_worker = int(envs_per_worker_raw)
    real_envs_per_worker = int(
        OmegaConf.select(
            cfg,
            "manual_cotrain.real_envs_per_worker",
            default=envs_per_worker,
        )
    )
    wm_envs_per_worker = int(
        OmegaConf.select(
            cfg,
            "manual_cotrain.wm_envs_per_worker",
            default=envs_per_worker,
        )
    )
    rollout_epoch = int(OmegaConf.select(cfg, "manual_cotrain.rollout_epoch", default=1))
    real_rollout_epoch_raw = OmegaConf.select(
        cfg,
        "manual_cotrain.real_rollout_epoch",
        default=rollout_epoch,
    )
    real_rollout_epoch = (
        0
        if real_workers <= 0
        else int(rollout_epoch if real_rollout_epoch_raw is None else real_rollout_epoch_raw)
    )
    wm_rollout_epoch = int(
        OmegaConf.select(
            cfg,
            "manual_cotrain.wm_rollout_epoch",
            default=rollout_epoch,
        )
    )
    wm_rollout_target = OmegaConf.select(
        cfg,
        "manual_cotrain.wm_rollout_target_trajectories",
        default=None,
    )
    real_rollout_target = OmegaConf.select(
        cfg,
        "manual_cotrain.real_rollout_target_trajectories",
        default=None,
    )
    group_size = int(group_size_raw)
    if group_size <= 0:
        raise ValueError("actor.train_cfg.algorithm_cfg.group_size must be positive")

    if real_workers > 0 and real_rollout_target is not None:
        real_trajectory_count = int(real_rollout_target)
        if real_trajectory_count % real_envs_per_worker != 0:
            raise ValueError(
                "manual_cotrain.real_rollout_target_trajectories must be divisible by "
                "manual_cotrain.real_envs_per_worker: "
                f"real_rollout_target_trajectories={real_trajectory_count}, "
                f"real_envs_per_worker={real_envs_per_worker}"
            )
        total_real_worker_epochs = real_trajectory_count // real_envs_per_worker
        if total_real_worker_epochs < real_workers:
            raise ValueError(
                "manual_cotrain.real_rollout_target_trajectories is too small to give "
                "each real worker at least one rollout_epoch: "
                f"real_rollout_target_trajectories={real_trajectory_count}, "
                f"real_envs_per_worker={real_envs_per_worker}, "
                f"real_workers={real_workers}"
            )
    else:
        real_trajectory_count = real_envs_per_worker * real_workers * real_rollout_epoch
    if wm_workers > 0 and wm_rollout_target is not None:
        wm_trajectory_count = int(wm_rollout_target)
        if wm_trajectory_count % wm_envs_per_worker != 0:
            raise ValueError(
                "manual_cotrain.wm_rollout_target_trajectories must be divisible by "
                "manual_cotrain.wm_envs_per_worker: "
                f"wm_rollout_target_trajectories={wm_trajectory_count}, "
                f"wm_envs_per_worker={wm_envs_per_worker}"
            )
        total_wm_worker_epochs = wm_trajectory_count // wm_envs_per_worker
        if total_wm_worker_epochs < wm_workers:
            raise ValueError(
                "manual_cotrain.wm_rollout_target_trajectories is too small to give "
                "each WM worker at least one rollout_epoch: "
                f"wm_rollout_target_trajectories={wm_trajectory_count}, "
                f"wm_envs_per_worker={wm_envs_per_worker}, "
                f"wm_workers={wm_workers}"
            )
    else:
        wm_trajectory_count = wm_envs_per_worker * wm_workers * wm_rollout_epoch
    actor_trajectory_count = wm_trajectory_count
    if actor_trajectory_count % group_size != 0:
        raise ValueError(
            "manual cotrain actor WM trajectory count must be divisible by "
            "actor.train_cfg.algorithm_cfg.group_size: "
            f"manual_cotrain.ngpu={ngpu}, "
            f"manual_cotrain.envs_per_worker={envs_per_worker}, "
            f"manual_cotrain.real_envs_per_worker={real_envs_per_worker}, "
            f"manual_cotrain.wm_envs_per_worker={wm_envs_per_worker}, "
            f"manual_cotrain.rollout_epoch={rollout_epoch}, "
            f"manual_cotrain.real_rollout_epoch={real_rollout_epoch}, "
            f"manual_cotrain.wm_rollout_epoch={wm_rollout_epoch}, "
            f"manual_cotrain.wm_rollout_target_trajectories={wm_rollout_target}, "
            f"real replay trajectory count={real_trajectory_count}, "
            f"actor WM trajectory count={actor_trajectory_count}, "
            f"group_size={group_size}"
        )


def _validate_manual_actor_ppo_batches(cfg: DictConfig) -> None:
    """Validate the RLinf global-batch/micro-batch hierarchy before Ray starts."""

    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith(("CotrainRunner", "DreamerRunner")):
        return
    global_batch_raw = OmegaConf.select(
        cfg,
        "actor.train_cfg.global_batch_size",
        default=None,
    )
    micro_batch_raw = OmegaConf.select(
        cfg,
        "actor.train_cfg.micro_batch_size",
        default=None,
    )
    if global_batch_raw is None and micro_batch_raw is None:
        return
    if global_batch_raw is None or micro_batch_raw is None:
        raise ValueError(
            "actor.train_cfg.global_batch_size and micro_batch_size must be configured together"
        )
    global_batch = int(global_batch_raw)
    micro_batch = int(micro_batch_raw)
    actor_ranks = max(
        1,
        int(OmegaConf.select(cfg, "manual_cotrain.ngpu", default=1)),
    )
    if global_batch <= 0 or micro_batch <= 0:
        raise ValueError("Actor PPO global_batch_size and micro_batch_size must be positive")
    if global_batch % (micro_batch * actor_ranks) != 0:
        raise ValueError(
            "actor.train_cfg.global_batch_size must be divisible by "
            "micro_batch_size * Actor ranks: "
            f"{global_batch} % ({micro_batch} * {actor_ranks}) != 0"
        )

    trajectories = OmegaConf.select(
        cfg,
        "manual_cotrain.wm_rollout_target_trajectories",
        default=None,
    )
    max_steps = OmegaConf.select(
        cfg,
        "manual_cotrain.max_steps_per_rollout_epoch",
        default=None,
    )
    chunk_size = OmegaConf.select(
        cfg,
        "manual_cotrain.num_action_chunks",
        default=None,
    )
    if trajectories is None or max_steps is None or chunk_size is None:
        return
    flattened_samples = int(trajectories) * (int(max_steps) // int(chunk_size))
    if flattened_samples % global_batch != 0:
        raise ValueError(
            "manual cotrain flattened rollout samples must be divisible by Actor "
            "global_batch_size: "
            f"{flattened_samples} % {global_batch} != 0"
        )


def _validate_manual_cotrain_replay_window(cfg: DictConfig) -> None:
    """Validate that EnvWorker can produce replay windows used by LearnerWorker."""

    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith(("CotrainRunner", "DreamerRunner")):
        return

    sequence_length_raw = OmegaConf.select(cfg, "replay.cfg.sequence_length", default=None)
    max_steps_raw = OmegaConf.select(
        cfg,
        "manual_cotrain.max_steps_per_rollout_epoch",
        default=None,
    )
    if sequence_length_raw is None or max_steps_raw is None:
        return

    sequence_length = int(sequence_length_raw)
    max_steps = int(max_steps_raw)
    if sequence_length <= 0:
        raise ValueError("replay.cfg.sequence_length must be positive")
    if max_steps < sequence_length:
        raise ValueError(
            "manual_cotrain.max_steps_per_rollout_epoch must be >= "
            "replay.cfg.sequence_length so LearnerWorker can sample full replay "
            f"sequences; got {max_steps} and {sequence_length}"
        )


def _validate_manual_cotrain_classifier_window(cfg: DictConfig) -> None:
    """Validate rollout length against classifier replay-window sampling."""

    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith(("CotrainRunner", "DreamerRunner")):
        return

    max_steps_raw = OmegaConf.select(
        cfg,
        "manual_cotrain.max_steps_per_rollout_epoch",
        default=None,
    )
    window_raw = OmegaConf.select(
        cfg,
        "learner.model_cfg.classifier.kwargs.window",
        default=OmegaConf.select(
            cfg,
            "learner.model_cfg.classifier.window",
            default=None,
        ),
    )
    if max_steps_raw is None or window_raw is None:
        return

    chunk_size_raw = OmegaConf.select(
        cfg,
        "learner.model_cfg.classifier.kwargs.chunk_size",
        default=OmegaConf.select(
            cfg,
            "learner.model_cfg.classifier.chunk_size",
            default=OmegaConf.select(
                cfg,
                "manual_cotrain.num_action_chunks",
                default=None,
            ),
        ),
    )
    if chunk_size_raw is None:
        return

    max_steps = int(max_steps_raw)
    window = int(window_raw)
    chunk_size = int(chunk_size_raw)
    if window <= 0:
        raise ValueError("classifier window must be positive")
    if chunk_size <= 0:
        raise ValueError("classifier chunk_size must be positive")
    required_steps = window * chunk_size
    if max_steps < required_steps:
        raise ValueError(
            "manual_cotrain.max_steps_per_rollout_epoch must cover classifier "
            "window sampling: classifier window requires "
            f"{required_steps} physical steps "
            f"(window={window}, chunk_size={chunk_size}), got {max_steps}"
        )


def _validate_fsdp_config(cfg: DictConfig) -> None:
    """Fail fast on unusable FSDP blocks before any worker spawns.

    Workers build ``FSDPModelManager(***.train_cfg.fsdp)`` inside Ray actors, so
    a bad ``strategy``/``precision`` would otherwise only surface after the
    cluster is up. The accepted strategy set mirrors
    ``FSDPModelManager`` (none/ddp/fsdp/fsdp1/fsdp2).
    """

    for base in ("learner.train_cfg.fsdp", "actor.train_cfg.fsdp"):
        fsdp = OmegaConf.select(cfg, base, default=None)
        if fsdp is None:
            continue

        strategy = OmegaConf.select(fsdp, "strategy", default=None)
        if strategy is not None:
            normalized = str(strategy).strip().lower()
            if normalized not in {"", "none", "ddp", "fsdp", "fsdp1", "fsdp2"}:
                raise ValueError(
                    f"{base}.strategy must be one of "
                    f"none, ddp, fsdp, fsdp1, fsdp2; got {strategy!r}"
                )

        precision = OmegaConf.select(fsdp, "precision", default=None)
        if precision is not None:
            normalized = str(precision).strip().lower()
            if normalized not in {
                "fp32",
                "float32",
                "bf16",
                "bfloat16",
                "fp16",
                "float16",
            }:
                raise ValueError(
                    f"{base}.precision must be one of fp32, bf16, or fp16; got {precision!r}"
                )


def _validate_ray_single_node_placement(cfg: DictConfig) -> None:
    num_nodes = OmegaConf.select(
        cfg,
        "cluster.num_nodes",
        default=OmegaConf.select(cfg, "scheduler.cluster.num_nodes", default=None),
    )
    if num_nodes is not None and int(num_nodes) != 1:
        raise ValueError(
            "DreamerVLA Ray backend is currently single-node; "
            f"cluster.num_nodes={num_nodes!r} is not supported"
        )

    placement = OmegaConf.select(cfg, "learner.placement", default=None)
    if placement is None:
        return

    raw = OmegaConf.to_container(placement, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("learner.placement must be a mapping")
    strategy = str(raw.get("strategy", "node")).strip().lower()
    try:
        num_workers_raw = OmegaConf.select(cfg, "learner.num_workers", default=None)
        num_workers = int(num_workers_raw) if num_workers_raw is not None else None
        if strategy in {"", "node", "cpu"}:
            count = int(raw.get("count", num_workers or 1))
            if count <= 0:
                raise ValueError("count must be >= 1")
            if num_workers is not None and count != num_workers:
                raise ValueError(f"count must match learner.num_workers ({count} != {num_workers})")
        elif strategy == "packed":
            _validate_ray_packed_placement(raw, num_workers=num_workers)
        elif strategy == "flexible":
            groups = raw.get("accelerator_groups", raw.get("groups"))
            actual_workers = len(_normalize_accelerator_groups(groups))
            if num_workers is not None and actual_workers != num_workers:
                raise ValueError(
                    "accelerator_groups must match learner.num_workers "
                    f"({actual_workers} != {num_workers})"
                )
        else:
            raise ValueError(f"strategy must be one of node, packed, or flexible; got {strategy!r}")
    except Exception as exc:
        raise ValueError(f"learner.placement is invalid: {exc}") from exc


def _validate_ray_packed_placement(
    raw: dict[str, Any],
    *,
    num_workers: int | None,
) -> None:
    start_gpu = int(raw.get("start_gpu", 0))
    num_gpus_per_worker = int(raw.get("num_gpus_per_worker", 1))
    if start_gpu < 0:
        raise ValueError("start_gpu must be >= 0")
    if num_gpus_per_worker <= 0:
        raise ValueError("num_gpus_per_worker must be >= 1")
    if "end_gpu" in raw:
        end_gpu = int(raw["end_gpu"])
    else:
        workers = num_workers or 1
        end_gpu = start_gpu + workers * num_gpus_per_worker - 1
    if end_gpu < start_gpu:
        raise ValueError(f"invalid GPU range [{start_gpu}, {end_gpu}]")
    span = end_gpu - start_gpu + 1
    if span % num_gpus_per_worker != 0:
        raise ValueError("GPU span must be divisible by learner.placement.num_gpus_per_worker")
    actual_workers = span // num_gpus_per_worker
    if num_workers is not None and actual_workers != num_workers:
        raise ValueError(
            f"packed GPU span must match learner.num_workers ({actual_workers} != {num_workers})"
        )


def _normalize_accelerator_groups(groups: Any) -> list[list[int]]:
    if not groups:
        raise ValueError("accelerator_groups must not be empty")
    normalized = [_parse_accelerator_group(group) for group in list(groups)]
    seen: set[int] = set()
    for group in normalized:
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(f"duplicate accelerator ranks: {sorted(overlap)}")
        seen.update(group)
    return normalized


def _parse_accelerator_group(value: Any) -> list[int]:
    if isinstance(value, str):
        ranks: list[int] = []
        for raw_part in value.split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                if start < 0 or end < start:
                    raise ValueError(f"invalid accelerator range {part!r}")
                ranks.extend(range(start, end + 1))
            else:
                ranks.append(int(part))
    elif isinstance(value, (list, tuple, ListConfig)):
        ranks = [int(rank) for rank in value]
    else:
        raise ValueError(f"unsupported accelerator group {value!r}")
    if not ranks:
        raise ValueError("accelerator group must not be empty")
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"accelerator ranks must be >= 0, got {ranks}")
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"accelerator group contains duplicate ranks: {ranks}")
    return ranks


def _validate_existing_paths(cfg: DictConfig) -> None:
    for key in (
        "dataset.hdf5_dir",
        "dataset.hidden_dir",
        "dataset.config_path",
        "dataset_val_ind.config_path",
        "dataset_val_ood.config_path",
    ):
        value = OmegaConf.select(cfg, key, default=None)
        if value in (None, ""):
            continue
        if not Path(str(value)).expanduser().exists():
            raise ValueError(f"{key} does not exist: {value}")


def _looks_oft_sidecar_cfg(cfg: DictConfig) -> bool:
    expected_action_head = _select_str(cfg, "dataset.expected_action_head_type")
    task_action_head = _select_str(cfg, "task.openvla_oft.hidden_token.expected_action_head_type")
    expected_model_path = _select_str(cfg, "dataset.expected_model_path")
    oft_ckpt_path = _select_str(cfg, "task.openvla_oft.ckpt_path")
    target = _select_str(cfg, "_target_") or ""
    return (
        "OFT" in target
        or (expected_action_head is not None and expected_action_head == task_action_head)
        or (expected_model_path is not None and expected_model_path == oft_ckpt_path)
    )


def _looks_oft_hidden_token_cfg(cfg: DictConfig) -> bool:
    if not _looks_oft_sidecar_cfg(cfg):
        return False
    expected_source = _select_str(cfg, "dataset.expected_obs_hidden_source")
    task_source = _select_str(cfg, "task.openvla_oft.hidden_token.expected_obs_hidden_source")
    return expected_source is not None and expected_source == task_source


def _selected_sidecar_action_horizon_key(cfg: DictConfig) -> str | None:
    if _looks_oft_hidden_token_cfg(cfg):
        return "task.openvla_oft.hidden_token.chunk_size"
    return None


def _require_equal_if_present(
    cfg: DictConfig,
    left_key: str,
    right_key: str,
    *,
    message: str,
) -> None:
    left = OmegaConf.select(cfg, left_key, default=None)
    right = OmegaConf.select(cfg, right_key, default=None)
    if left is None or right is None:
        return
    if left != right:
        raise ValueError(f"{message} {left_key}={left!r}, {right_key}={right!r}")


def _require_positive_if_present(cfg: DictConfig, key: str) -> None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return
    if float(value) <= 0:
        raise ValueError(f"{key} must be > 0, got {value!r}")


def _require_non_negative_if_present(cfg: DictConfig, key: str) -> None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return
    if float(value) < 0:
        raise ValueError(f"{key} must be >= 0, got {value!r}")


def _require_non_negative_int_if_present(cfg: DictConfig, key: str) -> None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return
    int_value = int(value)
    if float(value) != float(int_value) or int_value < 0:
        raise ValueError(f"{key} must be a non-negative integer, got {value!r}")


def _require_positive_int_if_present(cfg: DictConfig, key: str) -> None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return
    int_value = int(value)
    if float(value) != float(int_value) or int_value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")


def _normalize_backends(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_backends = [value]
    elif isinstance(value, (list, tuple, ListConfig)):
        raw_backends = list(value)
    else:
        raw_backends = [value]

    backends: list[str] = []
    for backend in raw_backends:
        normalized = str(backend).strip().lower()
        if normalized in {"", "none", "null", "false", "off", "disabled"}:
            continue
        backends.append(normalized)
    return backends


def _select_str(cfg: DictConfig, key: str) -> str | None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return None
    return str(value)


def _select_int(cfg: DictConfig, key: str) -> int | None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return None
    return int(value)


def _iter_config_nodes(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield every resolved config node so nested aliases cannot bypass gates."""

    if path:
        yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_config_nodes(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from _iter_config_nodes(child, child_path)


def _is_exact_int(value: Any, expected: int) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return float(value) == float(expected) and int(value) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def _int_sequence(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None


def _resolve_world_size(world_size: int | None) -> int:
    if world_size is not None:
        return max(1, int(world_size))
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


__all__ = ["validate_cfg"]
