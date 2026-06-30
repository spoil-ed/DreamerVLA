from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.algorithms.registry import get_actor_update_route
from dreamervla.models.registry import validate_model_type
from dreamervla.utils.metric_logger import MetricLogger


def validate_cfg(cfg: DictConfig, *, world_size: int | None = None) -> DictConfig:
    """Validate high-value Dreamer-VLA config invariants before runner setup.

    The validation is intentionally lightweight: relationship checks are always
    enabled, while filesystem existence checks are opt-in via
    ``validation.require_existing_paths=true`` so config composition remains
    usable on machines without the full dataset mounted.
    """
    _validate_logger_backends(cfg)
    _validate_algorithm_routes(cfg)
    _validate_training_batch(cfg, world_size=_resolve_world_size(world_size))
    _validate_resume_paths(cfg)
    _validate_sidecar_routes(cfg)
    _validate_chunk_horizon_consistency(cfg)
    _validate_latent_dimension_contracts(cfg)
    _validate_model_registry_refs(cfg)
    _validate_online_cotrain_pipeline(cfg)
    _validate_ray_manual_resources(cfg)
    _validate_fsdp_config(cfg)
    if bool(OmegaConf.select(cfg, "validation.require_existing_paths", default=False)):
        _validate_existing_paths(cfg)
    return cfg


def _validate_logger_backends(cfg: DictConfig) -> None:
    backends = _normalize_backends(
        OmegaConf.select(cfg, "runner.logger.logger_backends", default=None)
    )
    unsupported = [
        backend for backend in backends if backend not in MetricLogger.supported_logger
    ]
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


def _validate_training_batch(cfg: DictConfig, *, world_size: int) -> None:
    batch_size = OmegaConf.select(cfg, "dataloader.batch_size", default=None)
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError(f"dataloader.batch_size must be > 0, got {batch_size!r}")

    grad_accum = int(
        OmegaConf.select(cfg, "training.gradient_accumulate_every", default=1) or 1
    )
    if grad_accum <= 0:
        raise ValueError(
            "training.gradient_accumulate_every must be > 0, "
            f"got {grad_accum!r}"
        )

    global_batch_size = OmegaConf.select(
        cfg, "training.global_batch_size", default=None
    )
    if global_batch_size is None:
        return

    global_batch_size = int(global_batch_size)
    divisor = max(1, int(world_size)) * grad_accum
    if global_batch_size <= 0:
        raise ValueError(
            f"training.global_batch_size must be > 0, got {global_batch_size!r}"
        )
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


def _validate_sidecar_routes(cfg: DictConfig) -> None:
    dataset_hidden = _select_str(cfg, "dataset.hidden_dir")
    if dataset_hidden is None:
        return

    rynn_hidden = _select_str(cfg, "task.rynnvla_action_hidden_dir")
    rynn_input_hidden = _select_str(cfg, "task.rynnvla_input_token_hidden_dir")
    oft_hidden = _select_str(cfg, "task.openvla_oft.action_hidden_dir")
    oft_input_hidden = _select_str(cfg, "task.openvla_oft.input_token_hidden_dir")

    for candidate in (rynn_hidden, rynn_input_hidden, oft_hidden, oft_input_hidden):
        if candidate is not None and dataset_hidden == candidate:
            return

    if rynn_hidden is not None and _looks_rynn_sidecar_cfg(cfg):
        raise ValueError(
            "dataset.hidden_dir must match task.rynnvla_action_hidden_dir for "
            f"RynnVLA action-hidden routes: {dataset_hidden!r} != {rynn_hidden!r}"
        )
    if rynn_input_hidden is not None and _looks_rynn_input_token_cfg(cfg):
        raise ValueError(
            "dataset.hidden_dir must match task.rynnvla_input_token_hidden_dir "
            f"for RynnVLA input-token routes: {dataset_hidden!r} != "
            f"{rynn_input_hidden!r}"
        )
    if oft_hidden is not None and (
        (rynn_hidden is None and rynn_input_hidden is None)
        or _looks_oft_action_hidden_cfg(cfg)
    ):
        raise ValueError(
            "dataset.hidden_dir must match task.openvla_oft.action_hidden_dir "
            f"for OpenVLA-OFT action-hidden routes: {dataset_hidden!r} != "
            f"{oft_hidden!r}"
        )
    if oft_input_hidden is not None and _looks_oft_input_token_cfg(cfg):
        raise ValueError(
            "dataset.hidden_dir must match task.openvla_oft.input_token_hidden_dir "
            f"for OpenVLA-OFT input-token routes: {dataset_hidden!r} != "
            f"{oft_input_hidden!r}"
        )


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


def _validate_latent_dimension_contracts(cfg: DictConfig) -> None:
    for key in (
        "task.legacy_action_hidden",
        "task.openvla_oft",
    ):
        _validate_latent_spec(
            cfg,
            key,
            obs_dim_field="wm_obs_dim",
            action_dim_key="task.action_dim",
            check_action_token_count=True,
        )
    for key in (
        "task.legacy_input_tokens",
        "task.openvla_oft.input_tokens",
    ):
        _validate_latent_spec(cfg, key, obs_dim_field="wm_obs_dim")
    for key in (
        "task.legacy_action_hidden",
        "task.legacy_input_tokens",
        "task.openvla_oft",
        "task.openvla_oft.input_tokens",
    ):
        _validate_latent_stage_value(cfg, key)
    _validate_oft_input_token_patch_contract(cfg)

    for key in (
        "world_model",
        "ray_components.world_model.kwargs",
        "learner.model_cfg.world_model.kwargs",
        "inference.cfg.world_model.kwargs",
    ):
        _validate_latent_spec(cfg, key, obs_dim_field="obs_dim")
        _validate_latent_stage_contract(cfg, key)
        _validate_chunk_wm_token_space(cfg, key)


def _validate_oft_input_token_patch_contract(cfg: DictConfig) -> None:
    key = "task.openvla_oft.input_tokens"
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

    for spec_key in (
        "task.legacy_action_hidden",
        "task.legacy_input_tokens",
        "task.openvla_oft",
        "task.openvla_oft.input_tokens",
    ):
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
        and _looks_oft_input_token_cfg(cfg)
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
                    "input-token query_before proprio/language conditioning"
                )
        for required_key in ("dataset.proprio_keys", "dataset.lang_emb_dir"):
            if OmegaConf.select(cfg, required_key, default=None) in (None, ""):
                raise ValueError(
                    f"{required_key} must be set for OpenVLA-OFT input-token "
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
            f"{key}.action_emb_dim must be set for "
            "ChunkAwareWorldModel WM concat conditioning"
        )
    if action_emb_dim < 1:
        raise ValueError(f"{key}.action_emb_dim must be > 0, got {action_emb_dim}")
    if num_action_repeat is None:
        num_action_repeat = 1
    if num_action_repeat < 1:
        raise ValueError(
            f"{key}.num_action_repeat must be > 0, got {num_action_repeat}"
        )
    if proprio_emb_dim is None:
        proprio_emb_dim = 0
    if proprio_emb_dim < 0:
        raise ValueError(
            f"{key}.proprio_emb_dim must be >= 0, got {proprio_emb_dim}"
        )
    if num_proprio_repeat is None:
        num_proprio_repeat = 1
    if num_proprio_repeat < 1:
        raise ValueError(
            f"{key}.num_proprio_repeat must be > 0, got {num_proprio_repeat}"
        )
    if lang_emb_dim is None:
        lang_emb_dim = 0
    if lang_emb_dim < 0:
        raise ValueError(f"{key}.lang_emb_dim must be >= 0, got {lang_emb_dim}")
    if num_lang_repeat is None:
        num_lang_repeat = 1
    if num_lang_repeat < 1:
        raise ValueError(
            f"{key}.num_lang_repeat must be > 0, got {num_lang_repeat}"
        )
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
    if (
        chunk_size is not None
        and time_horizon is not None
        and chunk_size != time_horizon
    ):
        raise ValueError(
            f"{key}.chunk_size must match {key}.time_horizon "
            f"({chunk_size} != {time_horizon})"
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


def _validate_online_cotrain_pipeline(cfg: DictConfig) -> None:
    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith("OnlineCotrainPipelineRunner"):
        return
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
        "warmup_checkpoint_every",
    ):
        val = int(OmegaConf.select(cfg, f"training.{key}", default=0))
        if val < 0:
            raise ValueError(f"training.{key} must be >= 0, got {val}")
    log_every = int(OmegaConf.select(cfg, "training.replay_warmup_log_every", default=1))
    if log_every < 1:
        raise ValueError(f"training.replay_warmup_log_every must be >= 1, got {log_every}")


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
            "OnlineCotrainRayRunner",
            "ManualCotrainRayRunner",
            "ColdStartRayCollectRunner",
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
    _require_non_negative_if_present(cfg, "env.cfg.egl_step_timeout_s")
    _require_non_negative_int_if_present(cfg, "manual_cotrain.ngpu")
    _require_non_negative_int_if_present(cfg, "manual_cotrain.task_id")
    _require_non_negative_if_present(cfg, "manual_cotrain.env_rollout_timeout_s")
    _require_non_negative_int_if_present(cfg, "manual_cotrain.checkpoint_every")
    _require_positive_if_present(cfg, "manual_cotrain.global_steps")
    _require_positive_if_present(cfg, "manual_cotrain.learner_update_step")
    _require_positive_if_present(cfg, "manual_cotrain.sync_every")
    _require_positive_if_present(cfg, "manual_cotrain.rollout_epoch")
    _require_positive_if_present(cfg, "manual_cotrain.max_steps_per_rollout_epoch")
    _require_positive_if_present(cfg, "manual_cotrain.wm_rollout_multiplier")
    _require_positive_if_present(cfg, "manual_cotrain.num_action_chunks")
    _require_positive_if_present(cfg, "manual_cotrain.envs_per_worker")
    _validate_ray_single_node_placement(cfg)

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
    _validate_manual_cotrain_group_geometry(cfg)
    _validate_manual_cotrain_replay_window(cfg)
    _validate_manual_cotrain_classifier_window(cfg)

    precision = OmegaConf.select(cfg, "learner.train_cfg.precision", default=None)
    if precision is not None:
        normalized = str(precision).strip().lower()
        if normalized not in {"fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}:
            raise ValueError(
                "learner.train_cfg.precision must be one of "
                f"fp32, bf16, or fp16; got {precision!r}"
            )


def _validate_manual_cotrain_group_geometry(cfg: DictConfig) -> None:
    """Validate manual-cotrain rollout slots against actor GRPO grouping."""

    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith("ManualCotrainRayRunner"):
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
    env_workers = ngpu if ngpu > 0 else 1
    envs_per_worker = int(envs_per_worker_raw)
    rollout_epoch = int(
        OmegaConf.select(cfg, "manual_cotrain.rollout_epoch", default=1)
    )
    group_size = int(group_size_raw)
    if group_size <= 0:
        raise ValueError("actor.train_cfg.algorithm_cfg.group_size must be positive")

    logical_trajectory_count = env_workers * envs_per_worker * rollout_epoch
    if logical_trajectory_count % group_size != 0:
        raise ValueError(
            "manual cotrain logical trajectory count must be divisible by "
            "actor.train_cfg.algorithm_cfg.group_size: "
            f"manual_cotrain.ngpu={ngpu}, "
            f"manual_cotrain.envs_per_worker={envs_per_worker}, "
            f"manual_cotrain.rollout_epoch={rollout_epoch}, "
            f"logical trajectory count={logical_trajectory_count}, "
            f"group_size={group_size}"
        )


def _validate_manual_cotrain_replay_window(cfg: DictConfig) -> None:
    """Validate that EnvWorker can produce replay windows used by LearnerWorker."""

    target = str(OmegaConf.select(cfg, "_target_", default="") or "")
    if not target.endswith("ManualCotrainRayRunner"):
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
    if not target.endswith("ManualCotrainRayRunner"):
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
                    f"{base}.precision must be one of fp32, bf16, or fp16; "
                    f"got {precision!r}"
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
                raise ValueError(
                    f"count must match learner.num_workers ({count} != {num_workers})"
                )
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
            raise ValueError(
                "strategy must be one of node, packed, or flexible; "
                f"got {strategy!r}"
            )
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
        raise ValueError(
            "GPU span must be divisible by learner.placement.num_gpus_per_worker"
        )
    actual_workers = span // num_gpus_per_worker
    if num_workers is not None and actual_workers != num_workers:
        raise ValueError(
            "packed GPU span must match learner.num_workers "
            f"({actual_workers} != {num_workers})"
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


def _looks_rynn_sidecar_cfg(cfg: DictConfig) -> bool:
    expected_action_head = _select_str(cfg, "dataset.expected_action_head_type")
    task_action_head = _select_str(
        cfg, "task.legacy_action_hidden.expected_action_head_type"
    )
    expected_source = _select_str(cfg, "dataset.expected_obs_hidden_source")
    task_source = _select_str(
        cfg, "task.legacy_action_hidden.expected_obs_hidden_source"
    )
    return (
        expected_action_head is not None
        and expected_action_head == task_action_head
        and expected_source == task_source
    )


def _looks_rynn_input_token_cfg(cfg: DictConfig) -> bool:
    expected_action_head = _select_str(cfg, "dataset.expected_action_head_type")
    task_action_head = _select_str(
        cfg, "task.legacy_input_tokens.expected_action_head_type"
    )
    expected_source = _select_str(cfg, "dataset.expected_obs_hidden_source")
    task_source = _select_str(
        cfg, "task.legacy_input_tokens.expected_obs_hidden_source"
    )
    return (
        expected_action_head is not None
        and expected_action_head == task_action_head
        and expected_source is not None
        and expected_source == task_source
    )


def _looks_oft_sidecar_cfg(cfg: DictConfig) -> bool:
    expected_action_head = _select_str(cfg, "dataset.expected_action_head_type")
    task_action_head = _select_str(cfg, "task.openvla_oft.expected_action_head_type")
    expected_model_path = _select_str(cfg, "dataset.expected_model_path")
    oft_ckpt_path = _select_str(cfg, "task.openvla_oft.ckpt_path")
    target = _select_str(cfg, "_target_") or ""
    return (
        "OFT" in target
        or (
            expected_action_head is not None
            and expected_action_head == task_action_head
        )
        or (
            expected_model_path is not None
            and expected_model_path == oft_ckpt_path
        )
    )


def _looks_oft_action_hidden_cfg(cfg: DictConfig) -> bool:
    if not _looks_oft_sidecar_cfg(cfg):
        return False
    expected_source = _select_str(cfg, "dataset.expected_obs_hidden_source")
    task_source = _select_str(cfg, "task.openvla_oft.expected_obs_hidden_source")
    return expected_source is None or expected_source == task_source


def _looks_oft_input_token_cfg(cfg: DictConfig) -> bool:
    if not _looks_oft_sidecar_cfg(cfg):
        return False
    expected_source = _select_str(cfg, "dataset.expected_obs_hidden_source")
    task_source = _select_str(
        cfg, "task.openvla_oft.input_tokens.expected_obs_hidden_source"
    )
    return expected_source is not None and expected_source == task_source


def _selected_sidecar_action_horizon_key(cfg: DictConfig) -> str | None:
    if _looks_oft_input_token_cfg(cfg):
        return "task.openvla_oft.input_tokens.chunk_size"
    if _looks_oft_action_hidden_cfg(cfg):
        return "task.openvla_oft.chunk_size"
    if _looks_rynn_input_token_cfg(cfg):
        return "task.legacy_input_tokens.chunk_size"
    if _looks_rynn_sidecar_cfg(cfg):
        return "task.legacy_action_hidden.chunk_size"
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
        raise ValueError(
            f"{message} {left_key}={left!r}, {right_key}={right!r}"
        )


def _require_positive_if_present(cfg: DictConfig, key: str) -> None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return
    if int(value) <= 0:
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


def _resolve_world_size(world_size: int | None) -> int:
    if world_size is not None:
        return max(1, int(world_size))
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


__all__ = ["validate_cfg"]
