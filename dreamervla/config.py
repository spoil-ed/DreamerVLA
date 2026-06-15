from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.algorithms.registry import get_actor_update_route
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
    oft_hidden = _select_str(cfg, "task.openvla_oft.action_hidden_dir")

    if rynn_hidden is not None and dataset_hidden == rynn_hidden:
        return
    if oft_hidden is not None and dataset_hidden == oft_hidden:
        return

    if rynn_hidden is not None and _looks_rynn_sidecar_cfg(cfg):
        raise ValueError(
            "dataset.hidden_dir must match task.rynnvla_action_hidden_dir for "
            f"RynnVLA action-hidden routes: {dataset_hidden!r} != {rynn_hidden!r}"
        )
    if oft_hidden is not None and (
        rynn_hidden is None or _looks_oft_sidecar_cfg(cfg)
    ):
        raise ValueError(
            "dataset.hidden_dir must match task.openvla_oft.action_hidden_dir "
            f"for OpenVLA-OFT action-hidden routes: {dataset_hidden!r} != "
            f"{oft_hidden!r}"
        )


def _validate_chunk_horizon_consistency(cfg: DictConfig) -> None:
    _require_equal_if_present(
        cfg,
        "world_model.chunk_size",
        "algorithm.wmpo.chunk_size",
        message="WMPO chunk size must match the world-model chunk size.",
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

    if _looks_oft_sidecar_cfg(cfg):
        _require_equal_if_present(
            cfg,
            "dataset.expected_time_horizon",
            "task.openvla_oft.time_horizon",
            message="OFT dataset horizon must match task.openvla_oft.time_horizon.",
        )
    else:
        _require_equal_if_present(
            cfg,
            "dataset.expected_time_horizon",
            "task.time_horizon",
            message="RynnVLA dataset horizon must match task.time_horizon.",
        )


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


def _resolve_world_size(world_size: int | None) -> int:
    if world_size is not None:
        return max(1, int(world_size))
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


__all__ = ["validate_cfg"]
