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
    for key in ("wm_warmup_steps", "classifier_warmup_steps"):
        val = int(OmegaConf.select(cfg, f"training.{key}", default=0))
        if val < 0:
            raise ValueError(f"training.{key} must be >= 0, got {val}")


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
    _validate_ray_single_node_placement(cfg)

    precision = OmegaConf.select(cfg, "learner.train_cfg.precision", default=None)
    if precision is not None:
        normalized = str(precision).strip().lower()
        if normalized not in {"fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}:
            raise ValueError(
                "learner.train_cfg.precision must be one of "
                f"fp32, bf16, or fp16; got {precision!r}"
            )


def _validate_fsdp_config(cfg: DictConfig) -> None:
    """Fail fast on an unusable learner FSDP block before any worker spawns.

    The learner builds ``FSDPModelManager(**learner.train_cfg.fsdp)`` inside the
    Ray actor, so a bad ``strategy``/``precision`` would otherwise only surface
    after the cluster is up. The accepted strategy set mirrors
    ``FSDPModelManager`` (none/ddp/fsdp/fsdp1).
    """

    fsdp = OmegaConf.select(cfg, "learner.train_cfg.fsdp", default=None)
    if fsdp is None:
        return

    strategy = OmegaConf.select(fsdp, "strategy", default=None)
    if strategy is not None:
        normalized = str(strategy).strip().lower()
        if normalized not in {"", "none", "ddp", "fsdp", "fsdp1"}:
            raise ValueError(
                "learner.train_cfg.fsdp.strategy must be one of "
                f"none, ddp, fsdp, fsdp1; got {strategy!r}"
            )

    precision = OmegaConf.select(fsdp, "precision", default=None)
    if precision is not None:
        normalized = str(precision).strip().lower()
        if normalized not in {"fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}:
            raise ValueError(
                "learner.train_cfg.fsdp.precision must be one of "
                f"fp32, bf16, or fp16; got {precision!r}"
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


def _require_positive_if_present(cfg: DictConfig, key: str) -> None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return
    if int(value) <= 0:
        raise ValueError(f"{key} must be > 0, got {value!r}")


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
