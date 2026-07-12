"""Launch cold-start collection followed by offline-warmup online cotrain."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Literal

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.dataset.collection_manifest import (
    format_collection_report,
    quarantine_corrupt_shards,
    quarantine_incomplete_shards,
    read_manifest,
    resume_plan,
    summarize_collection,
    write_manifest,
)
from dreamervla.runners.render_device_config import parse_device_ids
from dreamervla.utils.egl_device import apply_libero_render_regime
from dreamervla.utils.hydra_config import script_config
from dreamervla.utils.paths import data_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "scripts"

PipelineMode = Literal["ray", "noray"]
CotrainPhase = Literal["all", "warmup", "online"]
_ZERO_GPU_EGL_ERROR = (
    "render_backend=egl requires ngpu>=1; use render_backend=osmesa for ngpu=0"
)
_ZERO_GPU_NORAY_ERROR = (
    "mode=noray does not support ngpu=0 because the no-Ray OFT collector requires CUDA; "
    "use mode=ray render_backend=osmesa for zero-GPU startup"
)
_ZERO_GPU_COMPONENT_PLACEMENT_ERROR = (
    "cluster.component_placement.* overrides are not supported when ngpu=0; "
    "remove them or set ngpu>=1"
)
_DEBUG_ASYNC_ENVS_PER_WORKER = 4


@dataclass(frozen=True)
class PipelinePlan:
    mode: PipelineMode
    profile: str
    task: str
    run_root: Path
    collected_root: Path
    reward_dir: Path
    hidden_dir: Path
    collect_cmd: list[str]
    cotrain_cmd: list[str]
    # Async cotrain (cotrain_engine=async): cotrain splits into a sync warmup-only phase
    # then a ray async online phase initialized from the consolidated warmup ckpt. Empty
    # for the default sync engine.
    cotrain_engine: str = "sync"
    cotrain_phase: CotrainPhase = "all"
    cotrain_warmup_cmd: list[str] = field(default_factory=list)
    cotrain_online_cmd: list[str] = field(default_factory=list)
    warmup_wm_ckpt: Path | None = None
    warmup_cls_ckpt: Path | None = None
    ray_init_ckpt: Path | None = None
    eval_enabled: bool = False
    eval_interval_global_steps: int = 0
    eval_cfg: dict[str, Any] = field(default_factory=dict)
    task_suite: str = ""
    vla_ckpt_path: Path | None = None


def _normalize_mode(mode: str) -> PipelineMode:
    normalized = mode.strip().lower().replace("_", "-")
    if normalized == "ray":
        return "ray"
    if normalized in {"noray", "no-ray", "non-ray"}:
        return "noray"
    raise ValueError("mode must be one of: ray, noray")


def _normalize_cotrain_phase(phase: str) -> CotrainPhase:
    normalized = phase.strip().lower().replace("-", "_")
    if normalized in {"all", "full"}:
        return "all"
    if normalized in {"warmup", "warmup_only"}:
        return "warmup"
    if normalized in {"online", "online_only"}:
        return "online"
    raise ValueError("cotrain_phase must be one of: all, warmup, online")


def _normalise_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _resolve_task(task: str, task_specs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized = task.strip().lower().replace("-", "_")
    if normalized.startswith("libero_"):
        normalized = normalized.removeprefix("libero_")
    try:
        raw = task_specs[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(task_specs))
        raise ValueError(f"task must be one of: {allowed}") from exc
    return normalized, dict(raw)


def _select_mapping(mapping: dict[str, Any], key: str, *, label: str) -> Any:
    normalized = _normalise_key(key)
    try:
        return mapping[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(f"{label} must be one of: {allowed}") from exc


def _render_overrides(items: Sequence[Any], context: dict[str, Any]) -> list[str]:
    return [str(item).format(**context) for item in items]


def _override_key(item: str) -> str:
    return str(item).split("=", 1)[0].lstrip("+~")


def _has_override(overrides: Sequence[str], key: str) -> bool:
    return any(_override_key(item) == key for item in overrides)


def _has_override_or_child(overrides: Sequence[str], key: str) -> bool:
    prefix = f"{key}."
    return any(
        (override_key := _override_key(item)) == key
        or override_key.startswith(prefix)
        for item in overrides
    )


def _debug_collect_overrides(mode: PipelineMode) -> list[str]:
    """Tiny cold-start collection budget for launcher-level ``debug=true``."""
    overrides = [
        "collect.episodes_per_task=1",
        "collect.episode_horizon=16",
        "collect.memory_fraction=0.8",
    ]
    if mode == "ray":
        overrides.extend(
            [
                "env.num_workers=2",
                "collect.num_inference_workers=1",
            ]
        )
    else:
        overrides.append("collect.envs_per_gpu=1")
    return overrides


def _debug_sync_cotrain_overrides() -> list[str]:
    """Tiny sync warmup / online budget for launcher-level ``debug=true``."""
    return [
        "training.debug=true",
        "training.wm_warmup_steps=1",
        "training.classifier_warmup_steps=1",
        "training.warmup_replay_epochs=0",
        "training.classifier_batch_size=1",
        "dataloader.batch_size=1",
        "online_rollout.buffer_size=1000",
        "online_rollout.num_envs=1",
        "online_rollout.total_env_steps=0",
    ]


def _ceil_to_multiple(value: int, multiple: int) -> int:
    divisor = max(1, int(multiple))
    return ((max(0, int(value)) + divisor - 1) // divisor) * divisor


def _lcm_int(left: int, right: int) -> int:
    lhs = max(1, abs(int(left)))
    rhs = max(1, abs(int(right)))
    return lhs * rhs // gcd(lhs, rhs)


def _debug_async_online_overrides(
    *,
    ngpu: int | None,
    explicit_overrides: Sequence[str] = (),
    effective_overrides: Sequence[str] = (),
) -> list[str]:
    """Tiny manual-Ray online budget for launcher-level ``debug=true``.

    Debug keeps multi-worker real-env concurrency but caps the per-worker env width
    unless the caller explicitly sets it. This avoids serial smoke runs while keeping
    the generated trajectory budget small.
    """
    effective_items = [*effective_overrides, *explicit_overrides]
    real_env_workers = (
        _last_int_override(effective_items, "manual_cotrain.real_env_workers") or 1
    )
    envs_per_worker = _debug_async_env_width(
        key="manual_cotrain.envs_per_worker",
        explicit_overrides=explicit_overrides,
        effective_overrides=effective_items,
        default=2,
    )
    wm_envs_per_worker = _debug_async_env_width(
        key="manual_cotrain.wm_envs_per_worker",
        explicit_overrides=explicit_overrides,
        effective_overrides=effective_items,
        default=envs_per_worker,
    )
    group_size = (
        _last_int_override(
            effective_items,
            "actor.train_cfg.algorithm_cfg.group_size",
        )
        or _last_int_override(effective_items, "algorithm.group_size")
        or 8
    )
    num_action_chunks = (
        _last_int_override(effective_items, "manual_cotrain.num_action_chunks") or 8
    )
    replay_sequence_length = (
        _last_int_override(effective_items, "replay.cfg.sequence_length") or 12
    )
    classifier_window = (
        _last_int_override(
            effective_items,
            "learner.model_cfg.classifier.kwargs.window",
        )
        or _last_int_override(
            effective_items,
            "learner.model_cfg.classifier.window",
        )
        or 8
    )
    classifier_chunk_size = (
        _last_int_override(
            effective_items,
            "learner.model_cfg.classifier.kwargs.chunk_size",
        )
        or _last_int_override(
            effective_items,
            "learner.model_cfg.classifier.chunk_size",
        )
        or int(num_action_chunks)
    )
    actor_ranks = max(1, _requested_gpu_count(ngpu))
    wm_workers = max(0, actor_ranks - 1)
    real_target = _ceil_to_multiple(
        max(8, int(real_env_workers) * int(envs_per_worker)),
        int(envs_per_worker),
    )
    wm_target = _ceil_to_multiple(
        max(8, int(wm_workers) * int(wm_envs_per_worker)),
        _lcm_int(
            int(wm_envs_per_worker),
            int(group_size) * int(actor_ranks),
        ),
    )
    max_steps_per_rollout_epoch = _ceil_to_multiple(
        max(
            int(replay_sequence_length),
            int(classifier_window) * int(classifier_chunk_size),
        ),
        int(num_action_chunks),
    )
    ppo_samples = (
        int(wm_target)
        * int(max_steps_per_rollout_epoch)
        // int(num_action_chunks)
    )
    actor_batch_overrides: list[str] = []
    explicit_global_batch = _last_int_override(
        explicit_overrides,
        "actor.train_cfg.global_batch_size",
    )
    resolved_global_batch = explicit_global_batch or ppo_samples
    if explicit_global_batch is None:
        actor_batch_overrides.append(
            f"actor.train_cfg.global_batch_size={resolved_global_batch}"
        )
    if not _has_override(explicit_overrides, "actor.train_cfg.micro_batch_size"):
        per_rank_global_batch = max(1, resolved_global_batch // actor_ranks)
        actor_batch_overrides.append(
            "actor.train_cfg.micro_batch_size="
            f"{_largest_divisor_at_most(per_rank_global_batch, 32)}"
        )
    return [
        "++training.debug=true",
        "manual_cotrain.global_steps=1",
        "manual_cotrain.rollout_epoch=1",
        "manual_cotrain.real_rollout_epoch=1",
        "manual_cotrain.wm_rollout_epoch=1",
        f"manual_cotrain.real_rollout_target_trajectories={real_target}",
        f"manual_cotrain.wm_rollout_target_trajectories={wm_target}",
        f"manual_cotrain.max_steps_per_rollout_epoch={max_steps_per_rollout_epoch}",
        *actor_batch_overrides,
        *_debug_async_env_width_overrides(
            explicit_overrides=explicit_overrides,
            envs_per_worker=envs_per_worker,
            wm_envs_per_worker=wm_envs_per_worker,
        ),
    ]


def _debug_async_env_width(
    *,
    key: str,
    explicit_overrides: Sequence[str],
    effective_overrides: Sequence[str],
    default: int,
) -> int:
    explicit_value = _last_int_override(explicit_overrides, key)
    if explicit_value is not None:
        return int(explicit_value)
    effective_value = _last_int_override(effective_overrides, key)
    width = int(effective_value if effective_value is not None else default)
    return max(1, min(width, _DEBUG_ASYNC_ENVS_PER_WORKER))


def _debug_async_env_width_overrides(
    *,
    explicit_overrides: Sequence[str],
    envs_per_worker: int,
    wm_envs_per_worker: int,
) -> list[str]:
    overrides: list[str] = []
    if not _has_override(explicit_overrides, "manual_cotrain.envs_per_worker"):
        overrides.append(f"manual_cotrain.envs_per_worker={int(envs_per_worker)}")
    if not _has_override(explicit_overrides, "manual_cotrain.wm_envs_per_worker"):
        overrides.append(
            f"manual_cotrain.wm_envs_per_worker={int(wm_envs_per_worker)}"
        )
    return overrides


def _without_override_keys(overrides: Sequence[str], keys: set[str]) -> list[str]:
    return [item for item in overrides if _override_key(item) not in keys]


def _override_value(item: str) -> str | None:
    text = str(item)
    if "=" not in text:
        return None
    return text.split("=", 1)[1]


def _hydra_string_value(value: str) -> str:
    text = str(value)
    if "," not in text:
        return text
    escaped = text.replace("'", "\\'")
    return f"'{escaped}'"


def _last_int_override(overrides: Sequence[str], key: str) -> int | None:
    value: int | None = None
    for item in overrides:
        if _override_key(item) != key:
            continue
        raw = _override_value(item)
        if raw is None:
            continue
        text = raw.strip().strip("'\"")
        if text.lower() in {"", "null", "none"}:
            value = None
            continue
        try:
            value = int(float(text))
        except ValueError:
            value = None
    return value


def _normalize_render_backend(value: str) -> str:
    return str(value).strip().strip("'\"").lower()


def _effective_render_backend(
    default: str,
    overrides: Sequence[str],
    *,
    keys: Sequence[str],
) -> str:
    backend = str(default)
    allowed = set(keys)
    for item in overrides:
        if _override_key(item) not in allowed:
            continue
        value = _override_value(item)
        if value is not None:
            backend = value
    return _normalize_render_backend(backend)


def _validate_zero_gpu_render_backend(ngpu: int | None, render_backend: str) -> None:
    if (
        _requested_gpu_count(ngpu) == 0
        and _normalize_render_backend(render_backend) == "egl"
    ):
        raise ValueError(_ZERO_GPU_EGL_ERROR)


def _validate_zero_gpu_route(mode: PipelineMode, ngpu: int | None) -> None:
    if mode == "noray" and _requested_gpu_count(ngpu) == 0:
        raise ValueError(_ZERO_GPU_NORAY_ERROR)


def _validate_zero_gpu_component_placement_overrides(
    ngpu: int | None,
    overrides: Sequence[str],
) -> None:
    if _requested_gpu_count(ngpu) != 0:
        return
    if any(
        (key := _override_key(item)) == "cluster.component_placement"
        or key.startswith("cluster.component_placement.")
        for item in overrides
    ):
        raise ValueError(_ZERO_GPU_COMPONENT_PLACEMENT_ERROR)


def _zero_gpu_ray_collect_overrides(explicit_overrides: Sequence[str]) -> list[str]:
    defaults = [
        "collect.num_inference_workers=1",
        "env.num_workers=1",
        "++inference.device=cpu",
    ]
    return [
        override
        for override in defaults
        if not _has_override(explicit_overrides, _override_key(override))
    ]


def _zero_gpu_sync_cotrain_overrides(explicit_overrides: Sequence[str]) -> list[str]:
    defaults = [
        "trainer.device=cpu",
        "training.distributed_strategy=ddp",
        "training.fsdp_mixed_precision=fp32",
        "encoder.torch_dtype=fp32",
        "optim.precision=fp32",
        "online_rollout.num_envs=1",
    ]
    return [
        override
        for override in defaults
        if not _has_override(explicit_overrides, _override_key(override))
    ]


def _format_hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Sequence) and not isinstance(value, str):
        return "[" + ",".join(_format_hydra_value(item) for item in value) + "]"
    return str(value)


def _control_overrides(
    values: Any,
    mapping: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    values = _plain(values)
    mapping = _plain(mapping)
    if not isinstance(values, Mapping) or not isinstance(mapping, Mapping):
        return []

    provided = {str(key) for key, value in values.items() if value is not None}
    allowed = {str(key) for key in mapping}
    unsupported = sorted(provided - allowed)
    if unsupported:
        allowed_label = ", ".join(sorted(allowed))
        raise ValueError(
            f"{label} controls not supported for the selected route: "
            f"{', '.join(unsupported)} (allowed: {allowed_label})"
        )

    overrides: list[str] = []
    for source_key, target_key in mapping.items():
        value = values.get(source_key)
        if value is None:
            continue
        overrides.append(f"{target_key}={_format_hydra_value(value)}")
    return overrides


def _collect_control_mapping(cfg: dict[str, Any], mode: PipelineMode) -> dict[str, Any]:
    controls = _plain(cfg.get("control_overrides", {}))
    if not isinstance(controls, Mapping):
        return {}
    collect_controls = _plain(controls.get("collect", {}))
    if not isinstance(collect_controls, Mapping):
        return {}
    common = _plain(collect_controls.get("common", {}))
    mode_specific = _plain(collect_controls.get(mode, {}))
    merged: dict[str, Any] = {}
    if isinstance(common, Mapping):
        merged.update(common)
    if isinstance(mode_specific, Mapping):
        merged.update(mode_specific)
    return merged


def _warmup_control_mapping(cfg: dict[str, Any]) -> dict[str, Any]:
    controls = _plain(cfg.get("control_overrides", {}))
    if not isinstance(controls, Mapping):
        return {}
    warmup_controls = _plain(controls.get("warmup", {}))
    return dict(warmup_controls) if isinstance(warmup_controls, Mapping) else {}


def _requested_gpu_count(ngpu: int | None) -> int:
    count = int(ngpu or 0)
    if count < 0:
        raise ValueError(f"ngpu must be >= 0, got {ngpu!r}")
    return count


def _scale_gpu_count(ngpu: int | None) -> int:
    return max(1, _requested_gpu_count(ngpu))


def _scaled_profile_count(
    profile_cfg: Mapping[str, Any],
    key: str,
    *,
    ngpu: int,
) -> int | None:
    raw = _plain(profile_cfg).get(key)
    if raw is None:
        return None
    per_gpu = int(raw)
    if per_gpu < 1:
        raise ValueError(f"profile.{key} must be >= 1 when set")
    return _scale_gpu_count(ngpu) * per_gpu


def _profile_per_gpu_count(profile_cfg: Mapping[str, Any], key: str) -> int | None:
    raw = _plain(profile_cfg).get(key)
    if raw is None:
        return None
    per_gpu = int(raw)
    if per_gpu < 1:
        raise ValueError(f"profile.{key} must be >= 1 when set")
    return per_gpu


def _cap_at_ngpu(value: int, ngpu: int) -> int:
    """Cap a worker count at the selected GPU count with a floor of 1.

    Profiles declare a worker count (e.g. 4) that assumes at least that many GPUs;
    at a smaller GPU count Ray placement asks for a GPU index that does not exist.
    Capping at min(value, ngpu) leaves ngpu>=value byte-identical while letting
    ngpu<value run (ngpu=2 -> 2, ngpu>=4 -> 4).
    """
    return max(1, min(int(value), int(ngpu)))


def _cap_override_at_ngpu(
    overrides: Sequence[str],
    key: str,
    *,
    ngpu: int,
) -> list[str]:
    """Rewrite an integer override's value in-place to min(value, ngpu) (floor 1)."""
    capped: list[str] = []
    for item in overrides:
        raw = _override_value(item)
        if _override_key(item) != key or raw is None:
            capped.append(item)
            continue
        value = int(str(raw).strip().strip("'\""))
        capped.append(f"{key}={_cap_at_ngpu(value, ngpu)}")
    return capped


def _sync_cotrain_scale_overrides(
    profile_cfg: Mapping[str, Any],
    *,
    ngpu: int,
    explicit_overrides: Sequence[str],
) -> list[str]:
    if _requested_gpu_count(ngpu) == 0:
        return []
    if _has_override(explicit_overrides, "online_rollout.num_envs"):
        return []
    num_envs = _scaled_profile_count(
        profile_cfg,
        "online_rollout_envs_per_gpu",
        ngpu=ngpu,
    )
    return [] if num_envs is None else [f"online_rollout.num_envs={num_envs}"]


def _ray_online_scale_overrides(
    profile_cfg: Mapping[str, Any],
    *,
    ngpu: int,
    render_backend: str,
    explicit_overrides: Sequence[str],
) -> list[str]:
    overrides: list[str] = []
    requested_gpu_count = _requested_gpu_count(ngpu)
    gpu_count = _scale_gpu_count(ngpu)
    envs_per_gpu = _profile_per_gpu_count(
        profile_cfg,
        "online_rollout_envs_per_gpu",
    )
    real_envs_per_worker = _profile_per_gpu_count(
        profile_cfg,
        "ray_online_real_envs_per_worker",
    )
    scaled_envs = _scaled_profile_count(
        profile_cfg,
        "online_rollout_envs_per_gpu",
        ngpu=ngpu,
    )
    real_env_workers = _last_int_override(
        explicit_overrides,
        "manual_cotrain.real_env_workers",
    )
    if real_env_workers is None:
        raw_real_env_workers = _plain(profile_cfg).get("ray_online_real_env_workers", 1)
        real_env_workers = int(raw_real_env_workers)
    if not _has_override(explicit_overrides, "render_backend"):
        overrides.append(f"render_backend={render_backend}")

    normalized_render_backend = render_backend.strip().lower()
    if requested_gpu_count == 0:
        if normalized_render_backend == "egl":
            raise ValueError(_ZERO_GPU_EGL_ERROR)
        cpu_defaults = [
            "cluster.component_placement=null",
            "inference.placement.strategy=node",
            "++inference.device=cpu",
            "learner.placement.strategy=node",
            "learner.train_cfg.device=cpu",
            "learner.train_cfg.precision=fp32",
            "env.num_workers=1",
        ]
        for override in cpu_defaults:
            if not _has_override(explicit_overrides, _override_key(override)):
                overrides.append(override)
        return overrides

    if normalized_render_backend != "egl":
        if scaled_envs is not None and not _has_override(
            explicit_overrides, "env.num_workers"
        ):
            overrides.append(f"env.num_workers={scaled_envs}")
        return overrides

    egl_envs_per_worker = real_envs_per_worker or envs_per_gpu
    if egl_envs_per_worker is not None:
        if not _has_override(explicit_overrides, "env.num_workers"):
            overrides.append(f"env.num_workers={gpu_count}")
        if not _has_override(explicit_overrides, "env.envs_per_worker"):
            overrides.append(f"env.envs_per_worker={egl_envs_per_worker}")

    overrides.extend(
        _ray_online_egl_component_placement_overrides(
            ngpu=gpu_count,
            real_env_workers=real_env_workers,
            explicit_overrides=explicit_overrides,
        )
    )

    return overrides


def _ray_online_egl_component_placement_overrides(
    *,
    ngpu: int,
    real_env_workers: int,
    explicit_overrides: Sequence[str],
) -> list[str]:
    if _has_override(explicit_overrides, "cluster.component_placement"):
        return []

    overrides: list[str] = []
    gpu_count = _requested_gpu_count(ngpu)
    if gpu_count < 1:
        raise ValueError("EGL component placement requires ngpu>=1")
    env_ranks = "0" if gpu_count == 1 else f"0-{gpu_count - 1}"
    actor_rank = gpu_count - 1
    rollout_ranks = _manual_rollout_component_rank_map(
        env_worker_count=gpu_count,
        gpu_count=gpu_count,
        real_env_workers=real_env_workers,
    )
    defaults = {
        "cluster.component_placement.env": env_ranks,
        "cluster.component_placement.rollout": _hydra_string_value(rollout_ranks),
        "cluster.component_placement.actor": str(actor_rank),
    }
    for key, value in defaults.items():
        if not _has_override(explicit_overrides, key):
            overrides.append(f"{key}={value}")
    return overrides


def _manual_rollout_component_rank_map(
    *,
    env_worker_count: int,
    gpu_count: int,
    real_env_workers: int,
) -> str:
    workers = int(env_worker_count)
    if workers <= 0:
        raise ValueError("env_worker_count must be positive")
    count = int(gpu_count)
    if count <= 0:
        raise ValueError("gpu_count must be positive")
    if workers == 1 and count == 1:
        return "0"
    start_gpu = min(max(1, int(real_env_workers)), count - 1)
    compute_gpus = list(range(start_gpu, count)) or [count - 1]
    if len(compute_gpus) >= 3 and workers > 1:
        actor_gpu = compute_gpus[-1]
        non_actor_gpus = compute_gpus[:-1]
        distributed_workers = workers - 1
    else:
        actor_gpu = None
        non_actor_gpus = compute_gpus
        distributed_workers = workers
    base, extra = divmod(distributed_workers, len(non_actor_gpus))
    segments: list[str] = []
    next_rank = 0
    for index, gpu in enumerate(non_actor_gpus):
        process_count = base + (1 if index < extra else 0)
        if process_count <= 0:
            continue
        end_rank = next_rank + process_count - 1
        process_ranks = (
            str(next_rank) if next_rank == end_rank else f"{next_rank}-{end_rank}"
        )
        segments.append(f"{gpu}:{process_ranks}")
        next_rank = end_rank + 1
    if actor_gpu is not None:
        segments.append(f"{actor_gpu}:{next_rank}")
        next_rank += 1
    if next_rank != workers:
        raise RuntimeError("failed to assign every rollout worker rank")
    return ",".join(segments)


def _validate_online_wm_env_reservation(
    profile_cfg: Mapping[str, Any],
    *,
    real_env_workers: int | None,
    ngpu: int,
) -> None:
    """Fail fast when the WM-fed async actor would get zero wm_env workers.

    GPU placement assigns ``gpu < real_env_workers`` to ``real_env`` and the rest to
    ``wm_env``; the async actor is fed ONLY by wm_env rollout shards, so
    ``real_env_workers >= ngpu`` leaves no wm_env worker and the actor starves ~60 min
    into the run (``collate_trajectory_shards requires at least one shard``). Guard only
    the multi-GPU case (ngpu>=2): a single GPU cannot host a separate wm_env worker.
    """
    if int(ngpu) < 2 or real_env_workers is None:
        return
    wm_rollout_target = _plain(profile_cfg).get(
        "ray_online_wm_rollout_target_trajectories"
    )
    if wm_rollout_target is None or int(wm_rollout_target) <= 0:
        return
    if int(real_env_workers) < int(ngpu):
        return
    raise ValueError(
        "async online cotrain feeds the actor from world-model (wm_env) rollouts, but "
        f"real_env_workers ({int(real_env_workers)}) leaves no GPU for wm_env at ngpu "
        f"({int(ngpu)}); need ngpu > real_env_workers"
    )


def _largest_divisor_at_most(target: int, limit: int) -> int:
    """Largest positive divisor of ``target`` that is <= ``limit`` (floor 1)."""
    limit = min(int(limit), int(target))
    for candidate in range(max(1, limit), 0, -1):
        if int(target) % candidate == 0:
            return candidate
    return 1


def _cap_real_envs_per_worker(
    profile_width: int,
    target_trajectories: int | None,
    real_env_workers: int | None,
) -> int:
    """Cap the real rollout width so the runner rollout-distribution guard holds.

    Returns ``min(profile_width, target // workers)`` clamped DOWN to the largest
    divisor of ``target`` so both guard invariants hold at any GPU count:
    ``target % envs == 0`` and ``target // envs >= workers``. When the profile omits a
    real rollout target, no divisibility applies and the profile width passes through.
    """
    width = max(1, int(profile_width))
    if target_trajectories is None:
        return width
    target = int(target_trajectories)
    if target <= 0:
        return width
    workers = max(1, int(real_env_workers or 1))
    limit = min(width, target // workers)
    return _largest_divisor_at_most(target, max(1, limit))


def _manual_cotrain_online_overrides(
    profile_cfg: Mapping[str, Any],
    *,
    ngpu: int,
    render_backend: str,
    explicit_overrides: Sequence[str],
    online_budget_overrides: Sequence[str] = (),
) -> list[str]:
    overrides: list[str] = []
    defaults: list[str] = []
    if not _has_override_or_child(explicit_overrides, "cluster.component_placement"):
        defaults.append("cluster.component_placement=null")
    defaults.extend(
        [
            f"manual_cotrain.ngpu={_requested_gpu_count(ngpu)}",
            f"+cluster.num_gpus={_requested_gpu_count(ngpu)}",
        ]
    )
    if not _has_override(explicit_overrides, "env.cfg.render_backend"):
        defaults.append(f"env.cfg.render_backend={render_backend}")
    envs_per_worker = _profile_per_gpu_count(
        profile_cfg,
        "ray_online_real_envs_per_worker",
    )
    if envs_per_worker is None:
        envs_per_worker = _profile_per_gpu_count(
            profile_cfg,
            "online_rollout_envs_per_gpu",
        )
    profile_real_envs_per_worker = envs_per_worker or 8
    requested_gpu = _requested_gpu_count(ngpu)
    real_env_workers = _plain(profile_cfg).get("ray_online_real_env_workers")
    capped_real_env_workers: int | None = None
    if real_env_workers is not None:
        # One real-env worker binds one GPU. The async actor is fed ONLY by
        # world-model (wm_env) rollout shards, so reserve at least one GPU for a
        # wm_env worker by capping at ngpu-1 (floor 1) rather than the raw GPU count:
        # a profile count >= ngpu would otherwise leave zero wm_env workers and starve
        # the actor. ngpu>=5 stays min(4, ngpu-1)=4, byte-identical to the profile 4.
        capped_real_env_workers = (
            min(int(real_env_workers), max(1, requested_gpu - 1))
            if requested_gpu > 0
            else int(real_env_workers)
        )
        defaults.append(
            f"manual_cotrain.real_env_workers={capped_real_env_workers}"
        )
    effective_real_env_workers = _last_int_override(
        explicit_overrides, "manual_cotrain.real_env_workers"
    )
    if effective_real_env_workers is None:
        effective_real_env_workers = capped_real_env_workers
    # Cap the real rollout width so the runner rollout-distribution guard holds at ANY
    # GPU count: `target % envs == 0` AND `target // envs >= real_env_workers`. The
    # profile declares the RLinf rollout width (16); at low GPU counts we keep it, but
    # when `target // workers` drops below it we clamp DOWN to the largest divisor of
    # `target` that still gives every real-env worker at least one rollout_epoch.
    default_envs_per_worker = _cap_real_envs_per_worker(
        profile_real_envs_per_worker,
        _plain(profile_cfg).get("ray_online_real_rollout_target_trajectories"),
        effective_real_env_workers,
    )
    defaults.append(f"manual_cotrain.envs_per_worker={default_envs_per_worker}")
    _validate_online_wm_env_reservation(
        profile_cfg,
        real_env_workers=effective_real_env_workers,
        ngpu=requested_gpu,
    )
    real_render_backend = _plain(profile_cfg).get("ray_online_real_render_backend")
    if real_render_backend is not None:
        defaults.append(
            "manual_cotrain.real_render_backend="
            f"{str(real_render_backend).strip().lower()}"
        )
    wm_envs_per_worker = _plain(profile_cfg).get("ray_online_wm_envs_per_worker")
    if wm_envs_per_worker is not None:
        defaults.append(
            f"manual_cotrain.wm_envs_per_worker={int(wm_envs_per_worker)}"
        )
    real_rollout_target = _plain(profile_cfg).get(
        "ray_online_real_rollout_target_trajectories"
    )
    if real_rollout_target is not None:
        defaults.append(
            "manual_cotrain.real_rollout_target_trajectories="
            f"{int(real_rollout_target)}"
        )
    wm_rollout_target = _plain(profile_cfg).get(
        "ray_online_wm_rollout_target_trajectories"
    )
    if wm_rollout_target is not None:
        defaults.append(
            "manual_cotrain.wm_rollout_target_trajectories="
            f"{int(wm_rollout_target)}"
        )
    max_steps_per_rollout_epoch = _plain(profile_cfg).get(
        "ray_online_max_steps_per_rollout_epoch"
    )
    if max_steps_per_rollout_epoch is not None:
        defaults.append(
            "manual_cotrain.max_steps_per_rollout_epoch="
            f"{int(max_steps_per_rollout_epoch)}"
        )
    real_rollout_epoch = _plain(profile_cfg).get("ray_online_real_rollout_epoch")
    if real_rollout_epoch is not None:
        defaults.append(
            f"manual_cotrain.real_rollout_epoch={int(real_rollout_epoch)}"
        )
    global_steps = _manual_cotrain_global_steps_from_budget(
        online_budget_overrides,
        explicit_overrides=explicit_overrides,
        default_envs_per_worker=default_envs_per_worker,
        default_wm_rollout_target_trajectories=(
            None if wm_rollout_target is None else int(wm_rollout_target)
        ),
    )
    if global_steps is not None:
        defaults.append(f"manual_cotrain.global_steps={global_steps}")
    rollout_timeout_s = _manual_cotrain_env_rollout_timeout_s(profile_cfg)
    if rollout_timeout_s is not None:
        defaults.append(f"manual_cotrain.env_rollout_timeout_s={rollout_timeout_s}")
    if _requested_gpu_count(ngpu) == 0:
        defaults.extend(
            [
                "actor.train_cfg.fsdp.strategy=none",
                "actor.train_cfg.device=cpu",
                "learner.train_cfg.device=cpu",
                "rollout.train_cfg.device=cpu",
            ]
        )
    for override in defaults:
        if not _has_override(explicit_overrides, _override_key(override)):
            overrides.append(override)
    return overrides


def _manual_cotrain_env_rollout_timeout_s(
    profile_cfg: Mapping[str, Any],
) -> int | None:
    # Applies to every render backend. osmesa (the mainline default) is CPU
    # software rendering and SLOWER than egl, so it needs this long rollout
    # timeout even more than egl did — the previous egl-only gate left osmesa on
    # the short 600s default and timed out real-env rollouts on slow/few-GPU boxes.
    raw = _plain(profile_cfg).get("ray_online_env_rollout_timeout_s", 2400)
    if raw is None:
        return None
    return int(float(raw))


def _manual_cotrain_global_steps_from_budget(
    online_budget_overrides: Sequence[str],
    *,
    explicit_overrides: Sequence[str],
    default_envs_per_worker: int,
    default_wm_rollout_target_trajectories: int | None = None,
) -> int | None:
    if _has_override(explicit_overrides, "manual_cotrain.global_steps"):
        return None
    total_env_steps = _last_int_override(
        [*online_budget_overrides, *explicit_overrides],
        "online_rollout.total_env_steps",
    )
    if total_env_steps is None or total_env_steps <= 0:
        return None
    envs_per_worker = (
        _last_int_override(explicit_overrides, "manual_cotrain.envs_per_worker")
        or int(default_envs_per_worker)
    )
    rollout_epoch = (
        _last_int_override(explicit_overrides, "manual_cotrain.rollout_epoch")
        or _last_int_override(explicit_overrides, "algorithm.rollout_epoch")
        or 16
    )
    max_steps = (
        _last_int_override(
            explicit_overrides,
            "manual_cotrain.max_steps_per_rollout_epoch",
        )
        or 512
    )
    wm_rollout_target = (
        _last_int_override(
            explicit_overrides,
            "manual_cotrain.wm_rollout_target_trajectories",
        )
        or default_wm_rollout_target_trajectories
        or 128
    )
    if int(wm_rollout_target) > 0:
        steps_per_global_step = max(1, int(wm_rollout_target) * int(max_steps))
        return max(
            1,
            (int(total_env_steps) + steps_per_global_step - 1)
            // steps_per_global_step,
        )
    steps_per_global_step = max(
        1,
        int(envs_per_worker) * int(rollout_epoch) * int(max_steps),
    )
    return max(1, (int(total_env_steps) + steps_per_global_step - 1) // steps_per_global_step)


def build_pipeline_plan(
    *,
    mode: str | None = None,
    profile: str | None = None,
    task: str | None = None,
    run_root: str | Path,
    python: str | None = None,
    launcher_cfg: dict[str, Any] | None = None,
    collect_overrides: Sequence[str] = (),
    cotrain_overrides: Sequence[str] = (),
    common_overrides: Sequence[str] = (),
    ngpu: int | None = None,
    master_port: int | None = None,
    debug: bool | None = None,
    cotrain_phase: str | None = None,
) -> PipelinePlan:
    cfg = script_config("coldstart_warmup_cotrain") if launcher_cfg is None else launcher_cfg
    selected_mode = _normalize_mode(str(cfg["mode"] if mode is None else mode))
    selected_profile = _normalise_key(str(cfg["profile"] if profile is None else profile))
    selected_cotrain_phase = _normalize_cotrain_phase(
        str(cfg.get("cotrain_phase", "all") if cotrain_phase is None else cotrain_phase)
    )
    _select_mapping(dict(cfg["profiles"]), selected_profile, label="profile")
    task_name, task_spec = _resolve_task(str(cfg["task"] if task is None else task), dict(cfg["tasks"]))
    python_cmd = str(cfg["python"] if python is None else python)
    # Multi-GPU cotrain runs under torchrun DDP; collection stays single-process
    # (noray = vectorized one-GPU collector; ray = its own worker fan-out).
    selected_ngpu = _requested_gpu_count(
        cfg.get("ngpu", 1) if ngpu is None else ngpu
    )
    _validate_zero_gpu_route(selected_mode, selected_ngpu)
    selected_master_port = int(
        cfg.get("master_port", 29500) if master_port is None else master_port
    )
    distributed = bool(cfg.get("distributed", True))
    selected_render_backend = str(cfg.get("render_backend", "osmesa"))
    debug_enabled = bool(cfg.get("debug", False) if debug is None else debug)
    root = Path(run_root).expanduser()
    # Collected episodes live in a stable, per-suite UNIFIED space so reruns
    # accumulate/resume there; only the training outputs stay run-isolated.
    collected_root = _data_root() / "collected_rollouts" / str(task_spec["suite"])
    reward_dir = collected_root / "reward"
    hidden_dir = collected_root / "hidden"
    collect_out = root / "collect"
    cotrain_out = root / "cotrain"
    context = {
        **task_spec,
        "task": task_name,
        "mode": selected_mode,
        "profile": selected_profile,
        "run_root": str(root),
        "reward_dir": str(reward_dir),
        "hidden_dir": str(hidden_dir),
        "collect_out": str(collect_out),
        "cotrain_out": str(cotrain_out),
        "render_backend": selected_render_backend,
    }
    eval_cfg = dict(cfg.get("eval") or {})
    eval_enabled = bool(eval_cfg.get("enabled", False) or debug_enabled)
    eval_interval_global_steps = _eval_interval_global_steps(
        eval_cfg,
        debug=debug_enabled,
    )
    mode_cfg = _select_mapping(dict(cfg["modes"]), selected_mode, label="mode")
    profile_cfg = _select_mapping(dict(cfg["profiles"]), selected_profile, label="profile")
    collect_profile_cfg = _select_mapping(
        dict(profile_cfg["collect"]),
        selected_mode,
        label=f"profile.{selected_profile}.collect",
    )
    collect_profile_items = _render_overrides(collect_profile_cfg, context)
    if selected_ngpu == 0:
        collect_profile_items = _without_override_keys(
            collect_profile_items,
            {"collect.num_inference_workers", "env.num_workers"},
        )
    else:
        # A profile's ray inference-worker count (e.g. 4) needs that many GPUs; cap it
        # at the selected GPU count so ngpu<4 does not request a missing GPU index.
        collect_profile_items = _cap_override_at_ngpu(
            collect_profile_items,
            "collect.num_inference_workers",
            ngpu=selected_ngpu,
        )
    collect_control_items = _control_overrides(
        cfg.get("collect"),
        _collect_control_mapping(cfg, selected_mode),
        label="collect",
    )
    cotrain_profile_items = _render_overrides(profile_cfg["cotrain"], context)
    debug_collect_items = (
        _debug_collect_overrides(selected_mode) if debug_enabled else []
    )
    debug_cotrain_items = _debug_sync_cotrain_overrides() if debug_enabled else []
    warmup_control_items = _control_overrides(
        cfg.get("warmup"),
        _warmup_control_mapping(cfg),
        label="warmup",
    )
    cotrain_engine = str(cfg.get("cotrain_engine", "sync")).strip().lower()
    sync_render_backend = _effective_render_backend(
        selected_render_backend,
        [
            *cotrain_profile_items,
            *warmup_control_items,
            *common_overrides,
            *cotrain_overrides,
        ],
        keys=("online_rollout.render_backend",),
    )
    _validate_zero_gpu_render_backend(selected_ngpu, sync_render_backend)
    async_render_backend = _effective_render_backend(
        selected_render_backend,
        [*common_overrides, *cotrain_overrides],
        keys=("render_backend",),
    )
    if cotrain_engine == "async":
        _validate_zero_gpu_render_backend(selected_ngpu, async_render_backend)
    _validate_zero_gpu_component_placement_overrides(
        selected_ngpu,
        [*common_overrides, *cotrain_overrides],
    )
    zero_gpu_collect_items = (
        _zero_gpu_ray_collect_overrides(
            [
                *collect_profile_items,
                *collect_control_items,
                *common_overrides,
                *collect_overrides,
            ]
        )
        if selected_ngpu == 0 and selected_mode == "ray"
        else []
    )
    zero_gpu_cotrain_items = (
        _zero_gpu_sync_cotrain_overrides(
            [
                *cotrain_profile_items,
                *warmup_control_items,
                *common_overrides,
                *cotrain_overrides,
            ]
        )
        if selected_ngpu == 0
        else []
    )

    collect_launch = [python_cmd, "-m"]
    if selected_mode == "noray" and distributed and selected_ngpu > 1:
        # The no-Ray collector shards work by torchrun rank and binds gpu_id=local_rank
        # (collect_parallel_rollouts.collect_rollouts), so wrapping the collect command
        # in torch.distributed.run gives multi-GPU collection. Ray collection fans out
        # via its own worker groups instead (no torchrun).
        collect_launch += [
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={selected_ngpu}",
            f"--master_port={selected_master_port}",
            "-m",
        ]
    collect_cmd = [
        *collect_launch,
        "dreamervla.train",
        *_render_overrides(mode_cfg["collect"], context),
        *collect_profile_items,
    ]
    # Ray env-worker parallelism scales with the GPU count so it is not stuck at the
    # config default of a couple of CPU env workers (a throughput bottleneck on a
    # multi-GPU box). Emitted BEFORE the controls so an explicit collect.num_workers
    # still wins; noray sizes its env concurrency via collect.envs_per_gpu instead.
    ray_env_scale: list[str] = []
    explicit_collect_overrides = [
        *collect_profile_items,
        *zero_gpu_collect_items,
        *debug_collect_items,
        *collect_control_items,
        *common_overrides,
        *collect_overrides,
    ]
    if selected_mode == "ray" and selected_ngpu > 0 and not _has_override(
        explicit_collect_overrides,
        "env.num_workers",
    ):
        ray_env_scale = [f"env.num_workers={_scale_gpu_count(selected_ngpu) * 4}"]
    collect_cmd.extend(
        [
            f"task.openvla_oft.hdf5_reward_dir={reward_dir}",
            f"task.openvla_oft.hidden_token_dir={hidden_dir}",
            f"++collect.hdf5_reward_dir={reward_dir}",
            f"++collect.hidden_dir={hidden_dir}",
            f"training.out_dir={collect_out}",
            *ray_env_scale,
            *zero_gpu_collect_items,
            *debug_collect_items,
            *collect_control_items,
            *common_overrides,
            *collect_overrides,
        ]
    )
    cotrain_launch = [python_cmd, "-m"]
    if distributed and selected_ngpu > 1:
        cotrain_launch += [
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={selected_ngpu}",
            f"--master_port={selected_master_port}",
            "-m",
        ]
    cotrain_cmd = [
        *cotrain_launch,
        "dreamervla.train",
        *_render_overrides(cfg["cotrain"]["base"], context),
        f"training.out_dir={cotrain_out}",
        *cotrain_profile_items,
        *_sync_cotrain_scale_overrides(
            profile_cfg,
            ngpu=selected_ngpu,
            explicit_overrides=[*common_overrides, *cotrain_overrides],
        ),
        *zero_gpu_cotrain_items,
        *debug_cotrain_items,
        *warmup_control_items,
        *common_overrides,
        *cotrain_overrides,
    ]

    cotrain_warmup_cmd: list[str] = []
    cotrain_online_cmd: list[str] = []

    # cotrain_engine=async: run the sync pipeline warmup ONLY (writes wm/classifier warmup
    # ckpts), consolidate them, then run the ray async overlap online loop initialized from
    # that ckpt. The online phase is NOT wrapped in torchrun (Ray owns placement).
    ckpt_dir = cotrain_out / "ckpt"
    warmup_wm_ckpt = ckpt_dir / "wm_warmup.ckpt"
    warmup_cls_ckpt = ckpt_dir / "classifier_warmup.ckpt"
    ray_init_ckpt = None
    if selected_cotrain_phase != "all":
        cotrain_warmup_cmd = [*cotrain_cmd, "online_rollout.total_env_steps=0"]
        cotrain_online_cmd = [*cotrain_cmd, "training.resume=true"]
    if cotrain_engine == "async":
        async_exp = str(
            cfg.get("cotrain_async_experiment", "openvla_onetraj_libero_cotrain_ray")
        )
        ray_init_ckpt = ckpt_dir / "ray_async_init.ckpt"
        explicit_online_overrides = [*common_overrides, *cotrain_overrides]
        ray_online_overrides = _ray_online_scale_overrides(
            profile_cfg,
            ngpu=selected_ngpu,
            render_backend=async_render_backend,
            explicit_overrides=explicit_online_overrides,
        )
        manual_cotrain_overrides = (
            _manual_cotrain_online_overrides(
                profile_cfg,
                ngpu=selected_ngpu,
                render_backend=async_render_backend,
                explicit_overrides=[
                    *explicit_online_overrides,
                    *ray_online_overrides,
                ],
                online_budget_overrides=[
                    *cotrain_profile_items,
                    *warmup_control_items,
                ],
            )
            if async_exp.startswith("openvla_onetraj_libero_cotrain_ray")
            else []
        )
        debug_online_items = (
            _debug_async_online_overrides(
                ngpu=selected_ngpu,
                explicit_overrides=explicit_online_overrides,
                effective_overrides=[
                    *explicit_online_overrides,
                    *ray_online_overrides,
                    *manual_cotrain_overrides,
                ],
            )
            if debug_enabled
            else []
        )
        cotrain_warmup_cmd = [*cotrain_cmd, "online_rollout.total_env_steps=0"]
        cotrain_online_cmd = [
            python_cmd,
            "-m",
            "dreamervla.train",
            f"experiment={async_exp}",
            f"task={task_spec['hydra_task']}",
            f"training.out_dir={cotrain_out}",
            f"init.warmup_ckpt_path={ray_init_ckpt}",
            *ray_online_overrides,
            *manual_cotrain_overrides,
            *debug_online_items,
            *common_overrides,
            *cotrain_overrides,
        ]

    return PipelinePlan(
        mode=selected_mode,
        profile=selected_profile,
        task=task_name,
        run_root=root,
        collected_root=collected_root,
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        collect_cmd=collect_cmd,
        cotrain_cmd=cotrain_cmd,
        cotrain_engine=cotrain_engine,
        cotrain_phase=selected_cotrain_phase,
        cotrain_warmup_cmd=cotrain_warmup_cmd,
        cotrain_online_cmd=cotrain_online_cmd,
        warmup_wm_ckpt=warmup_wm_ckpt,
        warmup_cls_ckpt=warmup_cls_ckpt,
        ray_init_ckpt=ray_init_ckpt,
        eval_enabled=eval_enabled,
        eval_interval_global_steps=eval_interval_global_steps,
        eval_cfg=eval_cfg,
        task_suite=str(task_spec["suite"]),
        vla_ckpt_path=Path(str(task_spec["ckpt_path"])),
    )


def validate_input_assets(
    *,
    data_root: str | Path,
    task: str | None = None,
    launcher_cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return missing or malformed input assets for the default one-traj OFT route."""
    cfg = script_config("coldstart_warmup_cotrain") if launcher_cfg is None else launcher_cfg
    _task_name, task_spec = _resolve_task(str(cfg["task"] if task is None else task), dict(cfg["tasks"]))
    root = Path(data_root).expanduser()
    ckpt = root / "checkpoints" / "Openvla-oft-SFT-traj1" / str(task_spec["ckpt_name"])
    stats = ckpt / "dataset_statistics.json"
    libero = root / "datasets" / "libero" / str(task_spec["suite"])
    errors: list[str] = []

    if not ckpt.is_dir():
        errors.append(f"OpenVLA-OFT checkpoint directory not found: {ckpt}")
    if not stats.is_file():
        errors.append(f"OpenVLA-OFT dataset statistics not found: {stats}")
    else:
        try:
            stats_data = json.loads(stats.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"OpenVLA-OFT dataset statistics is not valid JSON: {stats} ({exc})")
        else:
            if str(task_spec["stats_key"]) not in stats_data:
                errors.append(
                    "OpenVLA-OFT dataset statistics missing key "
                    f"'{task_spec['stats_key']}': {stats}"
                )
    if not libero.is_dir():
        errors.append(f"LIBERO dataset directory not found: {libero}")
    elif not any(libero.rglob("*.hdf5")):
        errors.append(f"LIBERO dataset directory has no HDF5 files: {libero}")
    return errors


def validate_collected_outputs(*, reward_dir: str | Path, hidden_dir: str | Path) -> list[str]:
    """Validate reusable cold-start shards against the OpenVLA hidden-token schema."""

    from dreamervla.preprocess.sidecar_schema import validate_hidden_token_sidecar_dir

    reward = Path(reward_dir).expanduser()
    hidden = Path(hidden_dir).expanduser()
    errors: list[str] = []
    reward_shards: list[Path] = []
    if not reward.is_dir():
        errors.append(f"cold-start reward directory not found: {reward}")
    else:
        reward_shards = sorted(reward.glob("*.hdf5"))
        if not reward_shards:
            errors.append(f"cold-start reward directory has no HDF5 shards: {reward}")
    if not hidden.is_dir():
        errors.append(f"cold-start hidden directory not found: {hidden}")
    else:
        hidden_shards = sorted(hidden.glob("*.hdf5"))
        if not hidden_shards:
            errors.append(f"cold-start hidden directory has no HDF5 shards: {hidden}")
        if reward_shards and hidden_shards:
            reward_names = [path.name for path in reward_shards]
            hidden_names = {path.name for path in hidden_shards}
            extras = sorted(hidden_names.difference(reward_names))
            if extras:
                errors.append(
                    "cold-start hidden directory has unpaired shards: "
                    + ", ".join(extras)
                )
            try:
                validate_hidden_token_sidecar_dir(
                    hidden,
                    expected_filenames=reward_names,
                    reference_dir=reward,
                    require_reference_complete=True,
                    require_sparse_rewards=True,
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                errors.append(str(exc))
    return errors


def validate_warmup_outputs(*, cotrain_out: str | Path) -> list[str]:
    """Return missing split warmup checkpoints for an online-only cotrain resume."""
    root = Path(cotrain_out).expanduser()
    ckpt = root / "ckpt"
    errors: list[str] = []
    wm = ckpt / "wm_warmup.ckpt"
    cls = ckpt / "classifier_warmup.ckpt"
    if not wm.is_file() and not (ckpt / "wm_warmup_hf").is_dir():
        errors.append(f"warmup world-model checkpoint not found: {wm}")
    if not cls.is_file() and not (ckpt / "classifier_warmup_hf").is_dir():
        errors.append(f"warmup classifier checkpoint not found: {cls}")
    return errors


def _data_root() -> Path:
    """Return ``DVLA_DATA_ROOT`` or the ``DVLA_ROOT/data`` fallback."""

    return data_root()


def _parse_hydra_like_argv(argv: Sequence[str]) -> tuple[str, list[str]]:
    config_name = "coldstart_warmup_cotrain"
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--config-name":
            if i + 1 >= len(argv):
                raise SystemExit("--config-name requires a value")
            config_name = argv[i + 1]
            i += 2
            continue
        if item.startswith("--config-name="):
            config_name = item.split("=", 1)[1]
            i += 1
            continue
        overrides.append(item)
        i += 1
    return config_name, overrides


def _plain(value: Any) -> Any:
    return (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )


def _as_str_list(value: Any) -> list[str]:
    value = _plain(value)
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _eval_interval_global_steps(
    eval_cfg: Mapping[str, Any],
    *,
    debug: bool,
) -> int:
    key = "debug_interval_global_steps" if debug else "interval_global_steps"
    value = eval_cfg.get(key, eval_cfg.get("interval_global_steps", 0))
    interval = int(value or 0)
    if interval < 0:
        raise ValueError(f"eval.{key} must be non-negative, got {interval}")
    return interval


def run_async_online_with_in_run_eval_metrics(plan: PipelinePlan) -> None:
    """Run async online cotrain, optionally evaluating saved step checkpoints."""

    if not plan.eval_enabled or int(plan.eval_interval_global_steps) <= 0:
        subprocess.run(plan.cotrain_online_cmd, check=True)
        return

    target_step = _manual_cotrain_target_step(plan.cotrain_online_cmd)
    eval_steps = _post_step_eval_steps(
        target_step=target_step,
        interval=int(plan.eval_interval_global_steps),
    )
    resume_ckpt: Path | None = None
    for step in eval_steps:
        train_cmd = _manual_cotrain_segment_cmd(
            plan.cotrain_online_cmd,
            target_step=step,
            resume_ckpt=resume_ckpt,
        )
        subprocess.run(train_cmd, check=True)
        ckpt_path = _manual_cotrain_step_ckpt(plan, step)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"manual cotrain checkpoint for eval not found: {ckpt_path}"
            )
        eval_out_dir = plan.run_root / "cotrain" / "eval" / f"global_step_{step}"
        eval_cmd = _post_step_eval_cmd(
            plan,
            ckpt_path=ckpt_path,
            out_dir=eval_out_dir,
        )
        subprocess.run(eval_cmd, check=True, env=_post_step_eval_env(plan))
        metrics_path = eval_out_dir / "eval_libero_metrics.json"
        eval_metrics = _read_post_step_eval_metrics(metrics_path)
        _append_eval_summary(
            plan.run_root / "cotrain" / "eval" / "eval_summary.json",
            global_step=step,
            ckpt_path=ckpt_path,
            eval_out_dir=eval_out_dir,
            metrics=eval_metrics,
            significant_drop_threshold=float(
                plan.eval_cfg.get("significant_drop_threshold", 0.10)
            ),
        )
        resume_ckpt = ckpt_path


def _manual_cotrain_target_step(cmd: Sequence[str]) -> int:
    target = _last_int_override(cmd, "manual_cotrain.global_steps")
    if target is None or int(target) <= 0:
        raise ValueError("manual_cotrain.global_steps must be set for post-step eval")
    return int(target)


def _post_step_eval_steps(*, target_step: int, interval: int) -> list[int]:
    if int(interval) <= 0:
        return []
    steps = list(range(int(interval), int(target_step) + 1, int(interval)))
    if not steps or steps[-1] != int(target_step):
        steps.append(int(target_step))
    return steps


def _replace_overrides(cmd: Sequence[str], values: Mapping[str, str]) -> list[str]:
    keys = {_override_key(key) for key in values}
    out = [item for item in cmd if _override_key(str(item)) not in keys]
    out.extend(f"{key}={value}" for key, value in values.items())
    return out


def _manual_cotrain_segment_cmd(
    cmd: Sequence[str],
    *,
    target_step: int,
    resume_ckpt: Path | None,
) -> list[str]:
    values = {
        "manual_cotrain.global_steps": str(int(target_step)),
        "manual_cotrain.checkpoint_every": "1",
    }
    if resume_ckpt is not None:
        ckpt = str(resume_ckpt)
        values.update(
            {
                "+manual_cotrain.resume_ckpt": ckpt,
                "+actor.init_ckpt.path": ckpt,
                "+actor.init_ckpt.components": "[policy]",
                "learner.init_ckpt.path": ckpt,
                "learner.init_ckpt.components": (
                    "[world_model,classifier,world_model_optimizer,"
                    "classifier_optimizer]"
                ),
            }
        )
    return _replace_overrides(cmd, values)


def _manual_cotrain_step_ckpt(plan: PipelinePlan, step: int) -> Path:
    return (
        plan.run_root
        / "cotrain"
        / "checkpoints"
        / f"manual_cotrain_step_{int(step)}"
        / "manual_cotrain.ckpt"
    )


def _post_step_eval_cmd(
    plan: PipelinePlan,
    *,
    ckpt_path: Path,
    out_dir: Path,
) -> list[str]:
    eval_cfg = dict(plan.eval_cfg)
    cmd = [
        str(plan.cotrain_online_cmd[0] if plan.cotrain_online_cmd else "python"),
        "-m",
        "dreamervla.launchers.train",
        "--config-name",
        "eval_libero_vla",
        "experiment=eval_libero_vla",
        f"out_dir={out_dir}",
        f"gpus={eval_cfg.get('gpus', '0')}",
        f"eval.ckpt_path={ckpt_path}",
        "eval.ckpt_kind=dreamer",
        f"eval.task_suite_name={eval_cfg.get('task_suite_name', plan.task_suite)}",
    ]
    if plan.vla_ckpt_path is not None:
        cmd.append(f"init.vla_ckpt_path={plan.vla_ckpt_path}")
    for source_key, target_key in (
        ("num_episodes_per_task", "eval.num_episodes_per_task"),
        ("task_ids", "eval.task_ids"),
        ("max_tasks", "eval.max_tasks"),
        ("max_steps", "eval.max_steps"),
        ("action_postprocess", "eval.action_postprocess"),
        ("seed", "eval.seed"),
    ):
        value = eval_cfg.get(source_key)
        if value is not None:
            cmd.append(f"{target_key}={_format_hydra_value(value)}")
    return cmd


def _post_step_eval_env(plan: PipelinePlan) -> dict[str, str]:
    env = os.environ.copy()
    backend = str(plan.eval_cfg.get("render_backend", "egl")).strip().lower()
    if backend not in {"egl", "osmesa"}:
        raise ValueError(f"eval.render_backend must be 'egl' or 'osmesa', got {backend!r}")
    return _libero_render_env(
        env,
        backend=backend,
        shard_id=0,
        gpu_pool=_post_step_eval_gpu_pool(plan.eval_cfg, backend=backend),
    )


def _libero_render_env(
    env: Mapping[str, str],
    *,
    backend: str,
    shard_id: int,
    gpu_pool: list[int],
) -> dict[str, str]:
    original = os.environ.copy()
    os.environ.clear()
    os.environ.update({str(key): str(value) for key, value in dict(env).items()})
    try:
        apply_libero_render_regime(backend, int(shard_id), gpu_pool)
        return dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(original)


def _post_step_eval_gpu_pool(
    eval_cfg: Mapping[str, Any],
    *,
    backend: str,
) -> list[int]:
    if str(backend).strip().lower() != "egl":
        return []
    for key in ("gpu_pool", "render_devices", "egl_device_pool"):
        devices = parse_device_ids(eval_cfg.get(key))
        if devices:
            return devices
    return [int(_post_step_eval_egl_device_id(eval_cfg))]


def _post_step_eval_egl_device_id(eval_cfg: Mapping[str, Any]) -> str:
    explicit = eval_cfg.get("egl_device_id")
    if explicit not in (None, ""):
        return str(int(explicit))
    gpus = str(eval_cfg.get("gpus", "0") if eval_cfg.get("gpus", "0") is not None else "0")
    parts = [part.strip() for part in gpus.split(",") if part.strip()]
    return str(int(parts[-1] if parts else "0"))


def _read_post_step_eval_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"eval metrics not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"eval metrics must be a JSON object: {path}")
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def _append_eval_summary(
    path: Path,
    *,
    global_step: int,
    ckpt_path: Path,
    eval_out_dir: Path,
    metrics: Mapping[str, float],
    significant_drop_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        summary = {"records": []}
    records = list(summary.get("records", [])) if isinstance(summary, dict) else []
    previous_rate = (
        float(records[-1]["eval_success_rate"])
        if records and "eval_success_rate" in records[-1]
        else None
    )
    success_rate = float(metrics.get("eval_success_rate", 0.0))
    best_rate = max(
        [success_rate, *[float(record.get("eval_success_rate", 0.0)) for record in records]]
    )
    drop = (
        max(0.0, float(previous_rate) - success_rate)
        if previous_rate is not None
        else 0.0
    )
    delta = (
        round(success_rate - float(previous_rate), 12)
        if previous_rate is not None
        else 0.0
    )
    record = {
        "global_step": int(global_step),
        "ckpt_path": str(ckpt_path),
        "eval_out_dir": str(eval_out_dir),
        "eval_success_rate": success_rate,
        "eval_success_rate_delta": float(delta),
        "eval_success_rate_trend": float(best_rate),
        "eval_best_success_rate": float(best_rate),
        "eval_success_rate_drop": float(drop),
        "eval_significant_drop": float(
            drop > max(0.0, float(significant_drop_threshold))
        ),
    }
    for key, value in metrics.items():
        record.setdefault(str(key), float(value))
    records.append(record)
    path.write_text(
        json.dumps(
            {
                "significant_drop_threshold": float(significant_drop_threshold),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _consolidate_warmup_state_dicts(wm_path: Path, cls_path: Path, out_path: Path) -> None:
    """Merge the sync pipeline's per-component warmup ckpts into one runner-format file.

    The pipeline writes wm_warmup.ckpt ({"world_model": sd, ...}) and classifier_warmup.ckpt
    ({"classifier": sd, "classifier_threshold": ...}); the ray async runner loads a single
    init.warmup_ckpt_path via _load_runner_state_dicts, which expects {"state_dicts": {...}}.
    """
    import torch  # local import: the launcher has no module-level torch dependency

    wm = torch.load(wm_path, map_location="cpu", weights_only=False)
    cls = torch.load(cls_path, map_location="cpu", weights_only=False)
    payload: dict[str, Any] = {
        "state_dicts": {"world_model": wm["world_model"], "classifier": cls["classifier"]}
    }
    if isinstance(cls, dict) and "classifier_threshold" in cls:
        payload["classifier_threshold"] = float(cls["classifier_threshold"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def collect_resume(
    plan: PipelinePlan,
    *,
    target_episodes: int | None,
    num_tasks: int,
    skip_collect: bool,
) -> dict[str, Any]:
    """Inspect collected data, resume missing episodes, and write the manifest.

    Resume is episode-level: only complete reward/hidden pairs count. Missing
    or incomplete ``(task_id, episode_id)`` entries below the per-task target are
    collected into new shards by the downstream collector.
    """
    target = (
        _derive_collect_target_from_cmd(plan.collect_cmd, num_tasks=num_tasks)
        if target_episodes is None
        else int(target_episodes)
    )
    quarantined = quarantine_corrupt_shards(plan.reward_dir, plan.hidden_dir)
    quarantined += quarantine_incomplete_shards(plan.reward_dir, plan.hidden_dir)
    if quarantined:
        print(
            f"[collect] quarantined {len(quarantined)} incomplete/corrupt shard(s): "
            f"{', '.join(quarantined)}",
            flush=True,
        )

    has_existing_shards = any(plan.reward_dir.glob("*.hdf5")) or any(
        plan.hidden_dir.glob("*.hdf5")
    )
    if skip_collect or has_existing_shards:
        schema_errors = validate_collected_outputs(
            reward_dir=plan.reward_dir,
            hidden_dir=plan.hidden_dir,
        )
        if schema_errors:
            joined = "\n  - ".join(schema_errors)
            raise ValueError(
                "cold-start collection does not match the required OpenVLA "
                f"hidden-token schema:\n  - {joined}"
            )

    if skip_collect:
        print("PHASE 1/2 SKIPPED: cold-start collection", flush=True)
        _write_collection_manifest(plan, target_episodes=target, num_tasks=num_tasks)
        return {"ran_collect": False, "target_episodes": target}

    summary = summarize_collection(
        plan.reward_dir,
        plan.hidden_dir,
        target_total=target,
        num_tasks=num_tasks,
    )
    print(format_collection_report(summary, root=plan.collected_root), flush=True)
    if summary["complete"]:
        print(f"PHASE 1/2 SKIPPED: target {target} already collected", flush=True)
        _write_collection_manifest(plan, target_episodes=target, num_tasks=num_tasks)
        return {"ran_collect": False, "target_episodes": target}

    collect_cmd = list(plan.collect_cmd)
    target_per_task = int(summary["target_per_task"] or 0)
    if target_per_task > 0:
        collect_cmd.append(f"collect.episodes_per_task={target_per_task}")
    rp = resume_plan(
        target_total=int(target),
        num_tasks=int(num_tasks),
        collected=int(summary["total"]),
    )
    print(
        f"[resume] topping up {rp['remaining']} episodes "
        f"({target_per_task}/task target, appending shards)",
        flush=True,
    )
    print("PHASE 1/2 START: cold-start collection", flush=True)
    subprocess.run(collect_cmd, check=True)
    schema_errors = validate_collected_outputs(
        reward_dir=plan.reward_dir,
        hidden_dir=plan.hidden_dir,
    )
    if schema_errors:
        joined = "\n  - ".join(schema_errors)
        raise ValueError(
            "cold-start collection does not match the required OpenVLA "
            f"hidden-token schema after collection:\n  - {joined}"
        )
    post = summarize_collection(
        plan.reward_dir,
        plan.hidden_dir,
        target_total=target,
        num_tasks=num_tasks,
    )
    print("PHASE 1/2 collected (aggregate across all processes):", flush=True)
    print(format_collection_report(post, root=plan.collected_root), flush=True)
    _write_collection_manifest(plan, target_episodes=target, num_tasks=num_tasks)
    return {"ran_collect": True, "target_episodes": target}


def _derive_collect_target_from_cmd(cmd: Sequence[str], *, num_tasks: int) -> int:
    episodes_per_task: int | None = None
    for item in cmd:
        text = str(item)
        if text.startswith("collect.episodes_per_task="):
            episodes_per_task = int(text.split("=", 1)[1])
    if episodes_per_task is None:
        raise ValueError("cannot derive collect target: collect.episodes_per_task is absent")
    return int(episodes_per_task) * int(num_tasks)


def main(argv: Sequence[str] | None = None) -> int:
    register_dreamervla_resolvers()
    config_name, overrides = _parse_hydra_like_argv(
        list(sys.argv[1:] if argv is None else argv)
    )
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="coldstart_warmup_cotrain",
        version_base=None,
    ):
        cfg_obj = compose(config_name=config_name, overrides=overrides)
    cfg: dict[str, Any] = OmegaConf.to_container(cfg_obj, resolve=True)  # type: ignore[assignment]
    previous_data_root = os.environ.get("DVLA_DATA_ROOT")
    os.environ["DVLA_DATA_ROOT"] = str(cfg.get("data_root") or _data_root())
    try:
        return _run_pipeline_main(cfg, overrides)
    finally:
        if previous_data_root is None:
            os.environ.pop("DVLA_DATA_ROOT", None)
        else:
            os.environ["DVLA_DATA_ROOT"] = previous_data_root


def _run_pipeline_main(cfg: dict[str, Any], overrides: Sequence[str]) -> int:
    python_cmd = str(cfg["python"])
    if python_cmd == "python" and not _has_override(overrides, "python"):
        python_cmd = sys.executable
    plan = build_pipeline_plan(
        mode=str(cfg["mode"]),
        profile=str(cfg["profile"]),
        task=str(cfg["task"]),
        run_root=str(cfg["run_root"]),
        python=python_cmd,
        launcher_cfg=cfg,
        collect_overrides=_as_str_list(cfg.get("collect_overrides")),
        cotrain_overrides=_as_str_list(cfg.get("cotrain_overrides")),
        common_overrides=_as_str_list(cfg.get("common_overrides")),
        debug=bool(cfg.get("debug", False)),
        cotrain_phase=str(cfg.get("cotrain_phase", "all")),
    )
    print(f"mode: {plan.mode}")
    print(f"profile: {plan.profile}")
    print(f"task: {plan.task}")
    print(f"cotrain_phase: {plan.cotrain_phase}")
    print(f"run_root: {plan.run_root}")
    print(f"reward_dir: {plan.reward_dir}")
    print(f"hidden_dir: {plan.hidden_dir}")
    print(f"collect: {shlex.join(plan.collect_cmd)}")
    print(f"cotrain: {shlex.join(plan.cotrain_cmd)}")
    if plan.cotrain_phase != "all" or plan.cotrain_engine == "async":
        print(f"cotrain_warmup: {shlex.join(plan.cotrain_warmup_cmd)}")
        print(f"cotrain_online: {shlex.join(plan.cotrain_online_cmd)}")
    if plan.eval_enabled:
        print(f"eval_interval_global_steps: {plan.eval_interval_global_steps}")
    if bool(cfg.get("dry_run", False)):
        return 0

    if not bool(cfg.get("skip_asset_check", False)):
        if plan.cotrain_phase == "online":
            errors = validate_warmup_outputs(cotrain_out=plan.run_root / "cotrain")
        elif bool(cfg.get("skip_collect", False)):
            errors = validate_collected_outputs(
                reward_dir=plan.reward_dir,
                hidden_dir=plan.hidden_dir,
            )
        else:
            errors = validate_input_assets(
                data_root=_data_root(),
                task=plan.task,
                launcher_cfg=cfg,
            )
        if errors:
            print("asset check failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            print(
                f"Assets are resolved under data_root={_data_root()} "
                "(= DVLA_DATA_ROOT, else <DVLA_ROOT>/data). DVLA_ROOT (code) and "
                "DVLA_DATA_ROOT (data) are independent — if your checkpoints/ and "
                "datasets/ live elsewhere, set DVLA_DATA_ROOT=<that dir> or pass "
                "data_root=<that dir>. Use skip_asset_check=true only when custom "
                "Hydra overrides already provide the assets.",
                file=sys.stderr,
            )
            return 2

    target_episodes = cfg.get("collect_target_episodes")
    num_tasks = int(cfg.get("collect_num_tasks", 10) or 10)
    collect_resume(
        plan,
        target_episodes=target_episodes,
        num_tasks=num_tasks,
        skip_collect=bool(cfg.get("skip_collect", False)) or plan.cotrain_phase == "online",
    )
    if plan.cotrain_phase == "warmup":
        print("PHASE 2/2 START: offline warmup only", flush=True)
        subprocess.run(plan.cotrain_warmup_cmd, check=True)
        if plan.cotrain_engine == "async":
            print("PHASE 2/2: consolidate warmup ckpts -> ray async init", flush=True)
            _consolidate_warmup_state_dicts(
                plan.warmup_wm_ckpt, plan.warmup_cls_ckpt, plan.ray_init_ckpt
            )
    elif plan.cotrain_phase == "online":
        print("PHASE 2/2 START: online cotrain resume from warmup ckpts", flush=True)
        if (
            plan.cotrain_engine == "async"
            and plan.ray_init_ckpt is not None
            and not plan.ray_init_ckpt.is_file()
        ):
            print("PHASE 2/2: consolidate warmup ckpts -> ray async init", flush=True)
            _consolidate_warmup_state_dicts(
                plan.warmup_wm_ckpt,
                plan.warmup_cls_ckpt,
                plan.ray_init_ckpt,
            )
        if plan.cotrain_engine == "async":
            run_async_online_with_in_run_eval_metrics(plan)
        else:
            subprocess.run(plan.cotrain_online_cmd, check=True)
    elif plan.cotrain_engine == "async":
        print("PHASE 2a/3: offline warmup (sync, writes warmup ckpts)", flush=True)
        subprocess.run(plan.cotrain_warmup_cmd, check=True)
        print("PHASE 2b/3: consolidate warmup ckpts -> ray async init", flush=True)
        _consolidate_warmup_state_dicts(
            plan.warmup_wm_ckpt, plan.warmup_cls_ckpt, plan.ray_init_ckpt
        )
        print("PHASE 2c/3: async online cotrain (ray overlap)", flush=True)
        run_async_online_with_in_run_eval_metrics(plan)
    else:
        print("PHASE 2/2 START: offline-warmup online cotrain", flush=True)
        subprocess.run(plan.cotrain_cmd, check=True)
    return 0


def _write_collection_manifest(
    plan: PipelinePlan, *, target_episodes: int | None, num_tasks: int
) -> None:
    """Write metadata + config next to the unified collected_rollouts data."""
    summary = summarize_collection(
        plan.reward_dir,
        plan.hidden_dir,
        target_total=target_episodes,
        num_tasks=num_tasks,
    )
    existing = read_manifest(plan.collected_root) or {}
    shards = (
        sorted(p.name for p in plan.reward_dir.glob("*.hdf5"))
        if plan.reward_dir.is_dir()
        else []
    )
    hidden_schema: dict[str, object] = {}
    preprocess_path = plan.hidden_dir / "preprocess_config.json"
    preprocess: dict[str, Any] = {}
    if preprocess_path.is_file():
        try:
            preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            preprocess = {}
        for key in (
            "hidden_key",
            "chunk_size",
            "token_count",
            "token_dim",
            "obs_hidden_source",
            "obs_embedding_shape",
            "hidden_storage_format",
            "output_dtype",
            "hidden_dim",
        ):
            if key in preprocess:
                hidden_schema[key] = preprocess[key]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    resolved = plan.run_root / "collect" / "resolved_config.yaml"
    resolved_snapshot = (
        resolved.read_text(encoding="utf-8") if resolved.is_file() else None
    )
    suite = str(preprocess.get("task_suite_name") or plan.collected_root.name)
    collected_counts = {
        "total": int(summary["total"]),
        "per_task": {str(k): int(v) for k, v in summary["per_task"].items()},
    }
    resume_status = {
        "complete": bool(summary["complete"]),
        "remaining": summary["remaining"],
        "target_total": summary["target_total"],
        "target_per_task": summary["target_per_task"],
        "num_tasks": int(summary["num_tasks"]),
    }
    write_manifest(
        plan.collected_root,
        {
            "suite": suite,
            "task": plan.task,
            "mode": plan.mode,
            "profile": plan.profile,
            "backend": os.environ.get("MUJOCO_GL", "unknown"),
            "reward_dir": str(plan.reward_dir),
            "hidden_dir": str(plan.hidden_dir),
            "policy_checkpoint": _policy_checkpoint_from_cmd(plan.collect_cmd),
            "hidden_schema": hidden_schema,
            "target_episodes": target_episodes,
            "num_tasks": num_tasks,
            "collected_counts": collected_counts,
            "collected_episodes": summary["total"],
            "episodes_per_task": summary["per_task"],
            "shards": shards,
            "status": "complete" if summary["complete"] else "in_progress",
            "resume_status": resume_status,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "resolved_config_snapshot": resolved_snapshot,
            "collect_cmd": list(plan.collect_cmd),
        },
    )
    # Co-locate the resolved collection config with the data (best-effort).
    if resolved.is_file():
        shutil.copy2(resolved, plan.collected_root / "resolved_config.yaml")


def _policy_checkpoint_from_cmd(cmd: Sequence[str]) -> str | None:
    for item in cmd:
        text = str(item)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in {
            "init.vla_ckpt_path",
            "collect.model_path",
            "task.openvla_oft.ckpt_path",
            "policy.ckpt_path",
        }:
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
