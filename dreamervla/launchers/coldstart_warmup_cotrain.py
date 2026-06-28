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


def _without_override_keys(overrides: Sequence[str], keys: set[str]) -> list[str]:
    return [item for item in overrides if _override_key(item) not in keys]


def _override_value(item: str) -> str | None:
    text = str(item)
    if "=" not in text:
        return None
    return text.split("=", 1)[1]


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
    scaled_envs = _scaled_profile_count(
        profile_cfg,
        "online_rollout_envs_per_gpu",
        ngpu=ngpu,
    )
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

    if envs_per_gpu is not None:
        if not _has_override(explicit_overrides, "env.num_workers"):
            overrides.append(f"env.num_workers={gpu_count}")
        if not _has_override(explicit_overrides, "env.envs_per_worker"):
            overrides.append(f"env.envs_per_worker={envs_per_gpu}")

    overrides.extend(
        _ray_online_egl_component_placement_overrides(
            ngpu=gpu_count,
            explicit_overrides=explicit_overrides,
        )
    )

    egl_cfg = _plain(profile_cfg).get("ray_online_egl_spawn", {})
    if not isinstance(egl_cfg, Mapping):
        return overrides
    egl_targets = {
        "stagger_s": "env.cfg.egl_spawn_stagger_s",
        "init_timeout_s": "env.cfg.egl_spawn_init_timeout_s",
    }
    for source_key, target_key in egl_targets.items():
        value = egl_cfg.get(source_key)
        if value is None or _has_override(explicit_overrides, target_key):
            continue
        overrides.append(f"++{target_key}={_format_hydra_value(value)}")
    return overrides


def _ray_online_egl_component_placement_overrides(
    *,
    ngpu: int,
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
    defaults = {
        "cluster.component_placement.env": env_ranks,
        "cluster.component_placement.rollout": "0",
        "cluster.component_placement.actor": str(actor_rank),
    }
    for key, value in defaults.items():
        if not _has_override(explicit_overrides, key):
            overrides.append(f"{key}={value}")
    return overrides


def _manual_cotrain_online_overrides(
    profile_cfg: Mapping[str, Any],
    *,
    ngpu: int,
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
    envs_per_worker = _profile_per_gpu_count(
        profile_cfg,
        "online_rollout_envs_per_gpu",
    )
    default_envs_per_worker = envs_per_worker or 8
    defaults.append(f"manual_cotrain.envs_per_worker={default_envs_per_worker}")
    global_steps = _manual_cotrain_global_steps_from_budget(
        online_budget_overrides,
        explicit_overrides=explicit_overrides,
        default_envs_per_worker=default_envs_per_worker,
    )
    if global_steps is not None:
        defaults.append(f"manual_cotrain.global_steps={global_steps}")
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


def _manual_cotrain_global_steps_from_budget(
    online_budget_overrides: Sequence[str],
    *,
    explicit_overrides: Sequence[str],
    default_envs_per_worker: int,
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
        or 256
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
    collect_control_items = _control_overrides(
        cfg.get("collect"),
        _collect_control_mapping(cfg, selected_mode),
        label="collect",
    )
    cotrain_profile_items = _render_overrides(profile_cfg["cotrain"], context)
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
            f"task.openvla_oft.input_token_hidden_dir={hidden_dir}",
            f"++collect.hdf5_reward_dir={reward_dir}",
            f"++collect.hidden_dir={hidden_dir}",
            f"training.out_dir={collect_out}",
            *ray_env_scale,
            *zero_gpu_collect_items,
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
        *warmup_control_items,
        *common_overrides,
        *cotrain_overrides,
    ]
    # debug=true overrides the profile's full values so the runner's central swap
    # takes over (small debug_* scale). Appended last so it wins in Hydra.
    if debug_enabled:
        cotrain_cmd.append("training.debug=true")

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
        async_exp = str(cfg.get("cotrain_async_experiment", "online_cotrain_ray_oft"))
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
                explicit_overrides=[
                    *explicit_online_overrides,
                    *ray_online_overrides,
                ],
                online_budget_overrides=[
                    *cotrain_profile_items,
                    *warmup_control_items,
                ],
            )
            if async_exp.startswith("manual_cotrain_")
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
            *common_overrides,
            *cotrain_overrides,
        ]
        if debug_enabled:
            cotrain_online_cmd.append("training.debug=true")

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
    """Return missing output shards when reusing an existing cold-start dump."""
    reward = Path(reward_dir).expanduser()
    hidden = Path(hidden_dir).expanduser()
    errors: list[str] = []
    if not reward.is_dir():
        errors.append(f"cold-start reward directory not found: {reward}")
    elif not any(reward.glob("*.hdf5")):
        errors.append(f"cold-start reward directory has no HDF5 shards: {reward}")
    if not hidden.is_dir():
        errors.append(f"cold-start hidden directory not found: {hidden}")
    elif not any(hidden.glob("*.hdf5")):
        errors.append(f"cold-start hidden directory has no HDF5 shards: {hidden}")
    elif not (hidden / "preprocess_config.json").is_file():
        errors.append(
            "cold-start hidden directory is missing preprocess_config.json: "
            f"{hidden / 'preprocess_config.json'}"
        )
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
    os.environ["DVLA_DATA_ROOT"] = str(cfg.get("data_root") or _data_root())
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
        subprocess.run(plan.cotrain_online_cmd, check=True)
    elif plan.cotrain_engine == "async":
        print("PHASE 2a/3: offline warmup (sync, writes warmup ckpts)", flush=True)
        subprocess.run(plan.cotrain_warmup_cmd, check=True)
        print("PHASE 2b/3: consolidate warmup ckpts -> ray async init", flush=True)
        _consolidate_warmup_state_dicts(
            plan.warmup_wm_ckpt, plan.warmup_cls_ckpt, plan.ray_init_ckpt
        )
        print("PHASE 2c/3: async online cotrain (ray overlap)", flush=True)
        subprocess.run(plan.cotrain_online_cmd, check=True)
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
