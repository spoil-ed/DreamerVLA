"""Launch cold-start collection followed by offline-warmup online cotrain."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.utils.hydra_config import script_config
from dreamervla.utils.paths import data_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "scripts"

PipelineMode = Literal["ray", "noray"]


@dataclass(frozen=True)
class PipelinePlan:
    mode: PipelineMode
    profile: str
    task: str
    run_root: Path
    reward_dir: Path
    hidden_dir: Path
    collect_cmd: list[str]
    cotrain_cmd: list[str]


def _normalize_mode(mode: str) -> PipelineMode:
    normalized = mode.strip().lower().replace("_", "-")
    if normalized == "ray":
        return "ray"
    if normalized in {"noray", "no-ray", "non-ray"}:
        return "noray"
    raise ValueError("mode must be one of: ray, noray")


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
) -> PipelinePlan:
    cfg = script_config("coldstart_warmup_cotrain") if launcher_cfg is None else launcher_cfg
    selected_mode = _normalize_mode(str(cfg["mode"] if mode is None else mode))
    selected_profile = _normalise_key(str(cfg["profile"] if profile is None else profile))
    _select_mapping(dict(cfg["profiles"]), selected_profile, label="profile")
    task_name, task_spec = _resolve_task(str(cfg["task"] if task is None else task), dict(cfg["tasks"]))
    python_cmd = str(cfg["python"] if python is None else python)
    root = Path(run_root).expanduser()
    reward_dir = root / "coldstart" / "reward"
    hidden_dir = root / "coldstart" / "hidden"
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
    }
    mode_cfg = _select_mapping(dict(cfg["modes"]), selected_mode, label="mode")
    profile_cfg = _select_mapping(dict(cfg["profiles"]), selected_profile, label="profile")
    collect_profile_cfg = _select_mapping(
        dict(profile_cfg["collect"]),
        selected_mode,
        label=f"profile.{selected_profile}.collect",
    )

    collect_cmd = [
        python_cmd,
        "-m",
        "dreamervla.train",
        *_render_overrides(mode_cfg["collect"], context),
        *_render_overrides(collect_profile_cfg, context),
    ]
    collect_cmd.extend(
        [
            f"task.openvla_oft.hdf5_reward_dir={reward_dir}",
            f"task.openvla_oft.action_hidden_dir={hidden_dir}",
            f"training.out_dir={collect_out}",
            *_control_overrides(
                cfg.get("collect"),
                _collect_control_mapping(cfg, selected_mode),
                label="collect",
            ),
            *common_overrides,
            *collect_overrides,
        ]
    )
    cotrain_cmd = [
        python_cmd,
        "-m",
        "dreamervla.train",
        *_render_overrides(cfg["cotrain"]["base"], context),
        f"training.out_dir={cotrain_out}",
        *_render_overrides(profile_cfg["cotrain"], context),
        *_control_overrides(
            cfg.get("warmup"),
            _warmup_control_mapping(cfg),
            label="warmup",
        ),
        *common_overrides,
        *cotrain_overrides,
    ]
    return PipelinePlan(
        mode=selected_mode,
        profile=selected_profile,
        task=task_name,
        run_root=root,
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        collect_cmd=collect_cmd,
        cotrain_cmd=cotrain_cmd,
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


def main(argv: Sequence[str] | None = None) -> int:
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
    plan = build_pipeline_plan(
        mode=str(cfg["mode"]),
        profile=str(cfg["profile"]),
        task=str(cfg["task"]),
        run_root=str(cfg["run_root"]),
        python=str(cfg["python"]),
        launcher_cfg=cfg,
        collect_overrides=_as_str_list(cfg.get("collect_overrides")),
        cotrain_overrides=_as_str_list(cfg.get("cotrain_overrides")),
        common_overrides=_as_str_list(cfg.get("common_overrides")),
    )
    print(f"mode: {plan.mode}")
    print(f"profile: {plan.profile}")
    print(f"task: {plan.task}")
    print(f"run_root: {plan.run_root}")
    print(f"reward_dir: {plan.reward_dir}")
    print(f"hidden_dir: {plan.hidden_dir}")
    print(f"collect: {shlex.join(plan.collect_cmd)}")
    print(f"cotrain: {shlex.join(plan.cotrain_cmd)}")
    if bool(cfg.get("dry_run", False)):
        return 0

    if not bool(cfg.get("skip_asset_check", False)):
        if bool(cfg.get("skip_collect", False)):
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
                "Use skip_asset_check=true only when custom Hydra overrides provide assets.",
                file=sys.stderr,
            )
            return 2

    if not bool(cfg.get("skip_collect", False)):
        subprocess.run(plan.collect_cmd, check=True)
    subprocess.run(plan.cotrain_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
