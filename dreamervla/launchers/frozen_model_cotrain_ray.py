"""Eight-GPU frozen-model Ray launch from explicit checkpoint assignments."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dreamervla.launchers.frozen_model_pre_mainline import (
    resolve_available_classifier_threshold,
    select_available_classifier_checkpoint,
    select_available_world_model_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISIBLE_GPUS = tuple(str(gpu) for gpu in range(8))


@dataclass(frozen=True)
class FrozenRayLaunch:
    """Resolved subprocess launch without any side effects."""

    command: list[str]
    env: dict[str, str]
    out_dir: Path
    visible_gpus: tuple[str, ...]
    resume: bool


def _existing_file(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _component_checkpoint(value: str, *, component: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        if component == "world_model":
            return select_available_world_model_checkpoint(path)
        if component == "classifier":
            return select_available_classifier_checkpoint(path)
        raise ValueError(f"unknown frozen component {component!r}")
    raise FileNotFoundError(f"{component} checkpoint/run does not exist: {path}")


def _visible_gpus() -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = (
        tuple(part.strip() for part in raw.split(",") if part.strip())
        if raw
        else DEFAULT_VISIBLE_GPUS
    )
    if len(gpus) != 8:
        raise ValueError(
            "frozen-model Ray cotrain requires exactly 8 visible GPUs; "
            f"got {len(gpus)} from CUDA_VISIBLE_DEVICES={raw!r}"
        )
    if len(set(gpus)) != len(gpus):
        raise ValueError(
            "frozen-model Ray cotrain requires eight distinct visible GPU IDs; "
            f"got CUDA_VISIBLE_DEVICES={raw!r}"
        )
    return gpus


def _default_out_dir() -> Path:
    explicit = os.environ.get("COTRAIN_RUN_ROOT") or os.environ.get("RUN_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    data_root = Path(os.environ.get("DVLA_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (data_root / "outputs" / "pre_mainline" / "frozen_cotrain_ray" / timestamp).resolve()


def _resume_out_dir(resume_ckpt: Path) -> Path:
    for parent in resume_ckpt.parents:
        if parent.name == "checkpoints":
            return parent.parent.resolve()
    raise ValueError(
        "cannot infer the resume run root from the policy checkpoint; assign COTRAIN_RUN_ROOT=/path"
    )


def _launch_env(visible_gpus: tuple[str, ...]) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
    env.setdefault("DVLA_ROOT", str(PROJECT_ROOT))
    env.setdefault("DVLA_DATA_ROOT", str(PROJECT_ROOT / "data"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("NCCL_NVLS_ENABLE", "0")
    env.setdefault("RAY_DEDUP_LOGS", "0")
    pythonpath = env.get("PYTHONPATH", "")
    entries = [entry for entry in pythonpath.split(":") if entry]
    if str(PROJECT_ROOT) not in entries:
        entries.insert(0, str(PROJECT_ROOT))
    env["PYTHONPATH"] = ":".join(entries)
    return env


def _required_environment_path(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name}=/path is required; use an explicit environment assignment")
    return value


def _hydra_overrides(argv: list[str]) -> list[str]:
    overrides: list[str] = []
    for item in argv:
        if "=" not in item:
            raise ValueError(
                f"expected Hydra key=value override, got {item!r}; "
                "set WORLD_MODEL_CKPT=/path and CLASSIFIER_CKPT=/path before the command"
            )
        overrides.append(str(item))
    return overrides


def _has_hydra_override(overrides: list[str], key: str) -> bool:
    return any(item.split("=", 1)[0].lstrip("+~") == key for item in overrides)


def _environment_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean assignment, got {raw!r}")


def build_launch(argv: list[str]) -> FrozenRayLaunch:
    """Resolve checkpoint paths, Hydra overrides, and the exact 8-GPU command."""

    hydra_overrides = _hydra_overrides(argv)
    wm_ckpt = _component_checkpoint(
        _required_environment_path("WORLD_MODEL_CKPT"),
        component="world_model",
    )
    classifier_ckpt = _component_checkpoint(
        _required_environment_path("CLASSIFIER_CKPT"),
        component="classifier",
    )
    classifier_threshold = None
    if not _has_hydra_override(
        hydra_overrides,
        "algorithm.lumos.classifier_threshold",
    ):
        classifier_threshold = resolve_available_classifier_threshold(
            classifier_ckpt,
            default=0.5,
        )
    resume_value = os.environ.get("COTRAIN_RESUME_CKPT", "").strip()
    resume_ckpt = (
        _existing_file(resume_value, label="policy resume checkpoint") if resume_value else None
    )
    explicit_run_root = os.environ.get("COTRAIN_RUN_ROOT", "").strip()
    if explicit_run_root:
        out_dir = Path(explicit_run_root).expanduser().resolve()
    elif resume_ckpt is not None:
        out_dir = _resume_out_dir(resume_ckpt)
    else:
        out_dir = _default_out_dir()
    visible_gpus = _visible_gpus()
    command = [
        sys.executable,
        "-m",
        "dreamervla.train",
        "experiment=dreamervla_frozen_models_rl_ray",
        "task=openvla_onetraj_libero",
        f"training.out_dir={out_dir}",
        f"init.world_model_state_ckpt={wm_ckpt}",
        f"init.classifier_state_ckpt={classifier_ckpt}",
        "manual_cotrain.ngpu=8",
        "cluster.num_gpus=8",
    ]
    if classifier_threshold is not None:
        command.append(
            "algorithm.lumos.classifier_threshold="
            f"{float(classifier_threshold)!r}"
        )
    if resume_ckpt is not None:
        command.extend(
            [
                "training.resume=true",
                f"training.resume_dir={out_dir}",
                f"manual_cotrain.resume_ckpt={resume_ckpt}",
            ]
        )
    command.extend(hydra_overrides)
    return FrozenRayLaunch(
        command=command,
        env=_launch_env(visible_gpus),
        out_dir=out_dir,
        visible_gpus=visible_gpus,
        resume=resume_ckpt is not None,
    )


def _print_launch(launch: FrozenRayLaunch) -> None:
    print(
        "[frozen-model-ray] "
        f"gpus={','.join(launch.visible_gpus)} "
        f"resume={str(launch.resume).lower()} "
        f"run_root={launch.out_dir}",
        flush=True,
    )
    print(
        "[frozen-model-ray] command: " + " ".join(shlex.quote(part) for part in launch.command),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    launch = build_launch(raw_args)
    _print_launch(launch)
    if _environment_bool("COTRAIN_DRY_RUN"):
        return 0
    launch.out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        launch.command,
        cwd=PROJECT_ROOT,
        env=launch.env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FrozenRayLaunch", "build_launch", "main"]
