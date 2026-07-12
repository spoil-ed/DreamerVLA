"""One-command launcher for eight-GPU frozen-model Ray policy training."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dreamervla.launchers.frozen_model_pre_mainline import (
    select_classifier_checkpoint,
    select_world_model_checkpoint,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the policy on 8 GPUs with frozen pretrained WM/CLS Ray workers."
        )
    )
    parser.add_argument("world_model_ckpt")
    parser.add_argument("classifier_ckpt")
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


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
            return select_world_model_checkpoint(path)
        if component == "classifier":
            return select_classifier_checkpoint(path)
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
    data_root = Path(
        os.environ.get("DVLA_DATA_ROOT", PROJECT_ROOT / "data")
    ).expanduser()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        data_root
        / "outputs"
        / "pre_mainline"
        / "frozen_cotrain_ray"
        / timestamp
    ).resolve()


def _resume_out_dir(resume_ckpt: Path) -> Path:
    for parent in resume_ckpt.parents:
        if parent.name == "checkpoints":
            return parent.parent.resolve()
    raise ValueError(
        "cannot infer the resume run root from the policy checkpoint; "
        "pass --run-root"
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


def build_launch(argv: list[str]) -> FrozenRayLaunch:
    """Resolve checkpoint paths, Hydra overrides, and the exact 8-GPU command."""

    args, hydra_overrides = _parser().parse_known_args(argv)
    wm_ckpt = _component_checkpoint(
        args.world_model_ckpt,
        component="world_model",
    )
    classifier_ckpt = _component_checkpoint(
        args.classifier_ckpt,
        component="classifier",
    )
    resume_ckpt = (
        _existing_file(args.resume_ckpt, label="policy resume checkpoint")
        if args.resume_ckpt
        else None
    )
    if args.run_root:
        out_dir = Path(args.run_root).expanduser().resolve()
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
        "[frozen-model-ray] command: "
        + " ".join(shlex.quote(part) for part in launch.command),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    launch = build_launch(raw_args)
    args, _ = _parser().parse_known_args(raw_args)
    _print_launch(launch)
    if bool(args.dry_run):
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
