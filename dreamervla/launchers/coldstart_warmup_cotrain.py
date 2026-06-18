"""Launch cold-start collection followed by offline-warmup online cotrain."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

PipelineMode = Literal["ray", "noray"]
PipelineTask = Literal["goal", "object", "spatial"]


@dataclass(frozen=True)
class TaskSpec:
    name: PipelineTask
    hydra_task: str
    suite: str
    ckpt_name: str
    stats_key: str


_TASK_SPECS: dict[str, TaskSpec] = {
    "goal": TaskSpec(
        name="goal",
        hydra_task="OpenVLA_Onetraj_ColdStart_LIBERO",
        suite="libero_goal",
        ckpt_name="Openvla-oft-SFT-libero-goal-traj1",
        stats_key="libero_goal_no_noops",
    ),
    "object": TaskSpec(
        name="object",
        hydra_task="OpenVLA_Onetraj_ColdStart_LIBERO_Object",
        suite="libero_object",
        ckpt_name="Openvla-oft-SFT-libero-object-traj1",
        stats_key="libero_object_no_noops",
    ),
    "spatial": TaskSpec(
        name="spatial",
        hydra_task="OpenVLA_Onetraj_ColdStart_LIBERO_Spatial",
        suite="libero_spatial",
        ckpt_name="Openvla-oft-SFT-libero-spatial-traj1",
        stats_key="libero_spatial_no_noops",
    ),
}


@dataclass(frozen=True)
class PipelinePlan:
    mode: PipelineMode
    task: PipelineTask
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


def _resolve_task(task: str) -> TaskSpec:
    normalized = task.strip().lower().replace("-", "_")
    if normalized.startswith("libero_"):
        normalized = normalized.removeprefix("libero_")
    try:
        return _TASK_SPECS[normalized]
    except KeyError as exc:
        raise ValueError("task must be one of: goal, object, spatial") from exc


def build_pipeline_plan(
    *,
    mode: str = "ray",
    task: str = "goal",
    run_root: str | Path,
    python: str = sys.executable,
    collect_overrides: Sequence[str] = (),
    cotrain_overrides: Sequence[str] = (),
    common_overrides: Sequence[str] = (),
) -> PipelinePlan:
    selected_mode = _normalize_mode(mode)
    task_spec = _resolve_task(task)
    root = Path(run_root).expanduser()
    reward_dir = root / "coldstart" / "reward"
    hidden_dir = root / "coldstart" / "hidden"
    collect_out = root / "collect"
    cotrain_out = root / "cotrain"

    collect_cmd = [python, "-m", "dreamervla.train"]
    if selected_mode == "ray":
        collect_cmd.extend(
            [
                "experiment=collect_rollouts_ray",
                f"task={task_spec.hydra_task}",
                "logger=tensorboard",
                "collect.task_ids=[0]",
                "collect.episodes_per_task=4",
                "collect.episode_horizon=64",
                "env.num_workers=2",
                "rollout.target_episodes=4",
                "rollout.max_steps=256",
            ]
        )
    else:
        collect_cmd.extend(
            [
                "experiment=collect_rollouts_onetraj",
                f"task={task_spec.hydra_task}",
                "logger=tensorboard",
                "collect.task_ids=[0]",
                "collect.episodes_per_task=4",
                "collect.episode_horizon=64",
                "collect.envs_per_gpu=1",
                "collect.gpu_id=0",
            ]
        )
    collect_cmd.extend(
        [
            f"task.openvla_oft.hdf5_reward_dir={reward_dir}",
            f"task.openvla_oft.action_hidden_dir={hidden_dir}",
            f"training.out_dir={collect_out}",
            *common_overrides,
            *collect_overrides,
        ]
    )
    cotrain_cmd = [
        python,
        "-m",
        "dreamervla.train",
        "experiment=online_cotrain_pipeline_oft_action_hidden",
        f"task={task_spec.hydra_task}",
        "logger=tensorboard",
        "training.debug=true",
        f"offline_warmup.data_dir={reward_dir}",
        f"offline_warmup.hidden_dir={hidden_dir}",
        "offline_warmup.task_id=0",
        f"env.task_suite_name={task_spec.suite}",
        f"training.out_dir={cotrain_out}",
        *common_overrides,
        *cotrain_overrides,
    ]
    return PipelinePlan(
        mode=selected_mode,
        task=task_spec.name,
        run_root=root,
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        collect_cmd=collect_cmd,
        cotrain_cmd=cotrain_cmd,
    )


def validate_input_assets(*, data_root: str | Path, task: str = "goal") -> list[str]:
    """Return missing or malformed input assets for the default one-traj OFT route."""
    task_spec = _resolve_task(task)
    root = Path(data_root).expanduser()
    ckpt = root / "checkpoints" / "Openvla-oft-SFT-traj1" / task_spec.ckpt_name
    stats = ckpt / "dataset_statistics.json"
    libero = root / "datasets" / "libero" / task_spec.suite
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
            if task_spec.stats_key not in stats_data:
                errors.append(
                    "OpenVLA-OFT dataset statistics missing key "
                    f"'{task_spec.stats_key}': {stats}"
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
    return errors


def _default_run_root() -> Path:
    data_root = Path(os.environ.get("DVLA_DATA_ROOT", "data")).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return data_root / "outputs" / "coldstart_warmup_cotrain" / stamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run cold-start collection, then run OnlineCotrainPipelineRunner with "
            "offline_warmup pointing at the collected HDF5/sidecar outputs."
        )
    )
    parser.add_argument("--mode", default="ray", choices=["ray", "noray", "no-ray", "non-ray"])
    parser.add_argument("--task", default="goal", choices=["goal", "object", "spatial"])
    parser.add_argument("--run-root", default=str(_default_run_root()))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument(
        "--skip-asset-check",
        action="store_true",
        help="Do not validate default ckpt/dataset inputs before launching.",
    )
    parser.add_argument(
        "--common-override",
        action="append",
        default=[],
        help="Hydra override appended to both collect and cotrain commands.",
    )
    parser.add_argument(
        "--collect-override",
        action="append",
        default=[],
        help="Hydra override appended only to the cold-start collect command.",
    )
    parser.add_argument(
        "--cotrain-override",
        action="append",
        default=[],
        help="Hydra override appended only to the cotrain command.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    plan = build_pipeline_plan(
        mode=args.mode,
        task=args.task,
        run_root=args.run_root,
        python=args.python,
        collect_overrides=args.collect_override,
        cotrain_overrides=args.cotrain_override,
        common_overrides=args.common_override,
    )
    print(f"mode: {plan.mode}")
    print(f"task: {plan.task}")
    print(f"run_root: {plan.run_root}")
    print(f"reward_dir: {plan.reward_dir}")
    print(f"hidden_dir: {plan.hidden_dir}")
    print(f"collect: {shlex.join(plan.collect_cmd)}")
    print(f"cotrain: {shlex.join(plan.cotrain_cmd)}")
    if args.dry_run:
        return 0

    if not args.skip_asset_check:
        if args.skip_collect:
            errors = validate_collected_outputs(
                reward_dir=plan.reward_dir,
                hidden_dir=plan.hidden_dir,
            )
        else:
            errors = validate_input_assets(
                data_root=os.environ.get("DVLA_DATA_ROOT", "data"),
                task=plan.task,
            )
        if errors:
            print("asset check failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            print("Use --skip-asset-check only when custom Hydra overrides provide assets.", file=sys.stderr)
            return 2

    if not args.skip_collect:
        subprocess.run(plan.collect_cmd, check=True)
    subprocess.run(plan.cotrain_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
