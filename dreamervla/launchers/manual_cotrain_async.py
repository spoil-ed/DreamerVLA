"""User-facing launcher for the manual async OpenVLA-OFT cotrain route."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GPUS = "0,1,2,3,4,5"
DEFAULT_TASK = "goal"
DEFAULT_RESUME_EXTRA_STEPS = 1_000_000
DEFAULT_GLOBAL_STEPS = 1_000_000
DEFAULT_CHECKPOINT_EVERY = 5
DEFAULT_ENVS_PER_WORKER = 8
DEFAULT_ROLLOUT_EPOCH = 1
DEFAULT_MAX_STEPS_PER_ROLLOUT_EPOCH = 64
DEFAULT_WM_ROLLOUT_MULTIPLIER = 4


@dataclass(frozen=True)
class ManualCotrainLaunch:
    command: list[str]
    env: dict[str, str]
    out_dir: Path
    visible_gpus: tuple[str, ...]
    resume: bool
    resume_step: int | None
    target_step: int | None


def _parse_key_values(argv: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in argv:
        if "=" not in item:
            raise ValueError(
                f"expected key=value argument, got {item!r}; "
                "supported keys: resume, ckpt, gpus, dry_run"
            )
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"empty key in argument {item!r}")
        values[key] = value.strip()
    return values


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def _parse_gpus(value: str | None) -> tuple[str, ...]:
    raw = value or os.environ.get("CUDA_VISIBLE_DEVICES") or DEFAULT_GPUS
    gpus = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not gpus:
        raise ValueError("gpus must contain at least one visible GPU id")
    return gpus


def _data_root() -> Path:
    return Path(os.environ.get("DVLA_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()


def _checkpoint_payload_path(path: Path) -> Path:
    if path.name != "manual_cotrain_manifest.json":
        return path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    components = manifest.get("components", {})
    policy = components.get("policy", {}) if isinstance(components, dict) else {}
    payload_name = str(policy.get("path", "manual_cotrain.ckpt"))
    return path.parent / payload_name


def _resume_step_from_checkpoint(path: Path) -> int:
    payload_path = _checkpoint_payload_path(path.expanduser())
    manifest_path = payload_path.with_name("manual_cotrain_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            return int(manifest["global_step"])
        except KeyError as exc:
            raise ValueError(
                f"manual cotrain manifest missing global_step: {manifest_path}"
            ) from exc
    match = re.search(r"manual_cotrain_step_(\d+)", str(payload_path))
    if match:
        return int(match.group(1))
    raise ValueError(
        "cannot infer resume global_step; pass a checkpoint under "
        "manual_cotrain_step_<N>/ or include manual_cotrain_manifest.json"
    )


def _run_tag(*, resume: bool, resume_step: int | None, ngpu: int) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if resume:
        step = "unknown" if resume_step is None else f"{resume_step:06d}"
        return f"manual_goal_resume{step}_{ngpu}gpu_{timestamp}"
    return f"manual_goal_fresh_{ngpu}gpu_{timestamp}"


def _base_env(visible_gpus: tuple[str, ...]) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("NCCL_NVLS_ENABLE", "0")
    env.setdefault("RAY_DEDUP_LOGS", "0")
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in env["PYTHONPATH"].split(":"):
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
    return env


def build_launch(argv: list[str]) -> ManualCotrainLaunch:
    values = _parse_key_values(argv)
    unsupported = sorted(set(values) - {"resume", "ckpt", "gpus", "dry_run"})
    if unsupported:
        raise ValueError(f"unsupported arguments: {', '.join(unsupported)}")

    resume = _parse_bool(values.get("resume"), default=False)
    visible_gpus = _parse_gpus(values.get("gpus"))
    ngpu = len(visible_gpus)
    env = _base_env(visible_gpus)

    if resume:
        raw_ckpt = values.get("ckpt")
        if not raw_ckpt:
            raise ValueError("resume=true requires ckpt=/path/to/manual_cotrain.ckpt")
        ckpt = Path(raw_ckpt).expanduser()
        payload_path = _checkpoint_payload_path(ckpt)
        if not payload_path.is_file():
            raise FileNotFoundError(f"manual cotrain checkpoint not found: {payload_path}")
        resume_step = _resume_step_from_checkpoint(ckpt)
        target_step = resume_step + DEFAULT_RESUME_EXTRA_STEPS
        out_dir = _data_root() / "outputs" / _run_tag(
            resume=True,
            resume_step=resume_step,
            ngpu=ngpu,
        ) / "cotrain"
        command = [
            sys.executable,
            "-m",
            "dreamervla.train",
            "experiment=openvla_onetraj_libero_cotrain_ray",
            "task=openvla_onetraj_coldstart_libero",
            "render_backend=osmesa",
            f"training.out_dir={out_dir}",
            f"manual_cotrain.ngpu={ngpu}",
            f"+cluster.num_gpus={ngpu}",
            f"manual_cotrain.envs_per_worker={DEFAULT_ENVS_PER_WORKER}",
            f"manual_cotrain.rollout_epoch={DEFAULT_ROLLOUT_EPOCH}",
            (
                "manual_cotrain.max_steps_per_rollout_epoch="
                f"{DEFAULT_MAX_STEPS_PER_ROLLOUT_EPOCH}"
            ),
            f"manual_cotrain.wm_rollout_multiplier={DEFAULT_WM_ROLLOUT_MULTIPLIER}",
            f"manual_cotrain.global_steps={target_step}",
            "manual_cotrain.learner_update_step=1",
            "manual_cotrain.wm_env_write_replay=false",
            "manual_cotrain.requires_bootstrap_value=false",
            f"+manual_cotrain.resume_ckpt={ckpt}",
            f"+manual_cotrain.checkpoint_every={DEFAULT_CHECKPOINT_EVERY}",
            "+manual_cotrain.env_rollout_timeout_s=1200",
            f"+actor.init_ckpt.path={ckpt}",
            "+actor.init_ckpt.components=[policy]",
            f"+learner.init_ckpt.path={ckpt}",
            "+learner.init_ckpt.components=[world_model,classifier]",
        ]
        return ManualCotrainLaunch(
            command=command,
            env=env,
            out_dir=out_dir,
            visible_gpus=visible_gpus,
            resume=True,
            resume_step=resume_step,
            target_step=target_step,
        )

    out_dir = _data_root() / "outputs" / _run_tag(
        resume=False,
        resume_step=None,
        ngpu=ngpu,
    )
    command = [
        sys.executable,
        "-m",
        "dreamervla.launchers.coldstart_warmup_cotrain",
        "mode=ray",
        f"task={DEFAULT_TASK}",
        f"ngpu={ngpu}",
        "profile=multi_gpu",
        "cotrain_engine=async",
        "render_backend=osmesa",
        f"run_root={out_dir}",
        f"manual_cotrain.envs_per_worker={DEFAULT_ENVS_PER_WORKER}",
        f"manual_cotrain.rollout_epoch={DEFAULT_ROLLOUT_EPOCH}",
        (
            "manual_cotrain.max_steps_per_rollout_epoch="
            f"{DEFAULT_MAX_STEPS_PER_ROLLOUT_EPOCH}"
        ),
        f"manual_cotrain.wm_rollout_multiplier={DEFAULT_WM_ROLLOUT_MULTIPLIER}",
        f"manual_cotrain.global_steps={DEFAULT_GLOBAL_STEPS}",
        "manual_cotrain.learner_update_step=1",
        "manual_cotrain.wm_env_write_replay=false",
        "manual_cotrain.requires_bootstrap_value=false",
        f"+manual_cotrain.checkpoint_every={DEFAULT_CHECKPOINT_EVERY}",
        "+manual_cotrain.env_rollout_timeout_s=1200",
    ]
    return ManualCotrainLaunch(
        command=command,
        env=env,
        out_dir=out_dir,
        visible_gpus=visible_gpus,
        resume=False,
        resume_step=None,
        target_step=DEFAULT_GLOBAL_STEPS,
    )


def _print_launch(launch: ManualCotrainLaunch) -> None:
    print(
        "[manual-cotrain-launch] "
        f"gpus={','.join(launch.visible_gpus)} "
        f"resume={str(launch.resume).lower()} "
        f"out_dir={launch.out_dir}",
        flush=True,
    )
    if launch.resume:
        print(
            "[manual-cotrain-launch] "
            f"resume_step={launch.resume_step} target_step={launch.target_step}",
            flush=True,
        )
    print("[manual-cotrain-launch] command:", flush=True)
    print(" ".join(shlex.quote(part) for part in launch.command), flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    launch = build_launch(argv)
    _print_launch(launch)
    if _parse_bool(_parse_key_values(argv).get("dry_run"), default=False):
        return 0
    launch.out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(launch.command, env=launch.env, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
