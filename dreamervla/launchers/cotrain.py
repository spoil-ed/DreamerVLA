"""One-command launcher for the trainable WM/CLS/VLA cotrain mainline."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "dreamervla_wmcls_cotrain_ray"


@dataclass(frozen=True)
class CotrainLaunch:
    """Fully resolved mainline cotrain command and process environment."""

    command: tuple[str, ...]
    env: dict[str, str]
    cfg: DictConfig


def _hydra_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _overrides(argv: list[str]) -> list[str]:
    for item in argv:
        if "=" not in item:
            raise ValueError(f"expected a Hydra key=value override, got {item!r}")
    return list(argv)


def _has_override(values: list[str], key: str) -> bool:
    return any(item.split("=", 1)[0].lstrip("+~") == key for item in values)


def _component_override(
    values: list[str],
    *,
    env_name: str,
    hydra_key: str,
) -> None:
    if _has_override(values, hydra_key):
        return
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return
    path = Path(raw).expanduser().resolve()
    if not (path.is_file() or path.is_dir()):
        raise FileNotFoundError(f"{env_name} checkpoint does not exist: {path}")
    values.append(f"{hydra_key}={_hydra_string(path)}")


def _component_overrides(values: list[str]) -> None:
    wm_key = "init.world_model_state_ckpt"
    classifier_key = "init.classifier_state_ckpt"
    wm_supplied = _has_override(values, wm_key) or bool(
        os.environ.get("WORLD_MODEL_CKPT", "").strip()
    )
    classifier_supplied = _has_override(values, classifier_key) or bool(
        os.environ.get("CLASSIFIER_CKPT", "").strip()
    )
    if not wm_supplied and not classifier_supplied:
        raise ValueError(
            "set WORLD_MODEL_CKPT and CLASSIFIER_CKPT to pretrained component "
            "checkpoints before launching cotrain"
        )
    if wm_supplied != classifier_supplied:
        raise ValueError(
            "set both WORLD_MODEL_CKPT and CLASSIFIER_CKPT for a warm start, "
            "or train the missing component with its independent runner first"
        )
    _component_override(
        values,
        env_name="WORLD_MODEL_CKPT",
        hydra_key=wm_key,
    )
    _component_override(
        values,
        env_name="CLASSIFIER_CKPT",
        hydra_key=classifier_key,
    )


def _runtime_overrides(values: list[str]) -> None:
    key = "manual_cotrain.global_steps"
    raw = os.environ.get("WMCLS_COTRAIN_GLOBAL_STEPS", "").strip()
    if not raw or _has_override(values, key):
        return
    try:
        global_steps = int(raw)
    except ValueError as exc:
        raise ValueError(
            "WMCLS_COTRAIN_GLOBAL_STEPS must be a positive integer"
        ) from exc
    if global_steps <= 0:
        raise ValueError("WMCLS_COTRAIN_GLOBAL_STEPS must be a positive integer")
    values.append(f"{key}={global_steps}")


def _compose(values: list[str]) -> DictConfig:
    register_dreamervla_resolvers()
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"),
        job_name="cotrain_launcher",
        version_base=None,
    ):
        cfg = compose(config_name="train", overrides=values)
    OmegaConf.resolve(cfg)
    return cfg


def _process_env(cfg: DictConfig) -> dict[str, str]:
    count = int(OmegaConf.select(cfg, "manual_cotrain.ngpu", default=1))
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible = [item.strip() for item in raw.split(",") if item.strip()]
    if not visible:
        visible = [str(index) for index in range(count)]
    if len(visible) != count or len(set(visible)) != count:
        raise ValueError(
            f"cotrain requires {count} distinct visible GPUs; "
            f"CUDA_VISIBLE_DEVICES={raw!r}"
        )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    env.setdefault("DVLA_ROOT", str(PROJECT_ROOT))
    env.setdefault("DVLA_DATA_ROOT", str(PROJECT_ROOT / "data"))
    env.setdefault(
        "LIBERO_CONFIG_PATH",
        str((Path(env["DVLA_DATA_ROOT"]) / ".libero").resolve()),
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("NCCL_NVLS_ENABLE", "0")
    env.setdefault("RAY_DEDUP_LOGS", "0")
    entries = [item for item in env.get("PYTHONPATH", "").split(":") if item]
    if str(PROJECT_ROOT) not in entries:
        entries.insert(0, str(PROJECT_ROOT))
    env["PYTHONPATH"] = ":".join(entries)
    return env


def build_launch(argv: list[str]) -> CotrainLaunch:
    """Build one direct train-only cotrain command from Hydra configuration."""

    values = _overrides(argv)
    if not _has_override(values, "experiment"):
        values.insert(0, f"experiment={DEFAULT_EXPERIMENT}")
    _component_overrides(values)
    _runtime_overrides(values)
    cfg = _compose(values)
    command = (sys.executable, "-m", "dreamervla.train", *values)
    return CotrainLaunch(command=command, env=_process_env(cfg), cfg=cfg)


def _print_launch(launch: CotrainLaunch) -> None:
    cfg = launch.cfg
    debug = bool(OmegaConf.select(cfg, "training.debug", default=False))
    global_steps = 10 if debug else OmegaConf.select(cfg, "manual_cotrain.global_steps")
    configured_eval_every = int(
        OmegaConf.select(cfg, "manual_cotrain.eval_interval_global_steps", default=0)
    )
    eval_every = 1 if debug else configured_eval_every
    save_every = 1 if debug else OmegaConf.select(cfg, "manual_cotrain.checkpoint_every")
    print(
        "[cotrain] "
        f"debug={str(debug).lower()} "
        f"global_steps={global_steps} "
        f"eval_every={eval_every} "
        f"save_every={save_every} "
        f"ngpu={OmegaConf.select(cfg, 'manual_cotrain.ngpu')} "
        f"gpus={launch.env['CUDA_VISIBLE_DEVICES']}",
        flush=True,
    )
    print(
        "[cotrain] checkpoints "
        f"vla={OmegaConf.select(cfg, 'init.vla_ckpt_path')} "
        f"world_model={OmegaConf.select(cfg, 'init.world_model_state_ckpt')} "
        f"classifier={OmegaConf.select(cfg, 'init.classifier_state_ckpt')}",
        flush=True,
    )
    print(
        "[cotrain] command: " + " ".join(shlex.quote(item) for item in launch.command),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    launch = build_launch(list(sys.argv[1:] if argv is None else argv))
    _print_launch(launch)
    if os.environ.get("COTRAIN_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return 0
    return int(
        subprocess.run(
            launch.command,
            cwd=PROJECT_ROOT,
            env=launch.env,
            check=False,
        ).returncode
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CotrainLaunch", "build_launch", "main"]
