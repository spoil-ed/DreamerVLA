"""One-command launcher for the trainable WM/CLS/VLA cotrain mainline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "openvla_libero"


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


def _public_cli_overrides(argv: list[str]) -> list[str]:
    """Translate the readable cotrain CLI into Hydra overrides."""

    parser = argparse.ArgumentParser(
        prog="dreamervla-cotrain",
        description="Launch OpenVLA LIBERO cotrain from Hydra configuration.",
    )
    parser.add_argument("--config", help="Hydra experiment name")
    parser.add_argument("--wm_ckpt", help="world-model checkpoint file or directory")
    parser.add_argument("--cls_ckpt", help="classifier checkpoint file or directory")
    args, remaining = parser.parse_known_args(argv)
    values = _overrides(remaining)

    if args.config is not None and _has_override(values, "experiment"):
        raise ValueError("--config cannot be combined with the Hydra experiment override")
    if not _has_override(values, "experiment"):
        experiment = args.config or DEFAULT_EXPERIMENT
        values.insert(0, f"experiment={experiment}")

    if (args.wm_ckpt is None) != (args.cls_ckpt is None):
        raise ValueError("--wm_ckpt and --cls_ckpt must be supplied together")
    for option, raw, hydra_key in (
        ("--wm_ckpt", args.wm_ckpt, "init.world_model_state_ckpt"),
        ("--cls_ckpt", args.cls_ckpt, "init.classifier_state_ckpt"),
    ):
        if raw is None:
            continue
        if _has_override(values, hydra_key):
            raise ValueError(f"{option} cannot be combined with the Hydra {hydra_key} override")
        path = Path(raw).expanduser().resolve()
        if not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(f"{option} checkpoint does not exist: {path}")
        values.append(f"{hydra_key}={_hydra_string(path)}")
    return values


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


def _nested_output_dim(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    direct = value.get("output_dim")
    if direct is not None:
        try:
            output_dim = int(direct)
        except (TypeError, ValueError):
            output_dim = 0
        if output_dim in {1, 2}:
            return output_dim
    for key in ("init_args", "classifier", "config"):
        inferred = _nested_output_dim(value.get(key))
        if inferred is not None:
            return inferred
    return None


def _classifier_checkpoint_output_dim(path: str | Path) -> int | None:
    """Read the binary-head contract without constructing the classifier."""

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
        if not config_path.is_file():
            return None
        with config_path.open(encoding="utf-8") as handle:
            return _nested_output_dim(json.load(handle))
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return None
    state_dicts = payload.get("state_dicts")
    state = payload.get("classifier")
    if state is None and isinstance(state_dicts, Mapping):
        state = state_dicts.get("classifier", state_dicts.get("model"))
    if state is None:
        state = payload.get("model")
    if not isinstance(state, Mapping):
        return _nested_output_dim(payload)
    for name, tensor in state.items():
        if (
            isinstance(name, str)
            and (name == "head.weight" or name.endswith(".head.weight"))
            and isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and int(tensor.shape[0]) in {1, 2}
        ):
            return int(tensor.shape[0])
    return _nested_output_dim(payload)


def _classifier_contract_overrides(values: list[str], cfg: DictConfig) -> bool:
    checkpoint = OmegaConf.select(cfg, "init.classifier_state_ckpt", default=None)
    if checkpoint in {None, ""}:
        return False
    output_dim = _classifier_checkpoint_output_dim(str(checkpoint))
    if output_dim is None:
        return False

    output_key = "ray_components.classifier.kwargs.output_dim"
    loss_key = "learner.train_cfg.classifier_loss_type"
    configured_output = int(OmegaConf.select(cfg, output_key))
    configured_loss = str(OmegaConf.select(cfg, loss_key)).lower()
    expected_loss = "bce" if output_dim == 1 else "ce"
    if _has_override(values, output_key) and configured_output != output_dim:
        raise ValueError(
            f"classifier checkpoint has output_dim={output_dim}, but {output_key}="
            f"{configured_output} was explicitly requested"
        )
    if _has_override(values, loss_key) and configured_loss != expected_loss:
        raise ValueError(
            f"classifier checkpoint output_dim={output_dim} requires {loss_key}="
            f"{expected_loss}, got {configured_loss}"
        )

    changed = False
    if not _has_override(values, output_key) and configured_output != output_dim:
        values.append(f"{output_key}={output_dim}")
        changed = True
    if not _has_override(values, loss_key) and configured_loss != expected_loss:
        values.append(f"{loss_key}={expected_loss}")
        changed = True
    return changed


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

    values = _public_cli_overrides(argv)
    _component_overrides(values)
    _runtime_overrides(values)
    cfg = _compose(values)
    if _classifier_contract_overrides(values, cfg):
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
