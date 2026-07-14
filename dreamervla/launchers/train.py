"""Launch one Hydra experiment, optionally through local torchrun."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

_LAUNCHER_KEYS = {
    "batch_size",
    "data_root",
    "distributed",
    "dry_run",
    "gpus",
    "master_port",
    "max_steps",
    "ngpu",
    "num_epochs",
    "num_workers",
    "out_dir",
    "print_config",
    "python",
    "write_libero_config",
}


def _parse_args(
    argv: Sequence[str],
) -> tuple[str, dict[str, str], list[str]]:
    """Split the experiment choice, launcher aliases, and Hydra overrides."""

    experiment: str | None = None
    launcher: dict[str, str] = {}
    overrides: list[str] = []
    items = list(argv)
    index = 0
    passthrough = False
    while index < len(items):
        item = items[index]
        if passthrough:
            overrides.append(item)
            index += 1
            continue
        if item == "--":
            passthrough = True
            index += 1
            continue
        if item in {"--config", "--config-name"}:
            if index + 1 >= len(items):
                raise SystemExit(f"{item} requires an experiment name")
            experiment = items[index + 1]
            index += 2
            continue
        if item.startswith("--config=") or item.startswith("--config-name="):
            experiment = item.split("=", 1)[1]
            index += 1
            continue
        if item.startswith("--"):
            raise SystemExit(
                f"Unsupported launcher flag {item!r}. Use Hydra key=value overrides."
            )
        if "=" not in item:
            raise SystemExit(f"expected a Hydra key=value override, got {item!r}")

        raw_key, raw_value = item.split("=", 1)
        key = raw_key.lstrip("+")
        if key == "experiment":
            experiment = raw_value
        elif key in _LAUNCHER_KEYS:
            launcher[key] = raw_value
        else:
            overrides.append(item)
        index += 1

    if not experiment:
        raise SystemExit("select an experiment with --config <name> or experiment=<name>")
    return experiment, launcher, overrides


def _parse_value(value: str) -> Any:
    if "," in value and not value.startswith(("[", "{", "'", '"')):
        return value
    return OmegaConf.from_dotlist([f"value={value}"]).value


def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _has_path(cfg: DictConfig, dotted: str) -> bool:
    sentinel = object()
    return OmegaConf.select(cfg, dotted, default=sentinel) is not sentinel


def _compose(experiment: str, overrides: Sequence[str]) -> DictConfig:
    register_dreamervla_resolvers()
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="experiment_launcher",
        version_base=None,
    ):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", *overrides],
        )
    OmegaConf.resolve(cfg)
    return cfg


def _target_overrides(
    cfg: DictConfig,
    launcher: Mapping[str, str],
) -> list[str]:
    mapping = {
        "batch_size": (
            "training.global_batch_size",
            "dataloader.batch_size",
            "training.batch_size",
        ),
        "num_workers": ("dataloader.num_workers", "training.num_workers"),
        "max_steps": ("training.max_steps", "training.max_train_steps"),
        "num_epochs": ("training.num_epochs",),
        "out_dir": ("training.out_dir",),
    }
    overrides: list[str] = []
    for alias, candidates in mapping.items():
        value = launcher.get(alias)
        if value in (None, ""):
            continue
        target = next((key for key in candidates if _has_path(cfg, key)), None)
        if target is None:
            raise ValueError(
                f"experiment does not expose a target for launcher alias {alias!r}"
            )
        overrides.append(f"{target}={value}")
    return overrides


def _launch_value(
    cfg: DictConfig,
    launcher: Mapping[str, str],
    key: str,
    default: Any,
) -> Any:
    if key in launcher:
        return _parse_value(launcher[key])
    return _select(cfg, f"launch.{key}", default)


def _write_libero_config(data_root: str) -> None:
    config_root = Path(
        os.environ.get("LIBERO_CONFIG_PATH", str(Path(data_root) / ".libero"))
    )
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {PROJECT_ROOT}/third_party/LIBERO/libero/libero",
                f"bddl_files: {PROJECT_ROOT}/third_party/LIBERO/libero/libero/bddl_files",
                f"init_states: {PROJECT_ROOT}/third_party/LIBERO/libero/libero/init_files",
                f"datasets: {data_root}/datasets/libero",
                f"assets: {PROJECT_ROOT}/third_party/LIBERO/libero/libero/assets",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _missing_required_values(cfg: DictConfig) -> list[str]:
    required = _select(cfg, "launch.required_target_values", []) or []
    return [
        str(key)
        for key in required
        if _select(cfg, str(key), None) in (None, "", "???")
    ]


def _build_env(
    cfg: DictConfig,
    launcher: Mapping[str, str],
    *,
    data_root: str,
) -> dict[str, str]:
    env = dict(os.environ)
    env["DVLA_ROOT"] = str(PROJECT_ROOT)
    env["DVLA_DATA_ROOT"] = data_root
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    gpus = _launch_value(cfg, launcher, "gpus", None)
    if gpus not in (None, ""):
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    pythonpath = [item for item in env.get("PYTHONPATH", "").split(":") if item]
    if str(PROJECT_ROOT) not in pythonpath:
        pythonpath.insert(0, str(PROJECT_ROOT))
    env["PYTHONPATH"] = ":".join(pythonpath)
    launch_env = _select(cfg, "launch.env", {}) or {}
    plain_env = (
        OmegaConf.to_container(launch_env, resolve=True)
        if isinstance(launch_env, DictConfig)
        else dict(launch_env)
    )
    for key, value in plain_env.items():
        if value is not None:
            env[str(key)] = str(value)
    return env


def _command(
    cfg: DictConfig,
    launcher: Mapping[str, str],
    experiment: str,
    overrides: Sequence[str],
) -> list[str]:
    python = str(_launch_value(cfg, launcher, "python", "python"))
    ngpu = int(_launch_value(cfg, launcher, "ngpu", 1))
    distributed = bool(_launch_value(cfg, launcher, "distributed", False))
    command = [python, "-m"]
    if distributed and ngpu > 1:
        port = int(_launch_value(cfg, launcher, "master_port", 29500))
        command.extend(
            [
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={ngpu}",
                f"--master_port={port}",
                "-m",
            ]
        )
    command.extend(
        ["dreamervla.train", "--config-name", "train", f"experiment={experiment}"]
    )
    command.extend(overrides)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    experiment, launcher, overrides = _parse_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    cfg = _compose(experiment, overrides)
    alias_overrides = _target_overrides(cfg, launcher)
    if alias_overrides:
        overrides = [*overrides, *alias_overrides]
        cfg = _compose(experiment, overrides)

    missing = _missing_required_values(cfg)
    if missing:
        rendered = ", ".join(f"{key}=<value>" for key in missing)
        print(
            f"[experiment:{experiment}] missing required Hydra override(s): {rendered}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    data_root = str(
        _launch_value(
            cfg,
            launcher,
            "data_root",
            os.environ.get("DVLA_DATA_ROOT", PROJECT_ROOT / "data"),
        )
    )
    if bool(_launch_value(cfg, launcher, "write_libero_config", True)):
        _write_libero_config(data_root)
    env = _build_env(cfg, launcher, data_root=data_root)
    command = _command(cfg, launcher, experiment, overrides)
    if bool(_launch_value(cfg, launcher, "print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))
    print(
        f"[experiment:{experiment}] target={cfg._target_} "
        f"ngpu={_launch_value(cfg, launcher, 'ngpu', 1)} "
        f"gpus={env.get('CUDA_VISIBLE_DEVICES', '<all>')}",
        flush=True,
    )
    print(f"[experiment:{experiment}] command: {shlex.join(command)}", flush=True)
    if bool(_launch_value(cfg, launcher, "dry_run", False)):
        return 0
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
