"""Launch one Hydra experiment, optionally through local torchrun."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.launchers.contracts import DefaultLaunchContract, LaunchContract
from dreamervla.utils.run_paths import infer_run_root, resolve_resume_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

_LAUNCHER_KEYS = {
    "data_root",
    "distributed",
    "dry_run",
    "gpus",
    "master_port",
    "ngpu",
    "print_config",
    "python",
    "write_libero_config",
}

_REMOVED_SEMANTIC_ALIASES = {
    "batch_size",
    "max_steps",
    "num_epochs",
    "num_workers",
    "out_dir",
}


@dataclass(frozen=True)
class ExperimentLaunch:
    """Fully resolved command, environment, and configuration for one experiment."""

    experiment: str
    command: tuple[str, ...]
    env: dict[str, str]
    cfg: DictConfig
    dry_run: bool
    print_config: bool
    ngpu: int
    summary_lines: tuple[str, ...] = ()


def _experiment_from_argv(argv: Sequence[str]) -> str:
    """Discover the experiment without rejecting contract-specific flags."""

    experiment: str | None = None
    items = list(argv)
    index = 0
    while index < len(items):
        item = items[index]
        if item in {"--config", "--config-name"}:
            if index + 1 >= len(items):
                raise SystemExit(f"{item} requires an experiment name")
            experiment = items[index + 1]
            index += 2
            continue
        if item.startswith(("--config=", "--config-name=")):
            experiment = item.split("=", 1)[1]
        elif "=" in item and item.split("=", 1)[0].lstrip("+") == "experiment":
            experiment = item.split("=", 1)[1]
        index += 1
    if not experiment:
        raise SystemExit("select an experiment with --config <name> or experiment=<name>")
    return experiment


def _parse_args(
    argv: Sequence[str],
) -> tuple[str, dict[str, str], list[str]]:
    """Split experiment choice, launch mechanics, and Hydra overrides."""

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
        if item == "--resume":
            if index + 1 >= len(items):
                raise SystemExit("--resume requires a run or checkpoint path")
            launcher["resume"] = items[index + 1]
            index += 2
            continue
        if item.startswith("--resume="):
            launcher["resume"] = item.split("=", 1)[1]
            index += 1
            continue
        if item.startswith(("--config=", "--config-name=")):
            experiment = item.split("=", 1)[1]
            index += 1
            continue
        if item.startswith("--"):
            raise SystemExit(f"Unsupported launcher flag {item!r}. Use Hydra key=value overrides.")
        if "=" not in item:
            raise SystemExit(f"expected a Hydra key=value override, got {item!r}")

        raw_key, raw_value = item.split("=", 1)
        key = raw_key.lstrip("+")
        if key == "experiment":
            experiment = raw_value
        elif key in _REMOVED_SEMANTIC_ALIASES:
            raise SystemExit(
                f"Launcher alias {key!r} was removed. Use the experiment's explicit "
                "Hydra key=value override."
            )
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


def _build_contract(cfg: DictConfig) -> LaunchContract:
    contract_cfg = _select(cfg, "launch.contract", None)
    if contract_cfg in (None, ""):
        return DefaultLaunchContract()
    contract = instantiate(contract_cfg)
    required_methods = (
        "normalize_argv",
        "derive_overrides",
        "validate",
        "update_env",
        "summary_lines",
    )
    missing = [name for name in required_methods if not callable(getattr(contract, name, None))]
    if missing:
        raise TypeError(
            "launch.contract must satisfy LaunchContract; missing " + ", ".join(missing)
        )
    return contract


def _target_overrides(
    launcher: Mapping[str, str],
    raw_overrides: Sequence[str] = (),
) -> list[str]:
    overrides: list[str] = []
    resume_source = launcher.get("resume")
    if resume_source not in (None, ""):
        raw_keys = {item.split("=", 1)[0].lstrip("+~") for item in raw_overrides}
        conflicts = raw_keys.intersection(
            {
                "training.out_dir",
                "training.resume",
                "training.resume_dir",
                "training.resume_path",
            }
        )
        if conflicts:
            raise ValueError(
                "--resume cannot be combined with out_dir or training.out_dir/resume overrides"
            )
        source = Path(str(resume_source)).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"--resume path does not exist: {source}")
        checkpoint = resolve_resume_checkpoint(source)
        run_root = infer_run_root(source)
        overrides.extend(
            [
                "training.resume=true",
                f"training.resume_path={json.dumps(str(checkpoint))}",
                f"training.resume_dir={json.dumps(str(run_root))}",
                f"training.out_dir={json.dumps(str(run_root))}",
            ]
        )
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
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path(data_root) / ".libero")))
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
    return [str(key) for key in required if _select(cfg, str(key), None) in (None, "", "???")]


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
    command.extend(["dreamervla.train", "--config-name", "train", f"experiment={experiment}"])
    command.extend(overrides)
    return command


def build_launch(argv: Sequence[str]) -> ExperimentLaunch:
    """Resolve one generic or contract-specialized training launch."""

    raw_argv = list(argv)
    discovered_experiment = _experiment_from_argv(raw_argv)
    discovery_cfg = _compose(discovered_experiment, [])
    contract = _build_contract(discovery_cfg)
    normalized_argv = contract.normalize_argv(raw_argv)
    experiment, launcher, overrides = _parse_args(normalized_argv)
    cfg = _compose(experiment, overrides)

    alias_overrides = _target_overrides(launcher, overrides)
    if alias_overrides:
        overrides = [*overrides, *alias_overrides]
        cfg = _compose(experiment, overrides)

    derived_overrides = contract.derive_overrides(cfg, overrides)
    if derived_overrides:
        overrides = [*overrides, *derived_overrides]
        cfg = _compose(experiment, overrides)
    contract.validate(cfg)

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
    contract.update_env(cfg, env)
    command = _command(cfg, launcher, experiment, overrides)
    return ExperimentLaunch(
        experiment=experiment,
        command=tuple(command),
        env=env,
        cfg=cfg,
        dry_run=bool(_launch_value(cfg, launcher, "dry_run", False)),
        print_config=bool(_launch_value(cfg, launcher, "print_config", False)),
        ngpu=int(_launch_value(cfg, launcher, "ngpu", 1)),
        summary_lines=tuple(contract.summary_lines(cfg, env)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    launch = build_launch(list(sys.argv[1:] if argv is None else argv))
    experiment = launch.experiment
    cfg = launch.cfg

    missing = _missing_required_values(cfg)
    if missing:
        rendered = ", ".join(f"{key}=<value>" for key in missing)
        print(
            f"[experiment:{experiment}] missing required Hydra override(s): {rendered}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if launch.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    print(
        f"[experiment:{experiment}] target={cfg._target_} "
        f"ngpu={launch.ngpu} "
        f"gpus={launch.env.get('CUDA_VISIBLE_DEVICES', '<all>')}",
        flush=True,
    )
    for line in launch.summary_lines:
        print(line, flush=True)
    print(
        f"[experiment:{experiment}] command: {shlex.join(launch.command)}",
        flush=True,
    )
    if launch.dry_run:
        return 0
    subprocess.run(launch.command, cwd=PROJECT_ROOT, env=launch.env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ExperimentLaunch", "build_launch", "main"]
