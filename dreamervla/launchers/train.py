from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "scripts"
TRAIN_CONFIG_DIR = PROJECT_ROOT / "configs"

_LAUNCHER_KEYS = {
    "batch_size",
    "data_root",
    "distributed",
    "dry_run",
    "env",
    "gpus",
    "master_port",
    "max_steps",
    "name",
    "ngpu",
    "num_epochs",
    "num_workers",
    "out_dir",
    "print_config",
    "python",
    "required_target_values",
    "experiment",
    "route",
    "run_tag",
    "task",
    "write_libero_config",
}


def _parse_hydra_like_args(argv: Sequence[str]) -> tuple[str, list[str], list[str]]:
    config_name = "train_vla"
    launcher_overrides: list[str] = []
    experiment_overrides: list[str] = []
    i = 0
    passthrough = False
    while i < len(argv):
        item = argv[i]
        if passthrough:
            experiment_overrides.append(item)
            i += 1
            continue
        if item == "--":
            passthrough = True
            i += 1
            continue
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

        if item.startswith("--"):
            raise SystemExit(
                f"Unsupported launcher flag {item!r}. Use Hydra override syntax like "
                "experiment=<name>, task=<name>, gpus=0,1, batch_size=16."
            )

        if item == "-m" or item.startswith("-m="):
            experiment_overrides.append(item)
            i += 1
            continue

        raw_key = item.lstrip("+").split("=", 1)[0]
        key = raw_key.split(".", 1)[0]
        is_launcher_override = raw_key in _LAUNCHER_KEYS or key == "env"
        if "=" in item and not item.startswith("+") and is_launcher_override:
            if raw_key == "route":
                launcher_overrides.append(item.replace("route=", "experiment=", 1))
            else:
                if raw_key == "gpus":
                    raw_key, raw_value = item.split("=", 1)
                    if "," in raw_value and not raw_value.startswith(("'", '"', "[")):
                        item = f"{raw_key}='{raw_value}'"
                launcher_overrides.append(item)
        else:
            experiment_overrides.append(item)
        i += 1
    return config_name, launcher_overrides, experiment_overrides


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, (DictConfig, ListConfig)) else value


def _env_value(value: Any) -> str | None:
    value = _plain(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _gpu_count(gpus: Any) -> int:
    gpus = _plain(gpus)
    if gpus is None or gpus == "":
        return 1
    if isinstance(gpus, int):
        return 1
    text = str(gpus)
    return max(1, len([part for part in text.replace(",", " ").split() if part]))


def _append_if_set(
    overrides: list[str], cfg: Mapping[str, Any], key: str, value: Any
) -> None:
    rendered = _env_value(value)
    if rendered is not None and rendered != "":
        overrides.append(f"{key}={rendered}")


def _has_path(cfg: Mapping[str, Any], dotted: str) -> bool:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def _select_path(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _missing_required_target_values(
    cfg: Mapping[str, Any],
    experiment_overrides: Sequence[str],
) -> list[str]:
    required = [str(key) for key in cfg.get("required_target_values", [])]
    if not required:
        return []
    target_cfg = _compose_train_config(str(cfg["experiment"]), experiment_overrides)
    return [
        key
        for key in required
        if _select_path(target_cfg, key) in (None, "", "???")
    ]


def _compose_train_config(
    experiment_name: str, overrides: Sequence[str]
) -> dict[str, Any]:
    register_dreamervla_resolvers()
    with initialize_config_dir(
        config_dir=str(TRAIN_CONFIG_DIR),
        job_name="train_launcher_target",
        version_base=None,
    ):
        cfg_obj = compose(
            config_name="train",
            overrides=[f"experiment={experiment_name}", *overrides],
        )
    return OmegaConf.to_container(cfg_obj, resolve=False)  # type: ignore[return-value]


def _experiment_overrides(cfg: Mapping[str, Any], trailing: Sequence[str]) -> list[str]:
    experiment_name = str(cfg["experiment"])
    overrides = list(trailing)
    task = cfg.get("task")
    if task not in (None, ""):
        overrides.append(f"task={task}")
    target_cfg = _compose_train_config(experiment_name, overrides)
    mapping = {
        "batch_size": ("dataloader.batch_size", "training.batch_size"),
        "num_workers": ("dataloader.num_workers", "training.num_workers"),
        "max_steps": ("training.max_steps", "training.max_train_steps"),
        "num_epochs": ("training.num_epochs",),
        "out_dir": ("training.out_dir",),
    }
    for cfg_key, candidates in mapping.items():
        for target_key in candidates:
            if _has_path(target_cfg, target_key):
                _append_if_set(overrides, cfg, target_key, cfg.get(cfg_key))
                break
    return overrides


def _write_libero_config(data_root: str) -> None:
    libero_config_path = Path(
        os.environ.get("LIBERO_CONFIG_PATH", str(Path(data_root) / ".libero"))
    )
    libero_config_path.mkdir(parents=True, exist_ok=True)
    (libero_config_path / "config.yaml").write_text(
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


def _build_env(cfg: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["DVLA_ROOT"] = str(PROJECT_ROOT)
    env["DVLA_DATA_ROOT"] = str(cfg["data_root"])
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    gpus = cfg.get("gpus")
    if gpus not in (None, ""):
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    if cfg.get("out_dir") not in (None, ""):
        env["OUT_DIR"] = str(cfg["out_dir"])
    pythonpath = env.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{PROJECT_ROOT}{':' + pythonpath if pythonpath else ''}"
    for key, value in dict(cfg.get("env", {})).items():
        rendered = _env_value(value)
        if rendered is not None:
            env[str(key)] = rendered
    return env


def _command(cfg: Mapping[str, Any], experiment_overrides: Sequence[str]) -> list[str]:
    train_module = str(cfg.get("module", "dreamervla.train"))
    experiment_name = str(cfg["experiment"])
    python = str(cfg.get("python", "python"))
    ngpu = int(cfg.get("ngpu") or _gpu_count(cfg.get("gpus")))
    base = [python, "-m"]
    if bool(cfg.get("distributed", True)) and ngpu > 1:
        base.extend(
            [
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={ngpu}",
                f"--master_port={cfg.get('master_port', 29500)}",
                "-m",
            ]
        )
    base.extend([train_module, "--config-name", "train", f"experiment={experiment_name}"])
    base.extend(experiment_overrides)
    return base


def main(argv: Sequence[str] | None = None) -> int:
    register_dreamervla_resolvers()
    config_name, launcher_overrides, trailing = _parse_hydra_like_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="train_launcher", version_base=None):
        cfg_obj = compose(config_name=config_name, overrides=launcher_overrides)
    cfg: dict[str, Any] = OmegaConf.to_container(cfg_obj, resolve=True)  # type: ignore[assignment]
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg_obj, resolve=True))

    data_root = str(cfg["data_root"])
    if bool(cfg.get("write_libero_config", True)):
        _write_libero_config(data_root)

    overrides = _experiment_overrides(cfg, trailing)
    missing = _missing_required_target_values(cfg, overrides)
    if missing:
        rendered = ", ".join(f"{key}=<value>" for key in missing)
        print(
            f"[train:{cfg.get('name')}] missing required Hydra override(s): {rendered}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    command = _command(cfg, overrides)
    env = _build_env(cfg)
    print(
        f"[train:{cfg.get('name')}] experiment={cfg['experiment']} "
        f"task={cfg.get('task') or '<experiment default>'} "
        f"ngpu={cfg.get('ngpu') or _gpu_count(cfg.get('gpus'))} "
        f"gpus={env.get('CUDA_VISIBLE_DEVICES', '<all>')}"
    )
    print(f"[train:{cfg.get('name')}] command: {shlex.join(command)}")
    if bool(cfg.get("dry_run", False)):
        return 0
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
