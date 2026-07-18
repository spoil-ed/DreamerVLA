from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.launchers.task_cli import normalize_task_flag

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "scripts"

_TASK_OVERRIDE_ROUTES: dict[str, tuple[str, bool]] = {
    "download/config": ("env.LIBERO_SUITES", True),
    "preprocess/preprocess_all": ("tasks", True),
    "preprocess/preprocess_libero": ("tasks", True),
    "preprocess/preprocess_suite": ("task", False),
    "preprocess/validate_libero_data": ("tasks", True),
}


def _parse_hydra_like_args(argv: Sequence[str]) -> tuple[str, list[str]]:
    config_name = "preprocess/preprocess_suite"
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
        if item.startswith("gpus="):
            raw_key, raw_value = item.split("=", 1)
            if "," in raw_value and not raw_value.startswith(("'", '"', "[")):
                item = f"{raw_key}='{raw_value}'"
        overrides.append(item)
        i += 1
    return config_name, overrides


def _plain(value: Any) -> Any:
    return (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )


def _as_list(value: Any) -> list[Any]:
    value = _plain(value)
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part for part in value.replace(",", " ").split() if part]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return list(value)
    return [value]


def _env_value(value: Any) -> str | None:
    value = _plain(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _step_selected(step: Mapping[str, Any], only: Sequence[str]) -> bool:
    if not only:
        return True
    script = str(step.get("script", ""))
    names = {
        str(step.get("id", "")),
        Path(script).name,
        Path(script).stem,
    }
    return any(
        selected in names or any(name.startswith(selected) for name in names) for selected in only
    )


def _replace_item(value: Any, item: Any) -> Any:
    if isinstance(value, str):
        return value.replace("{item}", str(item))
    if isinstance(value, list):
        return [_replace_item(child, item) for child in value]
    if isinstance(value, dict):
        return {key: _replace_item(child, item) for key, child in value.items()}
    return value


def _run_one_step(
    *,
    cfg: Mapping[str, Any],
    step: Mapping[str, Any],
    item: Any | None = None,
) -> None:
    step = _replace_item(dict(step), item) if item is not None else dict(step)
    script = PROJECT_ROOT / str(step["script"])
    command = ["bash", str(script), *[str(arg) for arg in _as_list(step.get("args"))]]

    env = os.environ.copy()
    env["DVLA_ROOT"] = str(PROJECT_ROOT)
    env["DVLA_DATA_ROOT"] = str(cfg["data_root"])
    env["PYTHON"] = str(cfg.get("python", "python"))
    pythonpath = env.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{PROJECT_ROOT}{':' + pythonpath if pythonpath else ''}"
    for key, value in dict(cfg.get("env", {})).items():
        rendered = _env_value(value)
        if rendered is not None:
            env[str(key)] = rendered
    for key, value in dict(step.get("env", {})).items():
        rendered = _env_value(value)
        if rendered is not None:
            env[str(key)] = rendered

    step_id = str(step.get("id", script.stem))
    state_dir = Path(str(cfg.get("state_dir", "")))
    marker = None
    if bool(step.get("marker", False)) and state_dir:
        marker = (
            PROJECT_ROOT / state_dir if not state_dir.is_absolute() else state_dir
        ) / f"{step_id}.done"
        if marker.exists() and not bool(cfg.get("force", False)):
            print(f"[workflow:{cfg.get('name')}] skip {step_id} marker={marker}")
            return

    print(f"[workflow:{cfg.get('name')}] run {step_id}: {shlex.join(command)}")
    if bool(cfg.get("dry_run", False)):
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()


def main(argv: Sequence[str] | None = None) -> int:
    register_dreamervla_resolvers()
    config_name, overrides = _parse_hydra_like_args(list(sys.argv[1:] if argv is None else argv))
    selected_task: str | None = None
    task_route = _TASK_OVERRIDE_ROUTES.get(config_name)
    if task_route is not None:
        overrides, task_override = normalize_task_flag(
            overrides,
            hydra_key=task_route[0],
            as_list=task_route[1],
        )
        if task_override is not None:
            overrides.append(task_override)
            selected_task = task_override.split("=", 1)[1].strip("[]")
    elif any(item == "--task" or item.startswith("--task=") for item in overrides):
        raise SystemExit(f"--task is not supported by workflow config {config_name!r}")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="workflow", version_base=None):
        cfg_obj = compose(config_name=config_name, overrides=overrides)
    cfg: dict[str, Any] = OmegaConf.to_container(cfg_obj, resolve=True)  # type: ignore[assignment]
    only = [str(item) for item in _as_list(cfg.get("only"))]

    print(
        f"[workflow:{cfg.get('name')}] root={PROJECT_ROOT} "
        f"data_root={cfg.get('data_root')} config={config_name}"
        f"{f' task={selected_task}' if selected_task else ''}"
    )
    for raw_step in list(cfg.get("steps", [])):
        step = dict(raw_step)
        if not bool(step.get("enabled", True)):
            continue
        if not _step_selected(step, only):
            print(f"[workflow:{cfg.get('name')}] skip {step.get('id')} only={only}")
            continue
        items = _as_list(step.get("for_each"))
        if items:
            for item in items:
                _run_one_step(cfg=cfg, step=step, item=item)
        else:
            _run_one_step(cfg=cfg, step=step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
