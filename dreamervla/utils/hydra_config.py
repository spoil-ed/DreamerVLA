"""Helpers for script entrypoints backed by Hydra YAML configs."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_CONFIG_DIR = PROJECT_ROOT / "configs" / "scripts"


def split_config_name(
    argv: Sequence[str] | None,
    *,
    default_config_name: str,
) -> tuple[str, list[str]]:
    config_name = default_config_name
    overrides: list[str] = []
    items = list(sys.argv[1:] if argv is None else argv)
    i = 0
    while i < len(items):
        item = items[i]
        if item == "--config-name":
            if i + 1 >= len(items):
                raise SystemExit("--config-name requires a value")
            config_name = items[i + 1]
            i += 2
            continue
        if item.startswith("--config-name="):
            config_name = item.split("=", 1)[1]
            i += 1
            continue
        if item.startswith("--"):
            raise SystemExit(
                f"Unsupported script flag {item!r}. Use Hydra override syntax key=value."
            )
        overrides.append(item)
        i += 1
    return config_name, overrides


def _plain(value: Any) -> Any:
    return (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )


def _namespace(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, dict):
        return SimpleNamespace(**{str(k): _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def script_config(
    default_config_name: str,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    register_dreamervla_resolvers()
    config_name, overrides = split_config_name(
        [] if argv is None else argv,
        default_config_name=default_config_name,
    )
    with initialize_config_dir(
        config_dir=str(SCRIPT_CONFIG_DIR),
        job_name=config_name,
        version_base=None,
    ):
        cfg_obj = compose(config_name=config_name, overrides=overrides)
    cfg = OmegaConf.to_container(cfg_obj, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Script config {config_name!r} must resolve to a mapping")
    return cfg


def script_namespace(
    default_config_name: str,
    argv: Sequence[str] | None = None,
) -> SimpleNamespace:
    return _namespace(script_config(default_config_name, sys.argv[1:] if argv is None else argv))
