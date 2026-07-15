"""Discover and load persisted Hydra run configuration."""

from __future__ import annotations

from pathlib import Path

from hydra.core.utils import setup_globals
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers


def _search_roots(path: Path) -> tuple[Path, ...]:
    if path.is_dir():
        start = path
    elif path.is_file() or path.suffix:
        start = path.parent
    else:
        start = path
    return (start, *start.parents)


def find_run_config(path: str | Path) -> Path:
    """Find the persisted config associated with a config, checkpoint, or run path.

    Native Hydra ``.hydra/config.yaml`` files take priority across the complete
    ancestor chain. Historical root ``resolved_config.yaml`` files are considered
    only when no native Hydra config exists.
    """

    requested = Path(path).expanduser()
    if requested.is_file() and requested.suffix.lower() in {".yaml", ".yml"}:
        return requested

    roots = _search_roots(requested)
    for root in roots:
        candidate = root / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    for root in roots:
        candidate = root / "resolved_config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"run config not found for input path: {requested}")


def load_run_config(path: str | Path) -> DictConfig:
    """Load and fully resolve the persisted config associated with ``path``."""

    setup_globals()
    register_dreamervla_resolvers()
    config_path = find_run_config(path)
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"run config must contain a mapping: {config_path}")
    OmegaConf.resolve(config)
    return config
