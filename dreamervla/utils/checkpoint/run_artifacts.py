"""Discover run roots, resumable checkpoints, and persisted Hydra configurations."""

from __future__ import annotations

import re
from pathlib import Path

from hydra.core.utils import setup_globals
from omegaconf import DictConfig, OmegaConf

from dreamervla.config_resolvers import register_dreamervla_resolvers
from dreamervla.utils.checkpoint.hf_checkpoint import is_hf_checkpoint, resolve_hf_checkpoint_dir

_STEP_DIR_RE = re.compile(r"(?:global_step_|manual_cotrain_step_)(\d+)$")
_STEP_FILE_RE = re.compile(r"(?:wm_step_|global_step_)(\d+)")


def infer_run_root(path: str | Path) -> Path:
    """Return the run root owning a run/checkpoint path.

    New runs write below ``checkpoints/``.  ``ckpt/`` remains recognized so an
    existing run can be resumed without migrating its files first.
    """

    candidate = Path(path).expanduser().resolve()
    directory = candidate.parent if candidate.is_file() else candidate
    for current in (directory, *directory.parents):
        if current.name in {"checkpoints", "ckpt", "checkpoint_hf"}:
            return current.parent.resolve()
    return directory.resolve()


def _step_value(path: Path) -> int:
    for part in reversed(path.parts):
        match = _STEP_DIR_RE.fullmatch(part) or _STEP_FILE_RE.search(part)
        if match:
            return int(match.group(1))
    return -1


def resolve_resume_checkpoint(path: str | Path) -> Path:
    """Resolve a run root or checkpoint path to the best checkpoint payload."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"resume path does not exist: {candidate}")
    run_root = infer_run_root(candidate)
    latest = run_root / "checkpoints" / "latest.ckpt"
    if latest.is_file():
        return latest.resolve()

    if is_hf_checkpoint(candidate):
        return resolve_hf_checkpoint_dir(candidate)

    manual = list(run_root.glob("checkpoints/global_step_*/manual_cotrain.ckpt"))
    if manual:
        return max(
            manual,
            key=lambda item: (_step_value(item), item.stat().st_mtime_ns),
        ).resolve()

    fixed_candidates = (
        run_root / "checkpoints" / "wm_warmup.ckpt",
        run_root / "checkpoints" / "classifier_warmup.ckpt",
        run_root / "ckpt" / "latest.ckpt",
        run_root / "ckpt" / "wm_warmup.ckpt",
        run_root / "ckpt" / "classifier_warmup.ckpt",
    )
    for fixed in fixed_candidates:
        if fixed.is_file():
            return fixed.resolve()

    patterns = (
        "checkpoints/manual_cotrain_step_*/*.ckpt",
        "checkpoints/warmup_progress/*.ckpt",
        "ckpt/manual_cotrain_step_*/*.ckpt",
        "ckpt/warmup_progress/*.ckpt",
        "latest.ckpt",
    )
    matches = [item for pattern in patterns for item in run_root.glob(pattern)]
    if matches:
        return max(matches, key=lambda item: (_step_value(item), item.stat().st_mtime_ns)).resolve()
    direct = [
        candidate / name
        for name in ("latest.ckpt", "manual_cotrain.ckpt", "model.ckpt")
        if (candidate / name).is_file()
    ]
    if direct:
        return direct[0].resolve()
    raise FileNotFoundError(
        "no resumable checkpoint found; "
        f"requested directory: {candidate}; expected latest: {latest}"
    )


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
