"""Canonical run-root and resume-checkpoint path discovery."""

from __future__ import annotations

import re
from pathlib import Path

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
        if current.name in {"checkpoints", "ckpt"}:
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
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return candidate

    if candidate.is_dir() and candidate.name not in {"checkpoints", "ckpt"}:
        direct = sorted(candidate.glob("*.ckpt"), key=lambda item: (_step_value(item), item.name))
        if direct:
            preferred = [
                item
                for item in direct
                if item.name in {"latest.ckpt", "manual_cotrain.ckpt", "model.ckpt"}
            ]
            return (preferred[-1] if preferred else direct[-1]).resolve()

    run_root = infer_run_root(candidate)
    fixed_candidates = (
        run_root / "checkpoints" / "latest.ckpt",
        run_root / "checkpoints" / "wm_warmup.ckpt",
        run_root / "ckpt" / "latest.ckpt",
        run_root / "ckpt" / "wm_warmup.ckpt",
        run_root / "latest.ckpt",
    )
    for fixed in fixed_candidates:
        if fixed.is_file():
            return fixed.resolve()

    patterns = (
        "checkpoints/global_step_*/*.ckpt",
        "checkpoints/manual_cotrain_step_*/*.ckpt",
        "checkpoints/warmup_progress/*.ckpt",
        "ckpt/manual_cotrain_step_*/*.ckpt",
        "ckpt/warmup_progress/*.ckpt",
    )
    matches = [item for pattern in patterns for item in run_root.glob(pattern)]
    if matches:
        return max(matches, key=lambda item: (_step_value(item), item.stat().st_mtime_ns)).resolve()
    raise FileNotFoundError(f"no resumable checkpoint found under run root: {run_root}")
