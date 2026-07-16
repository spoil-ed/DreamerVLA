"""Deterministic state and artifact helpers for the public reproduction workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class ReproductionError(RuntimeError):
    """Report a reproducibility contract violation with an actionable message."""


@dataclass(frozen=True)
class SelectedCheckpoint:
    """A metric-selected checkpoint and its immutable identity."""

    path: Path
    epoch: int
    metric_name: str
    value: float
    sha256: str


@dataclass(frozen=True)
class StageDecision:
    """The safe action for one training stage."""

    action: Literal["fresh", "resume", "skip"]
    resume_source: Path | None = None
    selected_checkpoint: Path | None = None


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of one regular file."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ReproductionError(f"cannot hash missing file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON document on the destination filesystem."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def select_metric_checkpoint(
    checkpoint_dir: str | Path,
    *,
    metric_name: str,
    mode: Literal["min", "max"],
) -> SelectedCheckpoint:
    """Select a flat metric checkpoint, breaking equal metrics by latest epoch."""

    directory = Path(checkpoint_dir).expanduser().resolve()
    if mode not in {"min", "max"}:
        raise ReproductionError(f"checkpoint selection mode must be min or max, got {mode!r}")
    if not directory.is_dir():
        raise ReproductionError(f"checkpoint directory does not exist: {directory}")
    pattern = re.compile(
        rf"^epoch=(?P<epoch>\d+)-{re.escape(metric_name)}="
        r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\.ckpt$"
    )
    candidates: list[tuple[float, int, Path]] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None and path.is_file() and path.stat().st_size > 0:
            candidates.append((float(match.group("value")), int(match.group("epoch")), path))
    if not candidates:
        raise ReproductionError(f"no {metric_name} metric checkpoints under {directory}")
    if mode == "min":
        value, epoch, path = min(candidates, key=lambda item: (item[0], -item[1]))
    else:
        value, epoch, path = max(candidates, key=lambda item: (item[0], item[1]))
    return SelectedCheckpoint(
        path=path.resolve(),
        epoch=epoch,
        metric_name=metric_name,
        value=value,
        sha256=sha256_file(path),
    )


def decide_stage(
    state: Mapping[str, Any],
    *,
    stage: str,
    run_root: str | Path,
    budget: int,
) -> StageDecision:
    """Choose fresh, resume, or validated skip without overwriting a run."""

    root = Path(run_root).expanduser().resolve()
    stages = state.get("stages", {})
    record = stages.get(stage, {}) if isinstance(stages, Mapping) else {}
    if isinstance(record, Mapping) and record.get("status") == "completed":
        recorded_budget = int(record.get("budget", -1))
        if recorded_budget != budget:
            raise ReproductionError(
                f"{stage} budget mismatch: recorded={recorded_budget} requested={budget}"
            )
        raw_selected = record.get("selected_checkpoint")
        selected = Path(str(raw_selected)).expanduser().resolve() if raw_selected else None
        if selected is None or not selected.is_file():
            raise ReproductionError(f"{stage} selected checkpoint is missing: {selected}")
        expected_hash = str(record.get("sha256", ""))
        actual_hash = sha256_file(selected)
        if expected_hash != actual_hash:
            raise ReproductionError(
                f"{stage} selected checkpoint hash mismatch: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        return StageDecision(action="skip", selected_checkpoint=selected)

    latest = root / "checkpoints" / "latest.ckpt"
    if latest.is_file() and latest.stat().st_size >= 0:
        return StageDecision(action="resume", resume_source=root)
    if root.exists() and any(root.iterdir()):
        raise ReproductionError(
            f"{stage} run root is non-empty but has no resumable checkpoint: {root}"
        )
    return StageDecision(action="fresh")


__all__ = [
    "ReproductionError",
    "SelectedCheckpoint",
    "StageDecision",
    "atomic_write_json",
    "decide_stage",
    "select_metric_checkpoint",
    "sha256_file",
]
