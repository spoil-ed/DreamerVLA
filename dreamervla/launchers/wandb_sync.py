"""Upload all offline W&B segments for one DreamerVLA logical run."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_SYNCED_SUFFIX = ".synced"


@dataclass(frozen=True)
class _Segment:
    stream: Path
    synced: bool


def _validate_run_id(run_id: str, *, source: Path) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid W&B run ID {run_id!r} in {source}")
    return run_id


def _stream_run_id(stream: Path) -> str:
    prefix = "run-"
    suffix = ".wandb"
    name = stream.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"invalid W&B stream filename: {stream}")
    run_id = name[len(prefix) : -len(suffix)]
    return _validate_run_id(run_id, source=stream)


def _scoped_path(path: Path, *, wandb_dir: Path) -> Path:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(wandb_dir)
    except ValueError as error:
        raise ValueError(f"W&B path escapes the requested directory: {path}") from error

    current = wandb_dir
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"W&B path must not contain a symlink: {path}")

    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"W&B path cannot be resolved safely: {path}") from error
    if not resolved.is_relative_to(wandb_dir):
        raise ValueError(f"W&B path escapes the requested directory: {path}")
    return resolved


def _run_directory_sort_key(run_directory: Path) -> str:
    name = run_directory.name
    for prefix in ("offline-run-", "run-"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _discover_segments(wandb_dir: Path) -> list[_Segment]:
    streams: dict[Path, bool] = {}
    for root in (wandb_dir, wandb_dir / "wandb"):
        if root.is_symlink():
            raise ValueError(f"W&B path must not contain a symlink: {root}")
        if not root.is_dir():
            continue
        for run_prefix in ("offline-run", "run"):
            for run_directory in root.glob(f"{run_prefix}-*"):
                if run_directory.is_symlink():
                    raise ValueError(f"W&B run directory must not be a symlink: {run_directory}")
                if not run_directory.is_dir():
                    continue
                run_directory = _scoped_path(run_directory, wandb_dir=wandb_dir)
                for stream in run_directory.glob("run-*.wandb"):
                    if stream.is_symlink():
                        raise ValueError(f"W&B stream must not be a symlink: {stream}")
                    if stream.is_file():
                        stream = _scoped_path(stream, wandb_dir=wandb_dir)
                        streams.setdefault(stream, False)
                for marker in run_directory.glob("run-*.wandb.synced"):
                    if marker.is_symlink():
                        raise ValueError(f"W&B sync marker must not be a symlink: {marker}")
                    if marker.is_file():
                        marker = _scoped_path(marker, wandb_dir=wandb_dir)
                        stream = Path(str(marker)[: -len(_SYNCED_SUFFIX)])
                        streams[stream] = True
    return sorted(
        (_Segment(stream=stream, synced=synced) for stream, synced in streams.items()),
        key=lambda segment: (
            _run_directory_sort_key(segment.stream.parent),
            segment.stream.name,
            str(segment.stream),
        ),
    )


def _resolve_run_id(wandb_dir: Path, segments: Sequence[_Segment]) -> str:
    identity_path = wandb_dir / "run_id.txt"
    if identity_path.is_file():
        run_id = _validate_run_id(
            identity_path.read_text(encoding="utf-8").strip(),
            source=identity_path,
        )
    else:
        run_id = _stream_run_id(segments[0].stream)

    for segment in segments:
        segment_run_id = _stream_run_id(segment.stream)
        if segment_run_id != run_id:
            raise ValueError(
                "conflicting W&B run IDs: "
                f"expected {run_id!r}, found {segment_run_id!r} in {segment.stream}"
            )
    return run_id


def _create_synced_marker(stream: Path, *, wandb_dir: Path) -> None:
    marker = stream.with_name(f"{stream.name}{_SYNCED_SUFFIX}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker, flags, 0o644)
    except FileExistsError:
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"unsafe W&B sync marker already exists: {marker}") from None
        _scoped_path(marker, wandb_dir=wandb_dir)
        return
    os.close(descriptor)


def sync_wandb_directory(wandb_directory: str | Path) -> None:
    """Synchronize supported offline segments below ``wandb_directory``."""

    wandb_dir = Path(wandb_directory).expanduser().resolve()
    if not wandb_dir.exists():
        raise FileNotFoundError(f"W&B directory does not exist: {wandb_dir}")
    if not wandb_dir.is_dir():
        raise NotADirectoryError(f"W&B directory is not a directory: {wandb_dir}")

    wandb_cli = shutil.which("wandb")
    if wandb_cli is None:
        raise RuntimeError("wandb CLI was not found on PATH; install W&B and run `wandb login`")

    segments = _discover_segments(wandb_dir)
    if not segments:
        raise RuntimeError(f"no supported offline W&B segment found in {wandb_dir}")
    run_id = _resolve_run_id(wandb_dir, segments)

    logical_run_already_synced = any(segment.synced for segment in segments)
    for segment in segments:
        if segment.synced:
            continue
        command = [wandb_cli, "sync", "--id", run_id]
        if logical_run_already_synced:
            command.append("--append")
        command.append(str(segment.stream))
        subprocess.run(command, check=True)
        _create_synced_marker(segment.stream, wandb_dir=wandb_dir)
        logical_run_already_synced = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload one DreamerVLA offline W&B directory to its logical online run."
    )
    parser.add_argument("wandb_directory", help="path to <run_root>/wandb")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for one-argument offline synchronization."""

    args = _parser().parse_args(argv)
    sync_wandb_directory(args.wandb_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
