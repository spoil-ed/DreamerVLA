from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the runtime data root.

    `DVLA_DATA_ROOT` is the release-facing override. When unset, runtime data
    resolves from the relative `data/` path of the current process.
    """

    return Path(os.environ.get("DVLA_DATA_ROOT", "data")).expanduser()


def data_path(*parts: str | os.PathLike[str]) -> Path:
    return data_root().joinpath(*map(Path, parts))


def checkpoints_path(*parts: str | os.PathLike[str]) -> Path:
    return data_path("checkpoints", *parts)


def processed_data_path(*parts: str | os.PathLike[str]) -> Path:
    return data_path("processed_data", *parts)


__all__ = [
    "PROJECT_ROOT",
    "checkpoints_path",
    "data_path",
    "data_root",
    "processed_data_path",
]
