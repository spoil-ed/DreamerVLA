from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dvla_root() -> Path:
    """Return the repository/source root.

    ``DVLA_ROOT`` is independent from ``DVLA_DATA_ROOT``. If the environment
    variable is not set, fall back to the imported package location so library
    code can still find repo-local source/config files.
    """

    env = os.environ.get("DVLA_ROOT")
    if env:
        return Path(env).expanduser()
    return PROJECT_ROOT


def data_root() -> Path:
    """Return the runtime data root.

    ``DVLA_DATA_ROOT`` has priority. When it is unset, runtime data is resolved
    under ``DVLA_ROOT / "data"``. ``DVLA_ROOT`` itself falls back to the imported
    package location for direct Python entrypoints.
    """

    env = os.environ.get("DVLA_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return dvla_root() / "data"


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
    "dvla_root",
    "processed_data_path",
]
