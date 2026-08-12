from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_OPENPI_ROOT = PROJECT_ROOT / "third_party" / "openpi"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("OPENPI_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.append(VENDORED_OPENPI_ROOT)
    return roots


def _is_openpi_tree(root: Path) -> bool:
    return (root / "src" / "openpi").is_dir()


def ensure_openpi_on_path() -> Path:
    """Put an official Physical Intelligence ``openpi`` checkout on ``sys.path``."""

    installed = importlib.util.find_spec("openpi")
    if installed is not None and installed.origin:
        return Path(installed.origin).resolve().parent.parent

    for candidate in _candidate_roots():
        root = candidate.resolve()
        if not _is_openpi_tree(root):
            continue
        source = str(root / "src")
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)
        return root
    raise FileNotFoundError(
        "Unable to find an official openpi checkout. Set OPENPI_ROOT or clone "
        f"https://github.com/Physical-Intelligence/openpi into {VENDORED_OPENPI_ROOT}."
    )


__all__ = ["ensure_openpi_on_path"]
