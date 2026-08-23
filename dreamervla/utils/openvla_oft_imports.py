from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT_OPENVLA_OFT_ROOT = PROJECT_ROOT / "dreamervla" / "models" / "embodiment" / "openvla_oft"
VENDORED_OPENVLA_OFT_ROOT = PROJECT_ROOT / "third_party" / "openvla-oft"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("OPENVLA_OFT_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(
        [
            EMBODIMENT_OPENVLA_OFT_ROOT,
            VENDORED_OPENVLA_OFT_ROOT,
        ]
    )
    return roots


def _is_openvla_oft_tree(root: Path) -> bool:
    return (root / "openvla_oft").is_dir() or (root / "prismatic").is_dir()


def default_openvla_oft_root() -> Path:
    for candidate in _candidate_roots():
        candidate = candidate.resolve()
        if _is_openvla_oft_tree(candidate):
            return candidate
    return EMBODIMENT_OPENVLA_OFT_ROOT


def _module_file(module_name: str) -> Path | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    module_file = getattr(module, "__file__", None)
    return None if module_file is None else Path(module_file).resolve()


def _evict_external_openvla_modules(root: Path) -> None:
    """Drop stale OpenVLA modules if they were imported from a different tree."""
    prefixes = ("openvla_oft", "openvla", "prismatic")
    to_delete: list[str] = []
    for module_name in list(sys.modules):
        if not (
            module_name in prefixes
            or module_name.startswith(tuple(f"{prefix}." for prefix in prefixes))
        ):
            continue
        module_file = _module_file(module_name)
        if module_file is not None and not module_file.is_relative_to(root):
            to_delete.append(module_name)
    for module_name in to_delete:
        del sys.modules[module_name]


def ensure_openvla_oft_on_path() -> Path:
    """Put the available OpenVLA-OFT tree first on sys.path."""
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    for candidate in _candidate_roots():
        candidate = candidate.resolve()
        if _is_openvla_oft_tree(candidate):
            candidate_str = str(candidate)
            if candidate_str in sys.path:
                sys.path.remove(candidate_str)
            sys.path.insert(0, candidate_str)
            _evict_external_openvla_modules(candidate)
            return candidate
    raise FileNotFoundError(
        "Unable to find an OpenVLA-OFT tree. Checked: "
        + ", ".join(str(path) for path in _candidate_roots())
    )


__all__ = ["default_openvla_oft_root", "ensure_openvla_oft_on_path"]
