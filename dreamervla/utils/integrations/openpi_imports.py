from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENDORED_OPENPI_ROOT = PROJECT_ROOT / "third_party" / "openpi"


def configure_openpi_jax_runtime() -> None:
    """Keep OpenPI's configuration-only JAX imports on the CPU.

    DreamerVLA's π0.5 model and optimizer are PyTorch/CUDA.  OpenPI still
    imports JAX for configuration, transforms, and pytree helpers, none of
    which need a JAX CUDA client on this route.  This guard must run before
    even discovering/importing OpenPI so training and standalone inference
    share the same isolation contract.
    """

    requested = os.environ.get("JAX_PLATFORMS")
    if requested is not None and requested.strip().lower() != "cpu":
        raise RuntimeError(
            f"DreamerVLA's OpenPI PyTorch route requires JAX_PLATFORMS=cpu; got {requested!r}"
        )
    if "jax" in sys.modules and requested is None:
        raise RuntimeError("JAX was imported before DreamerVLA could select its CPU backend")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


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

    configure_openpi_jax_runtime()
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


__all__ = ["configure_openpi_jax_runtime", "ensure_openpi_on_path"]
