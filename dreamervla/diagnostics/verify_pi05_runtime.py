"""Verify the isolated RLinf LIBERO + π0.5 runtime before train or eval."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from collections.abc import Callable

from dreamervla.utils.openpi_imports import ensure_openpi_on_path

CRITICAL_PI05_DISTRIBUTION_VERSIONS = {
    "bddl": "3.6.0",
    "future": "1.0.0",
    "lerobot": "0.3.3",
    "ray": "2.57.0",
    "rlinf-libero": "0.1.1",
    "rlinf-openpi": "0.1.1",
    "tokenizers": "0.21.4",
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "4.55.4",
}
_PYTORCH_DISTRIBUTIONS = {"torch", "torchaudio", "torchvision"}


def verify_pi05_distribution_versions(
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> None:
    """Reject missing or drifted packages in the locked π0.5 environment."""

    failures: list[str] = []
    for distribution, expected in CRITICAL_PI05_DISTRIBUTION_VERSIONS.items():
        try:
            installed = version_getter(distribution)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            failures.append(f"{distribution} is missing; expected {expected}")
            continue
        comparable = (
            installed.split("+", maxsplit=1)[0]
            if distribution in _PYTORCH_DISTRIBUTIONS
            else installed
        )
        if comparable != expected:
            failures.append(f"{distribution}=={installed}; expected {expected}")
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(
            "[verify_pi05_runtime] locked runtime mismatch:\n"
            f"{details}\n"
            "Re-run uv sync --all-extras --locked for the π0.5 project."
        )


def main() -> int:
    """Verify versions and the imports required by π0.5 LIBERO evaluation."""

    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"[verify_pi05_runtime] Python {sys.version.split()[0]} is active; expected Python 3.11"
        )
    verify_pi05_distribution_versions()
    ensure_openpi_on_path()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("[verify_pi05_runtime] OpenPI JAX backend is not isolated to CPU")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
        raise SystemExit("[verify_pi05_runtime] JAX preallocation must be disabled")
    for module_name in ("future", "lerobot", "openpi", "torch"):
        importlib.import_module(module_name)
    importlib.import_module("libero.libero.envs")
    print("π0.5 LIBERO runtime ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
