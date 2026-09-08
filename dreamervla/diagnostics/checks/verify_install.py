from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from collections.abc import Callable
from pathlib import Path

CRITICAL_DISTRIBUTION_VERSIONS = {
    "diffusers": "0.33.0",
    "draccus": "0.8.0",
    "jsonlines": "4.0.0",
    "mujoco": "3.8.0",
    "numpy": "1.26.4",
    "peft": "0.11.0",
    "protobuf": "4.25.9",
    "ray": "2.55.1",
    "sentencepiece": "0.1.99",
    "tensorflow": "2.15.0",
    "tensorflow-datasets": "4.9.3",
    "tensorflow-graphics": "2021.12.3",
    "tensorflow-metadata": "1.17.3",
    "timm": "0.9.10",
    "tokenizers": "0.19.1",
    "torch": "2.5.1",
    "torchaudio": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.40.1",
    "wandb": "0.26.1",
}
PYTORCH_DISTRIBUTIONS = {"torch", "torchaudio", "torchvision"}


def verify_distribution_versions(
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> None:
    """Fail when a critical runtime distribution is missing or has drifted."""
    failures: list[str] = []
    for distribution, expected in CRITICAL_DISTRIBUTION_VERSIONS.items():
        try:
            installed = version_getter(distribution)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            failures.append(f"{distribution} is missing; expected {expected}")
            continue
        comparable_installed = (
            installed.split("+", maxsplit=1)[0]
            if distribution in PYTORCH_DISTRIBUTIONS
            else installed
        )
        if comparable_installed != expected:
            failures.append(f"{distribution}=={installed}; expected {expected}")

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(
            "[verify_install] critical distribution version mismatch:\n"
            f"{details}\n"
            "Re-run scripts/install_env.sh with force=true for the affected install stages."
        )
    print("critical distribution versions ok")


def _import_paths(package_name: str) -> list[Path]:
    module = importlib.import_module(package_name)
    paths: list[Path] = []
    if getattr(module, "__file__", None):
        paths.append(Path(module.__file__).resolve())
    paths.extend(
        Path(item).resolve()
        for item in getattr(module, "__path__", [])
        if isinstance(item, str) and not item.startswith("__editable__")
    )
    return paths


def main() -> int:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"[verify_install] Python {sys.version.split()[0]} is active; expected Python 3.11"
        )
    verify_distribution_versions()

    for package_name in (
        "h5py",
        "hydra",
        "jsonlines",
        "mujoco",
        "omegaconf",
        "tensorflow_datasets",
        "transformers",
    ):
        importlib.import_module(package_name)
    libero = importlib.import_module("libero")
    torch = importlib.import_module("torch")

    dvla_root = Path(os.environ["DVLA_ROOT"]).resolve()
    expected_third_party_imports = {
        "libero": dvla_root / "third_party/LIBERO",
        "robosuite": dvla_root / "third_party/robosuite",
        "robomimic": dvla_root / "third_party/robomimic",
        "mimicgen": dvla_root / "third_party/mimicgen",
    }

    print(
        "torch",
        torch.__version__,
        "cuda",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    print("deps ok")
    print("libero", libero.__path__)

    for package_name, expected_root in expected_third_party_imports.items():
        paths = _import_paths(package_name)
        if not any(str(path).startswith(str(expected_root)) for path in paths):
            raise SystemExit(
                f"{package_name} imported from {paths}; expected a path under {expected_root}"
            )
        print(package_name, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
