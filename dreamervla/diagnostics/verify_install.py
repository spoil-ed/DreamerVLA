from __future__ import annotations

import importlib
import os
from pathlib import Path

import h5py  # noqa: F401
import hydra  # noqa: F401
import libero
import omegaconf  # noqa: F401
import torch
import transformers  # noqa: F401


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
