#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

install_log "checking imports in conda env=${CONDA_ENV_NAME} python=${PYTHON}"
install_log "verifying imports and CUDA visibility"
"${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch
import h5py
import hydra
import omegaconf
import transformers
import libero

dvla_root = Path(os.environ["DVLA_ROOT"]).resolve()
expected_third_party_imports = {
    "libero": dvla_root / "third_party/LIBERO",
    "robosuite": dvla_root / "third_party/robosuite",
    "robomimic": dvla_root / "third_party/robomimic",
    "mimicgen": dvla_root / "third_party/mimicgen",
}


def import_paths(package_name: str) -> list[Path]:
    module = importlib.import_module(package_name)
    paths: list[Path] = []
    if getattr(module, "__file__", None):
        paths.append(Path(module.__file__).resolve())
    for item in getattr(module, "__path__", []):
        if isinstance(item, str) and not item.startswith("__editable__"):
            paths.append(Path(item).resolve())
    return paths


print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("deps ok")
print("libero", libero.__path__)

for package_name, expected_root in expected_third_party_imports.items():
    paths = import_paths(package_name)
    if not any(str(path).startswith(str(expected_root)) for path in paths):
        raise SystemExit(
            f"{package_name} imported from {paths}; expected a path under {expected_root}"
        )
    print(package_name, paths)
PY
