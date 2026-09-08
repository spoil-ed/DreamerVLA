#!/usr/bin/env bash
set -euo pipefail

/usr/local/bin/smoke-mujoco-system

echo "[vla-smoke] checking the complete uv-installed environment"
uv --version
uv pip check
python -m dreamervla.diagnostics.checks.verify_install

python - <<'PY'
from __future__ import annotations

import importlib

for package in (
    "flash_attn",
    "libero",
    "mujoco",
    "ray",
    "robomimic",
    "robosuite",
    "tensorflow",
    "torch",
    "transformers",
):
    module = importlib.import_module(package)
    print(f"[vla-smoke] {package}: {getattr(module, '__version__', 'imported')}")
PY

echo "[vla-smoke] complete environment is ready"
