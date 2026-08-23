#!/usr/bin/env bash
set -euo pipefail

for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  if [[ -n "${!name:-}" ]]; then
    echo "[system-smoke] proxy variable ${name} must be empty" >&2
    exit 1
  fi
done

echo "[system-smoke] checking NVIDIA runtime"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[system-smoke] checking development tools"
python --version
uv --version
nvcc --version | tail -n 1
ffmpeg -version | sed -n '1p'

python - <<'PY'
from __future__ import annotations

import importlib.util

for package in ("isaacsim", "isaaclab"):
    if importlib.util.find_spec(package) is not None:
        raise SystemExit(f"unexpected package installed: {package}")

print("[system-smoke] Isaac Sim/Lab are absent")
PY
