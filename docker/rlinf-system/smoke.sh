#!/usr/bin/env bash
set -euo pipefail

for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  if [[ -n "${!name:-}" ]]; then
    echo "[rlinf-system-smoke] proxy variable ${name} must be empty" >&2
    exit 1
  fi
done

echo "[rlinf-system-smoke] checking NVIDIA runtime"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[rlinf-system-smoke] checking native development tools"
for command_name in cc cmake ffmpeg git git-lfs nvcc patchelf python3; do
  command -v "${command_name}" >/dev/null
done
nvcc --version | tail -n 1
ffmpeg -version | sed -n '1p'

echo "[rlinf-system-smoke] checking headless rendering libraries"
ldconfig -p | grep 'libEGL\.so' >/dev/null
ldconfig -p | grep 'libEGL_nvidia\.so' >/dev/null
ldconfig -p | grep 'libOSMesa\.so' >/dev/null
ldconfig -p | grep 'libGLX_nvidia\.so' >/dev/null
/usr/local/bin/egl-device-smoke

echo "[rlinf-system-smoke] checking that the uv/application layer is absent"
if command -v uv >/dev/null 2>&1; then
  echo "uv must be mounted from outside the system image" >&2
  exit 1
fi

python3 - <<'PY'
from __future__ import annotations

import importlib.util

for package in ("libero", "openpi", "ray", "rlinf", "torch"):
    if importlib.util.find_spec(package) is not None:
        raise SystemExit(f"unexpected Python package in system image: {package}")

print("[rlinf-system-smoke] external uv/application boundary is clean")
PY

echo "[rlinf-system-smoke] system image is ready"
