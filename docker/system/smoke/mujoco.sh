#!/usr/bin/env bash
set -euo pipefail

/usr/local/bin/smoke-system-common

[[ "${MUJOCO_GL:-}" == "egl" ]]
[[ "${PYOPENGL_PLATFORM:-}" == "egl" ]]
[[ "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" == "/usr/share/glvnd/egl_vendor.d/10_nvidia.json" ]]
test -r "${__EGL_VENDOR_LIBRARY_FILENAMES}"

echo "[mujoco-smoke] checking native rendering libraries"
ldconfig -p | grep 'libEGL\.so' >/dev/null
ldconfig -p | grep 'libEGL_nvidia\.so' >/dev/null
ldconfig -p | grep 'libOSMesa\.so' >/dev/null
ldconfig -p | grep 'libGLX_nvidia\.so' >/dev/null

echo "[mujoco-smoke] initializing a headless EGL device"
egl_summary="$(/usr/local/bin/egl-device-smoke)"
printf '%s\n' "${egl_summary}"
grep -Eiq 'NVIDIA' <<<"${egl_summary}"

echo "[mujoco-smoke] system environment is ready"
