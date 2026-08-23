#!/usr/bin/env bash
set -euo pipefail

/usr/local/bin/smoke-system-common

[[ "${VK_ICD_FILENAMES:-}" == "/usr/share/vulkan/icd.d/nvidia_icd.json" ]]
test -r "${VK_ICD_FILENAMES}"
test -r /usr/share/glvnd/egl_vendor.d/10_nvidia.json

echo "[sapien-smoke] checking native rendering libraries"
ldconfig -p | grep 'libvulkan\.so' >/dev/null
ldconfig -p | grep 'libGLX_nvidia\.so' >/dev/null

echo "[sapien-smoke] enumerating headless Vulkan devices"
summary="$(vulkaninfo --summary 2>&1)"
grep -E 'Vulkan Instance Version|deviceName|driverName|driverInfo' <<<"${summary}"
grep -Eiq 'NVIDIA|GeForce|RTX' <<<"${summary}"

echo "[sapien-smoke] system environment is ready"
