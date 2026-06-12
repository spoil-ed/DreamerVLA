#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
INSTALL_APT_TOOLS="${INSTALL_APT_TOOLS:-1}"
cd "${DVLA_ROOT}"

if [[ "${INSTALL_APT_TOOLS}" != "1" ]]; then
  echo "[install:00_apt_tools] skip apt tools; set INSTALL_APT_TOOLS=0/false when system packages are already installed"
  exit 0
fi

APT_BIN="${APT_BIN:-apt}"
if ! command -v "${APT_BIN}" >/dev/null 2>&1; then
  echo "${APT_BIN} is unavailable; set INSTALL_APT_TOOLS=false after installing system packages manually." >&2
  exit 2
elif [[ "${EUID}" -eq 0 ]]; then
  APT_RUNNER="${APT_BIN}"
elif command -v sudo >/dev/null 2>&1; then
  APT_RUNNER="sudo ${APT_BIN}"
else
  echo "system packages require root or sudo; set INSTALL_APT_TOOLS=false after installing them manually." >&2
  exit 2
fi

echo "[install:00_apt_tools] system packages: build tools, git/git-lfs, ffmpeg, OpenGL/OSMesa, ninja, wget"
echo "[install:00_apt_tools] running apt update/install through: ${APT_RUNNER}"
${APT_RUNNER} update
${APT_RUNNER} install -y \
  build-essential cmake curl ffmpeg git git-lfs libgl1 libopengl0 \
  libgl1-mesa-dri libgl1-mesa-glx libosmesa6 libosmesa6-dev ninja-build wget
