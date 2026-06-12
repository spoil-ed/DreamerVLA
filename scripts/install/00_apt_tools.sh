#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

if [[ "${INSTALL_APT_TOOLS:-1}" != "1" ]]; then
  install_log "skip apt tools because INSTALL_APT_TOOLS=${INSTALL_APT_TOOLS}"
  exit 0
fi

install_log "system packages: build tools, git/git-lfs, ffmpeg, OpenGL/OSMesa, ninja, wget"
APT_BIN="${APT_BIN:-apt}"
APT_RUNNER=()
if ! command -v "${APT_BIN}" >/dev/null 2>&1; then
  install_log "${APT_BIN} is unavailable; set INSTALL_APT_TOOLS=0 after installing system packages manually."
  exit 2
elif [[ "${EUID}" -eq 0 ]]; then
  APT_RUNNER=("${APT_BIN}")
elif command -v sudo >/dev/null 2>&1; then
  APT_RUNNER=(sudo "${APT_BIN}")
else
  install_log "system packages require root or sudo; set INSTALL_APT_TOOLS=0 after installing them manually."
  exit 2
fi

install_log "installing apt tools"
install_log "running apt update/install through: ${APT_RUNNER[*]}"
"${APT_RUNNER[@]}" update
"${APT_RUNNER[@]}" install -y \
  build-essential cmake curl ffmpeg git git-lfs libgl1 libopengl0 \
  libgl1-mesa-dri libgl1-mesa-glx libosmesa6 libosmesa6-dev ninja-build wget
