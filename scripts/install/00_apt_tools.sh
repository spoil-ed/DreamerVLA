#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

if [[ "${INSTALL_APT_TOOLS:-1}" != "1" ]]; then
  install_log "skip apt tools because INSTALL_APT_TOOLS=${INSTALL_APT_TOOLS}"
  exit 0
fi

install_log "installing apt tools"
sudo apt update
sudo apt install -y \
  build-essential cmake curl ffmpeg git git-lfs libgl1 libopengl0 \
  libgl1-mesa-dri libgl1-mesa-glx libosmesa6 libosmesa6-dev ninja-build wget
