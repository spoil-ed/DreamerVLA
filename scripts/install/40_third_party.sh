#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

clone_repo() {
  local url="$1"
  local dest="$2"
  local ref="$3"
  install_log "ensuring $(basename "${dest}") ref=${ref} -> ${dest}"
  if [[ ! -d "${dest}/.git" ]]; then
    git clone "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --all --tags
  git -C "${dest}" checkout "${ref}"
}

install_log "third_party_dir=${DVLA_ROOT}/third_party"
install_log "optional_third_party=${INSTALL_OPTIONAL_THIRD_PARTY}"
install_log "opensora_third_party=${INSTALL_OPENSORA_THIRD_PARTY}"
install_log "openvla_oft_third_party=${INSTALL_OPENVLA_OFT_THIRD_PARTY}"
install_log "cloning pinned LIBERO and robosuite-family repositories"
mkdir -p "${DVLA_ROOT}/third_party"

# Step 1: LIBERO benchmark package used by the offline and online LIBERO routes.
clone_repo https://github.com/Lifelong-Robot-Learning/LIBERO.git "${DVLA_ROOT}/third_party/LIBERO" 8f1084e

# Step 2: WMPO-compatible robosuite stack, using the same refs as the related WMPO installer.
clone_repo https://github.com/ARISE-Initiative/robosuite.git "${DVLA_ROOT}/third_party/robosuite" b9d8d3de5e3dfd1724f4a0e6555246c460407daa
clone_repo https://github.com/ARISE-Initiative/robosuite-task-zoo "${DVLA_ROOT}/third_party/robosuite-task-zoo" 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
clone_repo https://github.com/ARISE-Initiative/robomimic.git "${DVLA_ROOT}/third_party/robomimic" d0b37cf214bd24fb590d182edb6384333f67b661
clone_repo https://github.com/NVlabs/mimicgen.git "${DVLA_ROOT}/third_party/mimicgen" 72bd767

install_log "installing third_party editable packages"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite-task-zoo"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robomimic"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/mimicgen"

# Step 3: LIBERO needs lightweight packaging metadata before editable install.
install_log "applying LIBERO packaging compatibility"
cat > "${DVLA_ROOT}/third_party/LIBERO/pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "libero"
version = "0.1.0"
description = "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"
requires-python = ">=3.8"

[tool.setuptools.packages.find]
include = ["libero*"]
namespaces = true

[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
"*" = ["*"]
EOF
cat > "${DVLA_ROOT}/third_party/LIBERO/setup.py" <<'EOF'
from setuptools import setup

setup()
EOF
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/LIBERO"

# Step 4: OpenSora is vendored for OpenVLA-OFT/WMPO-compatible components.
if [[ "${INSTALL_OPENSORA_THIRD_PARTY}" == "1" ]]; then
  if [[ -d "${DVLA_ROOT}/third_party/opensora" ]]; then
    install_log "installing third_party/opensora"
    uv pip install --python "${PYTHON}" -v -e "${DVLA_ROOT}/third_party/opensora"
  else
    install_log "skip third_party/opensora because the directory is absent"
  fi
else
  install_log "skip third_party/opensora because INSTALL_OPENSORA_THIRD_PARTY=${INSTALL_OPENSORA_THIRD_PARTY}"
fi

# Step 5: OpenVLA-OFT follows the related WMPO install style.
if [[ "${INSTALL_OPENVLA_OFT_THIRD_PARTY}" == "1" ]]; then
  if [[ -d "${DVLA_ROOT}/third_party/openvla-oft" ]]; then
    install_log "installing third_party/openvla-oft and WMPO OpenVLA-OFT helper packages"
    if [[ -f "${DVLA_ROOT}/third_party/openvla-oft/install_mujoco.sh" ]]; then
      (cd "${DVLA_ROOT}/third_party/openvla-oft" && bash ./install_mujoco.sh)
    fi
    uv pip install --python "${PYTHON}" --no-deps -e "${DVLA_ROOT}/third_party/openvla-oft"
    uv pip install --python "${PYTHON}" --no-deps git+https://github.com/moojink/dlimp_openvla
    uv pip install --python "${PYTHON}" --no-deps "git+https://github.com/moojink/transformers-openvla-oft.git"
  else
    install_log "skip third_party/openvla-oft because the directory is absent"
  fi
else
  install_log "skip third_party/openvla-oft because INSTALL_OPENVLA_OFT_THIRD_PARTY=${INSTALL_OPENVLA_OFT_THIRD_PARTY}"
fi
