#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"
activate_conda_env

clone_repo() {
  local url="$1"
  local dest="$2"
  local ref="$3"
  if [[ ! -d "${dest}/.git" ]]; then
    git clone "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --all --tags
  git -C "${dest}" checkout "${ref}"
}

install_log "cloning third_party repositories"
mkdir -p "${DVLA_ROOT}/third_party"
clone_repo https://github.com/Lifelong-Robot-Learning/LIBERO.git "${DVLA_ROOT}/third_party/LIBERO" 8f1084e
clone_repo https://github.com/ARISE-Initiative/robosuite.git "${DVLA_ROOT}/third_party/robosuite" b9d8d3de5e3dfd1724f4a0e6555246c460407daa
clone_repo https://github.com/ARISE-Initiative/robosuite-task-zoo "${DVLA_ROOT}/third_party/robosuite-task-zoo" 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
clone_repo https://github.com/ARISE-Initiative/robomimic.git "${DVLA_ROOT}/third_party/robomimic" d0b37cf214bd24fb590d182edb6384333f67b661
clone_repo https://github.com/NVlabs/mimicgen.git "${DVLA_ROOT}/third_party/mimicgen" 72bd767
clone_repo https://github.com/StanfordVL/egl_probe.git "${DVLA_ROOT}/third_party/egl_probe" 3ddf90d

if [[ "${INSTALL_OPTIONAL_THIRD_PARTY}" == "1" ]]; then
  clone_repo https://github.com/NVIDIA/apex.git "${DVLA_ROOT}/third_party/apex" 5daec2a
  clone_repo https://github.com/hpcaitech/TensorNVMe.git "${DVLA_ROOT}/third_party/TensorNVMe" 6403388
fi

install_log "installing third_party editable packages"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite-task-zoo"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robomimic"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/mimicgen"

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

install_log "installing egl_probe"
if [[ -f "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" ]]; then
  sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' \
    "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" || true
fi
uv pip install --python "${PYTHON}" --no-build-isolation "${DVLA_ROOT}/third_party/egl_probe"

if [[ "${INSTALL_OPTIONAL_THIRD_PARTY}" == "1" ]]; then
  uv pip install --python "${PYTHON}" -e "${DVLA_ROOT}/third_party/TensorNVMe" || true
  uv pip install --python "${PYTHON}" -v --no-build-isolation "${DVLA_ROOT}/third_party/apex" || true
fi
