#!/usr/bin/env bash
# Formal single-machine DreamerVLA environment installer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.1.post1}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_OPTIONAL_THIRD_PARTY="${INSTALL_OPTIONAL_THIRD_PARTY:-0}"

echo "[install_env] root=${DVLA_ROOT}"
echo "[install_env] conda_env=${CONDA_ENV_NAME} python=${PYTHON_VERSION}"

echo "[install_env] apt tools"
sudo apt update
sudo apt install -y \
  build-essential cmake curl ffmpeg git git-lfs libgl1 libopengl0 \
  libgl1-mesa-dri libgl1-mesa-glx libosmesa6 libosmesa6-dev ninja-build wget

echo "[install_env] conda environment"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
export PYTHON="$(command -v python)"

echo "[install_env] uv"
python -m pip install --upgrade pip setuptools wheel uv
uv pip install --python "${PYTHON}" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url "${CUDA_INDEX_URL}"
uv pip install --python "${PYTHON}" -e "${DVLA_ROOT}"
uv pip install --python "${PYTHON}" -r "${DVLA_ROOT}/requirements.txt"
uv pip install --python "${PYTHON}" transformers==4.40.1

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "[install_env] flash-attn wheel"
  mkdir -p "${DVLA_ROOT}/data/wheels"
  FLASH_ATTN_WHEEL="${DVLA_ROOT}/data/wheels/$(basename "${FLASH_ATTN_WHEEL_URL}")"
  if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    curl -L "${FLASH_ATTN_WHEEL_URL}" -o "${FLASH_ATTN_WHEEL}"
  fi
  uv pip install --python "${PYTHON}" "${FLASH_ATTN_WHEEL}"
fi

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

echo "[install_env] third_party clone"
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

echo "[install_env] third_party editable installs"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite-task-zoo"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/robomimic"
uv pip install --python "${PYTHON}" --no-build-isolation -e "${DVLA_ROOT}/third_party/mimicgen"

echo "[install_env] LIBERO packaging compatibility"
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

echo "[install_env] egl_probe"
if [[ -f "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" ]]; then
  sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' \
    "${DVLA_ROOT}/third_party/egl_probe/egl_probe/CMakeLists.txt" || true
fi
uv pip install --python "${PYTHON}" --no-build-isolation "${DVLA_ROOT}/third_party/egl_probe"

if [[ "${INSTALL_OPTIONAL_THIRD_PARTY}" == "1" ]]; then
  uv pip install --python "${PYTHON}" -e "${DVLA_ROOT}/third_party/TensorNVMe" || true
  uv pip install --python "${PYTHON}" -v --no-build-isolation "${DVLA_ROOT}/third_party/apex" || true
fi

echo "[install_env] verify"
"${PYTHON}" - <<'PY'
import torch
import h5py
import hydra
import omegaconf
import transformers
import libero

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("deps ok")
print("libero", libero.__path__)
PY
