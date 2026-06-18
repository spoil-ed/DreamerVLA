#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
INSTALL_OPTIONAL_THIRD_PARTY="${INSTALL_OPTIONAL_THIRD_PARTY:-0}"
INSTALL_OPENSORA_THIRD_PARTY="${INSTALL_OPENSORA_THIRD_PARTY:-1}"
INSTALL_OPENVLA_OFT_THIRD_PARTY="${INSTALL_OPENVLA_OFT_THIRD_PARTY:-1}"
cd "${DVLA_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required before running this install step." >&2
  exit 2
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "[install:40_third_party] third_party_dir=${DVLA_ROOT}/third_party"
echo "[install:40_third_party] optional_third_party=${INSTALL_OPTIONAL_THIRD_PARTY}"
echo "[install:40_third_party] opensora_third_party=${INSTALL_OPENSORA_THIRD_PARTY}"
echo "[install:40_third_party] openvla_oft_third_party=${INSTALL_OPENVLA_OFT_THIRD_PARTY}"
mkdir -p "${DVLA_ROOT}/third_party"

if [[ ! -d "${DVLA_ROOT}/third_party/LIBERO/.git" ]]; then git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${DVLA_ROOT}/third_party/LIBERO"; fi
git -C "${DVLA_ROOT}/third_party/LIBERO" fetch --all --tags
git -C "${DVLA_ROOT}/third_party/LIBERO" checkout 8f1084e

if [[ ! -d "${DVLA_ROOT}/third_party/robosuite/.git" ]]; then git clone https://github.com/ARISE-Initiative/robosuite.git "${DVLA_ROOT}/third_party/robosuite"; fi
git -C "${DVLA_ROOT}/third_party/robosuite" fetch --all --tags
git -C "${DVLA_ROOT}/third_party/robosuite" checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa

if [[ ! -d "${DVLA_ROOT}/third_party/robosuite-task-zoo/.git" ]]; then git clone https://github.com/ARISE-Initiative/robosuite-task-zoo "${DVLA_ROOT}/third_party/robosuite-task-zoo"; fi
git -C "${DVLA_ROOT}/third_party/robosuite-task-zoo" fetch --all --tags
git -C "${DVLA_ROOT}/third_party/robosuite-task-zoo" checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed

if [[ ! -d "${DVLA_ROOT}/third_party/robomimic/.git" ]]; then git clone https://github.com/ARISE-Initiative/robomimic.git "${DVLA_ROOT}/third_party/robomimic"; fi
git -C "${DVLA_ROOT}/third_party/robomimic" fetch --all --tags
git -C "${DVLA_ROOT}/third_party/robomimic" checkout d0b37cf214bd24fb590d182edb6384333f67b661

if [[ ! -d "${DVLA_ROOT}/third_party/mimicgen/.git" ]]; then git clone https://github.com/NVlabs/mimicgen.git "${DVLA_ROOT}/third_party/mimicgen"; fi
git -C "${DVLA_ROOT}/third_party/mimicgen" fetch --all --tags
git -C "${DVLA_ROOT}/third_party/mimicgen" checkout 72bd767

uv pip install --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite"
uv pip install --no-build-isolation -e "${DVLA_ROOT}/third_party/robosuite-task-zoo"
uv pip install --no-build-isolation -e "${DVLA_ROOT}/third_party/robomimic"
uv pip install --no-build-isolation -e "${DVLA_ROOT}/third_party/mimicgen"

cat > "${DVLA_ROOT}/third_party/LIBERO/pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "libero"
version = "0.1.0"
description = "LIBERO benchmark package"
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
uv pip install --no-build-isolation -e "${DVLA_ROOT}/third_party/LIBERO"

if [[ "${INSTALL_OPENSORA_THIRD_PARTY}" == "1" && -d "${DVLA_ROOT}/third_party/opensora" ]]; then
  uv pip install -v -e "${DVLA_ROOT}/third_party/opensora"
fi

if [[ "${INSTALL_OPENVLA_OFT_THIRD_PARTY}" == "1" && -d "${DVLA_ROOT}/third_party/openvla-oft" ]]; then
  if [[ -f "${DVLA_ROOT}/third_party/openvla-oft/install_mujoco.sh" ]]; then
    (cd "${DVLA_ROOT}/third_party/openvla-oft" && bash ./install_mujoco.sh)
  fi
  uv pip install --no-deps -e "${DVLA_ROOT}/third_party/openvla-oft"
  uv pip install --no-deps git+https://github.com/moojink/dlimp_openvla
  # OpenVLA-OFT REQUIRES moojink's custom transformers fork: it patches the Llama
  # attention to bidirectional (is_causal=False) for OFT parallel action-chunk
  # decoding. Vanilla transformers yields 0% / garbage OFT actions even though BOTH
  # report __version__ "4.40.1". This is the single authoritative transformers;
  # --force-reinstall so the fork overrides anything pulled in transitively by
  # 30_python_deps (peft/diffusers). Offline: point TRANSFORMERS_OFT_FORK_SRC at a
  # local checkout / wheel / sdist instead of GitHub.
  TRANSFORMERS_OFT_FORK_SRC="${TRANSFORMERS_OFT_FORK_SRC:-git+https://github.com/moojink/transformers-openvla-oft.git}"
  uv pip install --no-deps --force-reinstall "${TRANSFORMERS_OFT_FORK_SRC}"
fi
