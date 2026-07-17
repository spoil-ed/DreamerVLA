#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
INSTALL_OPTIONAL_THIRD_PARTY="${INSTALL_OPTIONAL_THIRD_PARTY:-0}"
INSTALL_OPENSORA_THIRD_PARTY="${INSTALL_OPENSORA_THIRD_PARTY:-0}"
INSTALL_OPENVLA_OFT_THIRD_PARTY="${INSTALL_OPENVLA_OFT_THIRD_PARTY:-1}"
OPENVLA_OFT_REPO_URL="${OPENVLA_OFT_REPO_URL:-https://github.com/moojink/openvla-oft.git}"
OPENVLA_OFT_REVISION="${OPENVLA_OFT_REVISION:-e4287e94541f459edc4feabc4e181f537cd569a8}"
OPENSORA_REPO_URL="${OPENSORA_REPO_URL:-https://github.com/hpcaitech/Open-Sora.git}"
OPENSORA_REVISION="${OPENSORA_REVISION:-17cce908b22283acc3c946816c81f46dd442a453}"
DLIMP_OPENVLA_SRC="${DLIMP_OPENVLA_SRC:-git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46}"
TRANSFORMERS_OFT_FORK_SRC="${TRANSFORMERS_OFT_FORK_SRC:-git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab}"
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

LIBERO_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
LIBERO_PTH="${LIBERO_SITE_PACKAGES}/dreamervla_libero.pth"
printf '%s\n' "${DVLA_ROOT}/third_party/LIBERO" > "${LIBERO_PTH}"
echo "[install:40_third_party] libero_path_file=${LIBERO_PTH}"

if [[ "${INSTALL_OPENSORA_THIRD_PARTY}" == "1" ]]; then
  if [[ ! -e "${DVLA_ROOT}/third_party/opensora" ]]; then
    git clone "${OPENSORA_REPO_URL}" "${DVLA_ROOT}/third_party/opensora"
  elif [[ ! -d "${DVLA_ROOT}/third_party/opensora" ]]; then
    echo "third_party/opensora exists but is not a directory." >&2
    exit 2
  fi
  if [[ -d "${DVLA_ROOT}/third_party/opensora/.git" ]]; then
    git -C "${DVLA_ROOT}/third_party/opensora" fetch --all --tags
    git -C "${DVLA_ROOT}/third_party/opensora" checkout "${OPENSORA_REVISION}"
  else
    echo "[install:40_third_party] using existing unversioned Open-Sora source tree"
  fi
  uv pip install -v -e "${DVLA_ROOT}/third_party/opensora"
fi

if [[ "${INSTALL_OPENVLA_OFT_THIRD_PARTY}" == "1" ]]; then
  if [[ ! -e "${DVLA_ROOT}/third_party/openvla-oft" ]]; then
    git clone "${OPENVLA_OFT_REPO_URL}" "${DVLA_ROOT}/third_party/openvla-oft"
  elif [[ ! -d "${DVLA_ROOT}/third_party/openvla-oft" ]]; then
    echo "third_party/openvla-oft exists but is not a directory." >&2
    exit 2
  fi
  if [[ -d "${DVLA_ROOT}/third_party/openvla-oft/.git" ]]; then
    git -C "${DVLA_ROOT}/third_party/openvla-oft" fetch --all --tags
    git -C "${DVLA_ROOT}/third_party/openvla-oft" checkout "${OPENVLA_OFT_REVISION}"
  else
    echo "[install:40_third_party] using existing unversioned OpenVLA-OFT source tree"
  fi
  uv pip uninstall openvla-oft >/dev/null 2>&1 || true
  uv pip install --no-deps "${DLIMP_OPENVLA_SRC}"
  # OFT and dlimp are installed --no-deps above so their large, pin-conflicting
  # dependency trees do not override the curated env. That drops a few packages OFT
  # genuinely needs at runtime, so re-add them explicitly with the pins declared in
  # third_party/openvla-oft/pyproject.toml. tensorflow_datasets is what the RLDS data
  # pipeline (prismatic/vla/datasets/rlds) imports; rich/future are pulled in by the
  # OFT scripts and were missing on a clean --no-deps install.
  uv pip install rich==15.0.0 future==1.0.0 json-numpy==2.1.1 jsonlines==4.0.0 \
    tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 \
    tensorflow_metadata==1.17.3
  # OpenVLA-OFT REQUIRES moojink's custom transformers fork: it patches the Llama
  # attention to bidirectional (is_causal=False) for OFT parallel action-chunk
  # decoding. Vanilla transformers yields 0% / garbage OFT actions even though BOTH
  # report __version__ "4.40.1". This is the single authoritative transformers;
  # --force-reinstall so the fork overrides anything pulled in transitively by
  # 30_python_deps (peft/diffusers). Offline: point TRANSFORMERS_OFT_FORK_SRC at a
  # local checkout / wheel / sdist instead of GitHub.
  uv pip install --no-deps --force-reinstall "${TRANSFORMERS_OFT_FORK_SRC}"
fi
