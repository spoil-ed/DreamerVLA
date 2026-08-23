#!/usr/bin/env bash
# Download OpenVLA-OFT HDF5 SFT component checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
OPENVLA_OFT_REPO="${OPENVLA_OFT_REPO:-}"
OPENVLA_OFT_DOWNLOAD_METHOD="${OPENVLA_OFT_DOWNLOAD_METHOD:-hf}"
OPENVLA_OFT_CKPT_ROOT="${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT"
cd "${DVLA_ROOT}"

if [[ -z "${OPENVLA_OFT_REPO}" ]]; then
  echo "[download:00_openvla_oft] OPENVLA_OFT_REPO is empty; no checkpoint downloaded."
  exit 0
fi

repo="${OPENVLA_OFT_REPO%%:*}"
local_name="${OPENVLA_OFT_REPO#*:}"
if [[ "${local_name}" == "${OPENVLA_OFT_REPO}" ]]; then
  local_name="$(basename "${repo}")"
fi
target="${OPENVLA_OFT_CKPT_ROOT}/${local_name}"

if [[ "${OPENVLA_OFT_DOWNLOAD_METHOD}" == "hf" || "${OPENVLA_OFT_DOWNLOAD_METHOD}" == "huggingface-hub" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required: install huggingface-hub." >&2
    exit 2
  fi
  hf download "${repo}" --local-dir "${target}"
elif [[ "${OPENVLA_OFT_DOWNLOAD_METHOD}" == "git" ]]; then
  if ! command -v git >/dev/null 2>&1 || ! git lfs version >/dev/null 2>&1; then
    echo "git and git-lfs are required for OPENVLA_OFT_DOWNLOAD_METHOD=git." >&2
    exit 2
  fi
  git lfs install
  if [[ -d "${target}/.git" ]]; then
    git -C "${target}" pull --ff-only
  elif [[ -e "${target}" ]]; then
    echo "Target exists but is not a git checkout: ${target}" >&2
    exit 2
  else
    git clone "https://huggingface.co/${repo}" "${target}"
  fi
else
  echo "Unsupported OPENVLA_OFT_DOWNLOAD_METHOD=${OPENVLA_OFT_DOWNLOAD_METHOD}; use hf or git." >&2
  exit 2
fi
