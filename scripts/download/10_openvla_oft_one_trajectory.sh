#!/usr/bin/env bash
# Download OpenVLA-OFT one-trajectory SFT checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
OPENVLA_ONE_TRAJ_ROOT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1"
OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD="${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD:-hf}"
OPENVLA_ONE_TRAJ_REPO="${OPENVLA_ONE_TRAJ_REPO:-Haozhan72/Openvla-oft-SFT-libero-goal-traj1:Openvla-oft-SFT-libero-goal-traj1}"
cd "${DVLA_ROOT}"

repo="${OPENVLA_ONE_TRAJ_REPO%%:*}"
local_name="${OPENVLA_ONE_TRAJ_REPO#*:}"
if [[ "${local_name}" == "${OPENVLA_ONE_TRAJ_REPO}" ]]; then
  local_name="$(basename "${repo}")"
fi
target="${OPENVLA_ONE_TRAJ_ROOT}/${local_name}"

if [[ "${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD}" == "hf" || "${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD}" == "huggingface-hub" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required: install huggingface-hub." >&2
    exit 2
  fi
  hf download "${repo}" --local-dir "${target}"
elif [[ "${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD}" == "git" ]]; then
  if ! command -v git >/dev/null 2>&1 || ! git lfs version >/dev/null 2>&1; then
    echo "git and git-lfs are required for OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=git." >&2
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
  echo "Unsupported OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD}; use hf or git." >&2
  exit 2
fi
