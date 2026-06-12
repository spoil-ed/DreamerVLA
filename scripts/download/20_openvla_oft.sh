#!/usr/bin/env bash
# Download OpenVLA-OFT HDF5 SFT component checkpoints.
#
# The active task configs expect directories under:
#   ${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/<run>/
#
# Pass repos as repo:local_name pairs. The local_name must match the config path
# you want to satisfy, for example:
#   OPENVLA_OFT_REPOS="owner/repo:libero_goal_hdf5_latest_6650 owner/repo2:libero_object" bash scripts/download/20_openvla_oft.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

OPENVLA_OFT_DOWNLOAD_METHOD="${OPENVLA_OFT_DOWNLOAD_METHOD:-hf}"

if [[ -z "${OPENVLA_OFT_REPOS}" ]]; then
  download_log "OPENVLA_OFT_REPOS is empty; no OpenVLA-OFT HDF5 SFT checkpoints were downloaded."
  download_log "set OPENVLA_OFT_REPOS='owner/repo:libero_goal_hdf5_latest_6650 owner/repo2:libero_object' and rerun."
  exit 0
fi

download_hf_repo() {
  local repo="$1"
  local target="$2"

  case "${OPENVLA_OFT_DOWNLOAD_METHOD}" in
    hf|huggingface-hub)
      hf download "${repo}" --local-dir "${target}"
      ;;
    git)
      git lfs install
      if [[ -d "${target}/.git" ]]; then
        git -C "${target}" pull --ff-only
      elif [[ -e "${target}" ]]; then
        echo "Target exists but is not a git checkout: ${target}" >&2
        exit 2
      else
        git clone "https://huggingface.co/${repo}" "${target}"
      fi
      ;;
    *)
      echo "Unsupported OPENVLA_OFT_DOWNLOAD_METHOD=${OPENVLA_OFT_DOWNLOAD_METHOD}; use hf or git." >&2
      exit 2
      ;;
  esac
}

for spec in $(normalize_list "${OPENVLA_OFT_REPOS}"); do
  [[ -n "${spec}" ]] || continue
  repo="${spec%%:*}"
  local_name="${spec#*:}"
  [[ "${local_name}" != "${spec}" ]] || local_name="$(basename "${repo}")"
  target="${OPENVLA_OFT_CKPT_ROOT}/${local_name}"

  # Weight: OpenVLA-OFT HDF5 SFT checkpoint components such as dataset_statistics,
  # action_head--*_checkpoint.pt, and proprio_projector--*_checkpoint.pt.
  download_log "OpenVLA-OFT ${repo} -> ${target}"
  download_hf_repo "${repo}" "${target}"
done
