#!/usr/bin/env bash
# Download OpenVLA-OFT one-trajectory SFT checkpoints.
#
# Method 1, git clone with Git LFS:
#   OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=git bash scripts/download/30_openvla_oft_one_trajectory.sh
#
# Method 2, huggingface-hub:
#   OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=hf bash scripts/download/30_openvla_oft_one_trajectory.sh
#
# For faster downloads in China, callers may set:
#   export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD="${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD:-hf}"

download_one_repo() {
  local repo="$1"
  local target="$2"

  case "${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD}" in
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
      echo "Unsupported OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=${OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD}; use hf or git." >&2
      exit 2
      ;;
  esac
}

for spec in $(normalize_list "${OPENVLA_ONE_TRAJ_REPOS}"); do
  [[ -n "${spec}" ]] || continue
  repo="${spec%%:*}"
  local_name="${spec#*:}"
  [[ "${local_name}" != "${spec}" ]] || local_name="$(basename "${repo}")"
  target="${OPENVLA_ONE_TRAJ_ROOT}/${local_name}"

  # Weight: one-trajectory OpenVLA-OFT SFT checkpoint consumed by
  # scripts/eval/launch_openvla_oft_official_libero_eval.sh via CKPT_ROOT.
  download_log "OpenVLA-OFT one-trajectory ${repo} -> ${target}"
  download_one_repo "${repo}" "${target}"
done
