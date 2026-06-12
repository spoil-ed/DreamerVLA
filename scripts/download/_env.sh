#!/usr/bin/env bash
# Shared environment for DreamerVLA asset download steps.
#
# Child scripts should source this file, then do one focused thing:
#   source "${SCRIPT_DIR}/_env.sh"
#   run the downloader for that asset family
#
# Add variables here only when two or more child scripts need the same knob.
set -euo pipefail

# Step 0: Resolve roots without assuming the data root lives under the repo.
DOWNLOAD_STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${DOWNLOAD_STEP_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"

# Step 1: Make package modules importable for helper scripts that call Python.
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

# Step 2: Shared remote repositories and suite lists.
#
# RynnVLA-002's README points Chameleon tokenizer/base/starting-point downloads
# at the older WorldVLA HF repo. Keep that source explicit without presenting it
# as a separate model family.
export RYNNVLA_CHAMELEON_REPO="${RYNNVLA_CHAMELEON_REPO:-${WORLDVLA_REPO:-Alibaba-DAMO-Academy/WorldVLA}}"
export LUMINA_REPO="${LUMINA_REPO:-Alpha-VLLM/Lumina-mGPT-7B-768}"
export RYNNVLA_REPO="${RYNNVLA_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}"
export LIBERO_SUITES="${LIBERO_SUITES:-libero_goal libero_object libero_spatial libero_10}"
export DOWNLOAD_ACTION_WM="${DOWNLOAD_ACTION_WM:-1}"
export OPENVLA_OFT_REPOS="${OPENVLA_OFT_REPOS:-}"
export OPENVLA_ONE_TRAJ_REPOS="${OPENVLA_ONE_TRAJ_REPOS:-Haozhan72/Openvla-oft-SFT-libero-spatial-traj1:Openvla-oft-SFT-libero-spatial-traj1 Haozhan72/Openvla-oft-SFT-libero-object-traj1:Openvla-oft-SFT-libero-object-traj1 Haozhan72/Openvla-oft-SFT-libero-goal-traj1:Openvla-oft-SFT-libero-goal-traj1 Haozhan72/Openvla-oft-SFT-libero10-traj1:Openvla-oft-SFT-libero10-traj1}"

# Step 3: Canonical local target directories.
CHECKPOINT_DIR="${DVLA_DATA_ROOT}/checkpoints"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${DVLA_DATA_ROOT}/datasets/libero}"
CALVIN_DIR="${CALVIN_DIR:-${DVLA_DATA_ROOT}/datasets/calvin}"
OPENVLA_OFT_CKPT_ROOT="${OPENVLA_OFT_CKPT_ROOT:-${CHECKPOINT_DIR}/OpenVLA-OFT}"
OPENVLA_ONE_TRAJ_ROOT="${OPENVLA_ONE_TRAJ_ROOT:-${CHECKPOINT_DIR}/Openvla-oft-SFT-traj1}"

cd "${DVLA_ROOT}"

download_log() {
  printf '[download:%s] %s\n' "$(basename "$0")" "$*"
}

# Accept either comma-separated or space-separated lists in env vars.
normalize_list() {
  printf '%s\n' "$1" | tr ',' ' '
}
