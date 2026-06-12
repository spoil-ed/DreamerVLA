#!/usr/bin/env bash
# Shared environment for DreamerVLA asset download steps.
set -euo pipefail

DOWNLOAD_STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${DOWNLOAD_STEP_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

export WORLDVLA_REPO="${WORLDVLA_REPO:-Alibaba-DAMO-Academy/WorldVLA}"
export LUMINA_REPO="${LUMINA_REPO:-Alpha-VLLM/Lumina-mGPT-7B-768}"
export RYNNVLA_REPO="${RYNNVLA_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}"
export LIBERO_SUITES="${LIBERO_SUITES:-libero_goal libero_object libero_spatial libero_10}"
export DOWNLOAD_ACTION_WM="${DOWNLOAD_ACTION_WM:-1}"

CHECKPOINT_DIR="${DVLA_DATA_ROOT}/checkpoints"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${DVLA_DATA_ROOT}/datasets/libero}"
CALVIN_DIR="${CALVIN_DIR:-${DVLA_DATA_ROOT}/datasets/calvin}"

cd "${DVLA_ROOT}"

download_log() {
  printf '[download:%s] %s\n' "$(basename "$0")" "$*"
}

normalize_list() {
  printf '%s\n' "$1" | tr ',' ' '
}
