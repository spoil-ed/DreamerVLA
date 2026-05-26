#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
source "${SCRIPT_DIR}/lib/output_layout.sh"
PYTHON_BIN="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-openvla_oft_libero_goal}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${OUT_DIR:-}" ]]; then
  OUTPUT_ARCH="${OUTPUT_ARCH:-openvla_oft}"
  OUTPUT_CONFIG="${OUTPUT_CONFIG:-$(output_slug "${CONFIG_NAME#openvla_oft_}")}"
  OUTPUT_EXPERIMENT="${OUTPUT_EXPERIMENT:-${TIMESTAMP}}"
  OUT_DIR="$(output_layout_path "${PROJECT_ROOT}" vla "${OUTPUT_ARCH}" "${OUTPUT_CONFIG}" "${OUTPUT_EXPERIMENT}")"
fi
NUM_GPUS="${NUM_GPUS:-1}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/openvla-oft:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${PROJECT_ROOT}"

echo "[openvla-oft] config: ${CONFIG_NAME}"
echo "[openvla-oft] output: ${OUT_DIR}"
echo "[openvla-oft] gpus:   ${CUDA_VISIBLE_DEVICES:-all visible} (nproc=${NUM_GPUS})"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS}" \
  --module src.cli.train \
  --config-name "${CONFIG_NAME}" \
  training.out_dir="${OUT_DIR}" \
  "$@"
