#!/usr/bin/env bash
# OpenVLA-OFT LIBERO eval one-command wrapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
SUITE="${SUITE:-libero_goal}"
CKPT="${CKPT:-${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1}"
TASK_IDS="${TASK_IDS:-}"
NUM_TRIALS="${NUM_TRIALS:-10}"
GPU_ID="${GPU_ID:-0}"
POLICY_MODE="${POLICY_MODE:-discrete}"
CAMERA_INPUTS="${CAMERA_INPUTS:-primary}"
NUM_IMAGES="${NUM_IMAGES:-1}"
USE_PROPRIO="${USE_PROPRIO:-0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
ENV_IMG_RES="${ENV_IMG_RES:-256}"
SEED="${SEED:-7}"
OUTPUT_DIR="${OUTPUT_DIR:-${DVLA_DATA_ROOT}/outputs/eval/openvla_oft_official_libero}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-${DVLA_ROOT}/../WMPO/dependencies/openvla-oft}"

export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
cd "${DVLA_ROOT}"

if [[ "${USE_PROPRIO}" == "1" ]]; then
  PROPRIO_FLAG="--use-proprio"
else
  PROPRIO_FLAG="--no-use-proprio"
fi

python -m dreamervla.diagnostics.eval_openvla_oft_libero \
  --ckpt "${CKPT}" \
  --suite "${SUITE}" \
  --task-ids "${TASK_IDS}" \
  --num-trials "${NUM_TRIALS}" \
  --gpu-id "${GPU_ID}" \
  --seed "${SEED}" \
  --policy-mode "${POLICY_MODE}" \
  --camera-inputs "${CAMERA_INPUTS}" \
  --num-images "${NUM_IMAGES}" \
  ${PROPRIO_FLAG} \
  --num-open-loop-steps "${NUM_OPEN_LOOP_STEPS}" \
  --env-img-res "${ENV_IMG_RES}" \
  --output-dir "${OUTPUT_DIR}" \
  --openvla-oft-root "${OPENVLA_OFT_ROOT}" \
  "$@"
