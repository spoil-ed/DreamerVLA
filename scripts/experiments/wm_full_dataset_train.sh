#!/usr/bin/env bash
# Train the configured Chunk-WM on the complete original LIBERO replay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[wm-full-dataset] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[wm-full-dataset] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" -m dreamervla.diagnostics.experiment_stage_checks libero-original-warmup-run \
  --experiment "${ORIGINAL_COTRAIN_EXPERIMENT:-openvla_onetraj_libero_cotrain_noray}" \
  --task "${ORIGINAL_TASK:-openvla_onetraj_libero}" \
  --python "${PYTHON_BIN}" \
  --hidden-dir "${ORIGINAL_HIDDEN_DIR:-}" \
  --ngpu "${NGPU:-1}" \
  --master-port "${MASTER_PORT:-29500}" \
  --run-root "${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/wm_full_dataset/$(date +%Y%m%d_%H%M%S)}" \
  --wm-steps "${WM_WARMUP_STEPS:-20000}" \
  --classifier-steps 0 \
  --replay-epochs "${WARMUP_REPLAY_EPOCHS:-5}" \
  --checkpoint-every "${WARMUP_CHECKPOINT_EVERY:-500}" \
  --topk-k "${WARMUP_TOPK_K:-3}" \
  --wm-batch-size "${WM_BATCH_SIZE:-32}" \
  --classifier-batch-size 1 \
  --buffer-size "${WARMUP_BUFFER_SIZE:-160000}" \
  --task-ids "${ORIGINAL_TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}" \
  "$@"
