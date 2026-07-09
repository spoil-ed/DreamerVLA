#!/usr/bin/env bash
# Stage LIBERO-ORIG-00R: reprocess original LIBERO goal data into the OpenVLA artifact root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[libero-original-reprocess] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[libero-original-reprocess] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_BIN="${PYTHON:-python}"
export PYTHON="${PYTHON_BIN}"
PREPROCESS_GPUS="${PREPROCESS_GPUS:-${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}}"
PREPROCESS_NGPU="${PREPROCESS_NGPU:-${NGPU:-1}}"
PREPROCESS_NUM_PROCS="${PREPROCESS_NUM_PROCS:-${PRETOKENIZE_PROCS:-8}}"
PREPROCESS_OVERWRITE="${PREPROCESS_OVERWRITE:-true}"
OFT_CKPT_PATH="${OFT_CKPT:-${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1}"
OFT_POLICY_MODE_VALUE="${OFT_POLICY_MODE:-discrete}"
OFT_LATENT_SCHEME_VALUE="${OFT_LATENT_SCHEME:-both}"
OFT_HISTORY_VALUE="${OFT_HISTORY:-1}"
OFT_IMAGE_KEYS_VALUE="${OFT_IMAGE_KEYS:-agentview_rgb}"
OFT_ACTION_HIDDEN_GPUS_VALUE="${OFT_ACTION_HIDDEN_GPUS:-${PREPROCESS_NGPU}}"
export OFT_CHUNK_SIZE="${OFT_CHUNK_SIZE:-1}"

bash scripts/preprocess/prepare_libero_data.sh \
  task=libero_goal \
  libero_suite=libero_goal \
  task_name=openvla_onetraj_libero \
  artifact_name=OpenVLA_Onetraj_LIBERO_libero_goal \
  "only=[10_hdf5_reward,20_pretokenize_dataset,35_oft_action_hidden,40_validate]" \
  "gpus=${PREPROCESS_GPUS}" \
  "ngpu=${PREPROCESS_NGPU}" \
  "num_procs=${PREPROCESS_NUM_PROCS}" \
  "overwrite=${PREPROCESS_OVERWRITE}" \
  "python=${PYTHON_BIN}" \
  "env.OFT_CKPT=${OFT_CKPT_PATH}" \
  "env.OFT_POLICY_MODE=${OFT_POLICY_MODE_VALUE}" \
  "env.OFT_LATENT_SCHEME=${OFT_LATENT_SCHEME_VALUE}" \
  "env.OFT_HISTORY=${OFT_HISTORY_VALUE}" \
  "env.OFT_IMAGE_KEYS=${OFT_IMAGE_KEYS_VALUE}" \
  "env.OFT_ACTION_HIDDEN_GPUS=${OFT_ACTION_HIDDEN_GPUS_VALUE}" \
  "$@"
