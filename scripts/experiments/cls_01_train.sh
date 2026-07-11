#!/usr/bin/env bash
# Stage CLS-01: train the WMPO token success classifier.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT"
fi
echo "[cls-train] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[cls-train] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_BIN="${PYTHON:-python}"
CLS_EXPERIMENT="${CLS_EXPERIMENT:-wmpo_token_classifier_openvla_onetraj_libero_goal_h1}"
CLS_RESUME="${CLS_RESUME:-${RESUME:-false}}"
if [[ "${CLS_RESUME}" == "true" && -z "${CLS_RUN_ROOT:-}" && -z "${CLS_RESUME_DIR:-}" ]]; then
  echo "[cls-train] CLS_RESUME=true requires CLS_RUN_ROOT or CLS_RESUME_DIR" >&2
  exit 2
fi
if [[ -n "${CLS_RESUME_DIR:-}" ]]; then
  export CLS_RUN_ROOT="${CLS_RUN_ROOT:-${CLS_RESUME_DIR}}"
else
  export CLS_RUN_ROOT="${CLS_RUN_ROOT:-${DVLA_DATA_ROOT}/outputs/classifier/wmpo_token_cls_openvla_onetraj_libero_goal_h1/$(date +%Y%m%d_%H%M%S)}"
fi
CLS_RESUME_DIR="${CLS_RESUME_DIR:-${CLS_RUN_ROOT}}"
CLS_CKPT_EVERY="${CLS_CKPT_EVERY:-250}"
if [[ "${CLS_RESUME}" == "true" ]]; then
  if [[ -d "${CLS_RESUME_DIR}" ]]; then
    if [[ ! -f "${CLS_RESUME_DIR}/checkpoints/latest.ckpt" \
      && ! -f "${CLS_RESUME_DIR}/ckpt/latest.ckpt" \
      && ! -f "${CLS_RESUME_DIR}/latest.ckpt" ]]; then
      echo "[cls-train] no latest classifier checkpoint under CLS_RESUME_DIR=${CLS_RESUME_DIR}" >&2
      exit 2
    fi
  else
    echo "[cls-train] classifier resume run directory not found: ${CLS_RESUME_DIR}" >&2
    exit 2
  fi
fi

echo "[cls-train] EXPERIMENT = ${CLS_EXPERIMENT}" >&2
echo "[cls-train] RUN_ROOT   = ${CLS_RUN_ROOT}" >&2
echo "[cls-train] RESUME     = ${CLS_RESUME}" >&2
echo "[cls-train] CKPT_EVERY = ${CLS_CKPT_EVERY}" >&2

"${PYTHON_BIN}" -m dreamervla.train \
  "experiment=${CLS_EXPERIMENT}" \
  training.out_dir="${CLS_RUN_ROOT}" \
  training.resume="${CLS_RESUME}" \
  ++training.resume_dir="${CLS_RESUME_DIR}" \
  training.ckpt_every="${CLS_CKPT_EVERY}" \
  "$@"
