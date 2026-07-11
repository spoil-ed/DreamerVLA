#!/usr/bin/env bash
# Summarize a classifier training run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  DATA_ROOT_SOURCE="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  DATA_ROOT_SOURCE="DEFAULTED from DVLA_ROOT"
fi
echo "[classifier-training-eval] DVLA_ROOT      = ${DVLA_ROOT}" >&2
echo "[classifier-training-eval] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT} (${DATA_ROOT_SOURCE})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_EXECUTABLE="${PYTHON:-python}"
CLASSIFIER_EXPERIMENT="${CLASSIFIER_EXPERIMENT:-wmpo_token_classifier_openvla_onetraj_libero_goal_h1}"
if [[ -n "${CLASSIFIER_RUN_ROOT:-}" && -n "${CLASSIFIER_EVAL_OUTPUT:-}" ]]; then
  "${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.experiment_stage_checks cls-eval \
    --family "${CLASSIFIER_EXPERIMENT}" \
    --run-dir "${CLASSIFIER_RUN_ROOT}" \
    --out "${CLASSIFIER_EVAL_OUTPUT}" \
    "$@"
elif [[ -n "${CLASSIFIER_RUN_ROOT:-}" ]]; then
  "${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.experiment_stage_checks cls-eval \
    --family "${CLASSIFIER_EXPERIMENT}" \
    --run-dir "${CLASSIFIER_RUN_ROOT}" \
    "$@"
elif [[ -n "${CLASSIFIER_EVAL_OUTPUT:-}" ]]; then
  "${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.experiment_stage_checks cls-eval \
    --family "${CLASSIFIER_EXPERIMENT}" \
    --out "${CLASSIFIER_EVAL_OUTPUT}" \
    "$@"
else
  "${PYTHON_EXECUTABLE}" -m dreamervla.diagnostics.experiment_stage_checks cls-eval \
    --family "${CLASSIFIER_EXPERIMENT}" \
    "$@"
fi
