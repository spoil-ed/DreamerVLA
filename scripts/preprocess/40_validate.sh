#!/usr/bin/env bash
# Validate one generated LIBERO preprocessing artifact tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
TASK="${TASK:-libero_goal}"
LIBERO_SUITE="${LIBERO_SUITE:-${TASK}}"
TASK_NAME="${TASK_NAME:-${TASK}}"
if [[ "${LIBERO_SUITE}" == "${TASK}" ]]; then
  case "${TASK_NAME}" in
    RynnVLA_LIBERO|OpenVLA_Onetraj_LIBERO) LIBERO_SUITE="libero_goal" ;;
  esac
fi
ARTIFACT_NAME="${ARTIFACT_NAME:-${TASK_NAME}}"
if [[ "${ARTIFACT_NAME}" == "${TASK_NAME}" && "${TASK_NAME}" != "${LIBERO_SUITE}" ]]; then
  ARTIFACT_NAME="${TASK_NAME}_${LIBERO_SUITE}"
fi
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${ARTIFACT_NAME}"

python -m dreamervla.preprocess.validate_libero_data_prep \
  --data-root "${DVLA_DATA_ROOT}" \
  --processed-data-root "${PROCESSED_DATA_ROOT}" \
  --suites "${ARTIFACT_NAME}" \
  --his 1 \
  --action-horizon 1 \
  --image-resolution 256 \
  --check-action-hidden
