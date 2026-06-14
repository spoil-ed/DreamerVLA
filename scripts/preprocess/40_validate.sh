#!/usr/bin/env bash
# Validate one generated LIBERO preprocessing artifact tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
TASK="${TASK:-libero_goal}"
TASK_NAME="${TASK_NAME:-${TASK}}"
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${TASK_NAME}"

python -m dreamer_vla.preprocess.validate_libero_data_prep \
  --data-root "${DVLA_DATA_ROOT}" \
  --processed-data-root "${PROCESSED_DATA_ROOT}" \
  --suites "${TASK_NAME}" \
  --his 1 \
  --action-horizon 1 \
  --image-resolution 256 \
  --check-action-hidden
