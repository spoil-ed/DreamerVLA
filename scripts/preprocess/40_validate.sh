#!/usr/bin/env bash
# Validate the generated LIBERO preprocessing artifact tree for one suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
PYTHON="${PYTHON:-python}"
TASK="${TASK:-libero_goal}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --data-root) DVLA_DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
export PYTHON
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data"

"${PYTHON}" -m dreamer_vla.preprocess.validate_libero_data_prep \
  --data-root "${DVLA_DATA_ROOT}" \
  --processed-data-root "${PROCESSED_DATA_ROOT}" \
  --suites "${TASK}" \
  --his 1 \
  --action-horizon 1 \
  --image-resolution 256 \
  --check-action-hidden \
  "$@"
