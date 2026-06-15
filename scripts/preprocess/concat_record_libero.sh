#!/usr/bin/env bash
# Concatenate LIBERO token record shards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
TASK="${TASK:-libero_goal}"
TOKENS_DIR="${TOKENS_DIR:-${DVLA_DATA_ROOT}/processed_data/${TASK}/tokens}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

python -m dreamervla.preprocess.concat_record_libero --base-dir "${TOKENS_DIR}"
