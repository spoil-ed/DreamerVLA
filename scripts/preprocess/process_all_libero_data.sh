#!/usr/bin/env bash
# Compatibility wrapper for pretokenizing several LIBERO suites.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
PYTHON="${PYTHON:-python}"
SUITES="${SUITES:-libero_10 libero_object libero_spatial}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-8}"
GPUS="${GPUS:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --suites) SUITES="$2"; shift 2 ;;
    --data-root) DVLA_DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --gpus) GPUS="$2"; export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --num-procs) PRETOKENIZE_PROCS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
export DVLA_DATA_ROOT PYTHON PRETOKENIZE_PROCS GPUS
cd "${DVLA_ROOT}"

overall_rc=0
for suite in ${SUITES}; do
  echo "[process_all_libero_data] TASK=${suite}"
  if ! bash scripts/preprocess/20_pretokenize_dataset.sh --task "${suite}"; then
    echo "[process_all_libero_data] ${suite} failed" >&2
    overall_rc=1
  fi
done

exit "${overall_rc}"
