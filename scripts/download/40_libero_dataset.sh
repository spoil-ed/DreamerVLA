#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

mkdir -p "${LIBERO_DATASET_DIR}"

if [[ ! -f "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" ]]; then
  echo "Missing third_party/LIBERO. Run scripts/install_env.sh first." >&2
  exit 2
fi

for suite in $(normalize_list "${LIBERO_SUITES}"); do
  [[ -n "${suite}" ]] || continue
  download_log "LIBERO ${suite} dataset -> ${LIBERO_DATASET_DIR}/${suite}"
  "${PYTHON}" "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" \
    --download-dir "${LIBERO_DATASET_DIR}" \
    --datasets "${suite}" --use-huggingface
done
