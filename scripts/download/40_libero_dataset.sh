#!/usr/bin/env bash
# Download raw LIBERO demonstrations through LIBERO's official helper.
#
# Output for each suite in LIBERO_SUITES:
#   ${DVLA_DATA_ROOT}/datasets/libero/<suite>/*.hdf5
#
# New benchmark suites should keep the same datasets/libero/<suite>/ shape so
# preprocess and eval scripts can keep using one LIBERO_CONFIG_PATH.
set -euo pipefail

# Step 0: Load shared roots, Python, and suite list.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

# Step 1: LIBERO dataset destination.
mkdir -p "${LIBERO_DATASET_DIR}"

# Step 2: LIBERO's downloader lives in third_party, installed by install_env.
if [[ ! -f "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" ]]; then
  echo "Missing third_party/LIBERO. Run scripts/install_env.sh first." >&2
  exit 2
fi

# Step 3: Download each suite serially into datasets/libero/<suite>/.
for suite in $(normalize_list "${LIBERO_SUITES}"); do
  [[ -n "${suite}" ]] || continue
  download_log "LIBERO ${suite} dataset -> ${LIBERO_DATASET_DIR}/${suite}"
  "${PYTHON}" "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" \
    --download-dir "${LIBERO_DATASET_DIR}" \
    --datasets "${suite}" --use-huggingface
done
