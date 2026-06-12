#!/usr/bin/env bash
# Download raw LIBERO demonstrations through LIBERO's official helper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
SUITE="${SUITE:-libero_goal}"
LIBERO_DATASET_DIR="${DVLA_DATA_ROOT}/datasets/libero"
cd "${DVLA_ROOT}"

mkdir -p "${LIBERO_DATASET_DIR}"
if [[ ! -f "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" ]]; then
  echo "Missing third_party/LIBERO. Run scripts/install_env.sh first." >&2
  exit 2
fi

python "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" \
  --download-dir "${LIBERO_DATASET_DIR}" \
  --datasets "${SUITE}" --use-huggingface
