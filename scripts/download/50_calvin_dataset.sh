#!/usr/bin/env bash
# Download optional CALVIN datasets.
#
# Mirror example:
#   HF_ENDPOINT=https://hf-mirror.com CALVIN_DOWNLOAD_METHOD=hf_shards bash scripts/download/50_calvin_dataset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
CALVIN_DIR="${DVLA_DATA_ROOT}/datasets/calvin"
CALVIN_DOWNLOAD_METHOD="${CALVIN_DOWNLOAD_METHOD:-official}"
CALVIN_BASE_URL="${CALVIN_BASE_URL:-http://calvin.cs.uni-freiburg.de/dataset}"
CALVIN_TASK="${CALVIN_TASK:-task_ABCD_D}"
CALVIN_HF_SHARDS_REPO="${CALVIN_HF_SHARDS_REPO:-VyoJ/calvin-ABCD-D-shards}"
CALVIN_HF_SUBSETS_REPO="${CALVIN_HF_SUBSETS_REPO:-VyoJ/calvin-ABCD-D-subsets}"
CALVIN_OPENDATALAB_REPO="${CALVIN_OPENDATALAB_REPO:-OpenDataLab/CALVIN}"
EXTRACT_CALVIN="${EXTRACT_CALVIN:-0}"
cd "${DVLA_ROOT}"

mkdir -p "${CALVIN_DIR}"

if [[ "${CALVIN_DOWNLOAD_METHOD}" == "official" || "${CALVIN_DOWNLOAD_METHOD}" == "curl" || "${CALVIN_DOWNLOAD_METHOD}" == "freiburg" ]]; then
  zip_path="${CALVIN_DIR}/${CALVIN_TASK}.zip"
  if [[ ! -f "${zip_path}" ]]; then
    curl -L -C - "${CALVIN_BASE_URL}/${CALVIN_TASK}.zip" -o "${zip_path}"
  fi
  if [[ "${EXTRACT_CALVIN}" == "1" ]]; then
    python -m zipfile -e "${zip_path}" "${CALVIN_DIR}/${CALVIN_TASK}"
  fi
elif [[ "${CALVIN_DOWNLOAD_METHOD}" == "hf_shards" ]]; then
  echo "[download:50_calvin_dataset] hf_endpoint=${HF_ENDPOINT:-<default>}"
  hf download "${CALVIN_HF_SHARDS_REPO}" --repo-type dataset --local-dir "${CALVIN_DIR}/task_ABCD_D_shards"
elif [[ "${CALVIN_DOWNLOAD_METHOD}" == "hf_subsets" ]]; then
  echo "[download:50_calvin_dataset] hf_endpoint=${HF_ENDPOINT:-<default>}"
  hf download "${CALVIN_HF_SUBSETS_REPO}" --repo-type dataset --local-dir "${CALVIN_DIR}/task_ABCD_D_subsets"
elif [[ "${CALVIN_DOWNLOAD_METHOD}" == "opendatalab" ]]; then
  openxlab dataset get --dataset-repo "${CALVIN_OPENDATALAB_REPO}" --target-path "${CALVIN_DIR}/opendatalab"
else
  echo "Unsupported CALVIN_DOWNLOAD_METHOD=${CALVIN_DOWNLOAD_METHOD}; use official, hf_shards, hf_subsets, or opendatalab." >&2
  exit 2
fi
