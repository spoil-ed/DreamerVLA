#!/usr/bin/env bash
# Download optional CALVIN zip files.
#
# Output:
#   ${DVLA_DATA_ROOT}/datasets/calvin/<task>.zip
#   ${DVLA_DATA_ROOT}/datasets/calvin/<task>/  when EXTRACT_CALVIN=1
#   ${DVLA_DATA_ROOT}/datasets/calvin/task_ABCD_D_shards/  when CALVIN_DOWNLOAD_METHOD=hf_shards
#   ${DVLA_DATA_ROOT}/datasets/calvin/task_ABCD_D_subsets/ when CALVIN_DOWNLOAD_METHOD=hf_subsets
#   ${DVLA_DATA_ROOT}/datasets/calvin/opendatalab/         when CALVIN_DOWNLOAD_METHOD=opendatalab
#
# Add more CALVIN tasks by passing CALVIN_TASKS="task_A task_B"; keep all CALVIN
# files under datasets/calvin/ so they do not mix with LIBERO suites.
#
# Domestic / mirror-friendly examples:
#   HF_ENDPOINT=https://hf-mirror.com CALVIN_DOWNLOAD_METHOD=hf_shards bash scripts/download/50_calvin_dataset.sh
#   HF_ENDPOINT=https://hf-mirror.com CALVIN_DOWNLOAD_METHOD=hf_subsets bash scripts/download/50_calvin_dataset.sh
#   CALVIN_DOWNLOAD_METHOD=opendatalab bash scripts/download/50_calvin_dataset.sh
set -euo pipefail

# Step 0: Load shared roots and Python.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

# Step 1: CALVIN source and target task list.
CALVIN_DOWNLOAD_METHOD="${CALVIN_DOWNLOAD_METHOD:-official}"
CALVIN_BASE_URL="${CALVIN_BASE_URL:-http://calvin.cs.uni-freiburg.de/dataset}"
CALVIN_TASKS="${CALVIN_TASKS:-task_ABCD_D}"
CALVIN_HF_SHARDS_REPO="${CALVIN_HF_SHARDS_REPO:-VyoJ/calvin-ABCD-D-shards}"
CALVIN_HF_SUBSETS_REPO="${CALVIN_HF_SUBSETS_REPO:-VyoJ/calvin-ABCD-D-subsets}"
CALVIN_OPENDATALAB_REPO="${CALVIN_OPENDATALAB_REPO:-OpenDataLab/CALVIN}"
mkdir -p "${CALVIN_DIR}"

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI is required for CALVIN_DOWNLOAD_METHOD=${CALVIN_DOWNLOAD_METHOD}. Install with: pip install huggingface-hub" >&2
    exit 2
  fi
}

case "${CALVIN_DOWNLOAD_METHOD}" in
  official|curl|freiburg)
    # Step 2: Official Freiburg zip files, one task at a time; curl -C resumes partial files.
    for task in $(normalize_list "${CALVIN_TASKS}"); do
      [[ -n "${task}" ]] || continue
      zip_path="${CALVIN_DIR}/${task}.zip"
      if [[ ! -f "${zip_path}" ]]; then
        download_log "CALVIN ${task} archive -> ${zip_path}"
        curl -L -C - "${CALVIN_BASE_URL}/${task}.zip" -o "${zip_path}"
      else
        download_log "CALVIN ${task} archive already exists: ${zip_path}"
      fi

      # Step 3: Extraction is explicit because archives can be large.
      if [[ "${EXTRACT_CALVIN:-0}" == "1" ]]; then
        download_log "extract CALVIN ${task} -> ${CALVIN_DIR}/${task}"
        "${PYTHON}" -m zipfile -e "${zip_path}" "${CALVIN_DIR}/${task}"
      fi
    done
    ;;
  hf_shards)
    # Step 2: Hugging Face mirror of task_ABCD_D split into 30 GB multi-part zip shards.
    # Set HF_ENDPOINT=https://hf-mirror.com in mainland China if needed.
    ensure_hf_cli
    target="${CALVIN_DIR}/task_ABCD_D_shards"
    download_log "CALVIN task_ABCD_D shards from ${CALVIN_HF_SHARDS_REPO} -> ${target}"
    download_log "hf_endpoint=${HF_ENDPOINT:-<default>}"
    hf download "${CALVIN_HF_SHARDS_REPO}" --repo-type dataset --local-dir "${target}"
    download_log "reassemble hint: cd ${target} && zip -F calvin_ABCD_D.zip --out full_dataset.zip && unzip full_dataset.zip"
    ;;
  hf_subsets)
    # Step 2: Hugging Face mirror of task_ABCD_D split into complete structured subset zips.
    # Set HF_ENDPOINT=https://hf-mirror.com in mainland China if needed.
    ensure_hf_cli
    target="${CALVIN_DIR}/task_ABCD_D_subsets"
    download_log "CALVIN task_ABCD_D subsets from ${CALVIN_HF_SUBSETS_REPO} -> ${target}"
    download_log "hf_endpoint=${HF_ENDPOINT:-<default>}"
    hf download "${CALVIN_HF_SUBSETS_REPO}" --repo-type dataset --local-dir "${target}"
    download_log "subset hint: unzip the needed subset zip files under ${target}; each subset is self-contained."
    ;;
  opendatalab)
    # Step 2: OpenDataLab domestic platform mirror. Requires an OpenDataLab/OpenXLab account.
    if ! command -v openxlab >/dev/null 2>&1; then
      echo "openxlab CLI is required for CALVIN_DOWNLOAD_METHOD=opendatalab. Install with: pip install -U openxlab" >&2
      exit 2
    fi
    target="${CALVIN_DIR}/opendatalab"
    download_log "CALVIN OpenDataLab mirror ${CALVIN_OPENDATALAB_REPO} -> ${target}"
    openxlab dataset get --dataset-repo "${CALVIN_OPENDATALAB_REPO}" --target-path "${target}"
    ;;
  *)
    echo "Unsupported CALVIN_DOWNLOAD_METHOD=${CALVIN_DOWNLOAD_METHOD}; use official, hf_shards, hf_subsets, or opendatalab." >&2
    exit 2
    ;;
esac
