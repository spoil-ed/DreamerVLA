#!/usr/bin/env bash
# Download model weights and benchmark datasets used by DreamerVLA.
# All assets land under ${DVLA_DATA_ROOT} (default: <repo>/data).
set -euo pipefail

# ---- environment -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"
cd "${DVLA_ROOT}"

RYNNVLA_REPO="${RYNNVLA_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}"
WORLDVLA_REPO="${WORLDVLA_REPO:-Alibaba-DAMO-Academy/WorldVLA}"
LUMINA_REPO="${LUMINA_REPO:-Alpha-VLLM/Lumina-mGPT-7B-768}"
LIBERO_SUITES="${LIBERO_SUITES:-libero_goal libero_object libero_spatial libero_10}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_LIBERO="${DOWNLOAD_LIBERO:-1}"
DOWNLOAD_CALVIN="${DOWNLOAD_CALVIN:-0}"
DOWNLOAD_ACTION_WM="${DOWNLOAD_ACTION_WM:-1}"

CHECKPOINT_DIR="${DVLA_DATA_ROOT}/checkpoints"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${DVLA_DATA_ROOT}/datasets/libero}"
mkdir -p "${CHECKPOINT_DIR}" "${LIBERO_DATASET_DIR}"

echo "[download_assets] data_root=${DVLA_DATA_ROOT}"

normalize_list() {
  printf '%s\n' "$1" | tr ',' ' '
}

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  echo "[download_assets] Hugging Face weights -> ${CHECKPOINT_DIR}"
  hf download "${WORLDVLA_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}" \
    --include "chameleon/tokenizer/*" "chameleon/base_model/*" "base_model/*" "chameleon/starting_point/*"

  hf download "${LUMINA_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"

  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    hf download "${RYNNVLA_REPO}" --repo-type model \
      --local-dir "${CHECKPOINT_DIR}" \
      --include "VLA_model_256/${suite}/*"
    if [[ "${DOWNLOAD_ACTION_WM}" == "1" ]]; then
      hf download "${RYNNVLA_REPO}" --repo-type model \
        --local-dir "${CHECKPOINT_DIR}" \
        --include "Action_World_model_512/${suite}/*"
    fi
  done
fi

if [[ "${DOWNLOAD_LIBERO}" == "1" ]]; then
  echo "[download_assets] LIBERO datasets -> ${LIBERO_DATASET_DIR}"
  if [[ ! -f "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" ]]; then
    echo "Missing third_party/LIBERO. Run scripts/install_env.sh first." >&2
    exit 2
  fi
  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    "${PYTHON}" "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" \
      --download-dir "${LIBERO_DATASET_DIR}" \
      --datasets "${suite}" --use-huggingface
  done
fi

if [[ "${DOWNLOAD_CALVIN}" == "1" ]]; then
  echo "[download_assets] CALVIN datasets"
  CALVIN_BASE_URL="${CALVIN_BASE_URL:-http://calvin.cs.uni-freiburg.de/dataset}"
  CALVIN_TASKS="${CALVIN_TASKS:-task_ABCD_D}"
  CALVIN_DIR="${CALVIN_DIR:-${DVLA_DATA_ROOT}/datasets/calvin}"
  mkdir -p "${CALVIN_DIR}"
  for task in $(normalize_list "${CALVIN_TASKS}"); do
    [[ -n "${task}" ]] || continue
    zip_path="${CALVIN_DIR}/${task}.zip"
    if [[ ! -f "${zip_path}" ]]; then
      curl -L -C - "${CALVIN_BASE_URL}/${task}.zip" -o "${zip_path}"
    fi
    if [[ "${EXTRACT_CALVIN:-0}" == "1" ]]; then
      "${PYTHON}" -m zipfile -e "${zip_path}" "${CALVIN_DIR}/${task}"
    fi
  done
fi

echo "[download_assets] complete"
