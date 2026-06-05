#!/usr/bin/env bash
# Download model weights and benchmark datasets used by formal DreamerVLA flows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${DVLA_ROOT}"

RYNNVLA_REPO="${RYNNVLA_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}"
WORLDVLA_REPO="${WORLDVLA_REPO:-Alibaba-DAMO-Academy/WorldVLA}"
LUMINA_REPO="${LUMINA_REPO:-Alpha-VLLM/Lumina-mGPT-7B-768}"
LIBERO_SUITES="${LIBERO_SUITES:-libero_goal libero_object libero_spatial libero_10}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_LIBERO="${DOWNLOAD_LIBERO:-1}"
DOWNLOAD_CALVIN="${DOWNLOAD_CALVIN:-0}"
DOWNLOAD_ACTION_WM="${DOWNLOAD_ACTION_WM:-1}"

mkdir -p "${DVLA_ROOT}/data/ckpts"

normalize_list() {
  printf '%s\n' "$1" | tr ',' ' '
}

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  echo "[download_assets] Hugging Face weights"
  hf download "${WORLDVLA_REPO}" --repo-type model \
    --local-dir "${DVLA_ROOT}/data/ckpts" \
    --include "chameleon/tokenizer/*" "chameleon/base_model/*" "base_model/*" "chameleon/starting_point/*"

  hf download "${LUMINA_REPO}" --repo-type model \
    --local-dir "${DVLA_ROOT}/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"

  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    hf download "${RYNNVLA_REPO}" --repo-type model \
      --local-dir "${DVLA_ROOT}/data/ckpts" \
      --include "VLA_model_256/${suite}/*"
    if [[ "${DOWNLOAD_ACTION_WM}" == "1" ]]; then
      hf download "${RYNNVLA_REPO}" --repo-type model \
        --local-dir "${DVLA_ROOT}/data/ckpts" \
        --include "Action_World_model_512/${suite}/*"
    fi
  done
fi

if [[ "${DOWNLOAD_LIBERO}" == "1" ]]; then
  echo "[download_assets] LIBERO datasets"
  if [[ ! -f "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" ]]; then
    echo "Missing third_party/LIBERO. Run scripts/install_env.sh first." >&2
    exit 2
  fi
  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    "${PYTHON}" "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" \
      --datasets "${suite}" --use-huggingface
  done
fi

if [[ "${DOWNLOAD_CALVIN}" == "1" ]]; then
  echo "[download_assets] CALVIN datasets"
  CALVIN_BASE_URL="${CALVIN_BASE_URL:-http://calvin.cs.uni-freiburg.de/dataset}"
  CALVIN_TASKS="${CALVIN_TASKS:-task_ABCD_D}"
  CALVIN_DIR="${CALVIN_DIR:-${DVLA_ROOT}/data/dataset/calvin}"
  mkdir -p "${CALVIN_DIR}"
  for task in $(normalize_list "${CALVIN_TASKS}"); do
    [[ -n "${task}" ]] || continue
    archive="${CALVIN_DIR}/${task}.zip"
    if [[ ! -f "${archive}" ]]; then
      curl -L -C - "${CALVIN_BASE_URL}/${task}.zip" -o "${archive}"
    fi
    if [[ "${EXTRACT_CALVIN:-0}" == "1" ]]; then
      python -m zipfile -e "${archive}" "${CALVIN_DIR}/${task}"
    fi
  done
fi

echo "[download_assets] complete"
