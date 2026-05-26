#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${PROJECT_ROOT}/data"
RYNNVLA_REPO="${RYNNVLA_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}"
WORLDVLA_REPO="${WORLDVLA_REPO:-Alibaba-DAMO-Academy/WorldVLA}"
LUMINA_REPO="${LUMINA_REPO:-Alpha-VLLM/Lumina-mGPT-7B-768}"
LIBERO_SUITES="${LIBERO_SUITES:-libero_goal}"
DOWNLOAD_ACTION_WM="${DOWNLOAD_ACTION_WM:-1}"

mkdir -p "${ROOT_DIR}/ckpts"
mkdir -p "${ROOT_DIR}/ckpts/chameleon/tokenizer"
mkdir -p "${ROOT_DIR}/ckpts/chameleon/base_model"
mkdir -p "${ROOT_DIR}/ckpts/starting_point"

IFS=',' read -r -a SUITES <<< "${LIBERO_SUITES}"
for suite in "${SUITES[@]}"; do
  suite="$(echo "${suite}" | xargs)"
  if [[ -z "${suite}" ]]; then
    continue
  fi

  echo "Downloading RynnVLA VLA_model_256/${suite} ..."
  hf download "${RYNNVLA_REPO}" \
    --repo-type model \
    --local-dir "${ROOT_DIR}/ckpts" \
    --include "VLA_model_256/${suite}/*"

  if [[ "${DOWNLOAD_ACTION_WM}" == "1" ]]; then
    echo "Downloading RynnVLA Action_World_model_512/${suite} ..."
    hf download "${RYNNVLA_REPO}" \
      --repo-type model \
      --local-dir "${ROOT_DIR}/ckpts" \
      --include "Action_World_model_512/${suite}/*"
  fi
done

hf download "${WORLDVLA_REPO}" \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/chameleon/tokenizer" \
  --include "chameleon/tokenizer/*"

hf download "${WORLDVLA_REPO}" \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/chameleon/base_model" \
  --include "base_model/*"

hf download "${WORLDVLA_REPO}" \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/starting_point" \
  --include "chameleon/starting_point/*"

hf download "${LUMINA_REPO}" \
  --repo-type model \
  --local-dir "${ROOT_DIR}/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"
