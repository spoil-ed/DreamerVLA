#!/usr/bin/env bash
# Download RynnVLA-002 weights used by DreamerVLA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
CHECKPOINT_DIR="${DVLA_DATA_ROOT}/checkpoints"
RYNNVLA_CHAMELEON_REPO="${RYNNVLA_CHAMELEON_REPO:-Alibaba-DAMO-Academy/WorldVLA}"
LUMINA_REPO="${LUMINA_REPO:-Alpha-VLLM/Lumina-mGPT-7B-768}"
RYNNVLA_REPO="${RYNNVLA_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}"
SUITE="${SUITE:-libero_goal}"
DOWNLOAD_RYNNVLA_CHAMELEON="${DOWNLOAD_RYNNVLA_CHAMELEON:-1}"
DOWNLOAD_RYNNVLA_LUMINA="${DOWNLOAD_RYNNVLA_LUMINA:-1}"
DOWNLOAD_RYNNVLA_VLA="${DOWNLOAD_RYNNVLA_VLA:-1}"
DOWNLOAD_ACTION_WM="${DOWNLOAD_ACTION_WM:-1}"
cd "${DVLA_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"
echo "[download:10_rynnvla] chameleon=${DOWNLOAD_RYNNVLA_CHAMELEON} lumina=${DOWNLOAD_RYNNVLA_LUMINA} vla=${DOWNLOAD_RYNNVLA_VLA} action_wm=${DOWNLOAD_ACTION_WM}"

if [[ "${DOWNLOAD_RYNNVLA_CHAMELEON}" == "1" ]]; then
  hf download "${RYNNVLA_CHAMELEON_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}" \
    --include "chameleon/tokenizer/*" "chameleon/base_model/*" "base_model/*" "chameleon/starting_point/*"
fi

if [[ "${DOWNLOAD_RYNNVLA_LUMINA}" == "1" ]]; then
  hf download "${LUMINA_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"
fi

if [[ "${DOWNLOAD_RYNNVLA_VLA}" == "1" && -n "${SUITE}" ]]; then
  hf download "${RYNNVLA_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}" \
    --include "VLA_model_256/${SUITE}/*"
fi

if [[ "${DOWNLOAD_ACTION_WM}" == "1" && -n "${SUITE}" ]]; then
  hf download "${RYNNVLA_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}" \
    --include "Action_World_model_512/${SUITE}/*"
fi
