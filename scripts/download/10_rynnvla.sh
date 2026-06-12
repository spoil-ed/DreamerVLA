#!/usr/bin/env bash
# Download RynnVLA-002 weights used by DreamerVLA.
#
# Run all RynnVLA-related weights:
#   bash scripts/download/10_rynnvla.sh
#
# Download only one part by disabling the others:
#   DOWNLOAD_RYNNVLA_CHAMELEON=1 DOWNLOAD_RYNNVLA_LUMINA=0 DOWNLOAD_RYNNVLA_VLA=0 DOWNLOAD_ACTION_WM=0 bash scripts/download/10_rynnvla.sh
#   DOWNLOAD_RYNNVLA_CHAMELEON=0 DOWNLOAD_RYNNVLA_LUMINA=1 DOWNLOAD_RYNNVLA_VLA=0 DOWNLOAD_ACTION_WM=0 bash scripts/download/10_rynnvla.sh
#   DOWNLOAD_RYNNVLA_CHAMELEON=0 DOWNLOAD_RYNNVLA_LUMINA=0 DOWNLOAD_RYNNVLA_VLA=1 DOWNLOAD_ACTION_WM=0 LIBERO_SUITES=libero_goal bash scripts/download/10_rynnvla.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

DOWNLOAD_RYNNVLA_CHAMELEON="${DOWNLOAD_RYNNVLA_CHAMELEON:-1}"
DOWNLOAD_RYNNVLA_LUMINA="${DOWNLOAD_RYNNVLA_LUMINA:-1}"
DOWNLOAD_RYNNVLA_VLA="${DOWNLOAD_RYNNVLA_VLA:-1}"

mkdir -p "${CHECKPOINT_DIR}"

download_log "rynnvla chameleon=${DOWNLOAD_RYNNVLA_CHAMELEON} lumina=${DOWNLOAD_RYNNVLA_LUMINA} vla=${DOWNLOAD_RYNNVLA_VLA} action_wm=${DOWNLOAD_ACTION_WM}"

if [[ "${DOWNLOAD_RYNNVLA_CHAMELEON}" == "1" ]]; then
  # Weight 1: RynnVLA Chameleon tokenizer/VQGAN plus upstream-compatible
  # base/starting-point assets. RynnVLA-002 documents these under the older
  # Alibaba-DAMO-Academy/WorldVLA HF repo; DreamerVLA configs directly consume
  # checkpoints/chameleon/tokenizer/{text_tokenizer.json,vqgan.yaml,vqgan.ckpt}.
  download_log "RynnVLA Chameleon assets -> ${CHECKPOINT_DIR}/chameleon"
  hf download "${RYNNVLA_CHAMELEON_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}" \
    --include "chameleon/tokenizer/*" "chameleon/base_model/*" "base_model/*" "chameleon/starting_point/*"
fi

if [[ "${DOWNLOAD_RYNNVLA_LUMINA}" == "1" ]]; then
  # Weight 2: Lumina-mGPT tokenizer/backbone directory used by task.tokenizer_path
  # and RynnVLA pretokenization.
  download_log "Lumina-mGPT tokenizer/backbone -> ${CHECKPOINT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"
  hf download "${LUMINA_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"
fi

if [[ "${DOWNLOAD_RYNNVLA_VLA}" == "1" ]]; then
  # Weight 3: RynnVLA-002 VLA action-head checkpoints used by task.vla_ckpt_path.
  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    download_log "RynnVLA VLA_model_256/${suite} -> ${CHECKPOINT_DIR}/VLA_model_256/${suite}"
    hf download "${RYNNVLA_REPO}" --repo-type model \
      --local-dir "${CHECKPOINT_DIR}" \
      --include "VLA_model_256/${suite}/*"
  done
fi

if [[ "${DOWNLOAD_ACTION_WM}" == "1" ]]; then
  # Weight 4: RynnVLA-002 action world-model checkpoints used as optional
  # world-model initialization.
  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    download_log "RynnVLA Action_World_model_512/${suite} -> ${CHECKPOINT_DIR}/Action_World_model_512/${suite}"
    hf download "${RYNNVLA_REPO}" --repo-type model \
      --local-dir "${CHECKPOINT_DIR}" \
      --include "Action_World_model_512/${suite}/*"
  done
fi
