#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

mkdir -p "${CHECKPOINT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"

download_log "Lumina tokenizer/backbone -> ${CHECKPOINT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"
hf download "${LUMINA_REPO}" --repo-type model \
  --local-dir "${CHECKPOINT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"
