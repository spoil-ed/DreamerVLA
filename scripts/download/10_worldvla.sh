#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

CHECKPOINT_DIR="${DVLA_DATA_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

download_log "WorldVLA tokenizer and base weights -> ${CHECKPOINT_DIR}"
hf download "${WORLDVLA_REPO}" --repo-type model \
  --local-dir "${CHECKPOINT_DIR}" \
  --include "chameleon/tokenizer/*" "chameleon/base_model/*" "base_model/*" "chameleon/starting_point/*"
