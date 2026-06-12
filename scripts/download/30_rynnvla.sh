#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

mkdir -p "${CHECKPOINT_DIR}"

for suite in $(normalize_list "${LIBERO_SUITES}"); do
  [[ -n "${suite}" ]] || continue
  download_log "RynnVLA action-head weights for ${suite} -> ${CHECKPOINT_DIR}/VLA_model_256/${suite}"
  hf download "${RYNNVLA_REPO}" --repo-type model \
    --local-dir "${CHECKPOINT_DIR}" \
    --include "VLA_model_256/${suite}/*"

  if [[ "${DOWNLOAD_ACTION_WM}" == "1" ]]; then
    download_log "RynnVLA action world-model weights for ${suite} -> ${CHECKPOINT_DIR}/Action_World_model_512/${suite}"
    hf download "${RYNNVLA_REPO}" --repo-type model \
      --local-dir "${CHECKPOINT_DIR}" \
      --include "Action_World_model_512/${suite}/*"
  fi
done
