#!/usr/bin/env bash
# Validate the generated LIBERO preprocessing artifact tree for one suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

args=(
  --data-root "${DVLA_DATA_ROOT}"
  --processed-data-root "${PROCESSED_DATA_ROOT}"
  --suites "${TASK}"
  --his "${HIS}"
  --action-horizon "${ACTION_HORIZON}"
  --image-resolution "${IMAGE_RESOLUTION}"
)
if [[ "${VALIDATE_ACTION_HIDDEN}" == "1" ]]; then
  args+=(--check-action-hidden)
fi

"${PYTHON}" -m dreamer_vla.preprocess.validate_libero_data_prep "${args[@]}" "$@"
