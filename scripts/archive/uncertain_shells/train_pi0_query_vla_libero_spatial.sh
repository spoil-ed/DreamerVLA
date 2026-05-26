#!/usr/bin/env bash
# Run the pi0-query VLA non-goal pipeline for LIBERO-spatial only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUITES="${SUITES:-libero_spatial}"
export MASTER_PORT="${MASTER_PORT:-29549}"

exec bash "${SCRIPT_DIR}/train_pi0_query_vla_nongoal.sh" "$@"
