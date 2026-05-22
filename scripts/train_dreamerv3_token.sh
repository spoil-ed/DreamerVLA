#!/usr/bin/env bash
# SECONDARY baseline wrapper for DreamerV3 token world-model training.
# Current mainline uses pi0 action-hidden DreamerV3 WM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_KIND="${WM_KIND:-dreamerv3_token}" exec "${SCRIPT_DIR}/train_wm.sh" "$@"
