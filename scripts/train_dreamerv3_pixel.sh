#!/usr/bin/env bash
# SECONDARY baseline wrapper for DreamerV3 pixel world-model training.
# Current mainline uses pi0 action-hidden DreamerV3 WM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_KIND="${WM_KIND:-dreamerv3_pixel}" exec "${SCRIPT_DIR}/train_wm.sh" "$@"
