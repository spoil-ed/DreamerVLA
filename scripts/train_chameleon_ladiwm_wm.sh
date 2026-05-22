#!/usr/bin/env bash
# SECONDARY / ABLATION wrapper for Chameleon/LaDiWM-style world-model training.
# The current mainline is scripts/train_pi0_action_hidden_dreamerv3_wm.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_KIND="${WM_KIND:-chameleon}" exec "${SCRIPT_DIR}/train_wm.sh" "$@"
