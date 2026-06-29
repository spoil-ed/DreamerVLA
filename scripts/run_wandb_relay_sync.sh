#!/usr/bin/env bash
# Template launcher for the shared-disk W&B offline sync helper.
#
# Run this on the ONLINE machine (the one that can `wandb login`). It uploads the
# W&B offline runs that the OFFLINE GPU machine wrote onto the shared disk, by
# running `wandb sync` on them directly -- no SSH, no rsync, no copy.
# Edit the placeholders below first.
#
# This script NEVER stores a W&B API key. Authenticate separately on this machine
# with `wandb login` (or export WANDB_API_KEY in your own shell, outside the repo).
set -euo pipefail

# --- EDIT THESE PLACEHOLDERS -------------------------------------------------
# Shared-disk `wandb/` dir that DIRECTLY contains offline-run-*; for this repo
# that is the innermost `wandb/` under <run>/cotrain/log/wandb/.../wandb :
WANDB_DIR="/shared/DreamerVLA/data/outputs/<run>/cotrain/log/wandb/all/wandb"
WANDB_PROJECT="dreamervla"
WANDB_ENTITY="your-wandb-entity"
INTERVAL="60"                              # seconds between rounds
LOG_FILE="${HOME}/wandb_relay_sync.log"
# -----------------------------------------------------------------------------

python -m dreamervla.diagnostics.wandb_relay_sync \
  --wandb-dir "${WANDB_DIR}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --interval "${INTERVAL}" \
  --log-file "${LOG_FILE}" \
  "$@"
