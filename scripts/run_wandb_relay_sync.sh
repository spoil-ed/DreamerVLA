#!/usr/bin/env bash
# Template launcher for the W&B offline relay sync helper.
#
# Run this on the CPU "relay" machine (networked box that can BOTH `wandb login`
# AND ssh into the air-gapped GPU box). Edit the placeholders below first.
#
# This script NEVER stores a W&B API key. Authenticate separately on this machine
# with `wandb login` (or export WANDB_API_KEY in your own shell, outside the repo).
set -euo pipefail

# --- EDIT THESE PLACEHOLDERS -------------------------------------------------
REMOTE_HOST="gpu-host.example.com"        # GPU training machine hostname / IP
REMOTE_USER="your-remote-user"            # SSH user on the GPU machine
# Remote dir that DIRECTLY contains offline-run-*; for this repo that is the
# `wandb/` dir written under <run>/cotrain/log/wandb/.../wandb :
REMOTE_WANDB_DIR="/path/to/DreamerVLA/data/outputs/<run>/cotrain/log/wandb"
LOCAL_MIRROR_DIR="${HOME}/wandb_relay_mirror"
WANDB_PROJECT="dreamervla"
WANDB_ENTITY="your-wandb-entity"
SSH_PORT="22"
INTERVAL="60"                              # seconds between rounds
LOG_FILE="${LOCAL_MIRROR_DIR}/wandb_relay.log"
# -----------------------------------------------------------------------------

python -m dreamervla.diagnostics.wandb_relay_sync \
  --remote-host "${REMOTE_HOST}" \
  --remote-user "${REMOTE_USER}" \
  --remote-wandb-dir "${REMOTE_WANDB_DIR}" \
  --local-mirror-dir "${LOCAL_MIRROR_DIR}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --ssh-port "${SSH_PORT}" \
  --interval "${INTERVAL}" \
  --log-file "${LOG_FILE}" \
  "$@"
