#!/usr/bin/env bash
# ============================================================================
#  VLA SFT training
# ============================================================================
#  $CONFIG picks the VLA recipe; the LIBERO task lives inside the config
#  (default: libero_goal). Override anything on the Hydra CLI.
#
#  Available CONFIGs:
#    vla_pi0_query   (default)   pi0_query head, pretokenize SFT
#
#  Examples:
#    bash scripts/train_vla.sh
#    bash scripts/train_vla.sh task=libero_object
#    NGPU=4 bash scripts/train_vla.sh task=libero_10 training.num_epochs=5
#    OUT_DIR=data/outputs/vla/pi0_query/libero_object_run1 \
#        bash scripts/train_vla.sh task=libero_object
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- defaults --------------------------------------------------------------
CONFIG="${CONFIG:-vla_pi0_query}"
NGPU="${NGPU:-1}"
PYTHON="${PYTHON:-python}"
MASTER_PORT="${MASTER_PORT:-29501}"

# ---- env -------------------------------------------------------------------
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- launch ----------------------------------------------------------------
echo "[train_vla] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[train_vla] out_dir=${OUT_DIR:-<config default: data/outputs/vla/.../<timestamp>>}"
echo "[train_vla] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m src.cli.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m src.cli.train --config-name "${CONFIG}" "$@"
fi
