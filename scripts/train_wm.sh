#!/usr/bin/env bash
# ============================================================================
#  World-model training
# ============================================================================
#  Picks a WM recipe via $CONFIG; the LIBERO task lives inside the config
#  (default: libero_goal). Everything else is a YAML default — override on
#  the Hydra CLI as trailing args.
#
#  Available CONFIGs:
#    world_model_dinowm_step   (default)   DINO-WM, per-frame predictor
#    world_model_dinowm_chunk              DINO-WM, K-step chunk predictor
#    world_model_rssm_step                 RSSM action-hidden WM
#
#  Examples:
#    bash scripts/train_wm.sh
#    CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh
#    NGPU=4 bash scripts/train_wm.sh task=libero_object
#    OUT_DIR=/tmp/smoke bash scripts/train_wm.sh \
#        training.max_steps=1 dataloader.num_workers=0
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- defaults --------------------------------------------------------------
CONFIG="${CONFIG:-world_model_dinowm_step}"
NGPU="${NGPU:-1}"
PYTHON="${PYTHON:-python}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- env -------------------------------------------------------------------
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- launch ----------------------------------------------------------------
echo "[train_wm] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[train_wm] out_dir=${OUT_DIR:-<config default: data/outputs/worldmodel/.../<timestamp>>}"
echo "[train_wm] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m src.cli.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m src.cli.train --config-name "${CONFIG}" "$@"
fi
