#!/usr/bin/env bash
# ============================================================================
#  DreamerVLA training (joint WM SFT + actor-critic / PPO)
# ============================================================================
#  $CONFIG picks the joint-training route; the LIBERO task lives inside the
#  config (default: libero_goal). Override anything on the Hydra CLI.
#
#  Available CONFIGs:
#    dreamervla_pi0_action_hidden_head_actor   (default)   RSSM + pi0 head actor
#    dreamervla_rynn_dino_wm_actor_critic                  DINO-WM + DreamerV3 AC
#    dreamervla_rynn_dino_wm_wmpo_outcome                  DINO-WM + WMPO outcome PPO
#    dreamervla_oft_dino_wm_wmpo_outcome                   OpenVLA-OFT DINO-WM + WMPO outcome PPO
#
#  The OFT variant requires a pre-trained classifier checkpoint:
#    1. CONFIG=oft_latent_classifier_chunk bash scripts/train_wm.sh   → produces .ckpt
#    2. CONFIG=dreamervla_oft_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh \
#         init.classifier_state_ckpt=<path-from-step-1>
#
#  Examples:
#    bash scripts/train_dreamervla.sh
#    CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh
#    CONFIG=dreamervla_oft_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh \
#        task=libero_goal init.classifier_state_ckpt=path/to/classifier.ckpt
#    NGPU=4 CONFIG=dreamervla_rynn_dino_wm_actor_critic \
#        bash scripts/train_dreamervla.sh task=libero_object
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- defaults --------------------------------------------------------------
CONFIG="${CONFIG:-dreamervla_pi0_action_hidden_head_actor}"
NGPU="${NGPU:-1}"
PYTHON="${PYTHON:-python}"
MASTER_PORT="${MASTER_PORT:-29502}"

# ---- env -------------------------------------------------------------------
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- launch ----------------------------------------------------------------
echo "[train_dreamervla] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[train_dreamervla] out_dir=${OUT_DIR:-<config default: data/outputs/dreamervla/.../<timestamp>>}"
echo "[train_dreamervla] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m src.cli.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m src.cli.train --config-name "${CONFIG}" "$@"
fi
