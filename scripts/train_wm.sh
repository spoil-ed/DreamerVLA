#!/usr/bin/env bash
# ============================================================================
#  World-model training
# ============================================================================
#  Picks a WM recipe via $CONFIG; the LIBERO task lives inside the config
#  (default: libero_goal). Everything else is a YAML default — override on
#  the Hydra CLI as trailing args.
#
#  Available CONFIGs:
#    world_model_dinowm_chunk        (default)   DINO-WM, K-step chunk predictor
#    world_model_dinowm_step                     DINO-WM, per-frame predictor
#    oft_world_model_dinowm_chunk                 OpenVLA-OFT hidden (56x4096), chunk WM
#    latent_classifier_libero_goal_chunk          chunk latent success classifier
#    oft_latent_classifier_chunk                  OpenVLA-OFT chunk latent classifier
#
#  Examples:
#    bash scripts/train_wm.sh
#    CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh
#    CONFIG=oft_world_model_dinowm_chunk bash scripts/train_wm.sh task=libero_goal
#    CONFIG=oft_latent_classifier_chunk bash scripts/train_wm.sh task=libero_goal
#    NGPU=4 bash scripts/train_wm.sh task=libero_object
#    OUT_DIR=/tmp/smoke bash scripts/train_wm.sh \
#        training.max_steps=1 dataloader.num_workers=0
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${DVLA_ROOT}"

# ---- defaults --------------------------------------------------------------
CONFIG="${CONFIG:-world_model_dinowm_chunk}"
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- env -------------------------------------------------------------------

# ---- launch ----------------------------------------------------------------
echo "[train_wm] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[train_wm] out_dir=${OUT_DIR:-<config default: data/outputs/worldmodel/.../<timestamp>>}"
echo "[train_wm] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m dreamer_vla.cli.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m dreamer_vla.cli.train --config-name "${CONFIG}" "$@"
fi
