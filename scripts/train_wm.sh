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

# ---- environment -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# ---- LIBERO paths (datasets live under the data root) -----------------------
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/datasets/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi

cd "${DVLA_ROOT}"

# ---- knobs -------------------------------------------------------------------
CONFIG="${CONFIG:-world_model_dinowm_chunk}"
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- launch ------------------------------------------------------------------
echo "[train_wm] python=$(command -v "${PYTHON}")"
echo "[train_wm] root=${DVLA_ROOT}  data_root=${DVLA_DATA_ROOT}"
echo "[train_wm] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[train_wm] out_dir=${OUT_DIR:-<config default: \${DVLA_DATA_ROOT}/outputs/worldmodel/.../<timestamp>>}"
echo "[train_wm] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m dreamer_vla.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m dreamer_vla.train --config-name "${CONFIG}" "$@"
fi
