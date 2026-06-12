#!/usr/bin/env bash
# ============================================================================
#  DreamerVLA training (joint WM SFT + actor-critic / PPO)
# ============================================================================
#  $CONFIG picks the joint-training route; the LIBERO task lives inside the
#  config (default: libero_goal). Override anything on the Hydra CLI.
#
#  Available CONFIGs:
#    dreamervla_rynn_dino_wm_wmpo_outcome (default) DINO-WM + WMPO outcome PPO
#    dreamervla_rynn_dino_wm_actor_critic           DINO-WM + DreamerV3 AC
#    dreamervla_oft_dino_wm_wmpo_outcome            OpenVLA-OFT DINO-WM + WMPO outcome PPO
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

# ---- environment -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
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
cd "${DVLA_ROOT}"

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

# ---- knobs -------------------------------------------------------------------
CONFIG="${CONFIG:-dreamervla_rynn_dino_wm_wmpo_outcome}"
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"

# ---- launch ------------------------------------------------------------------
echo "[train_dreamervla] python=$(command -v "${PYTHON}")"
echo "[train_dreamervla] root=${DVLA_ROOT}  data_root=${DVLA_DATA_ROOT}"
echo "[train_dreamervla] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[train_dreamervla] out_dir=${OUT_DIR:-<config default: \${DVLA_DATA_ROOT}/outputs/dreamervla/.../<timestamp>>}"
echo "[train_dreamervla] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m dreamer_vla.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m dreamer_vla.train --config-name "${CONFIG}" "$@"
fi
