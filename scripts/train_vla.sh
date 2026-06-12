#!/usr/bin/env bash
# ============================================================================
#  VLA SFT training
# ============================================================================
#  $CONFIG picks the VLA recipe; the LIBERO task lives inside the config
#  (default: libero_goal). Override anything on the Hydra CLI.
#
#  Available CONFIGs:
#    vla_rynnvla_action_head             (default)   RynnVLA action head, pretokenize SFT
#    vla_sft_one_trajectory                          RynnVLA action head, one demo trajectory per task
#    openvla_oft_hdf5                                OpenVLA-OFT SFT on raw HDF5
#    openvla_oft_hdf5_one_trajectory                 OpenVLA-OFT LM-head action-token SFT, one random demo per task
#
#  Examples:
#    bash scripts/train_vla.sh
#    bash scripts/train_vla.sh task=libero_object
#    NGPU=4 bash scripts/train_vla.sh task=libero_10 training.num_epochs=5
#    CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_goal
#    CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh \
#        dataset.trajectory_offset=3
#    CONFIG=openvla_oft_hdf5 bash scripts/train_vla.sh task=libero_goal
#    CONFIG=openvla_oft_hdf5_one_trajectory bash scripts/train_vla.sh task=libero_goal
#    OUT_DIR=data/outputs/vla/rynnvla_action_head/libero_object_run1 \
#        bash scripts/train_vla.sh task=libero_object
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
CONFIG="${CONFIG:-vla_rynnvla_action_head}"
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

# ---- launch ------------------------------------------------------------------
echo "[train_vla] python=$(command -v "${PYTHON}")"
echo "[train_vla] root=${DVLA_ROOT}  data_root=${DVLA_DATA_ROOT}"
echo "[train_vla] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
if [[ -n "${OUT_DIR:-}" ]]; then
  echo "[train_vla] out_dir=${OUT_DIR}"
else
  echo "[train_vla] out_dir=<config default under \${DVLA_DATA_ROOT}/outputs/vla/.../<timestamp>>"
fi
echo "[train_vla] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m dreamer_vla.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m dreamer_vla.train --config-name "${CONFIG}" "$@"
fi
