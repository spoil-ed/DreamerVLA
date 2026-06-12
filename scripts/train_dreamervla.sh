#!/usr/bin/env bash
# ============================================================================
#  DreamerVLA training (joint WM SFT + actor-critic / PPO)
# ============================================================================
#  $CONFIG picks the joint-training route; the LIBERO task lives inside the
#  config (default: libero_goal). Override anything on the Hydra CLI.
#
#  Available CONFIGs:
#    dreamervla_rynn_dino_wm_wmpo_outcome (default) DINO-WM + WMPO outcome PPO
#    dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens RynnVLA frame-token Scheme B
#    dreamervla_rynn_dino_wm_actor_critic           DINO-WM + DreamerV3 AC
#    dreamervla_oft_dino_wm_wmpo_outcome            OpenVLA-OFT DINO-WM + WMPO outcome PPO
#    dreamervla_oft_dino_wm_wmpo_outcome_input_tokens OpenVLA-OFT frame-token Scheme B
#
#  The OFT variant requires a pre-trained classifier checkpoint:
#    1. bash scripts/train_wm.sh --config oft_latent_classifier_chunk  → produces .ckpt
#    2. bash scripts/train_dreamervla.sh --config dreamervla_oft_dino_wm_wmpo_outcome \
#         init.classifier_state_ckpt=<path-from-step-1>
#
#  Examples:
#    bash scripts/train_dreamervla.sh
#    bash scripts/train_dreamervla.sh --config dreamervla_rynn_dino_wm_wmpo_outcome
#    bash scripts/train_dreamervla.sh --config dreamervla_oft_dino_wm_wmpo_outcome \
#        --task libero_goal init.classifier_state_ckpt=path/to/classifier.ckpt
#    bash scripts/train_dreamervla.sh --config dreamervla_rynn_dino_wm_actor_critic \
#        --task libero_object --gpus 0,1,2,3 --ngpu 4
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
NGPU="${NGPU:-}"
MASTER_PORT="${MASTER_PORT:-29502}"
HYDRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --task)
      HYDRA_ARGS+=("task=$2")
      shift 2
      ;;
    --gpus)
      export CUDA_VISIBLE_DEVICES="$2"
      if [[ -z "${NGPU}" ]]; then
        gpu_count=0
        for _gpu in ${2//,/ }; do gpu_count=$((gpu_count + 1)); done
        NGPU="${gpu_count}"
      fi
      shift 2
      ;;
    --ngpu)
      NGPU="$2"
      shift 2
      ;;
    --batch-size)
      HYDRA_ARGS+=("dataloader.batch_size=$2")
      shift 2
      ;;
    --num-workers)
      HYDRA_ARGS+=("dataloader.num_workers=$2")
      shift 2
      ;;
    --out-dir)
      export OUT_DIR="$2"
      HYDRA_ARGS+=("training.out_dir=$2")
      shift 2
      ;;
    --max-steps)
      HYDRA_ARGS+=("training.max_steps=$2")
      shift 2
      ;;
    --)
      shift
      HYDRA_ARGS+=("$@")
      break
      ;;
    *)
      HYDRA_ARGS+=("$1")
      shift
      ;;
  esac
done
NGPU="${NGPU:-1}"

# ---- launch ------------------------------------------------------------------
echo "[train_dreamervla] python=$(command -v "${PYTHON}")"
echo "[train_dreamervla] root=${DVLA_ROOT}  data_root=${DVLA_DATA_ROOT}"
echo "[train_dreamervla] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
if [[ -n "${OUT_DIR:-}" ]]; then
  echo "[train_dreamervla] out_dir=${OUT_DIR}"
else
  echo "[train_dreamervla] out_dir=<config default under \${DVLA_DATA_ROOT}/outputs/dreamervla/.../<timestamp>>"
fi
echo "[train_dreamervla] hydra args: ${HYDRA_ARGS[*]}"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m dreamer_vla.train --config-name "${CONFIG}" "${HYDRA_ARGS[@]}"
else
  exec "${PYTHON}" -m dreamer_vla.train --config-name "${CONFIG}" "${HYDRA_ARGS[@]}"
fi
