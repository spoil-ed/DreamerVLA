#!/usr/bin/env bash
# Unified world-model training entrypoint.
#
# This script launches all current WM trainers. The workspace/model is selected
# by the Hydra config; this wrapper only standardizes defaults, output layout,
# launch mode, and common smoke/sequence overrides.
#
# Examples:
#   # Current DreamerV3 token WM
#   WM_KIND=dreamerv3_token bash scripts/train_wm.sh
#
#   # DreamerV3 pixel WM
#   WM_KIND=dreamerv3_pixel bash scripts/train_wm.sh
#
#   # Original/pretokenized token WM ablation
#   CONFIG_NAME=pretokenize_wm_libero_10_obs4096_minloss_rssm bash scripts/train_wm.sh
#
#   # Chameleon/LaDiWM-style WM
#   WM_KIND=chameleon BATCH_SIZE=1 GRAD_ACCUM=3 bash scripts/train_wm.sh
#
#   # Smoke test for distributed pretokenize WM configs
#   WM_SMOKE=1 WM_KIND=pretokenize CUDA_VISIBLE_DEVICES=4 NUM_GPUS=1 bash scripts/train_wm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-}"

infer_kind() {
  local config="$1"
  if [[ -n "${WM_KIND:-}" ]]; then
    echo "${WM_KIND}"
  elif [[ "${config}" == dreamerv3_token* ]]; then
    echo "dreamerv3_token"
  elif [[ "${config}" == dreamerv3_pixel* ]]; then
    echo "dreamerv3_pixel"
  elif [[ "${config}" == rynn_backbone_* ]]; then
    echo "rynn_backbone"
  elif [[ "${config}" == chameleon_* || "${config}" == *chameleon* ]]; then
    echo "chameleon"
  else
    echo "pretokenize"
  fi
}

if [[ -z "${CONFIG_NAME:-}" ]]; then
  case "${WM_KIND:-pretokenize}" in
    dreamerv3_token|token)
      CONFIG_NAME="dreamerv3_token_libero_goal"
      ;;
    dreamerv3_pixel|pixel)
      CONFIG_NAME="dreamerv3_pixel_libero_goal"
      ;;
    chameleon|ladiwm)
      CONFIG_NAME="chameleon_latent_action_wm_libero_goal"
      ;;
    rynn_backbone|rynn|strategy3)
      CONFIG_NAME="rynn_backbone_dreamerv3_pixel_wm_libero_goal"
      ;;
    pretokenize|tssm|rssm|transdreamer|"")
      if [[ "${WARMUP:-0}" == "1" || "${WM_WARMUP:-0}" == "1" ]]; then
        CONFIG_NAME="pretokenize_wm_libero_goal_warmup"
      else
        CONFIG_NAME="pretokenize_wm_libero_goal_transdreamer"
      fi
      ;;
    *)
    echo "ERROR: unknown WM_KIND='${WM_KIND}'. Use pretokenize, dreamerv3_token, dreamerv3_pixel, chameleon, or rynn_backbone." >&2
      exit 2
      ;;
  esac
fi

WM_KIND_RESOLVED="$(infer_kind "${CONFIG_NAME}")"
case "${WM_KIND_RESOLVED}" in
  token)
    WM_KIND_RESOLVED="dreamerv3_token"
    ;;
  pixel)
    WM_KIND_RESOLVED="dreamerv3_pixel"
    ;;
  ladiwm)
    WM_KIND_RESOLVED="chameleon"
    ;;
  rynn|strategy3)
    WM_KIND_RESOLVED="rynn_backbone"
    ;;
esac

case "${WM_KIND_RESOLVED}" in
  dreamerv3_token)
    DEFAULT_OUT_DIR_BASE="${PROJECT_ROOT}/data/outputs/worldmodel/dreamerv3_token"
    DEFAULT_LAUNCHER="single"
    ;;
  dreamerv3_pixel)
    DEFAULT_OUT_DIR_BASE="${PROJECT_ROOT}/data/outputs/worldmodel/dreamerv3_pixel"
    if [[ "${DREAMERV3_PIXEL_DDP:-${PIXEL_DDP:-${DDP:-0}}}" == "1" ]]; then
      DEFAULT_LAUNCHER="torchrun"
    else
      DEFAULT_LAUNCHER="single"
    fi
    ;;
  chameleon)
    DEFAULT_OUT_DIR_BASE="${PROJECT_ROOT}/data/outputs/worldmodel/chameleon_latent_action_wm"
    DEFAULT_LAUNCHER="torchrun"
    ;;
  rynn_backbone)
    DEFAULT_OUT_DIR_BASE="${PROJECT_ROOT}/data/outputs/worldmodel/rynn_backbone_dreamerv3_wm"
    if [[ "${RYNN_PIXEL_DDP:-${RYNN_BACKBONE_DDP:-${DDP:-0}}}" == "1" ]]; then
      DEFAULT_LAUNCHER="torchrun"
    else
      DEFAULT_LAUNCHER="single"
    fi
    ;;
  pretokenize|tssm|rssm|transdreamer)
    DEFAULT_OUT_DIR_BASE="${PROJECT_ROOT}/data/outputs/worldmodel/pretokenize_wm"
    DEFAULT_LAUNCHER="torchrun"
    ;;
  *)
    echo "ERROR: could not infer supported WM kind from CONFIG_NAME='${CONFIG_NAME}'." >&2
    exit 2
    ;;
esac

if [[ "${WM_KIND_RESOLVED}" == "dreamerv3_pixel" && "${DREAMERV3_PIXEL_DDP:-${PIXEL_DDP:-${DDP:-0}}}" == "1" && -z "${RUN_TAG}" ]]; then
  RUN_TAG="ddp_bs${BATCH_SIZE:-64}_nw${NUM_WORKERS:-16}_vizoff"
fi

if [[ "${WM_KIND_RESOLVED}" == "rynn_backbone" && "${RYNN_PIXEL_DDP:-${RYNN_BACKBONE_DDP:-${DDP:-0}}}" == "1" && -z "${RUN_TAG}" ]]; then
  if [[ "${CONFIG_NAME}" == *precomputed* ]]; then
    RUN_TAG="ddp_precomputed_bs${BATCH_SIZE:-96}_nw${NUM_WORKERS:-8}_viz"
  else
    RUN_TAG="ddp_bs${BATCH_SIZE:-2}_nw${NUM_WORKERS:-8}_rynn"
  fi
fi

if [[ "${WM_KIND_RESOLVED}" == "chameleon" && -z "${RUN_TAG}" ]]; then
  RUN_TAG="ladiwm_like_t4_tokens"
fi

OUT_DIR_BASE="${OUT_DIR_BASE:-${DEFAULT_OUT_DIR_BASE}}"
if [[ -n "${RUN_TAG}" ]]; then
  DEFAULT_RUN_NAME="${CONFIG_NAME}_${RUN_TAG}_${TIMESTAMP}"
else
  DEFAULT_RUN_NAME="${CONFIG_NAME}_${TIMESTAMP}"
fi
OUT_DIR="${OUT_DIR:-${OUT_DIR_BASE}/${DEFAULT_RUN_NAME}}"
LAUNCHER="${LAUNCHER:-${DEFAULT_LAUNCHER}}"
CONFIG_FILE="${PROJECT_ROOT}/configs/${CONFIG_NAME}.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: config not found: ${CONFIG_FILE}" >&2
  exit 2
fi

COMMON_OVERRIDES=(
  "training.out_dir=${OUT_DIR}"
)

SEQUENCE_OVERRIDES=()
WM_T="${WM_T:-}"
WM_STRIDE="${WM_STRIDE:-}"
if [[ -n "${WM_T}" ]]; then
  case "${WM_KIND_RESOLVED}" in
    pretokenize|tssm|rssm|transdreamer)
      SEQUENCE_OVERRIDES=(
        "++dataset.sequence_length=${WM_T}"
        "++dataset_val_ind.sequence_length=${WM_T}"
        "++dataset_val_ood.sequence_length=${WM_T}"
      )
      ;;
    dreamerv3_pixel|rynn_backbone)
      SEQUENCE_OVERRIDES=(
        "dataset.sequence_length=${WM_T}"
      )
      if [[ -n "${WM_STRIDE}" ]]; then
        SEQUENCE_OVERRIDES+=("dataset.stride=${WM_STRIDE}")
      fi
      ;;
  esac
fi

SMOKE_OVERRIDES=()
if [[ "${WM_SMOKE:-0}" == "1" ]]; then
  SMOKE_OVERRIDES=(
    "training.num_epochs=1"
    "dataloader.batch_size=${WM_SMOKE_BATCH_SIZE:-1}"
    "dataloader.num_workers=0"
    "dataloader.persistent_workers=false"
    "dataloader.pin_memory=false"
  )
  if [[ "${WM_KIND_RESOLVED}" == "dreamerv3_pixel" || "${WM_KIND_RESOLVED}" == "dreamerv3_token" || "${WM_KIND_RESOLVED}" == "rynn_backbone" ]]; then
    SMOKE_OVERRIDES+=(
      "training.max_steps=${WM_SMOKE_STEPS:-1}"
      "viz.enabled=false"
    )
  elif [[ "${WM_KIND_RESOLVED}" == "pretokenize" || "${WM_KIND_RESOLVED}" == "tssm" || "${WM_KIND_RESOLVED}" == "rssm" || "${WM_KIND_RESOLVED}" == "transdreamer" ]]; then
    SMOKE_OVERRIDES+=(
      "training.max_train_steps=${WM_SMOKE_STEPS:-1}"
      "dataset_val_ind=null"
      "dataset_val_ood=null"
      "viz.enabled=false"
      "checkpoint.save_last_ckpt=false"
      "checkpoint.topk.k=0"
    )
  elif [[ "${WM_KIND_RESOLVED}" == "chameleon" ]]; then
    SMOKE_OVERRIDES+=(
      "training.max_train_steps=${WM_SMOKE_STEPS:-1}"
      "dataset_val_ind=null"
      "dataset_val_ood=null"
      "checkpoint.save_last_ckpt=false"
      "checkpoint.topk.k=0"
    )
  fi
  if [[ "${WM_KIND_RESOLVED}" == "rynn_backbone" ]]; then
    SMOKE_OVERRIDES+=(
      "dataset.sequence_length=${WM_SMOKE_T:-2}"
      "dataset.stride=${WM_SMOKE_STRIDE:-8}"
      "training.encoder_chunk_size=${WM_SMOKE_ENCODER_CHUNK_SIZE:-2}"
    )
  fi
fi

KIND_OVERRIDES=()
if [[ "${WM_KIND_RESOLVED}" == "dreamerv3_pixel" ]]; then
  if [[ "${DREAMERV3_PIXEL_DDP:-${PIXEL_DDP:-${DDP:-0}}}" == "1" ]]; then
    # Aggressive H100-oriented defaults. In DDP these are per-rank values, so
    # global batch = BATCH_SIZE * NUM_GPUS.
    BATCH_SIZE="${BATCH_SIZE:-64}"
    NUM_WORKERS="${NUM_WORKERS:-16}"
    KIND_OVERRIDES=(
      "training.distributed_strategy=ddp"
      "training.data_parallel=false"
      "dataloader.batch_size=${BATCH_SIZE}"
      "dataloader.num_workers=${NUM_WORKERS}"
      "dataloader.persistent_workers=true"
      "dataloader.pin_memory=true"
      "viz.enabled=false"
    )
  else
    if [[ -n "${BATCH_SIZE:-}" ]]; then
      KIND_OVERRIDES+=("dataloader.batch_size=${BATCH_SIZE}")
    fi
    if [[ -n "${NUM_WORKERS:-}" ]]; then
      KIND_OVERRIDES+=("dataloader.num_workers=${NUM_WORKERS}")
    fi
  fi
elif [[ "${WM_KIND_RESOLVED}" == "chameleon" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  GRAD_ACCUM="${GRAD_ACCUM:-3}"
  NUM_WORKERS="${NUM_WORKERS:-8}"
  EPOCHS="${EPOCHS:-51}"
  LR="${LR:-5.0e-5}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
  GRAD_CLIP="${GRAD_CLIP:-10.0}"
  LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
  KIND_OVERRIDES=(
    "training.num_epochs=${EPOCHS}"
    "training.gradient_accumulate_every=${GRAD_ACCUM}"
    "training.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "dataloader.batch_size=${BATCH_SIZE}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "optim.grad_clip_norm=${GRAD_CLIP}"
    "optim.world_model.lr=${LR}"
    "optim.world_model.weight_decay=${WEIGHT_DECAY}"
  )
elif [[ "${WM_KIND_RESOLVED}" == "rynn_backbone" ]]; then
  if [[ "${RYNN_PIXEL_DDP:-${RYNN_BACKBONE_DDP:-${DDP:-0}}}" == "1" ]]; then
    if [[ "${CONFIG_NAME}" == *precomputed* ]]; then
      # Precomputed hidden avoids the frozen VLA backbone during training and
      # can use pure-pixel-like batches.
      BATCH_SIZE="${BATCH_SIZE:-96}"
      NUM_WORKERS="${NUM_WORKERS:-2}"
    else
      # Online Rynn-pixel carries a frozen VLA backbone on each rank, so the
      # useful per-GPU batch is much smaller than pure pixel DreamerV3.
      BATCH_SIZE="${BATCH_SIZE:-2}"
      NUM_WORKERS="${NUM_WORKERS:-2}"
    fi
    RYNN_ENCODER_CHUNK_SIZE="${RYNN_ENCODER_CHUNK_SIZE:-8}"
  else
    BATCH_SIZE="${BATCH_SIZE:-1}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    RYNN_ENCODER_CHUNK_SIZE="${RYNN_ENCODER_CHUNK_SIZE:-8}"
  fi
  PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
  PIN_MEMORY="${PIN_MEMORY:-false}"
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
  DATALOADER_MP_CONTEXT="${DATALOADER_MP_CONTEXT:-forkserver}"
  LR="${LR:-4.0e-5}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
  GRAD_CLIP="${GRAD_CLIP:-100.0}"
  LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
  KIND_OVERRIDES=(
    "training.distributed_strategy=$([[ "${RYNN_PIXEL_DDP:-${RYNN_BACKBONE_DDP:-${DDP:-0}}}" == "1" ]] && echo ddp || echo single)"
    "training.data_parallel=$([[ "${RYNN_PIXEL_DDP:-${RYNN_BACKBONE_DDP:-${DDP:-0}}}" == "1" ]] && echo false || echo true)"
    "training.encoder_chunk_size=${RYNN_ENCODER_CHUNK_SIZE}"
    "dataloader.batch_size=${BATCH_SIZE}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "dataloader.persistent_workers=${PERSISTENT_WORKERS}"
    "dataloader.pin_memory=${PIN_MEMORY}"
    "dataloader.prefetch_factor=${PREFETCH_FACTOR}"
    "dataloader.multiprocessing_context=${DATALOADER_MP_CONTEXT}"
    "optim.lr=${LR}"
    "optim.weight_decay=${WEIGHT_DECAY}"
    "optim.grad_clip=${GRAD_CLIP}"
    "optim.warmup=${LR_WARMUP_STEPS}"
  )
  if [[ -n "${VLA_INIT_CKPT:-}" ]]; then
    KIND_OVERRIDES+=(
      "init.vla_ckpt_path=${VLA_INIT_CKPT}"
      "encoder.model_path=${VLA_INIT_CKPT}"
    )
  fi
  if [[ -n "${ACTION_HORIZON:-${TIME_HORIZON:-}}" ]]; then
    KIND_OVERRIDES+=("encoder.time_horizon=${ACTION_HORIZON:-${TIME_HORIZON}}")
  fi
  if [[ "${CONFIG_NAME}" == *precomputed* && -n "${RYNN_HIDDEN_DIR:-}" ]]; then
    KIND_OVERRIDES+=(
      "dataset.hidden_dir=${RYNN_HIDDEN_DIR}"
      "dataset.expected_model_path=${VLA_INIT_CKPT:-}"
      "dataset.expected_encoder_state_ckpt=${VLA_STATE_CKPT:-${ENCODER_STATE_CKPT:-}}"
      "dataset.expected_time_horizon=${ACTION_HORIZON:-${TIME_HORIZON:-5}}"
    )
  fi
  if [[ "${CONFIG_NAME}" == *precomputed* && -n "${LOAD_ACTOR_SEQUENCE:-}" ]]; then
    KIND_OVERRIDES+=("dataset.load_actor_sequence=${LOAD_ACTOR_SEQUENCE}")
  fi
  if [[ "${CONFIG_NAME}" == *precomputed* && -n "${ACTOR_SEQUENCE_LENGTH:-}" ]]; then
    KIND_OVERRIDES+=(
      "dataset.actor_sequence_length=${ACTOR_SEQUENCE_LENGTH}"
      "world_model.actor_sequence_length=${ACTOR_SEQUENCE_LENGTH}"
    )
  fi
  if [[ "${CONFIG_NAME}" == *precomputed* && -n "${FULL_HIDDEN_REC_SCALE:-}" ]]; then
    KIND_OVERRIDES+=("world_model.full_hidden_rec_scale=${FULL_HIDDEN_REC_SCALE}")
  fi
fi

echo "WM kind:        ${WM_KIND_RESOLVED}"
echo "Config:         ${CONFIG_NAME}"
echo "Run output dir: ${OUT_DIR}"
echo "Launcher:       ${LAUNCHER}"
echo "Python:         ${PYTHON_BIN}"
echo "GPUs:           ${CUDA_VISIBLE_DEVICES}  (nproc_per_node=${NUM_GPUS})"
if ((${#SEQUENCE_OVERRIDES[@]})); then
  echo "WM sequence T:  ${WM_T}"
fi
if ((${#SMOKE_OVERRIDES[@]})); then
  echo "WM smoke mode:  enabled (${WM_SMOKE_STEPS:-1} step)"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=()
case "${LAUNCHER}" in
  single)
    CMD=("${PYTHON_BIN}" -m src.cli.train --config-name "${CONFIG_NAME}")
    ;;
  torchrun|distributed)
    CMD=(
      "${PYTHON_BIN}" -m torch.distributed.run
      --standalone
      --nnodes=1
      --nproc-per-node="${NUM_GPUS}"
      --master_port="${MASTER_PORT}"
      --module src.cli.train
      --config-name "${CONFIG_NAME}"
    )
    ;;
  *)
    echo "ERROR: unknown LAUNCHER='${LAUNCHER}'. Use single or torchrun." >&2
    exit 2
    ;;
esac

CMD+=(
  "${COMMON_OVERRIDES[@]}"
  "${SEQUENCE_OVERRIDES[@]}"
  "${KIND_OVERRIDES[@]}"
  "${SMOKE_OVERRIDES[@]}"
  "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN=1, not launching training. Command:\n'
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
