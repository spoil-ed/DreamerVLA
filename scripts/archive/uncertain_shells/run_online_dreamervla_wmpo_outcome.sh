#!/usr/bin/env bash
# DreamerVLA — ONLINE WMPO outcome training launcher (Path B).
#
# Real LIBERO env rollouts + ChunkAware WM imagine + LatentSuccessClassifier
# outcome reward + WMPO-style PPO. This is the production reward-loop.
#
# Stack:
#   classifier    = LatentSuccessClassifier (head_type=transformer, 11.3 M params)
#                   wmpo_aligned_small_tf episode F1 = 0.9514 @ threshold 0.59
#   world model   = ChunkAwareRynnDinoWMWorldModel, m1024_d6, chunk_size=5
#                   pinned step 17000 snapshot (today's training)
#   actor / critic = Pi0ActionHiddenActor + TwohotCritic (from config)
#   env           = LIBERO-goal sim, episode_horizon=200, MuJoCo+osmesa
#
# Usage:
#   bash scripts/run_online_dreamervla_wmpo_outcome.sh                       # GPU 6, libero_goal task 0
#   CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_online_dreamervla_wmpo_outcome.sh  # multi-GPU
#   TASK_IDS=0,1,2 bash scripts/run_online_dreamervla_wmpo_outcome.sh        # multi-task
#   TOTAL_ENV_STEPS=2000 bash scripts/run_online_dreamervla_wmpo_outcome.sh   # smoke test
#
# Overrides (env var → CLI flag):
#   CONFIG_NAME         = Hydra config name (without .yaml)
#   WM_CKPT             = path to ChunkAware WM ckpt
#   CLASSIFIER_CKPT     = path to LatentSuccessClassifier ckpt (online schema)
#   CLASSIFIER_THRESH   = override classifier success threshold (default: from ckpt)
#   OUT_DIR             = output dir under data/outputs/dreamervla/online_wmpo_outcome/
#   TASK_SUITE          = libero_goal (default)
#   TASK_IDS            = "0" or "0,1,2" (default "0")
#   TOTAL_ENV_STEPS     = total real env steps to collect
#   MAX_TRAIN_UPDATES   = cap on optimizer steps
#   BATCH_SIZE          = replay batch size
#   EPISODE_HORIZON     = real env episode length
#   TRAIN_RATIO         = DreamerV3 batched-steps-per-env-step ratio
#
# Outputs land under $OUT_DIR:
#   ├── checkpoints/latest.ckpt
#   ├── logs/online_wmpo_outcome.log
#   └── videos/  (if --video-every-env-steps > 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY=${PY:-/home/user01/miniconda3/envs/dreamervla/bin/python}
CONFIG_NAME=${CONFIG_NAME:-dreamer_vla_libero_goal_online_wmpo_outcome}

# Multi-GPU DDP support. If NUM_GPUS is unset and CUDA_VISIBLE_DEVICES has commas,
# we infer NUM_GPUS from the comma count; otherwise default to 1 (plain python).
if [[ -z "${NUM_GPUS:-}" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
    NUM_GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NUM_GPUS=1
  fi
fi
WM_CKPT=${WM_CKPT:-${REPO_ROOT}/data/outputs/worldmodel/rynn_dino_wm_action_hidden/chunkaware_pinned/step_00017000.ckpt}
CLASSIFIER_CKPT=${CLASSIFIER_CKPT:-${REPO_ROOT}/data/outputs/dreamervla/outcome_classifier/libero_goal/wmpo_aligned_small_tf/ckpt/best_episode_f10.9514_th0.59.ckpt}
CLASSIFIER_THRESH=${CLASSIFIER_THRESH:-}        # empty → use threshold from ckpt
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/data/outputs/dreamervla/online_wmpo_outcome/small_tf_ep0951_chunkwm17k/${RUN_TAG}}

TASK_SUITE=${TASK_SUITE:-libero_goal}
TASK_IDS=${TASK_IDS:-0}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-200000}
MAX_TRAIN_UPDATES=${MAX_TRAIN_UPDATES:-50000}
EPISODE_HORIZON=${EPISODE_HORIZON:-200}
BATCH_SIZE=${BATCH_SIZE:-4}
TRAIN_RATIO=${TRAIN_RATIO:-32}
MIN_REPLAY=${MIN_REPLAY:-64}
LOG_EVERY=${LOG_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-200}
# In DDP mode the default train_accum cadence (--train-every unset) deadlocks at
# zero updates because per-rank accumulators desync once first episodes complete
# at different env_step values (~ranks 0..N-1) and dist.all_reduce(MIN) takes 0
# forever after. Workaround: use deterministic cadence by default in multi-GPU
# mode. Override with TRAIN_EVERY="" to use train_accum anyway.
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  TRAIN_EVERY=${TRAIN_EVERY:-4}
  UPDATES_PER_TRAIN=${UPDATES_PER_TRAIN:-1}
else
  TRAIN_EVERY=${TRAIN_EVERY:-}
  UPDATES_PER_TRAIN=${UPDATES_PER_TRAIN:-1}
fi
VIDEO_EVERY_ENV_STEPS=${VIDEO_EVERY_ENV_STEPS:-0}   # set >0 to dump videos

# --------------- env --------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
unset MUJOCO_EGL_DEVICE_ID || true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-dvl-wmpo-outcome}

mkdir -p "${OUT_DIR}/videos" "${OUT_DIR}/logs" "${MPLCONFIGDIR}"

LOG="${OUT_DIR}/logs/online_wmpo_outcome.log"

# --------------- pre-flight sanity checks ----------------------------------
for p in "${WM_CKPT}" "${CLASSIFIER_CKPT}" "${REPO_ROOT}/configs/${CONFIG_NAME}.yaml"; do
  if [[ ! -f "${p}" ]]; then
    echo "[fatal] missing required file: ${p}" >&2
    exit 1
  fi
done

echo "===== online WMPO outcome — ${RUN_TAG} ====="
echo "  cfg              = ${CONFIG_NAME}"
echo "  wm ckpt          = ${WM_CKPT}"
echo "  classifier ckpt  = ${CLASSIFIER_CKPT}"
echo "  classifier thresh= ${CLASSIFIER_THRESH:-<from ckpt>}"
echo "  out dir          = ${OUT_DIR}"
echo "  task             = ${TASK_SUITE} ids=${TASK_IDS}"
echo "  CUDA_VISIBLE     = ${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS         = ${NUM_GPUS}"
echo "  total env steps  = ${TOTAL_ENV_STEPS}    max train upd = ${MAX_TRAIN_UPDATES}"
echo "  log              = ${LOG}"

# --------------- assemble args ---------------------------------------------
ARGS=(
  --config "${REPO_ROOT}/configs/${CONFIG_NAME}.yaml"
  --out-dir "${OUT_DIR}"
  --world-model-ckpt "${WM_CKPT}"
  --actor-update-kind outcome
  --classifier-ckpt "${CLASSIFIER_CKPT}"
  # Skip loading a separate fine-tuned encoder state — the default points to a
  # legacy 4×5120 action-head shape, incompatible with the current 4×35×1024
  # action-query model. The HF VLA dir loaded by build_encoder() already has
  # the right pretrained weights.
  --encoder-state-ckpt ""
  --task-suite "${TASK_SUITE}"
  --task-ids "${TASK_IDS}"
  --episode-horizon "${EPISODE_HORIZON}"
  --total-env-steps "${TOTAL_ENV_STEPS}"
  --max-train-updates "${MAX_TRAIN_UPDATES}"
  --train-ratio "${TRAIN_RATIO}"
  --batch-size "${BATCH_SIZE}"
  --min-replay "${MIN_REPLAY}"
  --log-every "${LOG_EVERY}"
  --save-every "${SAVE_EVERY}"
  --rssm-action-scale env
  --updates-per-train "${UPDATES_PER_TRAIN}"
)
if [[ -n "${TRAIN_EVERY}" ]]; then
  ARGS+=( --train-every "${TRAIN_EVERY}" )
fi
if [[ -n "${CLASSIFIER_THRESH}" ]]; then
  ARGS+=( --classifier-threshold "${CLASSIFIER_THRESH}" )
fi
if [[ "${VIDEO_EVERY_ENV_STEPS}" -gt 0 ]]; then
  ARGS+=(
    --video-every-env-steps "${VIDEO_EVERY_ENV_STEPS}"
    --video-fps 30
    --video-max-frames "${EPISODE_HORIZON}"
    --video-dir "${OUT_DIR}/videos"
  )
fi

# --------------- run -------------------------------------------------------
# Multi-GPU: launch via torchrun (script's _init_distributed reads LOCAL_RANK).
# Single-GPU: plain python (script falls back to single-process path).
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCHER=("${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}")
else
  LAUNCHER=("${PY}" -X faulthandler -u)
fi

{
  echo "===== launch $(date) ====="
  echo "launcher: ${LAUNCHER[*]}"
  echo "args: ${ARGS[*]}"
  "${LAUNCHER[@]}" scripts/train_online_pi0_action_hidden_dreamervla.py "${ARGS[@]}"
  code=$?
  echo "===== exit=${code} $(date) ====="
  exit "${code}"
} 2>&1 | tee -a "${LOG}"
