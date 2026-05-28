#!/usr/bin/env bash
# libero_object preprocessing (ACTION_HORIZON=1) + VLA SFT on GPUs 4,5.
# Invokes the python preprocess scripts directly with explicit paths
# (avoids the archive shells that miscompute ROOT_DIR).
#
# If WAIT_SESSION is non-empty and exists, wait for it to exit first.
#
# Usage:  tmux new-session -d -s vla_chain_object \
#           "bash scripts/chain_libero_object_after_10.sh"
set -uo pipefail

PROJECT_ROOT="/mnt/data/spoil/workspace/DreamerVLA"
cd "${PROJECT_ROOT}"

source /home/user01/miniconda3/etc/profile.d/conda.sh
conda activate dreamervla
PYTHON="/home/user01/miniconda3/envs/dreamervla/bin/python"

LOG_DIR="${PROJECT_ROOT}/data/logs/vla_libero_object"
mkdir -p "${LOG_DIR}"

WAIT_SESSION="${WAIT_SESSION:-}"
GPUS="${GPUS:-4,5}"
NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29548}"

LIBERO_TASK_SUITE=libero_object
TASK_NAME=object
IMAGE_RESOLUTION=256
ACTION_HORIZON=1
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-16}"

TOKENIZER_PATH="${PROJECT_ROOT}/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"
PROCESSED_DATA_ROOT="${PROJECT_ROOT}/data/processed_data"

RAW_DIR="${PROCESSED_DATA_ROOT}/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}"
IMG_STATE_DIR="${PROCESSED_DATA_ROOT}/${LIBERO_TASK_SUITE}_image_state_action_t_${IMAGE_RESOLUTION}"
CONVS_DIR="${PROCESSED_DATA_ROOT}/convs"
TOKENS_DIR="${PROCESSED_DATA_ROOT}/tokens"
CONCATE_DIR="${PROCESSED_DATA_ROOT}/concate_tokens"
CONFIG_DIR="${PROJECT_ROOT}/data/configs/${LIBERO_TASK_SUITE}"

mkdir -p "${CONVS_DIR}" "${TOKENS_DIR}" "${CONCATE_DIR}" "${CONFIG_DIR}"

# ───────── 0. optional wait ─────────
if [[ -n "${WAIT_SESSION}" ]] && tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
  echo "[$(date)] waiting for tmux session '${WAIT_SESSION}' to exit"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date)] '${WAIT_SESSION}' is gone — proceeding"
  sleep 20
fi

# ───────── 1. step 2 — extract images/state/action (CPU) ─────────
STEP2_PY="${PROJECT_ROOT}/src/utils/libero_utils/regenerate_libero_dataset_save_img_action_state_wrist.py"
echo "════════════════════════════════════════════════════════════════"
echo "[$(date)] === Step 2/5: extract img/state/action ==="
echo "════════════════════════════════════════════════════════════════"
if [[ -d "${IMG_STATE_DIR}" && $(ls "${IMG_STATE_DIR}" 2>/dev/null | wc -l) -gt 0 ]]; then
  echo "  ${IMG_STATE_DIR} already present, skipping"
else
  [[ -f "${STEP2_PY}" ]] || { echo "ERROR: missing ${STEP2_PY}"; exit 2; }
  "${PYTHON}" "${STEP2_PY}" \
    --libero_task_suite "${LIBERO_TASK_SUITE}" \
    --image_resolution "${IMAGE_RESOLUTION}" \
    --raw_data_dir "${RAW_DIR}" \
    --save_dir "${IMG_STATE_DIR}" \
    2>&1 | tee "${LOG_DIR}/step2_save_img_$(date +%Y%m%d_%H%M%S).log"
fi

# ───────── 2. step 3 — generate conversation JSONs (CPU) ─────────
echo "════════════════════════════════════════════════════════════════"
echo "[$(date)] === Step 3/5: generate conversation JSONs ==="
echo "════════════════════════════════════════════════════════════════"
(
  cd "${PROJECT_ROOT}/src/preprocess"
  "${PYTHON}" action_state_model_conv_generation.py \
    --base_dir "${IMG_STATE_DIR}" \
    --his 1 \
    --len_action "${ACTION_HORIZON}" \
    --task_name "${TASK_NAME}" \
    --resolution "${IMAGE_RESOLUTION}" \
    --with_state \
    --img_names imgs_third_view imgs_wrist \
    --output_dir "${CONVS_DIR}"
) 2>&1 | tee "${LOG_DIR}/step3_convs_$(date +%Y%m%d_%H%M%S).log"

# ───────── 3. step 4 — pretokenize + concat manifest (GPU) ─────────
echo "════════════════════════════════════════════════════════════════"
echo "[$(date)] === Step 4/5: pretokenize + concat manifest (GPU ${GPUS}) ==="
echo "════════════════════════════════════════════════════════════════"
(
  cd "${PROJECT_ROOT}/src/preprocess"
  "${PYTHON}" pretoken_state_action_model.py \
    --task "${TASK_NAME}" \
    --resolution "${IMAGE_RESOLUTION}" \
    --with_state \
    --img_names imgs_third_view imgs_wrist \
    --his 1 \
    --len_action "${ACTION_HORIZON}" \
    --num_procs "${PRETOKENIZE_PROCS}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --in_filename_dir "${CONVS_DIR}" \
    --out_root "${TOKENS_DIR}" \
    --gpu_devices "${GPUS}"

  bash concate_record_libero.sh "${TOKENS_DIR}"

  "${PYTHON}" concate_action_world_model_data_libero.py \
    --source_dir_patterns "libero_${TASK_NAME}_his_1_{}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
    --all_patterns "libero_${TASK_NAME}_his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
    --processed_data_root "${PROCESSED_DATA_ROOT}"
) 2>&1 | tee "${LOG_DIR}/step4_pretokenize_$(date +%Y%m%d_%H%M%S).log"

# ───────── 4. step 5 — write training yaml configs ─────────
echo "════════════════════════════════════════════════════════════════"
echo "[$(date)] === Step 5/5: write yaml configs ==="
echo "════════════════════════════════════════════════════════════════"
SUFFIX="his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"

cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize.yaml" <<EOF
META:
  - path: '${CONCATE_DIR}/libero_${TASK_NAME}_${SUFFIX}.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize_val_ind.yaml" <<EOF
META:
  - path: '${TOKENS_DIR}/libero_${TASK_NAME}_his_1_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize_val_ood.yaml" <<EOF
META:
  - path: '${TOKENS_DIR}/libero_${TASK_NAME}_his_1_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF
echo "Wrote configs to ${CONFIG_DIR}"

# Sanity
REQ=(
  "${CONCATE_DIR}/libero_${TASK_NAME}_${SUFFIX}.json"
  "${TOKENS_DIR}/libero_${TASK_NAME}_his_1_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json"
  "${TOKENS_DIR}/libero_${TASK_NAME}_his_1_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json"
  "${CONFIG_DIR}/${SUFFIX}_pretokenize.yaml"
)
for f in "${REQ[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[$(date)] ERROR: missing required artefact: $f"
    exit 5
  fi
done
echo "[$(date)] preprocessing artefacts OK"

# ───────── 5. launch libero_object VLA SFT ─────────
echo "════════════════════════════════════════════════════════════════"
echo "[$(date)] === Launch libero_object VLA SFT on GPUs ${GPUS} ==="
echo "════════════════════════════════════════════════════════════════"
TS=$(date +%Y%m%d_%H%M%S)
export CUDA_VISIBLE_DEVICES="${GPUS}"
export NGPU="${NGPU}"
export MASTER_PORT="${MASTER_PORT}"
export PYTHON
export OUT_DIR="data/outputs/vla/pi0_query/libero_object_${TS}"

bash "${PROJECT_ROOT}/scripts/train_vla.sh" \
  task=libero_object training.gradient_accumulate_every=2 \
  2>&1 | tee "${LOG_DIR}/train_libero_object_${TS}.log"

echo "[$(date)] chain done."
