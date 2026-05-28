#!/usr/bin/env bash
# ============================================================================
# Smoke test for the full pretokenize pipeline, starting from a raw HDF5 file.
#
# Pipeline:  raw HDF5 → extract images/actions/states
#          → generate conversations → tokenize → concat manifest → config
#
# Usage:
#   conda activate wmpo
#   bash scripts/preprocess/smoke_pretokenize.sh
#
# Optionally override any variable:
#   HDF5_FILE=... MAX_DEMOS=3 bash scripts/preprocess/smoke_pretokenize.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ----- tunables -----
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-1}"
MAX_DEMOS="${MAX_DEMOS:-2}"
LIBERO_TASK_NAME="${LIBERO_TASK_NAME:-goal}"

# Pick the smallest HDF5 by default (turn_on_the_stove, ~427M)
HDF5_FILE="${HDF5_FILE:-${ROOT_DIR}/data/libero/datasets/libero_goal/turn_on_the_stove_demo.hdf5}"

# Paths — everything writes to processed_data_smoke
PROCESSED_DATA_ROOT="${ROOT_DIR}/data/processed_data_smoke"
IMG_DIR="${PROCESSED_DATA_ROOT}/libero_goal_image_state_action_t_${IMAGE_RESOLUTION}"
CONVS_DIR="${PROCESSED_DATA_ROOT}/convs"
TOKENS_DIR="${PROCESSED_DATA_ROOT}/tokens"
CONCATE_DIR="${PROCESSED_DATA_ROOT}/concate_tokens"
CONFIG_DIR="${ROOT_DIR}/data/configs_smoke/libero_goal"
TOKENIZER_PATH="${ROOT_DIR}/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"

IMG_NAMES=(imgs_third_view imgs_wrist)

echo "========================================="
echo " Smoke pretokenize pipeline (from HDF5)"
echo " HDF5       : ${HDF5_FILE}"
echo " MAX_DEMOS  : ${MAX_DEMOS}"
echo " DATA ROOT  : ${PROCESSED_DATA_ROOT}"
echo "========================================="

# ---- Step 1: Extract images/actions/states from raw HDF5 ----
echo "[1/5] Extracting from raw HDF5 ..."
# Clean old smoke image data so we start fresh
rm -rf "${IMG_DIR}"
mkdir -p "${IMG_DIR}"

python -m dreamer_vla.preprocess.smoke_extract_hdf5 \
    --hdf5 "${HDF5_FILE}" \
    --save_dir "${IMG_DIR}" \
    --max_demos "${MAX_DEMOS}" \
    --resolution "${IMAGE_RESOLUTION}"

echo "  Extracted to ${IMG_DIR}"

# ---- Step 2: Generate conversations ----
echo "[2/5] Generating conversations ..."
rm -f "${CONVS_DIR}"/libero_${LIBERO_TASK_NAME}_his_1_*_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json 2>/dev/null || true
mkdir -p "${CONVS_DIR}"

cd "${ROOT_DIR}/src/preprocess"
python action_state_model_conv_generation.py \
    --base_dir "${IMG_DIR}" \
    --his 1 \
    --len_action "${ACTION_HORIZON}" \
    --task_name "${LIBERO_TASK_NAME}" \
    --resolution "${IMAGE_RESOLUTION}" \
    --with_state \
    --img_names "${IMG_NAMES[@]}" \
    --output_dir "${CONVS_DIR}"

echo "  Conversations written to ${CONVS_DIR}"

# ---- Step 3: Tokenize ----
echo "[3/5] Tokenizing ..."
# Clean stale tokens for this config
rm -rf "${TOKENS_DIR}"/libero_${LIBERO_TASK_NAME}_his_1_*_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION} 2>/dev/null || true

python pretoken_state_action_model.py \
    --task "${LIBERO_TASK_NAME}" \
    --resolution "${IMAGE_RESOLUTION}" \
    --with_state \
    --img_names "${IMG_NAMES[@]}" \
    --his 1 \
    --len_action "${ACTION_HORIZON}" \
    --num_procs "${PRETOKENIZE_PROCS}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --in_filename_dir "${CONVS_DIR}" \
    --out_root "${TOKENS_DIR}"

# ---- Step 4: Concat into manifest ----
echo "[4/5] Building manifest ..."
bash concate_record_libero.sh "${TOKENS_DIR}"

mkdir -p "${CONCATE_DIR}"
python concate_action_world_model_data_libero.py \
    --source_dir_patterns "libero_${LIBERO_TASK_NAME}_his_1_{}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
    --all_patterns "libero_${LIBERO_TASK_NAME}_his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
    --processed_data_root "${PROCESSED_DATA_ROOT}"

echo "  Manifest written to ${CONCATE_DIR}"

# ---- Step 5: Generate dataset config ----
echo "[5/5] Generating dataset config ..."
mkdir -p "${CONFIG_DIR}"

cat > "${CONFIG_DIR}/smoke_pretokenize.yaml" <<EOF
META:
  - path: '${CONCATE_DIR}/libero_${LIBERO_TASK_NAME}_his_1_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

echo "  Config written to ${CONFIG_DIR}/smoke_pretokenize.yaml"

echo ""
echo "========================================="
echo " Smoke pretokenize pipeline complete!"
echo ""
echo " To run smoke VLA training with this generated data config:"
echo "   python -m dreamer_vla.cli.train --config-name pretokenize_vla_libero_goal dataset.config_path=${CONFIG_DIR}/smoke_pretokenize.yaml dataset_val_ind=null dataset_val_ood=null training.num_epochs=1 training.max_train_steps=2"
echo "========================================="
