#!/usr/bin/env bash
# One-shot bootstrap: raw LIBERO-100 download → full preprocess pipeline for libero_10.
#
# Usage:
#   conda activate dreamervla
#   bash scripts/bootstrap_libero_10.sh
#
# Override defaults via env vars, e.g.:
#   GPU_DEVICES=0,1,2,3 PRETOKENIZE_PROCS=16 bash scripts/bootstrap_libero_10.sh
#
# Re-runs are safe: each step is skipped if its output already exists.
# To force re-run a step, delete its output directory first.

set -euo pipefail

# ---- config (override via env) ----
SUITE="${SUITE:-libero_10}"
TASK_NAME="${TASK_NAME:-10}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
ACTION_HORIZON="${ACTION_HORIZON:-10}"
GPU_DEVICES="${GPU_DEVICES:-0,1,4,5}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-32}"
DOWNLOAD_SUITE="${DOWNLOAD_SUITE:-libero_100}"   # libero_100 includes libero_10
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_TOKENIZERS_FIX="${SKIP_TOKENIZERS_FIX:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
RAW_DIR="$DATA_DIR/libero/datasets"
PROC_DIR="$DATA_DIR/processed_data"
CONFIG_DIR="$DATA_DIR/configs/$SUITE"

log() { printf '\n\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
skip() { printf '\033[0;33m[skip]\033[0m %s\n' "$*"; }

# ---- sanity: conda env ----
if [[ "${CONDA_DEFAULT_ENV:-}" != "dreamervla" ]]; then
  echo "ERROR: expected conda env 'dreamervla', got '${CONDA_DEFAULT_ENV:-<none>}'" >&2
  echo "Run: conda activate dreamervla" >&2
  exit 1
fi

cd "$ROOT_DIR"

# ---- Step 0: fix tokenizers/transformers version conflict ----
if [[ "$SKIP_TOKENIZERS_FIX" != "1" ]]; then
  log "Step 0: ensuring tokenizers 0.19.x (transformers 4.40.1 needs it)"
  if ! python -c "import transformers, peft" >/dev/null 2>&1; then
    pip install --quiet "tokenizers>=0.19,<0.20"
  fi
  python -c "import transformers, peft; print('  transformers=' + transformers.__version__, 'peft=' + peft.__version__)"
else
  skip "Step 0 (SKIP_TOKENIZERS_FIX=1)"
fi

# ---- Step 1: download raw LIBERO HDF5 ----
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  skip "Step 1 download (SKIP_DOWNLOAD=1)"
elif [[ -d "$RAW_DIR/$SUITE" ]] && compgen -G "$RAW_DIR/$SUITE/*.hdf5" > /dev/null; then
  skip "Step 1: $RAW_DIR/$SUITE already populated"
else
  log "Step 1: downloading $DOWNLOAD_SUITE from HuggingFace into $RAW_DIR"
  mkdir -p "$RAW_DIR"
  (
    cd "$ROOT_DIR/LIBERO/benchmark_scripts"
    python download_libero_datasets.py \
      --datasets "$DOWNLOAD_SUITE" \
      --use-huggingface \
      --download-dir "$RAW_DIR"
  )
fi

# ---- Step 1b: point ~/.libero/config.yaml at the new datasets dir ----
LIBERO_CFG="$HOME/.libero/config.yaml"
mkdir -p "$(dirname "$LIBERO_CFG")"
log "Step 1b: writing $LIBERO_CFG (datasets → $RAW_DIR)"
cat > "$LIBERO_CFG" <<EOF
assets: $ROOT_DIR/LIBERO/libero/libero/./assets
bddl_files: $ROOT_DIR/LIBERO/libero/libero/./bddl_files
benchmark_root: $ROOT_DIR/LIBERO/libero/libero
datasets: $RAW_DIR
init_states: $ROOT_DIR/LIBERO/libero/libero/./init_files
EOF

# ---- Step 2: filter no-op frames ----
STEP2_OUT="$PROC_DIR/${SUITE}_no_noops_t_${IMAGE_RESOLUTION}"
if [[ -d "$STEP2_OUT" ]] && compgen -G "$STEP2_OUT/*.hdf5" > /dev/null; then
  skip "Step 2: $STEP2_OUT already exists"
else
  log "Step 2: filter no-op → $STEP2_OUT"
  LIBERO_TASK_SUITE="$SUITE" IMAGE_RESOLUTION="$IMAGE_RESOLUTION" \
    bash "$SCRIPT_DIR/preprocess/processed_data_no_op.sh"
fi

# ---- Step 3: extract images / actions / states ----
STEP3_OUT="$PROC_DIR/${SUITE}_image_state_action_t_${IMAGE_RESOLUTION}"
if [[ -d "$STEP3_OUT" ]] && [[ -n "$(ls -A "$STEP3_OUT" 2>/dev/null)" ]]; then
  skip "Step 3: $STEP3_OUT already exists"
else
  log "Step 3: extract image/action/state/wrist → $STEP3_OUT"
  LIBERO_TASK_SUITE="$SUITE" IMAGE_RESOLUTION="$IMAGE_RESOLUTION" \
    bash "$SCRIPT_DIR/preprocess/processed_data_save_img_action_state_wrist.sh"
fi

# ---- Step 4: generate conversation JSONs ----
CONV_TAG="libero_${TASK_NAME}_his_2_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
STEP4_SENTINEL="$PROC_DIR/convs/${CONV_TAG}.json"  # wildcard split; we check for any
if compgen -G "$PROC_DIR/convs/libero_${TASK_NAME}_his_2_*_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json" > /dev/null; then
  skip "Step 4: convs for libero_${TASK_NAME} already exist"
else
  log "Step 4: build conversation JSON (base_dir=$STEP3_OUT)"
  LIBERO_TASK_NAME="$TASK_NAME" IMAGE_RESOLUTION="$IMAGE_RESOLUTION" ACTION_HORIZON="$ACTION_HORIZON" \
  BASE_DIR="$STEP3_OUT" \
    bash "$SCRIPT_DIR/preprocess/processed_data_generate_convs.sh"
fi

# ---- Step 5: pretokenize + merge manifest ----
MANIFEST="$PROC_DIR/concate_tokens/${CONV_TAG}.json"
if [[ -f "$MANIFEST" ]]; then
  skip "Step 5: manifest exists → $MANIFEST"
else
  log "Step 5: pretokenize on GPUs [$GPU_DEVICES] × $PRETOKENIZE_PROCS procs"
  PREPROCESS_GPU_DEVICES="$GPU_DEVICES" \
  TASK_NAME="$TASK_NAME" IMAGE_RESOLUTION="$IMAGE_RESOLUTION" ACTION_HORIZON="$ACTION_HORIZON" \
  PRETOKENIZE_PROCS="$PRETOKENIZE_PROCS" \
    bash "$SCRIPT_DIR/preprocess/processed_data_pretokenize.sh"
fi

# ---- Step 6: write training YAMLs ----
TRAIN_YAML="$CONFIG_DIR/his_2_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}_pretokenize.yaml"
if [[ -f "$TRAIN_YAML" ]]; then
  skip "Step 6: $TRAIN_YAML already exists"
else
  log "Step 6: write training YAMLs → $CONFIG_DIR"
  LIBERO_TASK_SUITE="$SUITE" TASK_NAME="$TASK_NAME" IMAGE_RESOLUTION="$IMAGE_RESOLUTION" ACTION_HORIZON="$ACTION_HORIZON" \
    bash "$SCRIPT_DIR/preprocess/prepare_train_configs.sh"
fi

log "Done. Key artefacts:"
echo "  raw hdf5       : $RAW_DIR/$SUITE"
echo "  filtered hdf5  : $STEP2_OUT"
echo "  extracted dirs : $STEP3_OUT"
echo "  conv JSONs     : $PROC_DIR/convs/libero_${TASK_NAME}_his_2_*_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
echo "  manifest       : $MANIFEST"
echo "  train config   : $TRAIN_YAML"
