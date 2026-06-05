#!/usr/bin/env bash
# Comprehensive LIBERO data preprocessing pipeline.
# For each suite: stage 2 (img/state/action extract, CPU)
#                 stage 3 (convs JSON,  CPU)
#                 stage 4 (pretokenize + concat manifest, GPU)
#                 stage 5 (write training yaml configs)
#
# libero_goal is fully processed already and is SKIPPED by default.
#
# Usage:
#   tmux new-session -d -s libero_data \
#     "bash scripts/preprocess/process_all_libero_data.sh"
#
# Env overrides:
#   SUITES="libero_10 libero_object libero_spatial"  (default)
#   GPUS=4,5
#   PRETOKENIZE_PROCS=16
#   FORCE=1   re-run even if outputs look complete
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
if [[ -f "${SCRIPT_DIR}/../common_env.sh" ]]; then
  source "${SCRIPT_DIR}/../common_env.sh"
  PROJECT_ROOT="${DVLA_ROOT}"
fi
cd "${PROJECT_ROOT}"

if [[ -n "${CONDA_SH:-}" ]]; then
  # Optional: CONDA_SH=/path/to/conda.sh CONDA_ENV=dreamervla bash ...
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-dreamervla}"
fi
PYTHON="${PYTHON:-python}"

LOG_DIR="${PROJECT_ROOT}/data/logs/libero_data_prep"
mkdir -p "${LOG_DIR}"

SUITES="${SUITES:-libero_10 libero_object libero_spatial}"
GPUS="${GPUS:-4,5}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-16}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"           # atomic pretokenize
HIS="${HIS:-1}"
FORCE="${FORCE:-0}"

TOKENIZER_PATH="${PROJECT_ROOT}/data/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"
PROCESSED_DATA_ROOT="${PROJECT_ROOT}/data/processed_data"

CONVS_DIR="${PROCESSED_DATA_ROOT}/convs"
TOKENS_DIR="${PROCESSED_DATA_ROOT}/tokens"
CONCATE_DIR="${PROCESSED_DATA_ROOT}/concate_tokens"
mkdir -p "${CONVS_DIR}" "${TOKENS_DIR}" "${CONCATE_DIR}"

suite_task_name() {
  printf '%s\n' "${1#libero_}"
}

json_len() {
  [[ -f "$1" ]] || { echo 0; return; }
  "${PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$1" 2>/dev/null || echo 0
}

pkl_count() {
  [[ -d "$1/files" ]] || { echo 0; return; }
  find "$1/files" -maxdepth 1 -name '*.pkl' | wc -l
}

process_one_suite() {
  local SUITE="$1"
  local TASK_NAME
  TASK_NAME="$(suite_task_name "${SUITE}")"
  local RAW_DIR="${PROCESSED_DATA_ROOT}/${SUITE}_no_noops_t_${IMAGE_RESOLUTION}"
  local IMG_STATE_DIR="${PROCESSED_DATA_ROOT}/${SUITE}_image_state_action_t_${IMAGE_RESOLUTION}"
  local CONFIG_DIR="${PROJECT_ROOT}/data/configs/${SUITE}"
  local SUFFIX="his_${HIS}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
  local MANIFEST="${CONCATE_DIR}/${SUITE}_${SUFFIX}.json"
  local VAL_IND_REC="${TOKENS_DIR}/${SUITE}_his_${HIS}_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json"
  local VAL_OOD_REC="${TOKENS_DIR}/${SUITE}_his_${HIS}_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json"
  local STAMP
  STAMP="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${CONFIG_DIR}"

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " SUITE = ${SUITE}   TASK = ${TASK_NAME}   ${STAMP}"
  echo "════════════════════════════════════════════════════════════════"

  # Sanity: raw must exist
  if [[ ! -d "${RAW_DIR}" ]]; then
    echo "ERROR: missing raw dir ${RAW_DIR}"; return 2
  fi
  local raw_n
  raw_n=$(ls "${RAW_DIR}"/*.hdf5 2>/dev/null | wc -l)
  echo "  raw hdf5 files: ${raw_n}"
  if [[ "${raw_n}" -lt 1 ]]; then echo "  ERROR: no hdf5 in raw"; return 2; fi

  # ── Stage 2 ── extract images/state/action ─────────────────────────────────
  local need_stage2=1
  if [[ -d "${IMG_STATE_DIR}" ]]; then
    local td=$(ls "${IMG_STATE_DIR}" 2>/dev/null | wc -l)
    if [[ "${td}" -ge "${raw_n}" ]] && [[ "${FORCE}" != "1" ]]; then
      echo "  [stage2] ${td} task dirs present, skipping"
      need_stage2=0
    else
      echo "  [stage2] only ${td}/${raw_n} task dirs — wiping and re-running"
      rm -rf "${IMG_STATE_DIR}"
    fi
  fi
  if [[ "${need_stage2}" == "1" ]]; then
    local log="${LOG_DIR}/${SUITE}_stage2_${STAMP}.log"
    echo "  [stage2] -> ${log}"
    "${PYTHON}" "${PROJECT_ROOT}/dreamer_vla/preprocess/libero_utils/regenerate_libero_dataset_save_img_action_state_wrist.py" \
      --libero_task_suite "${SUITE}" \
      --image_resolution "${IMAGE_RESOLUTION}" \
      --raw_data_dir "${RAW_DIR}" \
      --save_dir "${IMG_STATE_DIR}" \
      > "${log}" 2>&1
    local rc=$?
    local td=$(ls "${IMG_STATE_DIR}" 2>/dev/null | wc -l)
    echo "  [stage2] exit=${rc}  task dirs ${td}/${raw_n}"
    if [[ "${td}" -lt "${raw_n}" ]]; then
      echo "  [stage2] FAIL: only ${td}/${raw_n} dirs created — aborting suite"; return 3
    fi
  fi

  # ── Stage 3 ── generate conversation JSONs (his_1 / len_action=1) ──────────
  local CONV_TRAIN="${CONVS_DIR}/${SUITE}_his_${HIS}_train_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
  local CONV_VIND="${CONVS_DIR}/${SUITE}_his_${HIS}_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
  local CONV_VOOD="${CONVS_DIR}/${SUITE}_his_${HIS}_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
  if [[ -f "${CONV_TRAIN}" && -f "${CONV_VIND}" && -f "${CONV_VOOD}" && "${FORCE}" != "1" ]]; then
    echo "  [stage3] convs present, skipping"
  else
    local log="${LOG_DIR}/${SUITE}_stage3_${STAMP}.log"
    echo "  [stage3] -> ${log}"
    (
      cd "${PROJECT_ROOT}/dreamer_vla/preprocess"
      "${PYTHON}" action_state_model_conv_generation.py \
        --base_dir "${IMG_STATE_DIR}" \
        --his "${HIS}" \
        --len_action "${ACTION_HORIZON}" \
        --task_name "${TASK_NAME}" \
        --resolution "${IMAGE_RESOLUTION}" \
        --with_state \
        --img_names imgs_third_view imgs_wrist \
        --output_dir "${CONVS_DIR}"
    ) > "${log}" 2>&1
    local rc=$?
    echo "  [stage3] exit=${rc}"
    for f in "${CONV_TRAIN}" "${CONV_VIND}" "${CONV_VOOD}"; do
      [[ -f "$f" ]] || { echo "  [stage3] FAIL: missing $f"; return 4; }
    done
  fi

  # ── Stage 4 ── pretokenize + concat manifest (GPU ${GPUS}) ─────────────────
  local TOK_TRAIN="${TOKENS_DIR}/${SUITE}_his_${HIS}_train_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
  local TOK_VIND="${TOKENS_DIR}/${SUITE}_his_${HIS}_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
  local TOK_VOOD="${TOKENS_DIR}/${SUITE}_his_${HIS}_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
  local TRAIN_REC="${TOK_TRAIN}/record.json"
  local n_conv_train n_conv_vind n_conv_vood n_tok_train n_tok_vind n_tok_vood
  local n_rec_train n_rec_vind n_rec_vood n_manifest n_total
  n_conv_train=$(json_len "${CONV_TRAIN}")
  n_conv_vind=$(json_len "${CONV_VIND}")
  n_conv_vood=$(json_len "${CONV_VOOD}")
  n_tok_train=$(pkl_count "${TOK_TRAIN}")
  n_tok_vind=$(pkl_count "${TOK_VIND}")
  n_tok_vood=$(pkl_count "${TOK_VOOD}")
  n_rec_train=$(json_len "${TRAIN_REC}")
  n_rec_vind=$(json_len "${VAL_IND_REC}")
  n_rec_vood=$(json_len "${VAL_OOD_REC}")
  n_manifest=$(json_len "${MANIFEST}")
  n_total=$((n_tok_train + n_tok_vind + n_tok_vood))
  echo "  [stage4] counts train ${n_tok_train}/${n_conv_train}, val_ind ${n_tok_vind}/${n_conv_vind}, val_ood ${n_tok_vood}/${n_conv_vood}; records ${n_rec_train}/${n_rec_vind}/${n_rec_vood}; manifest ${n_manifest}/${n_total}"

  local tokens_complete=0 records_complete=0
  if [[ "${n_tok_train}" -eq "${n_conv_train}" && "${n_tok_vind}" -eq "${n_conv_vind}" && "${n_tok_vood}" -eq "${n_conv_vood}" ]]; then
    tokens_complete=1
  fi
  if [[ "${tokens_complete}" == "1" && "${n_rec_train}" -eq "${n_tok_train}" && "${n_rec_vind}" -eq "${n_tok_vind}" && "${n_rec_vood}" -eq "${n_tok_vood}" && "${n_manifest}" -eq "${n_total}" ]]; then
    records_complete=1
  fi

  if [[ "${tokens_complete}" == "1" && "${records_complete}" == "1" && "${FORCE}" != "1" ]]; then
    echo "  [stage4] pkl + records complete, skipping"
  else
    local log="${LOG_DIR}/${SUITE}_stage4_${STAMP}.log"
    echo "  [stage4] -> ${log}  (GPUs=${GPUS})"
    (
      cd "${PROJECT_ROOT}/dreamer_vla/preprocess"
      if [[ "${tokens_complete}" != "1" || "${FORCE}" == "1" ]]; then
        overwrite_arg=()
        [[ "${OVERWRITE_TOKENS:-0}" == "1" ]] && overwrite_arg=(--overwrite)
        "${PYTHON}" pretoken_state_action_model.py \
          --task "${TASK_NAME}" \
          --resolution "${IMAGE_RESOLUTION}" \
          --with_state \
          --img_names imgs_third_view imgs_wrist \
          --his "${HIS}" \
          --len_action "${ACTION_HORIZON}" \
          --num_procs "${PRETOKENIZE_PROCS}" \
          --tokenizer_path "${TOKENIZER_PATH}" \
          --in_filename_dir "${CONVS_DIR}" \
          --out_root "${TOKENS_DIR}" \
          --gpu_devices "${GPUS}" \
          "${overwrite_arg[@]}"
      else
        echo "  [stage4] token pkl complete; rebuilding record/manifest only"
      fi

      bash "${PROJECT_ROOT}/scripts/preprocess/concat_record_libero.sh" "${TOKENS_DIR}"

      "${PYTHON}" concat_action_world_model_data_libero.py \
        --source_dir_patterns "${SUITE}_his_${HIS}_{}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
        --all_patterns "${SUITE}_his_${HIS}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
        --processed_data_root "${PROCESSED_DATA_ROOT}"
    ) > "${log}" 2>&1
    local rc=$?
    echo "  [stage4] exit=${rc}"
    for f in "${MANIFEST}" "${VAL_IND_REC}" "${VAL_OOD_REC}"; do
      [[ -f "$f" ]] || { echo "  [stage4] FAIL: missing $f"; return 5; }
    done
    local n=$("${PYTHON}" -c "import json; print(len(json.load(open('${MANIFEST}'))))")
    echo "  [stage4] manifest entries: ${n}"
  fi

  # ── Stage 5 ── write training yaml configs ────────────────────────────────
  cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize.yaml" <<EOF
META:
  - path: '${MANIFEST}'
prompt_text: 'Finish the task: {task_text}.'
EOF
  cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize_val_ind.yaml" <<EOF
META:
  - path: '${VAL_IND_REC}'
prompt_text: 'Finish the task: {task_text}.'
EOF
  cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize_val_ood.yaml" <<EOF
META:
  - path: '${VAL_OOD_REC}'
prompt_text: 'Finish the task: {task_text}.'
EOF
  echo "  [stage5] wrote yaml configs to ${CONFIG_DIR}"
  echo "  ✓ SUITE ${SUITE} DONE"
}

OVERALL_RC=0
for suite in ${SUITES}; do
  if ! process_one_suite "${suite}"; then
    echo "✗ ${suite} FAILED"
    OVERALL_RC=1
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " ALL DONE  (overall_rc=${OVERALL_RC})  $(date)"
echo "════════════════════════════════════════════════════════════════"
for s in libero_goal libero_10 libero_object libero_spatial; do
  f="${CONCATE_DIR}/${s}_his_${HIS}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
  if [[ -f "$f" ]]; then
    n=$("${PYTHON}" -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null)
    echo "  ${s}: ${n} manifest entries"
  else
    echo "  ${s}: MISSING manifest"
  fi
done
exit ${OVERALL_RC}
