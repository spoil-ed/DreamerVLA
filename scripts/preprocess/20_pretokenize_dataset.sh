#!/usr/bin/env bash
# Build the pretokenized LIBERO dataset: image tree, convs, tokens, records, configs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/_env.sh"

if [[ "${FILTER_NOOPS}" != "1" ]]; then
  echo "Pretokenize configs currently target *_no_noops_t_* paths; set FILTER_NOOPS=1 or RUN_PRETOKENIZE=0." >&2
  exit 3
fi

require_hdf5_files "${HDF5_DIR}" "[preprocess:20_pretokenize_dataset.sh] missing final HDF5 input" 5

raw_n="$(hdf5_count "${HDF5_DIR}")"
preprocess_log "stage 4 input hdf5=${raw_n} image_state_dir=${IMG_STATE_DIR}"
need_image_state=1
if [[ -d "${IMG_STATE_DIR}" ]]; then
  task_dirs="$(child_dir_count "${IMG_STATE_DIR}")"
  if [[ "${task_dirs}" -ge "${raw_n}" && "${FORCE}" != "1" ]]; then
    preprocess_log "stage 4 skipped: ${task_dirs} task dirs present"
    need_image_state=0
  else
    preprocess_log "stage 4 has ${task_dirs}/${raw_n} task dirs; rebuilding"
    rm -rf "${IMG_STATE_DIR}"
  fi
fi

if [[ "${need_image_state}" == "1" ]]; then
  log="${LOG_DIR}/${TASK}_stage4_image_state_$(date +%Y%m%d_%H%M%S).log"
  preprocess_log "stage 4: image/state/action extraction -> ${log}"
  "${PYTHON}" -m dreamer_vla.preprocess.libero_utils.regenerate_libero_dataset_save_img_action_state_wrist \
    --libero_task_suite "${TASK}" \
    --image_resolution "${IMAGE_RESOLUTION}" \
    --raw_data_dir "${HDF5_DIR}" \
    --save_dir "${IMG_STATE_DIR}" \
    > "${log}" 2>&1
fi
task_dirs="$(child_dir_count "${IMG_STATE_DIR}")"
if [[ "${task_dirs}" -lt "${raw_n}" ]]; then
  echo "[preprocess:20_pretokenize_dataset.sh] only ${task_dirs}/${raw_n} task dirs created: ${IMG_STATE_DIR}" >&2
  exit 5
fi

CONV_TRAIN="${CONVS_DIR}/${TASK}_his_${HIS}_train_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
CONV_VIND="${CONVS_DIR}/${TASK}_his_${HIS}_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
CONV_VOOD="${CONVS_DIR}/${TASK}_his_${HIS}_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}.json"
if [[ -f "${CONV_TRAIN}" && -f "${CONV_VIND}" && -f "${CONV_VOOD}" && "${FORCE}" != "1" ]]; then
  preprocess_log "stage 5 skipped: convs present"
else
  log="${LOG_DIR}/${TASK}_stage5_convs_$(date +%Y%m%d_%H%M%S).log"
  preprocess_log "stage 5: conversation JSONs -> ${log}"
  "${PYTHON}" -m dreamer_vla.preprocess.action_state_model_conv_generation \
    --base_dir "${IMG_STATE_DIR}" \
    --his "${HIS}" \
    --len_action "${ACTION_HORIZON}" \
    --task_name "${TASK_NAME}" \
    --resolution "${IMAGE_RESOLUTION}" \
    --with_state \
    --img_names imgs_third_view imgs_wrist \
    --output_dir "${CONVS_DIR}" \
    > "${log}" 2>&1
fi
for conv in "${CONV_TRAIN}" "${CONV_VIND}" "${CONV_VOOD}"; do
  [[ -f "${conv}" ]] || { echo "[preprocess:20_pretokenize_dataset.sh] missing conv JSON: ${conv}" >&2; exit 5; }
done

TOK_TRAIN="${TOKENS_DIR}/${TASK}_his_${HIS}_train_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
TOK_VIND="${TOKENS_DIR}/${TASK}_his_${HIS}_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
TOK_VOOD="${TOKENS_DIR}/${TASK}_his_${HIS}_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}"
TRAIN_REC="${TOK_TRAIN}/record.json"
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
preprocess_log "stage 6 counts train ${n_tok_train}/${n_conv_train}, val_ind ${n_tok_vind}/${n_conv_vind}, val_ood ${n_tok_vood}/${n_conv_vood}; records ${n_rec_train}/${n_rec_vind}/${n_rec_vood}; manifest ${n_manifest}/${n_total}"

tokens_complete=0
records_complete=0
if [[ "${n_tok_train}" -eq "${n_conv_train}" && "${n_tok_vind}" -eq "${n_conv_vind}" && "${n_tok_vood}" -eq "${n_conv_vood}" ]]; then
  tokens_complete=1
fi
if [[ "${tokens_complete}" == "1" && "${n_rec_train}" -eq "${n_tok_train}" && "${n_rec_vind}" -eq "${n_tok_vind}" && "${n_rec_vood}" -eq "${n_tok_vood}" && "${n_manifest}" -eq "${n_total}" ]]; then
  records_complete=1
fi

if [[ "${tokens_complete}" == "1" && "${records_complete}" == "1" && "${FORCE}" != "1" ]]; then
  preprocess_log "stage 6 skipped: token pkl + records complete"
else
  if [[ "${tokens_complete}" != "1" || "${FORCE}" == "1" ]]; then
    if ! has_regular_files "${TOKENIZER_PATH}"; then
      echo "Missing Lumina tokenizer/backbone files for pretokenization: ${TOKENIZER_PATH}" >&2
      echo "Run: DOWNLOAD_LIBERO=0 bash scripts/download_assets.sh" >&2
      exit 4
    fi
  fi

  log="${LOG_DIR}/${TASK}_stage6_tokens_manifest_$(date +%Y%m%d_%H%M%S).log"
  preprocess_log "stage 6: pretokenize + records + manifest -> ${log} (GPUs=${GPUS})"
  set +e
  (
    set -e
    if [[ "${tokens_complete}" != "1" || "${FORCE}" == "1" ]]; then
      overwrite_arg=()
      [[ "${OVERWRITE_TOKENS}" == "1" ]] && overwrite_arg=(--overwrite)
      "${PYTHON}" -m dreamer_vla.preprocess.pretoken_state_action_model \
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
      echo "token pkl complete; rebuilding record/manifest only"
    fi

    bash "${DVLA_ROOT}/scripts/preprocess/concat_record_libero.sh" "${TOKENS_DIR}"

    "${PYTHON}" -m dreamer_vla.preprocess.concat_action_world_model_data_libero \
      --source_dir_patterns "${TASK}_his_${HIS}_{}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
      --all_patterns "${TASK}_${SUFFIX}" \
      --processed_data_root "${PROCESSED_DATA_ROOT}"
  ) > "${log}" 2>&1
  rc=$?
  set -e
  preprocess_log "stage 6 exit=${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[preprocess:20_pretokenize_dataset.sh] stage 6 failed; see ${log}" >&2
    echo "[preprocess:20_pretokenize_dataset.sh] last log lines:" >&2
    tail -n 40 "${log}" >&2
    exit 5
  fi
fi

"${PYTHON}" -m dreamer_vla.preprocess.validate_libero_data_prep \
  --data-root "${DVLA_DATA_ROOT}" \
  --processed-data-root "${PROCESSED_DATA_ROOT}" \
  --suites "${TASK}" \
  --his "${HIS}" \
  --action-horizon "${ACTION_HORIZON}" \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --skip-configs

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
preprocess_log "stage 7 wrote YAML configs to ${CONFIG_DIR}"

"${PYTHON}" -m dreamer_vla.preprocess.validate_libero_data_prep \
  --data-root "${DVLA_DATA_ROOT}" \
  --processed-data-root "${PROCESSED_DATA_ROOT}" \
  --suites "${TASK}" \
  --his "${HIS}" \
  --action-horizon "${ACTION_HORIZON}" \
  --image-resolution "${IMAGE_RESOLUTION}"
