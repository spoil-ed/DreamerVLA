#!/usr/bin/env bash
# Build LIBERO image/state trees, conversations, token records, manifests, configs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
TASK="${TASK:-libero_goal}"
LIBERO_SUITE="${LIBERO_SUITE:-${TASK}}"
TASK_NAME="${TASK_NAME:-${TASK}}"
if [[ "${LIBERO_SUITE}" == "${TASK}" ]]; then
  case "${TASK_NAME}" in
    RynnVLA_LIBERO|OpenVLA_Onetraj_LIBERO) LIBERO_SUITE="libero_goal" ;;
  esac
fi
ARTIFACT_NAME="${ARTIFACT_NAME:-${TASK_NAME}}"
if [[ "${ARTIFACT_NAME}" == "${TASK_NAME}" && "${TASK_NAME}" != "${LIBERO_SUITE}" ]]; then
  ARTIFACT_NAME="${TASK_NAME}_${LIBERO_SUITE}"
fi
HIS=1
ACTION_HORIZON=1
IMAGE_RESOLUTION=256
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-8}"
GPUS="${GPUS:-0}"
OVERWRITE="${OVERWRITE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS}}"
cd "${DVLA_ROOT}"

PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data/${ARTIFACT_NAME}"
HDF5_DIR="${PROCESSED_DATA_ROOT}/no_noops_t_256"
IMG_STATE_DIR="${PROCESSED_DATA_ROOT}/image_state_action_t_256"
CONVS_DIR="${PROCESSED_DATA_ROOT}/convs"
TOKENS_DIR="${PROCESSED_DATA_ROOT}/tokens"
CONCATE_DIR="${PROCESSED_DATA_ROOT}/concate_tokens"
CONFIG_DIR="${DVLA_DATA_ROOT}/configs/${ARTIFACT_NAME}"
TOKENIZER_PATH="${DVLA_DATA_ROOT}/checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768"
SUFFIX="his_1_third_view_wrist_w_state_1_256"
TASK_LABEL="${TASK_LABEL:-}"
if [[ -z "${TASK_LABEL}" ]]; then
  if [[ "${TASK_NAME}" == "${LIBERO_SUITE}" ]]; then
    TASK_LABEL="${LIBERO_SUITE#libero_}"
  else
    TASK_LABEL="${TASK_NAME}"
  fi
fi

mkdir -p "${CONVS_DIR}" "${TOKENS_DIR}" "${CONCATE_DIR}" "${CONFIG_DIR}"
if [[ -z "$(find "${HDF5_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print -quit 2>/dev/null || true)" ]]; then
  echo "No no-op-filtered HDF5 files found under: ${HDF5_DIR}" >&2
  echo "Run: bash scripts/preprocess/prepare_libero_data.sh task=${TASK} only=[10_hdf5_reward]" >&2
  exit 5
fi

if [[ "${OVERWRITE}" == "1" || ! -d "${IMG_STATE_DIR}" ]]; then
  [[ "${OVERWRITE}" == "1" ]] && rm -rf "${IMG_STATE_DIR}"
  python -m dreamervla.preprocess.libero_utils.regenerate_libero_dataset_save_img_action_state_wrist \
    --libero_task_suite "${LIBERO_SUITE}" \
    --image_resolution "${IMAGE_RESOLUTION}" \
    --raw_data_dir "${HDF5_DIR}" \
    --save_dir "${IMG_STATE_DIR}"
else
  echo "[20_pretokenize_dataset] skip image/state/action: ${IMG_STATE_DIR}"
fi

python -m dreamervla.preprocess.action_state_model_conv_generation \
  --base_dir "${IMG_STATE_DIR}" \
  --his "${HIS}" \
  --len_action "${ACTION_HORIZON}" \
  --task_name "${TASK_LABEL}" \
  --resolution "${IMAGE_RESOLUTION}" \
  --with_state \
  --img_names imgs_third_view imgs_wrist \
  --output_dir "${CONVS_DIR}"

if [[ "${OVERWRITE}" == "1" ]]; then
  python -m dreamervla.preprocess.pretoken_state_action_model \
    --task "${TASK_LABEL}" \
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
    --overwrite
else
  python -m dreamervla.preprocess.pretoken_state_action_model \
    --task "${TASK_LABEL}" \
    --resolution "${IMAGE_RESOLUTION}" \
    --with_state \
    --img_names imgs_third_view imgs_wrist \
    --his "${HIS}" \
    --len_action "${ACTION_HORIZON}" \
    --num_procs "${PRETOKENIZE_PROCS}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --in_filename_dir "${CONVS_DIR}" \
    --out_root "${TOKENS_DIR}" \
    --gpu_devices "${GPUS}"
fi

bash "${DVLA_ROOT}/scripts/preprocess/concat_record_libero.sh" "${TOKENS_DIR}"

python -m dreamervla.preprocess.concat_action_world_model_data_libero \
  --source_dir_patterns "libero_${TASK_LABEL}_his_${HIS}_{}_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}" \
  --all_patterns "${ARTIFACT_NAME}_${SUFFIX}" \
  --processed_data_root "${PROCESSED_DATA_ROOT}"

if [[ "${ARTIFACT_NAME}" == "${LIBERO_SUITE}" ]]; then
  python -m dreamervla.preprocess.validate_libero_data_prep \
    --data-root "${DVLA_DATA_ROOT}" \
    --processed-data-root "${PROCESSED_DATA_ROOT}" \
    --suites "${ARTIFACT_NAME}" \
    --his "${HIS}" \
    --action-horizon "${ACTION_HORIZON}" \
    --image-resolution "${IMAGE_RESOLUTION}" \
    --skip-configs
fi

cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize.yaml" <<EOF
META:
  - path: '${CONCATE_DIR}/${ARTIFACT_NAME}_${SUFFIX}.json'
prompt_text: 'Finish the task: {task_text}.'
EOF
cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize_val_ind.yaml" <<EOF
META:
  - path: '${TOKENS_DIR}/libero_${TASK_LABEL}_his_${HIS}_val_ind_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF
cat > "${CONFIG_DIR}/${SUFFIX}_pretokenize_val_ood.yaml" <<EOF
META:
  - path: '${TOKENS_DIR}/libero_${TASK_LABEL}_his_${HIS}_val_ood_third_view_wrist_w_state_${ACTION_HORIZON}_${IMAGE_RESOLUTION}/record.json'
prompt_text: 'Finish the task: {task_text}.'
EOF

if [[ "${ARTIFACT_NAME}" == "${LIBERO_SUITE}" ]]; then
  python -m dreamervla.preprocess.validate_libero_data_prep \
    --data-root "${DVLA_DATA_ROOT}" \
    --processed-data-root "${PROCESSED_DATA_ROOT}" \
    --suites "${ARTIFACT_NAME}" \
    --his "${HIS}" \
    --action-horizon "${ACTION_HORIZON}" \
    --image-resolution "${IMAGE_RESOLUTION}"
fi
