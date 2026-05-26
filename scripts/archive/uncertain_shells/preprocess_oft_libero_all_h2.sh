#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/data/spoil/workspace/DreamerVLA}
PYTHON=${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}
OPENVLA_OFT_DIR=${OPENVLA_OFT_DIR:-/mnt/data/spoil/workspace/openvla-oft}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
MASTER_PORT=${MASTER_PORT:-29517}
CHUNK_SIZE=${CHUNK_SIZE:-16}
SUITES=${SUITES:-goal object spatial}
SKIP_CD_SIDECARS=${SKIP_CD_SIDECARS:-1}
OVERWRITE=${OVERWRITE:-0}
MANUAL_SHARD=${MANUAL_SHARD:-1}

export CUDA_VISIBLE_DEVICES
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

run_suite() {
  local suite="$1"
  local hdf5_dir ckpt key tag

  case "$suite" in
    goal|libero_goal)
      suite="goal"
      hdf5_dir="$ROOT/data/processed_data/libero_goal_no_noops_t_256"
      ckpt="$ROOT/data/ckpts/OpenVLA-OFT/libero_goal_hdf5_latest_6650"
      key="libero_goal_no_noops"
      tag="hdf5_6650"
      ;;
    object|libero_object)
      suite="object"
      hdf5_dir="$ROOT/data/processed_data/libero_object_no_noops_t_256"
      ckpt="$ROOT/data/ckpts/OpenVLA-OFT/libero_object"
      key="libero_object_no_noops"
      tag="official"
      ;;
    spatial|libero_spatial)
      suite="spatial"
      hdf5_dir="$ROOT/data/processed_data/libero_spatial_no_noops_t_256"
      ckpt="$ROOT/data/ckpts/OpenVLA-OFT/libero_spatial"
      key="libero_spatial_no_noops"
      tag="official"
      ;;
    10|libero_10)
      suite="10"
      hdf5_dir="$ROOT/data/processed_data/libero_10_no_noops_t_256"
      ckpt="$ROOT/data/ckpts/OpenVLA-OFT/libero_10"
      key="libero_10_no_noops"
      tag="official"
      ;;
    *)
      echo "unknown suite: $suite" >&2
      return 2
      ;;
  esac

  local prefix="$ROOT/data/processed_data/libero_${suite}_no_noops_t_256_oft_${tag}"
  local out_c="${prefix}_action_hidden_c_h2_h8"
  local out_d="${prefix}_action_hidden_d_h2_h8"
  local out_action="${prefix}_legacy_action_hidden_vla_policy_h2"
  local log="$ROOT/data/outputs/preprocess_oft_libero_${suite}_${tag}_h2.log"

  mkdir -p "$ROOT/data/outputs"
  echo "[oft-libero] suite=$suite hdf5=$hdf5_dir ckpt=$ckpt log=$log"
  local cd_args=()
  if [[ "$SKIP_CD_SIDECARS" == "1" || "$SKIP_CD_SIDECARS" == "true" ]]; then
    cd_args+=(--skip-cd-sidecars)
  fi

  local common_args=(
    --openvla-oft-dir "$OPENVLA_OFT_DIR" \
    --hdf5-dir "$hdf5_dir" \
    --out-c-dir "$out_c" \
    --out-d-dir "$out_d" \
    --out-action-dir "$out_action" \
    --oft-ckpt "$ckpt" \
    --unnorm-key "$key" \
    --image-keys agentview_rgb eye_in_hand_rgb \
    --num-images-in-input 4 \
    --include-state \
    --center-crop \
    --rotate-images-180 \
    --history 2 \
    --chunk-size "$CHUNK_SIZE" \
    --output-dtype float16 \
    "${cd_args[@]}"
  )
  if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
    common_args+=(--overwrite)
  fi

  if [[ "$MANUAL_SHARD" == "1" || "$MANUAL_SHARD" == "true" ]]; then
    IFS=',' read -r -a shard_gpus <<< "$VISIBLE_DEVICES"
    local world_size=${#shard_gpus[@]}
    if [[ "$world_size" -lt 1 ]]; then
      echo "no CUDA devices configured for manual shard" >&2
      return 2
    fi
    local pids=()
    local rank
    for rank in "${!shard_gpus[@]}"; do
      local gpu="${shard_gpus[$rank]}"
      (
        CUDA_VISIBLE_DEVICES="$gpu" MANUAL_SHARD_RANK="$rank" MANUAL_SHARD_WORLD_SIZE="$world_size" \
          "$PYTHON" "$ROOT/scripts/preprocess_oft_action_hidden.py" "${common_args[@]}"
      ) 2>&1 | sed -u "s/^/[gpu${gpu}-rank${rank}] /" &
      pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  else
    "$PYTHON" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node=2 --master_port="$MASTER_PORT" \
      "$ROOT/scripts/preprocess_oft_action_hidden.py" \
      "${common_args[@]}" 2>&1
  fi | tee "$log"
}

cd "$ROOT"
for suite in $SUITES; do
  run_suite "$suite"
done
