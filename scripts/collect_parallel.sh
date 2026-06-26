#!/usr/bin/env bash
# Data-parallel cold-start collection: one Ray collect job per GPU, each on a
# task subset, writing a distinct shard; shards are merged into one coldstart
# dir that the warmup/cotrain launcher can consume with skip_collect=true.
#
# No code change to the collector is needed: the OFT collect path hard-codes the
# shard name (ray_shard_000.hdf5), so each job writes to its OWN dir and we merge
# afterwards with unique names. offline_seed reads every *.hdf5 in the dir and
# pairs reward/hidden shards by identical filename.
#
# Usage:
#   bash scripts/collect_parallel.sh task=goal ngpu=6 run_root=/path/to/run \
#        [episodes_per_task=50] [episode_horizon=300] [num_workers=4] \
#        [task_ids=0,1,2,3,4,5,6,7,8,9] [dry_run=false]
#
# Then run warmup + (full) cotrain on the merged data:
#   bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal \
#        skip_collect=true run_root=/path/to/run ngpu=6 profile=multi_gpu
set -euo pipefail

# ---- defaults -------------------------------------------------------------
task=goal
ngpu=6
run_root=""
episodes_per_task=50
episode_horizon=300
num_workers=4
task_ids="0,1,2,3,4,5,6,7,8,9"
dry_run=false
python_bin="${PYTHON:-python}"

for kv in "$@"; do
  key="${kv%%=*}"; val="${kv#*=}"
  case "$key" in
    task) task="$val" ;;
    ngpu) ngpu="$val" ;;
    run_root) run_root="$val" ;;
    episodes_per_task) episodes_per_task="$val" ;;
    episode_horizon) episode_horizon="$val" ;;
    num_workers) num_workers="$val" ;;
    task_ids) task_ids="$val" ;;
    dry_run) dry_run="$val" ;;
    *) echo "[collect_parallel] unknown arg: $kv" >&2; exit 2 ;;
  esac
done

# ---- task -> hydra task name (mirrors configs/scripts/coldstart_warmup_cotrain.yaml) ----
case "$task" in
  goal)    hydra_task=openvla_onetraj_coldstart_libero ;;
  object)  hydra_task=openvla_onetraj_coldstart_libero_object ;;
  spatial) hydra_task=openvla_onetraj_coldstart_libero_spatial ;;
  10)      hydra_task=openvla_onetraj_coldstart_libero_10 ;;
  *) echo "[collect_parallel] unknown task '$task' (expected goal|object|spatial|10)" >&2; exit 2 ;;
esac

if [[ -z "$run_root" ]]; then
  data_root="${DVLA_DATA_ROOT:-${DVLA_ROOT:-$(pwd -P)}/data}"
  run_root="${data_root}/outputs/coldstart_warmup_cotrain/parallel_$(date +%Y%m%d_%H%M%S)"
fi

IFS=',' read -r -a ALL_TASKS <<< "$task_ids"
ntasks="${#ALL_TASKS[@]}"
echo "[collect_parallel] task=$task ($hydra_task)  ngpu=$ngpu  tasks=$ntasks  run_root=$run_root"

# ---- round-robin task -> GPU assignment -----------------------------------
declare -a GPU_TASKS
for ((g=0; g<ngpu; g++)); do GPU_TASKS[$g]=""; done
for ((t=0; t<ntasks; t++)); do
  g=$(( t % ngpu ))
  if [[ -z "${GPU_TASKS[$g]}" ]]; then GPU_TASKS[$g]="${ALL_TASKS[$t]}"; else GPU_TASKS[$g]="${GPU_TASKS[$g]},${ALL_TASKS[$t]}"; fi
done

mkdir -p "$run_root/logs" "$run_root/coldstart/reward" "$run_root/coldstart/hidden"

# ---- launch one collect job per GPU that has tasks ------------------------
declare -a PIDS GPUS_USED
for ((g=0; g<ngpu; g++)); do
  subset="${GPU_TASKS[$g]}"
  [[ -z "$subset" ]] && continue
  rw="$run_root/coldstart_g${g}/reward"; hid="$run_root/coldstart_g${g}/hidden"
  log="$run_root/logs/collect_g${g}.log"
  cmd=( "$python_bin" -m dreamervla.train
        experiment=collect_rollouts_ray task="$hydra_task" logger=tensorboard
        "collect.task_ids=[${subset}]"
        collect.episodes_per_task="$episodes_per_task"
        collect.episode_horizon="$episode_horizon"
        collect.memory_fraction=0.9
        env.num_workers="$num_workers"
        task.openvla_oft.hdf5_reward_dir="$rw"
        task.openvla_oft.input_token_hidden_dir="$hid"
        "++collect.hdf5_reward_dir=$rw"
        "++collect.hidden_dir=$hid"
        "++collect.oft_latent_spec.expected_action_head_type=\${task.openvla_oft.input_tokens.expected_action_head_type}"
        "++collect.oft_latent_spec.expected_obs_hidden_source=\${task.openvla_oft.input_tokens.expected_obs_hidden_source}"
        "++collect.oft_latent_spec.expected_prompt_style=\${task.openvla_oft.input_tokens.expected_prompt_style}"
        "++collect.oft_latent_spec.expected_history=\${task.openvla_oft.input_tokens.expected_history}"
        "++collect.oft_latent_spec.expected_include_state=\${task.openvla_oft.input_tokens.expected_include_state}"
        "++collect.oft_latent_spec.expected_rotate_images_180=\${task.openvla_oft.input_tokens.expected_rotate_images_180}"
        "++collect.oft_latent_spec.token_dim=\${task.openvla_oft.input_tokens.token_dim}"
        "++collect.oft_latent_spec.token_count=\${task.openvla_oft.input_tokens.token_count}"
        "++collect.oft_latent_spec.wm_obs_dim=\${task.openvla_oft.input_tokens.wm_obs_dim}"
        "++collect.oft_latent_spec.chunk_size=\${task.openvla_oft.input_tokens.chunk_size}"
        training.out_dir="$run_root/collect_g${g}" )
  echo "[collect_parallel] GPU $g  tasks=[${subset}]  -> $log"
  if [[ "$dry_run" == "true" ]]; then
    echo "  CUDA_VISIBLE_DEVICES=$g ${cmd[*]}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$g" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NCCL_NVLS_ENABLE=0 \
    "${cmd[@]}" > "$log" 2>&1 &
  PIDS+=("$!"); GPUS_USED+=("$g")
done

if [[ "$dry_run" == "true" ]]; then
  echo "[collect_parallel] dry_run: not launched. Merge + follow-up below would run after collection."
  echo "[collect_parallel] follow-up: bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=$task skip_collect=true run_root=$run_root ngpu=$ngpu profile=multi_gpu"
  exit 0
fi

# ---- wait for all jobs, fail loudly if any died ---------------------------
fail=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[collect_parallel] FAILED: GPU ${GPUS_USED[$i]} (see $run_root/logs/collect_g${GPUS_USED[$i]}.log)" >&2
    fail=1
  fi
done
[[ "$fail" == "1" ]] && { echo "[collect_parallel] one or more collect jobs failed; aborting before merge." >&2; exit 1; }

# ---- merge shards into the shared coldstart dir (unique, matching names) ---
echo "[collect_parallel] merging shards -> $run_root/coldstart/{reward,hidden}"
for g in "${GPUS_USED[@]}"; do
  rwdir="$run_root/coldstart_g${g}/reward"
  [[ -d "$rwdir" ]] || { echo "[collect_parallel] WARN: no reward dir for GPU $g — skipping" >&2; continue; }
  shopt -s nullglob
  found=0
  for src_rw in "$rwdir"/*.hdf5; do
    base="$(basename "$src_rw")"; src_hd="$run_root/coldstart_g${g}/hidden/$base"
    [[ -f "$src_hd" ]] || { echo "[collect_parallel] WARN: missing hidden shard $base for GPU $g" >&2; continue; }
    # unique name keyed by GPU + original shard name, identical in reward/ and hidden/
    dst="g${g}_${base}"
    cp -f "$src_rw" "$run_root/coldstart/reward/$dst"
    cp -f "$src_hd" "$run_root/coldstart/hidden/$dst"
    found=1
  done
  shopt -u nullglob
  [[ "$found" == "0" ]] && echo "[collect_parallel] WARN: GPU $g produced no shards" >&2
done
# preprocess_config.json (identical across jobs) — copy one
pc="$(find "$run_root"/coldstart_g*/hidden -name preprocess_config.json 2>/dev/null | head -1 || true)"
[[ -n "$pc" ]] && cp -f "$pc" "$run_root/coldstart/hidden/preprocess_config.json"

nshards=$(find "$run_root/coldstart/reward" -name '*.hdf5' | wc -l | tr -d ' ')
echo "[collect_parallel] done. merged $nshards reward shard(s) into $run_root/coldstart"
echo "[collect_parallel] next (warmup + full cotrain on the merged data):"
echo "  NCCL_NVLS_ENABLE=0 bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=$task skip_collect=true run_root=$run_root ngpu=$ngpu profile=multi_gpu"
