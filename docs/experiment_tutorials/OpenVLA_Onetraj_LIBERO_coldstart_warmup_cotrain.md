# OpenVLA-OFT cold-start: collect + warmup + cotrain

Background (pipeline, WM architecture, memory/OOM breakdown, troubleshooting):
[EXPLAINED.md](EXPLAINED.md) · parameters: [../PARAMETERS.md](../PARAMETERS.md).
The e2e scripts take a suite shorthand `task=goal|object|spatial|10`.

## 0. Env

```bash
cd DreamerVLA
conda activate dreamervla
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-$(pwd -P)/data}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export NCCL_NVLS_ENABLE=0    # multi-GPU DDP cotrain
bash scripts/install/60_verify.sh
```

## 1. One-command e2e

```bash
# no-Ray
DVLA_ROOT=/path/to/DreamerVLA DVLA_DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal

# Ray
DVLA_ROOT=/path/to/DreamerVLA DVLA_DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal
```

Smoke / dry-run:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal debug=true
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal dry_run=true
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh   task=goal dry_run=true
```

Tuned smoke:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
    task=goal collect.envs_per_gpu=16 collect.episodes_per_task=2 \
    warmup.wm_steps=16 warmup.classifier_steps=16
```

## 2. Output layout

The one-command e2e is just orchestration — the two parts stay on disk separately:

- **collect** → collected episodes in the stable, resumable, per-suite space
  `${DVLA_DATA_ROOT}/collected_rollouts/<suite>/` with `reward/` + `hidden/` HDF5
  shards (`shard_NNN.hdf5`, appended on resume), `hidden/preprocess_config.json`,
  `collection_manifest.json` (counts, per-task breakdown, target, status), and a
  `resolved_config.yaml`.
- **cotrain** → run-isolated training outputs under `${RUN_ROOT}/cotrain/`
  (warmup ckpts + online ckpts + TensorBoard); the collect phase's own logs go to
  `${RUN_ROOT}/collect/`.

Resume/skip is automatic: set `collect_target_episodes=<N>` and a relaunch prints
an inspection report (collected / target, per-task counts, what is still needed)
and either tops up by appending shards or skips collection when the target is met.

## 3. Manual stages

```bash
SUITE=libero_goal
DATA="${DVLA_DATA_ROOT}/collected_rollouts/${SUITE}"
RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_manual"

# collect -> unified collected_rollouts/<suite> space
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_onetraj task=openvla_onetraj_coldstart_libero logger=tensorboard \
  collect.task_ids=all collect.episodes_per_task=4 collect.episode_horizon=300 \
  collect.envs_per_gpu=32 collect.memory_fraction=0.9 collect.gpu_id=0 \
  task.openvla_oft.hdf5_reward_dir="${DATA}/reward" task.openvla_oft.action_hidden_dir="${DATA}/hidden" \
  training.out_dir="${RUN_ROOT}/collect"

# warmup + cotrain -> reads the same collected_rollouts/<suite> space
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden task=openvla_onetraj_coldstart_libero logger=tensorboard \
  offline_warmup.data_dir="${DATA}/reward" offline_warmup.hidden_dir="${DATA}/hidden" offline_warmup.task_id=null \
  'env.task_ids=[0,1,2,3,4,5,6,7,8,9]' training.out_dir="${RUN_ROOT}/cotrain" \
  training.wm_warmup_steps=256 training.classifier_warmup_steps=256 \
  dataloader.batch_size=96 training.classifier_batch_size=512 \
  online_rollout.buffer_size=10000 online_rollout.total_env_steps=0
```

## 4. Validate

```bash
python -m pytest tests/e2e_tests/test_s6_ray_coldstart_collect.py -q
DVLA_GPU_E2E=1 python -m pytest tests/e2e_tests/test_s6_real_oft_coldstart.py -q -s
```

## 5. Inspect results

The collected-data summary lives in `collected_rollouts/<suite>/collection_manifest.json`
(total, per-task counts, target, status). For a deeper check (e.g. success rate):

```bash
python - <<'PY'
from pathlib import Path
import h5py, torch
reward_dir = Path("<DVLA_DATA_ROOT>/collected_rollouts/<suite>/reward")
total = success = 0
for path in sorted(reward_dir.glob("*.hdf5")):
    with h5py.File(path, "r") as handle:
        for key in handle["data"]:
            total += 1
            success += int(handle["data"][key]["sparse_rewards"][()].max() > 0)
print(f"success={success}/{total}")
ckpt = Path("<run_root>/cotrain/ckpt")
for name in ("wm_warmup.ckpt", "classifier_warmup.ckpt"):
    print(name, sorted(torch.load(ckpt / name, map_location="cpu", weights_only=False)))
PY
```
