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

## 2. Manual stages

```bash
RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_manual"
RW="${RUN_ROOT}/coldstart/reward"
HID="${RUN_ROOT}/coldstart/hidden"

# collect
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_onetraj task=openvla_onetraj_coldstart_libero logger=tensorboard \
  collect.task_ids=all collect.episodes_per_task=4 collect.episode_horizon=300 \
  collect.envs_per_gpu=32 collect.memory_fraction=0.9 collect.gpu_id=0 \
  task.openvla_oft.hdf5_reward_dir="${RW}" task.openvla_oft.action_hidden_dir="${HID}" \
  training.out_dir="${RUN_ROOT}/collect"

# warmup + cotrain
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden task=openvla_onetraj_coldstart_libero logger=tensorboard \
  offline_warmup.data_dir="${RW}" offline_warmup.hidden_dir="${HID}" offline_warmup.task_id=null \
  'env.task_ids=[0,1,2,3,4,5,6,7,8,9]' training.out_dir="${RUN_ROOT}/cotrain" \
  training.wm_warmup_steps=256 training.classifier_warmup_steps=256 \
  dataloader.batch_size=96 training.classifier_batch_size=512 \
  online_rollout.buffer_size=10000 online_rollout.total_env_steps=0
```

## 3. Validate

```bash
python -m pytest tests/e2e_tests/test_s6_ray_coldstart_collect.py -q
DVLA_GPU_E2E=1 python -m pytest tests/e2e_tests/test_s6_real_oft_coldstart.py -q -s
```

## 4. Inspect results

```bash
python - <<'PY'
from pathlib import Path
import h5py, torch
reward_dir = Path("<run_root>/coldstart/reward")
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
