# OpenVLA-OFT Cold-Start Collection And Warmup

This recipe reproduces the OpenVLA-OFT one-trajectory LIBERO cold-start flow:

```text
collect rollouts -> reward HDF5 + obs_embedding sidecar -> offline WM/classifier warmup
```

The default release path is warmup-only: `online_rollout.total_env_steps=0`.
Online cotrain remains available through Hydra overrides, but it is not the
default release check.

## Requirements

Activate the project environment and point the data root at the prepared assets:

```bash
cd DreamerVLA
conda activate dreamervla
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-$(pwd -P)/data}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

Required files for the selected suite:

```text
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/<suite-ckpt>/
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/<suite-ckpt>/dataset_statistics.json
${DVLA_DATA_ROOT}/datasets/libero/<suite>/*.hdf5
```

The OpenVLA-OFT checkpoint must use the OpenVLA-OFT transformers fork. Run the
install verifier before long jobs:

```bash
bash scripts/install/60_verify.sh
```

## Tasks

| Launcher `task=` | Hydra task | LIBERO suite |
| --- | --- | --- |
| `goal` | `OpenVLA_Onetraj_ColdStart_LIBERO` | `libero_goal` |
| `object` | `OpenVLA_Onetraj_ColdStart_LIBERO_Object` | `libero_object` |
| `spatial` | `OpenVLA_Onetraj_ColdStart_LIBERO_Spatial` | `libero_spatial` |
| `10` | `OpenVLA_Onetraj_ColdStart_LIBERO_10` | `libero_10` |

All model, world-model, actor, classifier, token, and action dimensions are
derived from `task.openvla_oft`.

## Run

No-Ray collector:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal
```

Ray collector:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal
```

The release profile defaults are intentionally configurable Hydra overrides:

| Parameter | Release default |
| --- | --- |
| `collect.task_ids` | `all` |
| `collect.episodes_per_task` | `4` |
| `collect.episode_horizon` | `300` |
| no-Ray `collect.envs_per_gpu` | `32` |
| Ray `collect.num_workers` -> `env.num_workers` | `16` |
| Ray `collect.max_steps` -> `rollout.max_steps` | `1200` |
| Ray stop count | derived from `episodes_per_task * selected task_ids` |
| `warmup.wm_steps` -> `training.wm_warmup_steps` | `256` |
| `warmup.classifier_steps` -> `training.classifier_warmup_steps` | `256` |
| `warmup.batch_size` -> `dataloader.batch_size` | `96` |
| `warmup.classifier_batch_size` -> `training.classifier_batch_size` | `512` |
| `warmup.total_env_steps` -> `online_rollout.total_env_steps` | `0` |

Adjust them with normal launcher overrides:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
    task=goal \
    collect.envs_per_gpu=16 \
    collect.episodes_per_task=2 \
    warmup.wm_steps=16 \
    warmup.classifier_steps=16
```

Print the exact commands without running them:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal dry_run=true
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal dry_run=true
```

## Output

Each run writes one root:

```text
<run_root>/
  collect/
  coldstart/
    reward/*.hdf5
    hidden/*.hdf5
    hidden/preprocess_config.json
  cotrain/
    ckpt/wm_warmup.ckpt
    ckpt/classifier_warmup.ckpt
```

Use `run_root=...` for a stable output path. Use `skip_collect=true` to reuse an
existing `<run_root>/coldstart/{reward,hidden}` pair and run warmup only.

## Inspect Results

Success count:

```bash
python - <<'PY'
from pathlib import Path
import h5py

reward_dir = Path("<run_root>/coldstart/reward")
total = success = 0
for path in sorted(reward_dir.glob("*.hdf5")):
    with h5py.File(path, "r") as handle:
        for key in handle["data"]:
            rewards = handle["data"][key]["sparse_rewards"][()]
            total += 1
            success += int(rewards.max() > 0)
print(f"success={success}/{total}")
PY
```

Sidecar shape:

```bash
python - <<'PY'
from pathlib import Path
import h5py

hidden = next(Path("<run_root>/coldstart/hidden").glob("*.hdf5"))
with h5py.File(hidden, "r") as handle:
    ds = handle["data"]["demo_0"]["obs_embedding"]
    print(ds.shape, ds.dtype)
PY
```

For OpenVLA-OFT action-query hidden, expect `(T, 229376)` and `float16`.

Warmup checkpoints:

```bash
python - <<'PY'
from pathlib import Path
import torch

ckpt = Path("<run_root>/cotrain/ckpt")
for name in ("wm_warmup.ckpt", "classifier_warmup.ckpt"):
    payload = torch.load(ckpt / name, map_location="cpu", weights_only=False)
    print(name, sorted(payload))
PY
```

## Manual Commands

No-Ray collect:

```bash
RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_manual"
RW="${RUN_ROOT}/coldstart/reward"
HID="${RUN_ROOT}/coldstart/hidden"

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_onetraj \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  logger=tensorboard \
  collect.task_ids=all \
  collect.episodes_per_task=4 \
  collect.episode_horizon=300 \
  collect.envs_per_gpu=32 \
  collect.memory_fraction=0.9 \
  collect.gpu_id=0 \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.action_hidden_dir="${HID}" \
  training.out_dir="${RUN_ROOT}/collect"
```

Warmup-only:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  logger=tensorboard \
  offline_warmup.data_dir="${RW}" \
  offline_warmup.hidden_dir="${HID}" \
  offline_warmup.task_id=null \
  'env.task_ids=[0,1,2,3,4,5,6,7,8,9]' \
  training.out_dir="${RUN_ROOT}/cotrain" \
  training.wm_warmup_steps=256 \
  training.classifier_warmup_steps=256 \
  dataloader.batch_size=96 \
  training.classifier_batch_size=512 \
  online_rollout.buffer_size=10000 \
  online_rollout.total_env_steps=0
```

Ray collect:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_ray \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  logger=tensorboard \
  collect.task_ids=all \
  collect.episodes_per_task=4 \
  collect.episode_horizon=300 \
  collect.memory_fraction=0.9 \
  env.num_workers=16 \
  rollout.max_steps=1200 \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.action_hidden_dir="${HID}" \
  training.out_dir="${RUN_ROOT}/collect"
```

## Validation Commands

Fast Ray contract test:

```bash
python -m pytest tests/e2e_tests/test_s6_ray_coldstart_collect.py -q
```

Real OFT Ray smoke is gated because it loads the checkpoint and LIBERO:

```bash
DVLA_GPU_E2E=1 python -m pytest tests/e2e_tests/test_s6_real_oft_coldstart.py -q -s
```

## Troubleshooting

- If collection produces no successful episodes, first confirm the route is
  using OpenVLA-OFT action chunks. The collector and Ray inference worker execute
  `task.openvla_oft.chunk_size` actions open-loop before consuming a new chunk.
- If warmup says replay is empty, collect episodes with
  `collect.episode_horizon >= online_rollout.sequence_length`.
- If sidecar validation fails, use one Hydra task consistently for collect and
  warmup; do not mix VLA checkpoint and LIBERO suite manually.
- If Ray hangs during startup, run the synthetic Ray test above and then retry
  with fewer `env.num_workers`.
