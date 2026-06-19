# OpenVLA-OFT Cold-Start Rollout Collection

This recipe collects fresh OpenVLA-OFT one-trajectory LIBERO rollouts into the
same reward-HDF5 and `obs_embedding` sidecar layout used by the world model and
offline warmup routes.

## Tasks

| Hydra task | Suite | Checkpoint |
| --- | --- | --- |
| `OpenVLA_Onetraj_ColdStart_LIBERO` | `libero_goal` | `Openvla-oft-SFT-libero-goal-traj1` |
| `OpenVLA_Onetraj_ColdStart_LIBERO_Object` | `libero_object` | `Openvla-oft-SFT-libero-object-traj1` |
| `OpenVLA_Onetraj_ColdStart_LIBERO_Spatial` | `libero_spatial` | `Openvla-oft-SFT-libero-spatial-traj1` |
| `OpenVLA_Onetraj_ColdStart_LIBERO_10` | `libero_10` | `Openvla-oft-SFT-libero10-traj1` |

The task config is the source of truth for checkpoint path, dataset statistics,
action head type, hidden dimensions, token count, chunk size, and output dirs.

## Run

Single GPU, vectorized no-Ray collector:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_onetraj \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  collect.task_ids=all \
  collect.episodes_per_task=4 \
  collect.episode_horizon=300 \
  collect.envs_per_gpu=32
```

Multi-GPU sharded no-Ray collector:

```bash
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=osmesa \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dreamervla.train \
  experiment=collect_rollouts_onetraj \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  collect.task_ids=all \
  collect.episodes_per_task=4 \
  collect.episode_horizon=300 \
  collect.envs_per_gpu=32
```

Ray collector:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_ray \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  collect.task_ids=all \
  collect.episodes_per_task=4 \
  collect.episode_horizon=300 \
  env.num_workers=16 \
  rollout.max_steps=1200
```

For a throwaway run, override both output dirs:

```bash
task.openvla_oft.hdf5_reward_dir=/tmp/dvla_collect/reward \
task.openvla_oft.action_hidden_dir=/tmp/dvla_collect/hidden
```

## Tunable Parameters

| Key | Meaning |
| --- | --- |
| `collect.task_ids` | `all`, a list, a comma-separated string, or a single task id. |
| `collect.episodes_per_task` | Number of trajectories per selected LIBERO task. |
| `collect.episode_horizon` | Episode cap; use 300 for release collection. |
| `collect.envs_per_gpu` | no-Ray env subprocesses per GPU. |
| `env.num_workers` | Ray env workers. |
| Ray stop count | Derived from `collect.episodes_per_task * selected task_ids` for OpenVLA-OFT collection. |
| `rollout.max_steps` | Ray driver ticks; use enough waves for the requested trajectories. |
| `collect.memory_fraction` | Per-process CUDA memory cap. |

OpenVLA-OFT action chunks are executed open-loop for
`task.openvla_oft.chunk_size` steps before a new chunk is consumed. The hidden
sidecar is still written for every observed frame.

## Output

Default output for the goal task:

```text
data/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward/
data/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1/
```

Expected sidecar for the discrete OpenVLA-OFT action-query route:

```text
obs_embedding shape: (T, 229376)
obs_embedding dtype: float16
chunk size: 8
token layout: 56 x 4096
```

## Inspect A Collection

```bash
python - <<'PY'
from pathlib import Path
import h5py

reward_dir = Path("/tmp/dvla_collect/reward")
hidden_dir = Path("/tmp/dvla_collect/hidden")

total = success = 0
for path in sorted(reward_dir.glob("*.hdf5")):
    with h5py.File(path, "r") as handle:
        for key in handle["data"]:
            rewards = handle["data"][key]["sparse_rewards"][()]
            total += 1
            success += int(rewards.max() > 0)
print(f"success={success}/{total}")

hidden = next(hidden_dir.glob("*.hdf5"))
with h5py.File(hidden, "r") as handle:
    ds = handle["data"]["demo_0"]["obs_embedding"]
    print(ds.shape, ds.dtype)
PY
```

## Follow-Up Training

Use the same task when consuming collected data:

```bash
python -m dreamervla.train \
  experiment=oft_discrete_token_world_model_dinowm_chunk \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  task.openvla_oft.hdf5_reward_dir=/tmp/dvla_collect/reward \
  task.openvla_oft.action_hidden_dir=/tmp/dvla_collect/hidden
```

For cold-start collection plus offline warmup, prefer the release launcher in
[OpenVLA-OFT Cold-Start Collection And Warmup](OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md).
