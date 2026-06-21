# OpenVLA-OFT cold-start rollout collection

Background, tunable parameters & output format: [EXPLAINED.md](EXPLAINED.md) ·
[../PARAMETERS.md](../PARAMETERS.md). Suites: `openvla_onetraj_coldstart_libero`
(goal), `..._object`, `..._spatial`, `..._10`.

## Collect (single GPU, vectorized no-Ray)

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_onetraj \
  task=openvla_onetraj_coldstart_libero \
  collect.task_ids=all collect.episodes_per_task=4 \
  collect.episode_horizon=300 collect.envs_per_gpu=32
```

## Collect (multi-GPU DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=osmesa \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dreamervla.train experiment=collect_rollouts_onetraj \
  task=openvla_onetraj_coldstart_libero \
  collect.task_ids=all collect.episodes_per_task=4 \
  collect.episode_horizon=300 collect.envs_per_gpu=32
```

## Collect (Ray)

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_ray \
  task=openvla_onetraj_coldstart_libero \
  collect.task_ids=all collect.episodes_per_task=4 \
  collect.episode_horizon=300 env.num_workers=16 rollout.max_steps=1200
```

## Follow-up: train a WM on the collected rollouts

```bash
python -m dreamervla.train \
  experiment=oft_discrete_token_world_model_dinowm_chunk \
  task=openvla_onetraj_coldstart_libero \
  task.openvla_oft.hdf5_reward_dir=/tmp/dvla_collect/reward \
  task.openvla_oft.action_hidden_dir=/tmp/dvla_collect/hidden
```

## Inspect a collection

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

Full cold-start + warmup + cotrain:
[OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md](OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md).
