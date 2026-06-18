# OpenVLA-OFT One-Trajectory — Cold-Start Rollout Collection (pure Hydra)

Drive the base OpenVLA-OFT one-trajectory VLA in LIBERO-Goal and **collect fresh
rollouts** into reward-dir HDF5 + OFT action-hidden sidecars, so the discrete
world model can be trained from cold-start-collected data with **zero collector
changes**.

This is the *online-collection* alternative to offline preprocessing: instead of
extracting action-hidden from pre-existing demos
([action-hidden WM recipe](OpenVLA_Onetraj_LIBERO_action_hidden_world_model.md),
step 1), the collector rolls the VLA out in the LIBERO env from sequential init
states and dumps the same artifact layout the WM/classifier already consume.

Latent route: **Scheme A — action hidden** (the action-slot hidden tokens the OFT
LM head consumes; discrete one-traj ckpt → `[T, 56, 4096]` per frame, flat
`obs_embedding` dim `56*4096 = 229376`).

```
collect_rollouts   (experiment=collect_rollouts_onetraj,  task=OpenVLA_Onetraj_ColdStart_LIBERO)
   → reward HDF5 + OFT action-hidden sidecar  (no_noops_t_256_remaining_reward + ..._h1)
   → discrete world model   (experiment=oft_discrete_token_world_model_dinowm_chunk,
                             task=OpenVLA_Onetraj_ColdStart_LIBERO)   ← SAME task=
```

## Single source of truth: one `task=` for write and read

The collector and the discrete WM read **the same task config**, so the sidecar the
collector writes always matches what `BalancedTerminalDataset` validates on read
(`model_path`, `time_horizon`, `action_head_type`, `obs_hidden_source`,
`prompt_style`, `history`, `include_state`, `rotate_images_180`).

Each LIBERO suite has its **own** one-trajectory discrete ckpt, bound in its own
cold-start `task=` config — VLA and suite are tied together, so you cannot
accidentally roll a goal VLA in spatial tasks. Pick the suite **explicitly**
(default = goal; append `task=…` to switch):

| `task=` | suite | VLA ckpt (`data/checkpoints/Openvla-oft-SFT-traj1/…`) | unnorm key |
| --- | --- | --- | --- |
| `OpenVLA_Onetraj_ColdStart_LIBERO`         | libero_goal    | `Openvla-oft-SFT-libero-goal-traj1`    | `libero_goal_no_noops` |
| `OpenVLA_Onetraj_ColdStart_LIBERO_10`      | libero_10      | `Openvla-oft-SFT-libero10-traj1`       | `libero_10_no_noops` |
| `OpenVLA_Onetraj_ColdStart_LIBERO_Object`  | libero_object  | `Openvla-oft-SFT-libero-object-traj1`  | `libero_object_no_noops` |
| `OpenVLA_Onetraj_ColdStart_LIBERO_Spatial` | libero_spatial | `Openvla-oft-SFT-libero-spatial-traj1` | `libero_spatial_no_noops` |

Before any rollout the collector asserts the ckpt-detected head mode (discrete /
no-proprio) against the task's `expected_*`, so a ckpt↔task mismatch fails fast.

Each config holds the discrete-extraction truth in one place — e.g.
`configs/task/OpenVLA_Onetraj_ColdStart_LIBERO.yaml` (the `_10` / `_Object` /
`_Spatial` variants inherit `libero_<suite>` and only swap ckpt, unnorm key, and
output namespace):

```yaml
defaults:
  - OpenVLA_Onetraj_LIBERO          # ckpt_path, hdf5_reward_dir, dataset_statistics_key, ...
  - _self_
name: OpenVLA_Onetraj_ColdStart_LIBERO
openvla_oft:
  action_hidden_dir: ${task.hdf5_dir}_oft_legacy_action_hidden_vla_policy_h1
  expected_action_head_type: oft_discrete_token   # discrete one-traj ckpt (no L1 head)
  expected_include_state: false                   # discrete ⇒ no proprio
  expected_history: 1                             # single-frame history (h1)
  time_horizon: 8
  token_dim: 4096
  chunk_size: 8
```

`expected_obs_hidden_source: action_query`, `expected_prompt_style: vla_policy`
and `expected_rotate_images_180: true` are inherited from the suite config.

> `num_images_in_input` is an OFT **deployment** param the checkpoint does not
> persist, so it is set centrally via `collect.num_images_in_input` (default `1`,
> single agentview — what the discrete one-traj ckpt was trained/evaluated with at
> ~50% success). It is **not** derived from `len(task.image_keys)` (the stored
> camera views, 2): feeding the discrete VLA 2 images collapses rollout success to
> ~0%. Both camera views are still **stored** in the HDF5 regardless.
> Env render resolution is `task.image_resolution` (256), **not** `task.image_size`
> (64, the WM latent grid).

Switch suite by appending `task=…` (example: libero_10):

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dreamervla.train experiment=collect_rollouts_onetraj \
  task=OpenVLA_Onetraj_ColdStart_LIBERO_10 \
  collect.task_ids=all collect.episodes_per_task=300 \
  collect.episode_horizon=300 collect.envs_per_gpu=8
# → outputs under data/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_10/...
```

## 0. System

```bash
cd DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
conda activate dreamervla
# LIBERO rendering on this host: EGL crashes in robosuite read_pixels; use osmesa.
export MUJOCO_GL=osmesa
```

Required assets:

```text
data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1   # OFT one-traj ckpt (discrete)
data/datasets/libero/libero_goal/*.hdf5                                    # raw LIBERO-Goal demos (init states + task defs)
```

To download them from scratch (same checkpoint + LIBERO-Goal suite), follow
[OpenVLA_Onetraj_LIBERO.md](OpenVLA_Onetraj_LIBERO.md) §0–§1 (install + download:
`download.openvla_one_traj=true`, `env.LIBERO_SUITES=[libero_goal]`).

## 1. Collect rollouts

The entry is the standard Hydra train module. Launch it with
`python -m torch.distributed.run` when sharding work across `M` ranks (one GPU
each); the collector reads
`RANK`/`WORLD_SIZE`/`LOCAL_RANK` for **work sharding only** and does **not**
initialize a torch process group (no DDP). Each rank writes its own shard
(`r{rank}_shard_000.hdf5`).

```bash
# 2 GPUs, all libero_goal tasks, 300 episodes each, K=8 within-rank envs:
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dreamervla.train experiment=collect_rollouts_onetraj \
  collect.task_ids=all collect.episodes_per_task=300 \
  collect.episode_horizon=300 collect.envs_per_gpu=8

# 1-GPU smoke (single in-process env per rank):
CUDA_VISIBLE_DEVICES=0 python -m dreamervla.train \
  experiment=collect_rollouts_onetraj \
  collect.task_ids=0 collect.episodes_per_task=2 \
  collect.episode_horizon=64 collect.envs_per_gpu=1
```

All overrides are normal Hydra `key=value`.

### `collect.*` knobs

| Key | Default | Meaning |
| --- | --- | --- |
| `collect.policy_mode` | `auto` | `auto` detects l1 vs discrete from the ckpt; the cold-start ckpt is discrete. The detected mode is asserted against the task's `expected_action_head_type`/`expected_include_state` before any rollout. |
| `collect.gpu_id` | `0` | Single-process device; ignored under `torchrun` (LOCAL_RANK wins). |
| `collect.task_ids` | `all` | `all`, csv (`0,2`), a single int, or a list. |
| `collect.episodes_per_task` | `2` | Episodes per LIBERO task. |
| `collect.episode_horizon` | `64` | Max steps per episode. Must be **≥ the WM `sequence_length` (36 for the discrete chunk WM)** so each episode yields at least one training window. |
| `collect.envs_per_gpu` | `1` | `K` within-rank envs. `>1` enables the batched vectorized path (one VLA forward over `K` observations). |

The checkpoint, dataset-statistics key, and output directories are **not** CLI
knobs — they come from `task=OpenVLA_Onetraj_ColdStart_LIBERO`. To collect into a
throwaway location (e.g. for a smoke), override the task dirs:
`task.openvla_oft.hdf5_reward_dir=/tmp/x/reward task.openvla_oft.action_hidden_dir=/tmp/x/hidden`.

## 2. Outputs

Collected (model-generated) rollouts go under **`data/collected_rollouts/`** — a
folder-level marker that keeps them separate from `data/processed_data/` (offline
preprocessed demos); the per-suite artifact name is the same, only the root differs:

```text
data/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward
data/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1
```

The hidden dir holds the per-frame `obs_embedding` and a `preprocess_config.json`
sidecar. For the cold-start discrete task it records:

```json
{"action_head_type": "oft_discrete_token", "history": 1, "include_state": false,
 "num_images_in_input": 1, "time_horizon": 8, "obs_hidden_source": "action_query",
 "prompt_style": "vla_policy", "rotate_images_180": true, "hidden_key": "obs_embedding"}
```

## 3. Consume in the discrete world model (zero collector change)

Point the discrete WM at the **same task**; it reads the collected reward HDF5 +
`_h1` sidecar through the standard `BalancedTerminalDataset`:

```bash
bash scripts/train_wm.sh \
  experiment=oft_discrete_token_world_model_dinowm_chunk \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=4
```

`configs/experiment/oft_discrete_token_world_model_dinowm_chunk.yaml` overrides
`/worldmodel: openvla_oft_discrete_token_action_chunk`; under
`task=OpenVLA_Onetraj_ColdStart_LIBERO` its discrete settings re-assert exactly
the values the task already carries (h1 / `oft_discrete_token` / no-state), so
sidecar validation passes with no override gymnastics.

Downstream stages (success classifier, DreamerVLA `wmpo_outcome`) are identical
to the offline action-hidden recipe; see
[the action-hidden WM tutorial](OpenVLA_Onetraj_LIBERO_action_hidden_world_model.md)
steps 3–5.

## Notes

- **Layer-1 sharding only.** `torchrun --nproc_per_node=M` gives `M` independent
  rank processes that split the `(task_id, episode)` work-list; there is no
  gradient sync. `training.distributed_strategy` being auto-flipped to `ddp` by
  `dreamervla/train.py` is harmless — `CollectRolloutsRunner` ignores it and never
  builds a process group.
- **Layer-2 within-rank batching.** `collect.envs_per_gpu=K>1` runs `K` LIBERO
  envs per rank and batches their observations through one OFT forward
  (`VecRolloutEnv` + `collect_vectorized`). `obs_embedding` is tolerance-equal to
  the single-env path; states/images/actions are byte-exact.
- **Pure-Hydra entry.** The legacy `argparse` entry
  (`python -m dreamervla.runners.collect_parallel_rollouts key=value`) is removed.
  The only entry is `python -m dreamervla.train experiment=collect_rollouts_onetraj`.

## Verified smoke (1-GPU, /tmp outputs)

The following was run on this host (8× H100) against the present OFT one-traj
checkpoint and **completes without error**, proving collect → sidecar → discrete-WM
consumption end-to-end. Outputs go to `/tmp` so the canonical namespace stays clean.

```bash
RW=/tmp/coldstart_smoke/reward; H1=/tmp/coldstart_smoke/hidden
PY=python   # after conda activate dreamervla

# 1) Collect (1 GPU, task 0, 2 episodes).
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa $PY -m dreamervla.train \
  experiment=collect_rollouts_onetraj \
  collect.task_ids=0 collect.episodes_per_task=2 collect.episode_horizon=64 collect.envs_per_gpu=1 \
  task.openvla_oft.hdf5_reward_dir=$RW task.openvla_oft.action_hidden_dir=$H1
# → 2 demos written to shard_000.hdf5; reward/hidden dirs printed; GPU ~14 GB (~18%).

# 2) Verify the sidecar
$PY -c "import json,glob; c=json.load(open(glob.glob('$H1/preprocess_config.json')[0])); print(c['action_head_type'],c['history'],c['include_state'],c['num_images_in_input'],c['time_horizon'])"
# → oft_discrete_token 1 False 1 8

# 3) Discrete-WM consumption (zero collector change) — point dataset at the /tmp products
MP=$($PY -c "import json,glob; print(json.load(open(glob.glob('$H1/preprocess_config.json')[0]))['model_path'])")
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa WANDB_MODE=disabled $PY -m dreamervla.train \
  experiment=oft_discrete_token_world_model_dinowm_chunk task=OpenVLA_Onetraj_ColdStart_LIBERO \
  logger=tensorboard training.out_dir=/tmp/coldstart_wm_smoke training.num_epochs=1 \
  dataloader.batch_size=1 dataloader.num_workers=0 \
  task.openvla_oft.hdf5_reward_dir=$RW task.openvla_oft.action_hidden_dir=$H1 \
  dataset.hdf5_dir=$RW dataset.hidden_dir=$H1 dataset.expected_model_path=$MP \
  dataset.max_files=1 dataset.max_demos_per_file=2 dataset.balanced_length=4
# → _validate_hidden_sidecar passes silently (no ValueError); ds[0] materializes;
#   4 WM train steps run; ckpt written to /tmp/coldstart_wm_smoke/ckpt/latest.ckpt.
```

Notes (general facts, not hacks):

1. **Use the same interpreter for the workers.** A bare `torchrun` on PATH may be a
   different Python than the conda env; the launcher therefore calls
   `${PYTHON:-python} -m torch.distributed.run`. `conda activate dreamervla` first, or
   set `PYTHON=python` after activating the environment. The 1-GPU smoke above uses the direct entry,
   which has no torchrun dependency.
2. **Smoke episodes are short.** A 2-episode/64-step smoke yields unsuccessful (timed-out)
   episodes — fine for verifying the pipeline; the discrete WM trains on the
   remaining-steps reward, not on binary success.

## Known gaps / not covered

- The multi-rank path via the launcher (`NUM_GPUS>1`) and the Layer-2 batched
  `collect.envs_per_gpu>1` path are unit-tested and were exercised in earlier
  sessions, but are not re-smoked in the verification above (which is 1-GPU,
  single-env).
- Classifier consumption (`experiment=oft_latent_classifier_chunk
  task=OpenVLA_Onetraj_ColdStart_LIBERO`) reads the same dirs and should consume the
  collected products the same way; not smoked here.
