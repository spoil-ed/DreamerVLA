# OpenVLA-OFT Cold-Start -> Offline Warmup -> Online Cotrain

This guide covers the end-to-end cold-start-to-cotrain handoff for the
one-trajectory OpenVLA-OFT LIBERO route. The launcher accepts one task selector:
`--task goal|object|spatial`. It provides two runnable launchers:

- Ray: `scripts/e2e_coldstart_warmup_cotrain_ray.sh`
- No Ray: `scripts/e2e_coldstart_warmup_cotrain_noray.sh`

Both launchers use the same data contract:

```text
cold-start collector
  -> reward HDF5 + obs_embedding sidecar
  -> OnlineReplay seed via offline_warmup.data_dir / hidden_dir
  -> OnlineCotrainPipelineRunner WM + classifier warmup
  -> online cotrain loop
```

The important contract is directory wiring. The collector writes reward-schema
HDF5 files and matching hidden sidecars. The cotrain pipeline reads those same
directories through `offline_warmup.*`, seeds `OnlineReplay`, warms the world
model and classifier, then enters the online cotrain loop.

## What The Two Scripts Do

Ray e2e:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh --task goal
```

No-Ray e2e:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh --task goal
```

The Ray script uses:

```text
collect:  experiment=collect_rollouts_ray
cotrain:  experiment=online_cotrain_pipeline_oft_action_hidden
```

The no-Ray script uses:

```text
collect:  experiment=collect_rollouts_onetraj
cotrain:  experiment=online_cotrain_pipeline_oft_action_hidden
```

Both scripts default to a small real-data warmup smoke:

```text
task id:               0
episodes:              4
collect horizon:       64
cotrain warmup:        1 WM step + 1 classifier step
online env steps:      0
sidecar shape:         OpenVLA-OFT action hidden, 56 * 4096 = 229376
logger:                logger=tensorboard
run root:              ${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/<timestamp>
```

Supported task values:

| `--task` | Hydra task | LIBERO suite | One-traj OFT ckpt |
| --- | --- | --- | --- |
| `goal` | `OpenVLA_Onetraj_ColdStart_LIBERO` | `libero_goal` | `Openvla-oft-SFT-libero-goal-traj1` |
| `object` | `OpenVLA_Onetraj_ColdStart_LIBERO_Object` | `libero_object` | `Openvla-oft-SFT-libero-object-traj1` |
| `spatial` | `OpenVLA_Onetraj_ColdStart_LIBERO_Spatial` | `libero_spatial` | `Openvla-oft-SFT-libero-spatial-traj1` |

Before launching, the scripts validate that the default checkpoint and LIBERO
dataset assets are already present. They activate `${DVLA_CONDA_ENV:-dreamervla}`
when `conda` is available, but they do not install packages or download
checkpoints.

Default assets checked:

```text
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/<task-specific-ckpt>
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/<task-specific-ckpt>/dataset_statistics.json
${DVLA_DATA_ROOT}/datasets/libero/<task-specific-suite>/*.hdf5
```

Use `--dry-run` to print commands without checking assets or launching jobs:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh --dry-run
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh --task object --dry-run
```

## Output Layout

Both launchers write one run root:

```text
<run_root>/
  collect/
  coldstart/
    reward/
      *.hdf5
    hidden/
      *.hdf5
      preprocess_config.json
  cotrain/
```

The handoff into cotrain is explicit:

```text
offline_warmup.data_dir   = <run_root>/coldstart/reward
offline_warmup.hidden_dir = <run_root>/coldstart/hidden
offline_warmup.task_id    = 0
```

## Full-Run Shape

Use `--run-root` to make output deterministic. Pass Hydra overrides through the
launcher with `--collect-override`, `--cotrain-override`, or
`--common-override`.

Ray example:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
    --task goal \
    --run-root "${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/ray_goal_full" \
    --collect-override collect.task_ids=all \
    --collect-override collect.episodes_per_task=300 \
    --collect-override collect.episode_horizon=300 \
    --collect-override env.num_workers=4 \
    --collect-override rollout.target_episodes=3000 \
    --collect-override rollout.max_steps=900000 \
    --cotrain-override training.wm_warmup_steps=2000 \
    --cotrain-override training.classifier_warmup_steps=2000 \
    --cotrain-override online_rollout.total_env_steps=200000
```

No-Ray example:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
    --task spatial \
    --run-root "${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/noray_spatial_full" \
    --collect-override collect.task_ids=all \
    --collect-override collect.episodes_per_task=300 \
    --collect-override collect.episode_horizon=300 \
    --collect-override collect.envs_per_gpu=1 \
    --cotrain-override training.wm_warmup_steps=2000 \
    --cotrain-override training.classifier_warmup_steps=2000 \
    --cotrain-override online_rollout.total_env_steps=200000
```

The default launcher cotrain config is intentionally smoke-sized. It proves the
handoff by warming the WM and classifier from the collected OFT sidecar, then
entering the cotrain loop and saving `cotrain/ckpt/latest.ckpt`. Full online
OFT cotrain still needs the online rollout encoder path to emit matching
OpenVLA-OFT action-hidden latents; the current online loop is still RynnVLA
encoder based.

If you already have cold-start output under the same `run_root`, skip collection
and run only warmup/cotrain:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  --run-root "${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/ray_goal_full" \
  --skip-collect
```

With `--skip-collect`, the launcher validates that
`<run_root>/coldstart/reward` and `<run_root>/coldstart/hidden` contain HDF5
shards before starting cotrain.

## Manual Equivalent

Ray collect:

```bash
RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/manual_ray"
RW="${RUN_ROOT}/coldstart/reward"
HID="${RUN_ROOT}/coldstart/hidden"

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_ray logger=tensorboard \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  collect.task_ids=[0] collect.episodes_per_task=4 collect.episode_horizon=64 \
  env.num_workers=2 rollout.target_episodes=4 rollout.max_steps=256 \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.action_hidden_dir="${HID}" \
  training.out_dir="${RUN_ROOT}/collect"
```

No-Ray collect:

```bash
RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/manual_noray"
RW="${RUN_ROOT}/coldstart/reward"
HID="${RUN_ROOT}/coldstart/hidden"

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=collect_rollouts_onetraj logger=tensorboard \
  task=OpenVLA_Onetraj_ColdStart_LIBERO \
  collect.task_ids=[0] collect.episodes_per_task=4 collect.episode_horizon=64 \
  collect.envs_per_gpu=1 collect.gpu_id=0 \
  task.openvla_oft.hdf5_reward_dir="${RW}" \
  task.openvla_oft.action_hidden_dir="${HID}" \
  training.out_dir="${RUN_ROOT}/collect"
```

Shared warmup + cotrain:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa python -m dreamervla.train \
  experiment=online_cotrain_pipeline_oft_action_hidden logger=tensorboard \
  +task=OpenVLA_Onetraj_ColdStart_LIBERO \
  offline_warmup.data_dir="${RW}" \
  offline_warmup.hidden_dir="${HID}" \
  offline_warmup.task_id=0 \
  env.task_suite_name=libero_goal \
  training.out_dir="${RUN_ROOT}/cotrain" \
  training.debug=false \
  training.wm_warmup_steps=1 \
  training.classifier_warmup_steps=1 \
  training.classifier_batch_size=1 \
  dataloader.batch_size=1 \
  online_rollout.sequence_length=9 \
  online_rollout.buffer_size=100 \
  online_rollout.total_env_steps=0 \
  world_model.obs_dim=229376 \
  world_model.token_count=56 \
  world_model.token_dim=4096 \
  world_model.chunk_size=8 \
  +world_model.time_horizon=8 \
  world_model.model_dim=128 \
  world_model.depth=1 \
  world_model.heads=4 \
  world_model.mlp_dim=256 \
  world_model.num_hist=1 \
  world_model.num_pred=1 \
  world_model.chunk_rollout_chunks=1 \
  policy._target_=dreamervla.models.actor.OpenVLADiscreteTokenActor \
  policy.action_hidden_dim=4096 \
  policy.time_horizon=8 \
  policy.head_type=oft_discrete_token \
  policy.adapter_hidden_dim=128 \
  +policy.init_lm_head_ckpt="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" \
  classifier.latent_dim=4096 \
  classifier.window=2 \
  +classifier.head_type=linear \
  classifier.hidden_dim=128 \
  classifier.num_layers=1 \
  classifier.num_heads=4 \
  classifier.chunk_size=8 \
  +classifier.token_dim=4096 \
  +classifier.token_pool=mean \
  +classifier.token_count=56 \
  critic.hidden_dim=128 \
  critic.critic_hidden_dim=128 \
  algorithm.wmpo.chunk_size=8
```

## Fake Data-Flow Proof

The fast fake proof is:

```bash
python -m pytest \
  tests/e2e_tests/test_s6_ray_coldstart_collect.py::test_fake_coldstart_50pct_success_seeds_cotrain_warmup -v
```

What it proves:

1. Cold-start collection runs with a fake alternating-success env.
2. The collector writes four demos.
3. The written sparse success labels are exactly `2/4 = 50%`.
4. `seed_replay_from_offline` loads the reward/hidden shards into `OnlineReplay`.
5. The replay reports two successes and two failures.
6. `OnlineCotrainPipelineRunner._offline_warmup_wm` and
   `_offline_warmup_classifier` consume batches from that replay.

This is the regression test for the plumbing contract:

```text
collector output dirs == cotrain offline_warmup input dirs
```

## Real OFT Collector Check

The real-policy collector e2e is gated because it needs GPU, LIBERO, and the
OpenVLA-OFT checkpoint:

```bash
DVLA_GPU_E2E=1 python -m pytest tests/e2e_tests/test_s6_real_oft_coldstart.py -v -s
```

Without `DVLA_GPU_E2E=1`, this test skips.

## Inspect Output

Check success rate from a reward shard:

```bash
python - <<'PY'
import h5py
from pathlib import Path

reward = next(Path("data/outputs/coldstart_warmup_cotrain/<run>/coldstart/reward").glob("*.hdf5"))
with h5py.File(reward, "r") as f:
    demos = [f["data"][key] for key in sorted(f["data"])]
    successes = [bool(demo["sparse_rewards"][-1]) for demo in demos]
print(sum(successes), "/", len(successes), "=", sum(successes) / max(len(successes), 1))
PY
```

Check sidecar shape:

```bash
python - <<'PY'
import h5py
from pathlib import Path

hidden = next(Path("data/outputs/coldstart_warmup_cotrain/<run>/coldstart/hidden").glob("*.hdf5"))
with h5py.File(hidden, "r") as f:
    ds = f["data"]["demo_0"]["obs_embedding"]
    print(ds.shape, ds.dtype)
PY
```

For real OFT action-query hidden, the sidecar shape is `(T, 229376)` with
`float16`.

## Current Boundary

The e2e launchers prove the real cold-start-to-warmup handoff with OFT
action-hidden sidecars. They do not yet prove long-horizon online OFT cotrain,
because the online rollout path still uses the RynnVLA encoder to create live
latents. Keep `online_rollout.total_env_steps=0` for the default OFT smoke
unless the online encoder path has been updated to emit the same OpenVLA-OFT
latent shape as the cold-start sidecars.

## Troubleshooting

- If the launcher fails before training, read the asset check output. Set
  `DVLA_DATA_ROOT` to the data root containing the checkpoint and LIBERO HDF5s.
- If you pass custom checkpoint or dataset Hydra overrides, use
  `--skip-asset-check` after verifying those custom paths yourself.
- If warmup says the replay is empty, increase `collect.episode_horizon` so
  collected episodes are at least `online_rollout.sequence_length`.
- If `offline_warmup.data_dir` has no HDF5 shards, the collect stage did not
  write into the run root you are passing to cotrain.
- If model warmup fails with tensor shape errors, the cotrain model config does
  not match the sidecar latent shape. Align `world_model.obs_dim`,
  `token_count`, `token_dim`, classifier config, and policy hidden dimensions
  with the collected sidecar.
