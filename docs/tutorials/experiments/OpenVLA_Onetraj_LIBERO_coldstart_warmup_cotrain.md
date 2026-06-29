# OpenVLA-OFT cold-start: collect + warmup + cotrain

Collect one-trajectory OpenVLA-OFT rollouts, warm up the world model and success
classifier on them, then cotrain WM/classifier with slow-policy RL — in one command.
Background and tuning live in [EXPLAINED.md](EXPLAINED.md) and
[../../PARAMETERS.md](../../PARAMETERS.md). The e2e scripts take a suite shorthand
`task=goal|object|spatial|10`.

## Environment

```bash
cd DreamerVLA
conda activate dreamervla
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-$(pwd -P)/data}"
mkdir -p logs
bash scripts/install/60_verify.sh
```

`60_verify.sh` also asserts `peft==0.11.0`: a newer peft imports
`transformers.EncoderDecoderCache`, which the OpenVLA-OFT transformers fork (4.40.1)
lacks, so OFT policy load raises `ImportError` — and only surfaces deep inside a Ray
inference worker. If it flags, run `pip install peft==0.11.0` (a stray openvla-oft
install without `--no-deps` upgrades it past the pin).

The only per-command env var is `CUDA_VISIBLE_DEVICES`; `MUJOCO_GL` defaults to osmesa
(the runner sets it) and `NCCL_NVLS_ENABLE=0` is set inside the e2e scripts. Pick the
render backend with the launcher knob `render_backend`, not an env var.

## Render backends

The online cotrain rollout has two backends, switched with `render_backend` in
the launcher/Ray route. The sync no-Ray route keeps the nested direct key
`online_rollout.render_backend`. Both write the same outputs (see [Output](#output));
collection always renders osmesa.

| Backend | Select | Implementation |
| --- | --- | --- |
| **egl** — GPU, RLinf-style | `render_backend=egl` | Ray mainline binds env workers with `cluster.component_placement.env`; each EnvWorker owns `CUDA_VISIBLE_DEVICES` + `MUJOCO_EGL_DEVICE_ID` and hosts `env.envs_per_worker` LIBERO spawn children on that render GPU. The legacy no-Ray vec env still uses `online_rollout.render_devices` |
| **osmesa** — CPU, stable | `render_backend=osmesa` or `num_envs=1` | the validated `VecRolloutEnv`; use this if egl aborts |

## Run

Default GPUs `0,1,2,3,4,5`; logs go to `logs/`. The four schemes are the cross of the
**collect backend** (`noray` = pure torchrun vectorized collector, `ray` = worker
fan-out) and the cotrain **rollout `render_backend`** (`osmesa` = CPU software,
`egl` = GPU offscreen). Only the script name and `render_backend` differ — collect
always renders osmesa. Add `cotrain_engine=async` when you want the Ray online
cotrain worker topology instead of the sync DDP cotrain stage.

```bash
# 1) no-Ray + osmesa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=osmesa > logs/cotrain_noray_osmesa.log 2>&1

# 2) no-Ray + egl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=egl > logs/cotrain_noray_egl.log 2>&1

# 3) Ray + osmesa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=osmesa > logs/cotrain_ray_osmesa.log 2>&1

# 4) Ray + egl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=egl > logs/cotrain_ray_egl.log 2>&1

# 4b) Ray async online + egl (6-GPU Ray worker topology)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=6 profile=multi_gpu \
  cotrain_engine=async render_backend=egl > logs/cotrain_ray_async_egl.log 2>&1
```

With `profile=multi_gpu ngpu=6`, the launcher expands the Hydra concurrency from
the profile: sync cotrain gets `online_rollout.num_envs=12`; Ray async manual cotrain
(`cotrain_engine=async`) derives `manual_cotrain.ngpu=6` and
`manual_cotrain.envs_per_worker=2` unless explicitly overridden. The manual route then
uses its own placement planner: GPU0 hosts the real env, rollout worker, and learner;
GPU1-5 host WMEnv workers, rollout workers, and ActorGroup FSDP ranks. Each EGL env
worker owns its render device; child env slots inherit that regime.

### Mainline Hydra Config

The default cotrain path in this tutorial is the sync Hydra route:

```yaml
# configs/scripts/coldstart_warmup_cotrain.yaml
cotrain:
  base:
    - experiment=online_cotrain_pipeline_oft_backbone_latent
```

That experiment composes
`configs/dreamervla/online_cotrain_pipeline_openvla_oft_backbone_latent.yaml`.
The success classifier is configured there, not as an ad-hoc launcher override:

```yaml
classifier:
  head_type: spatial_tf
  hidden_dim: 1024
  num_layers: 12
  num_heads: 8
  token_count: ${task.openvla_oft.input_tokens.token_count}
  token_dim: ${task.openvla_oft.input_tokens.token_dim}
```

The Ray async online phase is selected by `cotrain_engine=async`. The launcher first
runs the same sync warmup-only phase, writes `ray_async_init.ckpt`, then starts
`experiment=manual_cotrain_ray_oft_backbone_latent` by default. That route follows the
manual-notes four-group topology:

```yaml
_target_: dreamervla.runners.ManualCotrainRayRunner
manual_cotrain:
  ngpu: 6
  envs_per_worker: 2
  rollout_epoch: 16
  max_steps_per_rollout_epoch: 256
  num_action_chunks: ${task.openvla_oft.input_tokens.chunk_size}
```

For a 6-GPU production run, use the launcher form above
(`profile=multi_gpu ngpu=6 cotrain_engine=async render_backend=egl`). The effective
manual placement is:

```text
GPU0:
  RealEnvWorker
  MultiStepRolloutWorker
  LearnerWorker

GPU1..GPU5:
  WMEnvWorker
  MultiStepRolloutWorker
  EmbodiedFSDPActor rank
```

Do not tune this route through old `online_cotrain_ray_*` placement assumptions unless
you explicitly choose the legacy runner. Manual cotrain placement is driven by
`manual_cotrain.ngpu` and `manual_cotrain.envs_per_worker`.

### Run Stages Separately

Use one stable `RUN_ROOT` when debugging the later phases. The warmup stage writes
`${RUN_ROOT}/cotrain/ckpt/wm_warmup.ckpt` and
`${RUN_ROOT}/cotrain/ckpt/classifier_warmup.ckpt`; the online stage resumes those
files and skips replay loading + warmup.

```bash
export RUN_ROOT="${DVLA_DATA_ROOT}/outputs/coldstart_warmup_cotrain/goal_g67_split_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

# 1) Cold-start collection only. Skip this if
# ${DVLA_DATA_ROOT}/collected_rollouts/libero_goal already has the desired shards.
CUDA_VISIBLE_DEVICES=6,7 python -m dreamervla.train \
  experiment=collect_rollouts_ray \
  task=openvla_onetraj_coldstart_libero \
  logger=tensorboard \
  collect.task_ids=all \
  collect.episodes_per_task=50 \
  collect.episode_horizon=300 \
  collect.memory_fraction=0.9 \
  collect.num_inference_workers=2 \
  env.num_workers=8 \
  task.openvla_oft.hdf5_reward_dir="${DVLA_DATA_ROOT}/collected_rollouts/libero_goal/reward" \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT}/collected_rollouts/libero_goal/hidden" \
  training.out_dir="${RUN_ROOT}/collect" \
  > logs/cotrain_goal_g67_collect.log 2>&1

# 2) Offline replay warmup only. This runs the 1-epoch replay warmup and exits
# before online rollout.
CUDA_VISIBLE_DEVICES=6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=2 profile=multi_gpu collect.num_inference_workers=2 \
  skip_collect=true cotrain_phase=warmup run_root="${RUN_ROOT}" \
  > logs/cotrain_goal_g67_warmup.log 2>&1

# 3) Online cotrain only. This validates the split warmup ckpts under RUN_ROOT,
# appends training.resume=true, and starts directly from the online phase.
CUDA_VISIBLE_DEVICES=6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=2 profile=multi_gpu collect.num_inference_workers=2 \
  skip_collect=true cotrain_phase=online run_root="${RUN_ROOT}" \
  > logs/cotrain_goal_g67_online.log 2>&1
```

Ray collect fans out `collect.num_inference_workers` policy workers (the `multi_gpu`
profile sets 4); keep it **≤ the visible GPU count** — e.g. on two GPUs add
`collect.num_inference_workers=2` (and `CUDA_VISIBLE_DEVICES=6,7 ngpu=2`).

Variants (append the knob):

- **smoke** (cotrain at tiny step counts; collect unchanged): `debug=true`.
- **preview** the launch plan without running anything: `dry_run=true`.
- **fewer collect episodes** for a quick real run: `collect.episodes_per_task=2`.
- **slice collected data** into shards of N episodes (both backends; default 0 = one
  shard per rank/worker): `collect.demos_per_shard=N`. Sliced shards make a long collect
  crash-resilient (a crash only loses the last small shard) and finer-grained to resume;
  warmup globs `*.hdf5`, so loading is unchanged.

Collection is quieter by design: each rank/worker no longer prints every episode — only
rank 0 streams progress, and the launcher prints one **aggregate** summary
(`PHASE 1/2 collected (aggregate across all processes)`) once collection finishes.

`multi_gpu` batch sizes are per-GPU (global = value × `ngpu`); lower them on OOM. Add
`cotrain_engine=async` for the RLinf-style rollout⟂training overlap loop (Ray only).

### Manual Ray async control plane and WorldModelEnv

Ray async cotrain still uses the normal DreamerVLA Runner lifecycle. The
`ManualCotrainRayRunner` is the current control plane: it starts ReplayWorker,
LearnerGroup, ActorGroup, RolloutGroup, RealEnvGroup, and optional WMEnvGroup;
schedules global steps; tracks `policy`, `world_model`, and `classifier` versions;
and synchronizes those versions only at rollout/update boundaries.

There are two EnvWorker backends:

- Real environment backend: `RolloutWorker/Runner -> EnvWorker(real env)`, where
  the env returns LIBERO observations and rewards.
- World model backend: `RolloutWorker/Runner -> EnvWorker(WorldModelEnv)`, where
  `LatentWorldModelEnv` computes `next_obs, reward, done, info` from the current
  world model and classifier/verifier snapshot.

Policy hidden outputs are route-dependent. In the manual OpenVLA-OFT route,
RolloutGroup can encode real image observations through `OFTRolloutBundle`, emit
`obs_embedding`/`lang_emb`, and return complete `forward_inputs` for ActorGroup PPO.
WMEnv can bootstrap initial latent/lang/proprio state from replay when those sidecars
exist.

Use the manual tiny smoke before running large LIBERO cotrain changes:

```bash
PYTHONPATH=. WANDB_MODE=offline \
python -m dreamervla.train \
  experiment=manual_cotrain_ray_tiny \
  training.out_dir=/tmp/dvla_manual_cotrain_tiny_smoke
```

Expected output includes the manual group banner and one completed short global step.
The older `online_cotrain_ray_world_model_env_tiny` route remains a legacy
WorldModelEnv smoke, not the current manual-cotrain control plane.

## Output

The e2e is orchestration only; the two stages stay on disk separately:

- **collect** → `${DVLA_DATA_ROOT}/collected_rollouts/<suite>/` — a resumable per-suite
  space with `reward/` + `hidden/` HDF5 shards, `collection_manifest.json` (counts,
  target, status), and `resolved_config.yaml`. A relaunch tops up to
  `collect_target_episodes=<N>` or skips collection when the target is met. With
  `collect.demos_per_shard=N` the per-rank shard is sliced into N-episode files
  (`r{rank}_shard_{NNN}.hdf5`); default 0 keeps one growing shard per rank.
- **cotrain** → `${RUN_ROOT}/cotrain/` — warmup + online checkpoints and TensorBoard
  (the collect phase's own logs go to `${RUN_ROOT}/collect/`).
