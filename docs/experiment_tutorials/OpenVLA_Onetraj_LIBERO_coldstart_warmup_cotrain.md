# OpenVLA-OFT cold-start: collect + warmup + cotrain

Collect one-trajectory OpenVLA-OFT rollouts, warm up the world model and success
classifier on them, then cotrain WM/classifier with slow-policy RL — in one command.
Background and tuning live in [EXPLAINED.md](EXPLAINED.md) and
[../PARAMETERS.md](../PARAMETERS.md). The e2e scripts take a suite shorthand
`task=goal|object|spatial|10`.

## Environment

```bash
cd DreamerVLA
conda activate dreamervla
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-$(pwd -P)/data}"
mkdir -p logs
bash scripts/install/60_verify.sh
```

The only per-command env var is `CUDA_VISIBLE_DEVICES`; `MUJOCO_GL` defaults to osmesa
(the runner sets it) and `NCCL_NVLS_ENABLE=0` is set inside the e2e scripts. Pick the
render backend with the launcher knob `render_backend`, not an env var.

## Render backends

The online cotrain rollout has two backends, switched with `render_backend` (direct
experiment entry: `online_rollout.render_backend`). Both write the same outputs (see
[Output](#output)); collection always renders osmesa.

| Backend | Select | Implementation |
| --- | --- | --- |
| **egl** — GPU, RLinf-vendored | `render_backend=egl`, `num_envs=K` | one spawn subprocess per env through RLinf's `SubprocVectorEnv` (`dreamervla/envs/rlinf_venv.py` → `OnlineEglVecEnv`); each child forces `MUJOCO_GL=egl` + its own GPU. Keep **1–2 envs per GPU**; action_hidden only (`backbone_latent` needs `num_envs=1`) |
| **osmesa** — CPU, stable | `render_backend=osmesa` or `num_envs=1` | the validated `VecRolloutEnv`; use this if egl aborts |

## Run

Default GPUs `0,1,2,3,4,5`; logs go to `logs/`.

```bash
# no-Ray (pure torchrun) — collect -> warmup -> cotrain, osmesa rollout
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=osmesa > logs/cotrain_noray_osmesa.log 2>&1

# egl rollout (RLinf-vendored): same command, render_backend=egl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=egl > logs/cotrain_noray_egl.log 2>&1
```

Variants (append the knob):

- **Ray** collect + DDP cotrain: swap the script for `e2e_coldstart_warmup_cotrain_ray.sh`.
- **smoke** (cotrain at tiny step counts; collect unchanged): `debug=true`.
- **preview** the launch plan without running anything: `dry_run=true`.
- **fewer collect episodes** for a quick real run: `collect.episodes_per_task=2`.

`multi_gpu` batch sizes are per-GPU (global = value × `ngpu`); lower them on OOM. Add
`cotrain_engine=async` for the RLinf-style rollout⟂training overlap loop (Ray only).

## Output

The e2e is orchestration only; the two stages stay on disk separately:

- **collect** → `${DVLA_DATA_ROOT}/collected_rollouts/<suite>/` — a resumable per-suite
  space with `reward/` + `hidden/` HDF5 shards, `collection_manifest.json` (counts,
  target, status), and `resolved_config.yaml`. A relaunch tops up to
  `collect_target_episodes=<N>` or skips collection when the target is met.
- **cotrain** → `${RUN_ROOT}/cotrain/` — warmup + online checkpoints and TensorBoard
  (the collect phase's own logs go to `${RUN_ROOT}/collect/`).
