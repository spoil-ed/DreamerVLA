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

Render and GPU env vars are passed **inline per command** (see [Run](#run)), not exported
globally — so the same shell can launch either render backend without a stale `MUJOCO_GL`
locking the backend.

## Render backends

The online cotrain rollout has two backends, switched with the launcher knob
`render_backend` (the direct experiment entry is `online_rollout.render_backend`). Keep
`MUJOCO_GL=osmesa` inline either way — collection always renders osmesa and the egl rollout
forces egl inside each child. Both backends write the same outputs (see [Output](#output)).

| Backend | Select | Implementation |
| --- | --- | --- |
| **egl** — GPU, RLinf-vendored | `render_backend=egl`, `num_envs=K` | one spawn subprocess per env through RLinf's `SubprocVectorEnv` (`dreamervla/envs/rlinf_venv.py` → `OnlineEglVecEnv`); each child forces `MUJOCO_GL=egl` + its own `CUDA_VISIBLE_DEVICES`/`MUJOCO_EGL_DEVICE_ID` from the visible-GPU pool. Keep **1–2 envs per GPU**; action_hidden only (`backbone_latent` needs `num_envs=1`) |
| **osmesa** — CPU, stable | `render_backend=osmesa` or `num_envs=1` | the validated `VecRolloutEnv`; use this if egl aborts |

## Run

Env vars are inline; stdout/stderr go to `logs/`. Default GPUs `0,1,2,3,4,5`.

```bash
# Ray collect + DDP cotrain, osmesa rollout
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 MUJOCO_GL=osmesa NCCL_NVLS_ENABLE=0 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=osmesa > logs/cotrain_ray_osmesa.log 2>&1

# same, egl rollout (RLinf-vendored)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 MUJOCO_GL=osmesa NCCL_NVLS_ENABLE=0 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=6 profile=multi_gpu \
  render_backend=egl > logs/cotrain_ray_egl.log 2>&1

# no-Ray variant: swap the script name e2e_coldstart_warmup_cotrain_noray.sh
# fast end-to-end smoke (full pipeline, tiny step counts): add debug=true
# preview the launch plan without running anything: add dry_run=true
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 MUJOCO_GL=osmesa NCCL_NVLS_ENABLE=0 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=6 profile=multi_gpu \
  debug=true > logs/cotrain_ray_smoke.log 2>&1
```

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
