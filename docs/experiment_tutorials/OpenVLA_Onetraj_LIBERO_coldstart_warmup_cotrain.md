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
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export NCCL_NVLS_ENABLE=0          # multi-GPU DDP cotrain
bash scripts/install/60_verify.sh
```

## Render backends

The online rollout has two render backends, selected with
`online_rollout.render_backend` (launcher knob: `render_backend`). Both write the same
outputs (see [Output](#output)).

| Backend | Select | Env | Notes |
| --- | --- | --- | --- |
| **egl** — GPU, RLinf-vendored | `render_backend=egl`, `num_envs=K` | keep the osmesa exports; each child sets `MUJOCO_GL=egl` + its own `CUDA_VISIBLE_DEVICES`/`MUJOCO_EGL_DEVICE_ID` from the visible-GPU pool | one spawn subprocess per env through RLinf's `SubprocVectorEnv` (`dreamervla/envs/rlinf_venv.py` → `OnlineEglVecEnv`). Keep **1–2 envs per GPU**; action_hidden only (`backbone_latent` needs `num_envs=1`) |
| **osmesa** — CPU, stable | `render_backend=osmesa` or `num_envs=1` | `MUJOCO_GL=osmesa` | the validated path (`VecRolloutEnv`); use this if egl aborts |

## Run

```bash
# no-Ray
CUDA_VISIBLE_DEVICES=0 bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal
# Ray
CUDA_VISIBLE_DEVICES=0 bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal
# smoke / dry-run
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh   task=goal debug=true
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal dry_run=true
```

Multi-GPU (8× H100, single node) — the cotrain stage runs torchrun DDP when `ngpu>1`:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_NVLS_ENABLE=0
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=8 profile=multi_gpu
RAY_NUM_GPUS=8 bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=8 \
  profile=multi_gpu collect.num_inference_workers=4
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
