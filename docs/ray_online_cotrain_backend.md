# Ray Online Cotrain Backend

This is an opt-in backend for exercising Ray actor topologies. The default
Dreamer-VLA training path remains the single-machine Runner flow selected
through Hydra.

## Install

Ray is declared as an optional project extra:

```bash
pip install -e '.[ray]'
```

The tested minimum is `ray[default]>=2.47.0`.

Use the project Python 3.11 environment. The smoke commands below were
validated with:

```bash
export PYTHON=~/miniconda3/envs/dreamervla/bin/python
"${PYTHON}" --version
```

## Quick Proof: Run The Actual Train Entry

These commands do not need LIBERO assets or real checkpoints. They launch the
same Hydra entrypoint operators use in normal runs, start a local single-node
Ray runtime, and write standard run artifacts.

Ray online cotrain with the tiny DreamerVLA learner phases:

```bash
PYTHONPATH=. WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
"${PYTHON}" -m dreamervla.train \
  experiment=online_cotrain_ray_dreamervla_tiny \
  logger=tensorboard \
  training.out_dir=/tmp/dvla_ray_tiny_train_proof \
  rollout.steps=9
```

Expected proof artifacts:

```text
/tmp/dvla_ray_tiny_train_proof/resolved_config.yaml
/tmp/dvla_ray_tiny_train_proof/run_manifest.json
```

Check the runner identity:

```bash
"${PYTHON}" - <<'PY'
from pathlib import Path
import json

root = Path("/tmp/dvla_ray_tiny_train_proof")
manifest = json.loads((root / "run_manifest.json").read_text())
print(manifest["runner"]["class"], manifest["runner"]["name"])
PY
```

Expected output:

```text
OnlineCotrainRayRunner online_cotrain_ray
```

Ray cold-start synthetic collection:

```bash
PYTHONPATH=. WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
"${PYTHON}" -m dreamervla.train \
  experiment=collect_rollouts_ray_synthetic \
  logger=tensorboard \
  training.out_dir=/tmp/dvla_ray_collect_train_proof \
  dump.reward_dir=/tmp/dvla_ray_collect_train_proof/reward \
  dump.hidden_dir=/tmp/dvla_ray_collect_train_proof/hidden \
  rollout.target_episodes=4 \
  rollout.max_steps=12
```

Expected proof artifacts:

```text
/tmp/dvla_ray_collect_train_proof/resolved_config.yaml
/tmp/dvla_ray_collect_train_proof/run_manifest.json
/tmp/dvla_ray_collect_train_proof/reward/ray_shard_000.hdf5
/tmp/dvla_ray_collect_train_proof/hidden/ray_shard_000.hdf5
/tmp/dvla_ray_collect_train_proof/hidden/preprocess_config.json
```

Inspect the written HDF5 files:

```bash
"${PYTHON}" - <<'PY'
from pathlib import Path
import h5py

root = Path("/tmp/dvla_ray_collect_train_proof")
for kind in ("reward", "hidden"):
    path = root / kind / "ray_shard_000.hdf5"
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"].keys())
        print(kind, len(demos), demos[:3])
        if kind == "hidden":
            ds = handle["data"][demos[0]]["obs_embedding"]
            print("obs_embedding", ds.shape, ds.dtype)
        else:
            ds = handle["data"][demos[0]]["sparse_rewards"]
            print("sparse_rewards", ds.shape, ds.dtype, float(ds[()].max()))
print("preprocess_config", (root / "hidden" / "preprocess_config.json").exists())
PY
```

For the default synthetic route, expect four demos, `sparse_rewards` length 3,
and `obs_embedding` shape `(3, 4)`.

## Online Cotrain Smoke Route

The low-cost cotrain route launches real Ray actors for:

- scheduler primitives: `Cluster`, `Worker`, `WorkerGroup`, `Channel`,
  `Placement`
- rollout: one `EnvWorker` per env actor
- inference: batched encoder + world-model + policy actor
- replay: `ReplayWorker` around `OnlineReplay`
- learner: `LearnerWorker` running either the synthetic PPO-style smoke update
  or real DreamerVLA cotrain phases
- weight sync: Ray object-store state-dict handoff, bucketed transfer,
  patch/delta transfer, optional dtype compression, and tagged collective
  send/recv for single-node learner groups

Run it through the normal Hydra entry:

```bash
python -m dreamervla.train experiment=online_cotrain_ray_synthetic logger=tensorboard
```

The runner writes the standard `BaseRunner` run artifacts under
`training.out_dir`, including `resolved_config.yaml` and `run_manifest.json`.

For a learner path closer to production semantics, use:

```bash
python -m dreamervla.train \
  experiment=online_cotrain_ray_dreamervla_tiny \
  logger=tensorboard
```

This route uses tiny WM / classifier / actor modules but exercises the
`dreamervla_cotrain` learner phases and the same weight-sync boundary as the
real route.

## What The Overlap Does

The runner uses Ray `ObjectRef` handles, not Python `async/await`.

Inside `OnlineCotrainRayRunner._run_loop_overlap`:

- `InferenceWorker.forward_batch(...)` is submitted and kept in
  `pending_infers` without immediately blocking.
- Per-env `EnvWorker.step(...)` calls are submitted and kept in
  `pending_steps`.
- The driver drains ready inference and env refs with `ray.wait(...)`.
- A completed env step is the only event that re-enqueues that env's next obs,
  so each env remains strictly serial.
- Learner update and weight sync keep their existing async path.

This gives rollout-level `inference <-> env-step` overlap on one machine:

```text
old: infer.wait -> env.step.wait -> infer.wait -> env.step.wait
new: infer(t) in flight while env-step(t-1) refs are in flight
```

The relevant metrics are:

| Metric | Meaning |
| --- | --- |
| `time/rollout_overlap_events` | Number of rollout inference launches after the first batch or while env refs were already in flight. |
| `time/rollout_strict_overlap_events` | Inference launches while env-step refs were definitely pending. |
| `time/rollout_infer_ready_batches` | Number of inference batches drained from Ray. |
| `time/rollout_env_ready_batches` | Number of env-step results drained from Ray. |
| `time/infer_{encode,world_model,policy}_s` | Stage timing from `InferenceWorker.forward_batch`. |
| `time/{infer,env_step,learner,weight_sync,ray}_wait_s` | Driver-side wait time by stage. |
| `time/gpu_*`, `time/cuda_*` | Best-effort `nvidia-smi` and torch CUDA allocator metrics when available. |

## Single-Node Multi-GPU Placement

The Ray backend is intentionally single-machine / single-node. Multi-GPU
placement is manual and config-driven. The first production smoke should not
require FSDP: place inference on one GPU and the learner on another GPU.

```bash
CUDA_VISIBLE_DEVICES=2,3 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
PYTHONPATH=. "${PYTHON}" -m dreamervla.train \
  experiment=online_cotrain_ray_dreamervla_tiny \
  logger=tensorboard \
  +inference.placement.strategy=packed \
  +inference.placement.gpu_id=0 \
  inference.cfg.device=auto \
  +learner.placement.strategy=packed \
  +learner.placement.start_gpu=1 \
  +learner.placement.end_gpu=1 \
  +learner.placement.num_gpus_per_worker=1 \
  learner.train_cfg.device=auto \
  training.out_dir=/tmp/dvla_ray_tiny_2gpu_no_fsdp_proof \
  rollout.steps=9
```

With `CUDA_VISIBLE_DEVICES=2,3`, Ray placement index `0` maps to physical GPU 2
and placement index `1` maps to physical GPU 3. Inside each actor the selected
card is exposed as local `cuda:0`, so `device=auto` resolves correctly without
FSDP.

FSDP remains a separate manual learner-sharding option via `+parallelism=fsdp`;
it is not required to prove two-card Ray placement.

To exercise packed Ray placement with 3 or 4 GPU actors, use the e2e test:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
PYTHONPATH=. "${PYTHON}" -m pytest \
  tests/e2e_tests/test_s1_worker_group.py::test_worker_group_packed_gpu_env_maps_visible_devices
```

## No-Ray Multi-GPU Baseline Smoke

Use this when you want to verify the normal `torchrun` / NCCL path without Ray.
It does not load LIBERO assets or checkpoints; it only proves that two local
or more local processes bind to GPUs and can communicate.

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m dreamervla.diagnostics.smoke_torchrun_multigpu
```

Expected output contains one JSON line per rank:

```text
{"all_reduce_sum": 3.0, "backend": "nccl", ..., "local_rank": 0, "rank": 0, "ray_imported": false, "world_size": 2}
{"all_reduce_sum": 3.0, "backend": "nccl", ..., "local_rank": 1, "rank": 1, "ray_imported": false, "world_size": 2}
```

The matching e2e test is:

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. "${PYTHON}" -m pytest \
  tests/e2e_tests/test_noray_torchrun_multigpu.py
```

With `CUDA_VISIBLE_DEVICES=2,3,4,5`, that e2e covers 2, 3, and 4 ranks.

## Cold-Start Rollout Smoke Route

The cold-start route launches Ray env actors plus one batched inference actor
and writes reward HDF5 plus matching `obs_embedding` sidecar shards through
`RolloutDumpWriter`:

```bash
python -m dreamervla.train experiment=collect_rollouts_ray_synthetic
```

Default smoke outputs stay repo-relative:

```text
data/collected_rollouts/ray_synthetic/reward/ray_shard_000.hdf5
data/collected_rollouts/ray_synthetic/hidden/ray_shard_000.hdf5
data/collected_rollouts/ray_synthetic/hidden/preprocess_config.json
```

For real collected data, override `env.cfg`, `inference.cfg`,
`dump.reward_dir`, and `dump.hidden_dir` through Hydra. Keep runtime data under
`${DVLA_DATA_ROOT}` or a relative `data/...` path so runs remain portable.

## Real OFT / LIBERO Smoke Routes

These routes are gated because they load real checkpoints and LIBERO assets.

Real Ray online cotrain:

```bash
export DVLA_REAL_RAY_COTRAIN_SMOKE=1
export DVLA_RYNNVLA_CKPT=/path/to/rynnvla_or_action_head_ckpt
export DVLA_DREAMERVLA_WARMUP_CKPT=/path/to/warmup_runner.ckpt

PYTHONPATH=. "${PYTHON}" -m pytest tests/e2e_tests/test_s5_ray_real_cotrain.py
```

The matching Hydra route is:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. "${PYTHON}" -m dreamervla.train \
  experiment=online_cotrain_ray_oft \
  logger=tensorboard \
  init.vla_ckpt_path="${DVLA_RYNNVLA_CKPT}" \
  init.warmup_ckpt_path="${DVLA_DREAMERVLA_WARMUP_CKPT}" \
  env.num_workers=1 \
  env.cfg.kwargs.max_steps=8 \
  rollout.steps=16 \
  rollout.min_replay_episodes=1 \
  replay.cfg.sequence_length=8 \
  ray_data.sequence_length=8 \
  learner.train_cfg.batch_size=1 \
  learner.train_cfg.classifier_batch_size=1 \
  learner.model_cfg.classifier.kwargs.window=1 \
  learner.train_cfg.algorithm_cfg.wmpo.classifier_min_steps=1
```

The config keeps model and data concerns separate:

- model construction lives under `ray_components.{encoder,world_model,policy,classifier}`
- dataset/task/rollout details live under `task`, `env`, `replay`, and
  `ray_data`
- the runner consumes generic `learner.model_cfg` and `inference.cfg`; it does
  not branch on RynnVLA, LIBERO, or a concrete sidecar folder name

Real OpenVLA-OFT Ray cold-start collection:

```bash
export DVLA_REAL_OFT_COLLECT_SMOKE=1
export DVLA_OFT_CKPT=/path/to/openvla_oft_ckpt

PYTHONPATH=. "${PYTHON}" -m pytest tests/e2e_tests/test_s6_ray_real_oft_collect.py
```

The matching Hydra route is:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. "${PYTHON}" -m dreamervla.train \
  experiment=collect_rollouts_ray \
  logger=tensorboard \
  task.openvla_oft.ckpt_path="${DVLA_OFT_CKPT}" \
  task.openvla_oft.component_ckpt_dir="${DVLA_OFT_CKPT}" \
  task.openvla_oft.hdf5_reward_dir=/tmp/dvla_real_oft_collect/reward \
  task.openvla_oft.action_hidden_dir=/tmp/dvla_real_oft_collect/hidden \
  collect.task_ids=[0] \
  collect.episodes_per_task=1 \
  collect.envs_per_gpu=1 \
  rollout.target_episodes=1 \
  rollout.max_steps=300 \
  env.num_workers=1
```

## Tests

True Ray tests live under `tests/e2e_tests/`:

```bash
"${PYTHON}" -m pytest \
  tests/e2e_tests/test_s3_inference_worker.py \
  tests/e2e_tests/test_s5_ray_cotrain_smoke.py \
  tests/e2e_tests/test_s6_ray_coldstart_collect.py
```

Unit-level contract tests live under `tests/unit_tests/` and avoid starting Ray
unless explicitly placed in e2e.

## Current Boundaries

This backend is scoped to single-node Ray. It validates the online worker
boundaries, overlap loop, Ray learner boundary, FSDP/FSDP2 strategy entry
points, and weight-sync contracts. The remaining production work is intentionally
separate from the backend architecture:

- real multi-GPU CUDA smoke / long-run convergence validation on the target
  machine

Multi-node Ray is not a target for this backend.
