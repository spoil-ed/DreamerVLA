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

## Online Cotrain Smoke Route

The low-cost cotrain route launches real Ray actors for:

- scheduler primitives: `Cluster`, `Worker`, `WorkerGroup`, `Channel`,
  `Placement`
- rollout: one `EnvWorker` per env actor
- inference: batched encoder + world-model + policy actor
- replay: `ReplayWorker` around `OnlineReplay`
- learner: `LearnerWorker` running either the synthetic PPO-style smoke update
  or real DreamerVLA cotrain phases
- weight sync: Ray object-store state-dict handoff

Run it through the normal Hydra entry:

```bash
python -m dreamervla.train experiment=online_cotrain_ray_synthetic
```

The runner writes the standard `BaseRunner` run artifacts under
`training.out_dir`, including `resolved_config.yaml` and `run_manifest.json`.

## Single-Node Multi-GPU Learner

The Ray backend is intentionally single-machine / single-node. Multi-GPU
learner placement is manual and config-driven:

```bash
python -m dreamervla.train \
  experiment=online_cotrain_ray_dreamervla_tiny \
  +parallelism=fsdp \
  learner.num_workers=2 \
  learner.placement.end_gpu=1 \
  learner.train_cfg.fsdp.backend=nccl
```

Inside each Ray learner actor, `CUDA_VISIBLE_DEVICES` is isolated by placement,
so `learner.train_cfg.device=auto` resolves to local `cuda:0`. For CPU-only
debugging use `+parallelism=none`, or set `learner.placement.strategy=node`.
`validate_cfg` checks learner placement shapes before runner setup; the runners
also keep the backend scoped to one live Ray node.

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

## Tests

True Ray tests live under `tests/e2e_tests/`:

```bash
python -m pytest tests/e2e_tests -q
```

Unit-level contract tests live under `tests/unit_tests/` and avoid starting Ray
unless explicitly placed in e2e.

## Current Boundaries

This backend validates the online worker boundaries and overlap loop. The
remaining production integrations are intentionally separate steps:

- real LIBERO/VLA component adapters for the Ray runner config
- bucketed/patch/collective weight sync for large GPU-resident weights
- production LIBERO/OFT cold-start config binding for the Ray collector

Multi-node Ray is not a target for this backend.
