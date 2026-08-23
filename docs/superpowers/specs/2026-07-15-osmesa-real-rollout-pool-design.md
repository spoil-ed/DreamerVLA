# OSMesa Real Rollout Pool Design

## Goal

Run the Dreamer mainline real-LIBERO collection with 25 concurrent CPU OSMesa
workers while retaining eight GPU rollout-policy replicas and the existing
seven-worker imagined world-model phase.

## Topology

- Launch 25 `RealEnvWorker` actors with one environment slot per worker.
- Keep eight `MultiStepRolloutWorker` actors, one per GPU.
- Send every real-environment observation batch to one shared request queue.
  The eight rollout actors concurrently consume that queue, so available actors
  dynamically claim work rather than requiring `env_rank == rollout_rank`.
- Keep the response route rank-scoped. Each rollout result retains its original
  `env_rank` and is returned on that environment worker's response key.
- Keep imagined rollout rank-keyed and batched exactly as before. Real and
  imagined phases are sequential, so WM environment ranks use their own
  stage-local `0..N-1` namespace instead of being offset by the 25 real workers.

## Configuration

Introduce `manual_cotrain.real_envs_per_worker`, defaulting to the legacy
`envs_per_worker` value for compatibility. The Dreamer recipe sets:

```yaml
real_env_workers: 25
real_envs_per_worker: 1
real_rollout_target_trajectories: 32
```

This assigns two rollout epochs to seven workers and one epoch to the remaining
18 workers, producing exactly 32 completed trajectories in two waves.

## Control Flow

The real phase starts all 25 environment actors and all eight rollout actors.
Real actors publish `ObservationBatchMsg` values under a shared request key.
Rollout actors validate that each batch is internally rank-consistent, perform
inference, and publish `RolloutResultBatchMsg` under the original `env_rank`.
After every environment actor completes, the runner publishes eight stop
messages to the shared request key, one for each rollout consumer.

The WM and evaluation phases retain per-rank request keys and per-rank stop
messages. No extra VLA replicas are created.

## Validation and Observability

- Validation requires positive `real_envs_per_worker` and verifies that a fixed
  real trajectory target is divisible by it and large enough to schedule every
  configured real worker.
- Manual Ray validation applies to both `CotrainRunner` and `DreamerRunner`.
- Real rollout progress uses `real_envs_per_worker` and
  `real_max_steps_per_rollout_epoch`; imagined progress remains unchanged.
- Tests cover placement, exact 32-trajectory distribution, shared request and
  rank-scoped response routing, stop routing, and the resolved Dreamer recipe.

## Compatibility

The existing rank-keyed batch route remains the default. Only Dreamer real
collection selects the shared request key, so cotrain, imagined rollout, and
evaluation keep their current contracts.
