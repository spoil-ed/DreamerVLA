# Manual Cotrain Rollout Budget

This note defines the rollout budget terms used by the manual Ray cotrain route.

## Three Units

`env action step`

One low-level environment action. In LIBERO/OpenVLA-OFT, this is one action applied to
the real simulator or the world-model environment.

`chunk_step`

One policy forward that emits one action chunk. For OpenVLA-OFT, one chunk contains
`manual_cotrain.num_action_chunks` low-level actions. With the current OpenVLA-OFT
route, `num_action_chunks = 8`, so:

```text
1 chunk_step = 1 policy forward = 8 env action steps
```

`epoch` in these manual cotrain knobs

One trajectory shard per env slot. It is not a training epoch. One rollout epoch runs:

```text
chunk_steps_per_epoch = max_steps_per_rollout_epoch / num_action_chunks
```

Example with `max_steps_per_rollout_epoch=512` and `num_action_chunks=8`:

```text
chunk_steps_per_epoch = 512 / 8 = 64
```

So one epoch means each slot emits one trajectory shard of 512 low-level env action
steps, represented as 64 action chunks.

## Knobs

`manual_cotrain.max_steps_per_rollout_epoch`

The real-env trajectory horizon in low-level env action steps.

If this is `512`, each RealEnv epoch runs 512 low-level action steps per slot.

`manual_cotrain.wm_rollout_multiplier`

Multiplies the world-model trajectory horizon only:

```text
wm_max_steps_per_rollout_epoch =
  max_steps_per_rollout_epoch * wm_rollout_multiplier
```

If both RealEnv and WMEnv should use a 512-step horizon, set:

```yaml
manual_cotrain:
  max_steps_per_rollout_epoch: 512
  wm_rollout_multiplier: 1
```

`manual_cotrain.real_rollout_epoch`

How many trajectory shards each RealEnv slot emits per global step.

With `envs_per_worker=2` and `real_rollout_epoch=4`, the single RealEnv worker emits:

```text
2 slots * 4 epochs = 8 real trajectory shards
```

`manual_cotrain.wm_rollout_epoch`

How many trajectory shards each WMEnv slot emits per global step.

With 3 WMEnv workers, `envs_per_worker=2`, and `wm_rollout_epoch=16`, WMEnv emits:

```text
3 workers * 2 slots * 16 epochs = 96 WM trajectory shards
```

## Example: 4 GPUs, Real Fewer Shards, Same 512-Step Horizon

For a 4-GPU run with one RealEnv worker and three WMEnv workers:

```yaml
manual_cotrain:
  max_steps_per_rollout_epoch: 512
  wm_rollout_multiplier: 1
  real_rollout_epoch: 4
  wm_rollout_epoch: 16
  envs_per_worker: 2
```

This means:

```text
Real horizon per shard = 512 env action steps = 64 chunk_steps
WM horizon per shard   = 512 env action steps = 64 chunk_steps

Real shard count = 1 worker * 2 slots * 4 epochs  = 8
WM shard count   = 3 workers * 2 slots * 16 epochs = 96
Total shards     = 104
```

The horizons are equal. The number of shards is not equal: WM contributes many more
trajectory shards, while RealEnv contributes fewer expensive simulator trajectories.

The actor grouping constraint still applies:

```text
total shards must be divisible by algorithm.group_size
```

With `group_size=8`, `104 / 8 = 13`, so this budget is valid.

## Common Confusion

`real_rollout_epoch` and `wm_rollout_epoch` do not control trajectory length. They
control trajectory count.

`max_steps_per_rollout_epoch` and `wm_rollout_multiplier` control trajectory length.

For equal real/WM trajectory length, use `wm_rollout_multiplier=1`.

For fewer RealEnv data than WMEnv data, keep `real_rollout_epoch < wm_rollout_epoch`.
