# Failure-Conditioned Episode-Start Imagination Design

## Objective

Use real LIBERO collection only to maintain a pool of failed episodes, then train
the VLA exclusively with imagined PPO. The vision encoder, world model, and success
classifier stay frozen. Only the actor parameter partition is optimized.

This is deliberately an episode-start selector, not failure-near KIR: every imagined
rollout starts from the first valid hidden state of a failed real episode. The selector
name remains explicit so endpoint/window/classifier-guided anchors can be added later.

## Step Order

```text
collect completed real episodes
        |
append current episodes to historical replay (preserve success metadata)
        |
select failed_episode_start anchors from current + historical failures
        |
sample with replacement to fill imagined policy groups
        |
frozen WM rollout + frozen classifier reward
        |
imagined PPO updates actor parameters only
```

There is no encoder SFT, real-batch re-encoding, WM update, or classifier update.
The rollout policy is synchronized once at step entry and again after committed PPO.

## Anchor Contract

`OnlineReplay` owns selection. For `selector=failed_episode_start` it:

- filters to records whose explicit episode outcome is failure;
- takes `episode[0]` from each selected record;
- balances across requested tasks that have eligible failures;
- samples cyclically with replacement, allowing a failed episode to be reused;
- returns `anchor_step=0`, `is_failure_anchor=true`, and aligned requested sidecars.

Current real episodes are appended rather than replacing replay, so cold-start,
historical online, and current-step failures form one bounded pool. `RealTrajectory.success`
is authoritative when a Ray real batch is inserted.

If the replay has no failed episode, the global step skips imagined rollout and PPO
with an explicit metric. It must not silently fall back to successful/ordinary starts.

## Ownership

- `ReplayWorker` inserts real trajectories and exposes selector/count APIs.
- WM EnvWorker forwards the configured selector and repeats each sampled anchor by
  PPO `group_size`, keeping every comparison group on an identical real context.
- WM/classifier inference remains in WM EnvGroup and is no-grad/frozen.
- RolloutGroup performs policy inference only.
- ActorGroup receives imagined trajectories and performs PPO only.
- LearnerGroup may remain resident as the checkpoint owner in this first version,
  but its update method is never called in this mode.

## Configuration

Hydra selects:

- `manual_cotrain.training_mode: failure_imagined_rl`
- `manual_cotrain.initial_condition_selector: failed_episode_start`
- `manual_cotrain.learner_updates_enabled: false`
- `manual_cotrain.staged_policy_update: false`

The active cotrain recipe uses these values. Shell launchers contain no hidden
training defaults.

## Tests

1. Failed-only first-frame selection and explicit anchor metadata.
2. Repeated sampling fills batches and preserves task/sidecar alignment.
3. Appending real batches preserves historical failures and explicit outcomes.
4. WM bootstrap forwards the configured selector and preserves group repetition.
5. Cotrain order excludes encoder SFT, re-encoding, and learner updates while still
   running real collection, imagined rollout, and PPO.
6. No-failure steps skip imagined PPO instead of falling back or hanging.
