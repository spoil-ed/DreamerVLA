# Aggressive Dreamer Effect Experiment Design

## Objective

Add an opt-in Hydra experiment for one aggressive, short Dreamer run on eight
H100 GPUs. The experiment must leave the existing `openvla_libero` recipe
unchanged and must be selected explicitly at launch time.

The experiment is intended to establish whether a stronger failure-conditioned
imagined-RL update improves real LIBERO Goal success quickly. It is not intended
to identify the causal contribution of each changed hyperparameter.

## Scope

This change adds configuration, composition tests, and the configuration registry
entry only. It does not change runner code, PPO implementation, W&B aggregation,
the existing `openvla_libero` experiment, or shell-launcher behavior.

## Configuration Surface

Add `configs/experiment/openvla_libero_aggressive.yaml` as a complete experiment
recipe. It composes the same task, Dreamer route, world model, classifier, and
launcher as `openvla_libero.yaml`, then declares only the aggressive experiment's
own run name and overrides.

The recipe owns these values:

| Configuration path | Value |
|---|---:|
| `run.name` | `openvla_libero_aggressive` |
| `manual_cotrain.global_steps` | `20` |
| `manual_cotrain.checkpoint_every` | `5` |
| `manual_cotrain.eval_interval_global_steps` | `5` |
| `manual_cotrain.eval_initial_global_step` | `true` |
| `manual_cotrain.eval_protocol.num_episodes_per_task` | `10` |
| `manual_cotrain.real_rollout_target_trajectories` | `64` |
| `manual_cotrain.max_policy_kl` | `0.03` |
| `algorithm.group_size` | `16` |
| `algorithm.entropy_bonus` | `1.0e-3` |
| `ray_actor_optimizer.lr` | `1.0e-6` |
| `actor.train_cfg.lr` | `1.0e-6` |
| `actor.train_cfg.algorithm_cfg.ppo_update_epochs` | `2` |

The experiment deliberately keeps the following inherited values unchanged:

- 1,024 imagined trajectories per global step;
- global actor batch size 16,384 and microbatch size 8;
- PPO clip bounds, dual clip, reward coefficient, reward filtering, gamma, GAE,
  KL coefficient, optimizer gradient clip, replay capacity, task balancing, and
  per-step policy synchronization;
- frozen world-model and classifier checkpoints supplied by the launcher.

The duplicated actor learning-rate paths are intentional: the resolved config
contains a runner-level actor learning rate and a concrete optimizer learning
rate. Both must agree, and the composition test must prove that they do.

## Launch Contract

Training is selected explicitly without changing the default experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero_aggressive \
  --wm_ckpt /path/to/wm/checkpoints/latest.ckpt \
  --cls_ckpt /path/to/classifier/checkpoints/latest.ckpt
```

The launcher must continue to map `--config openvla_libero` to the original
recipe and `--config openvla_libero_aggressive` to the new recipe. No launcher
defaults or shell parsing are added.

The 100-episode resident evaluations at steps 0, 5, 10, 15, and 20 are screening
evaluations. A 300-episode confirmation is launched explicitly against a selected
checkpoint:

```bash
bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/aggressive-run \
  eval.num_episodes_per_task=30
```

The same task IDs, environment seed schedule, and initial-state protocol must be
used for the initial-policy baseline and the selected trained checkpoint so that
episode-level outcomes can be paired outside the training loop.

## Safety and Interpretation

`manual_cotrain.max_policy_kl=0.03` is the transactional hard guard. A policy
update above the limit must use the existing rollback behavior. This configuration
does not add dynamic learning-rate changes or custom early stopping because those
would require runner behavior changes and would make the single experiment harder
to reproduce.

Operational monitoring should stop the run for any NaN/Inf, KL rollback, worker
crash, discarded trajectory, or actor/rollout policy-version mismatch. The human
operator should also inspect active-sample density and real-rollout success at
each five-step checkpoint. Those monitoring rules are runbook guidance, not new
configuration semantics.

The experiment is considered effective only if the paired 300-episode evaluation
improves overall success by at least eight percentage points with a one-sided
paired test below 0.05 and no task loses more than ten percentage points. Because
there is no simultaneous factor or control run, the result establishes the effect
of the combined aggressive recipe relative to its initial policy, not the causal
effect of any one hyperparameter.

## Validation

Add a focused Hydra composition test that proves:

1. `experiment=openvla_libero_aggressive` resolves to `DreamerRunner` and all
   values in the table above;
2. interpolated group size, entropy coefficient, and optimizer learning rate reach
   the actor configuration;
3. the resolved configuration passes `validate_cfg`;
4. `experiment=openvla_libero` retains its existing run name, 20,000-step budget,
   group size 8, policy learning rate `5.0e-7`, PPO epoch count 1, entropy
   coefficient 0, 32 real trajectories, and KL limit 0.1.

Update the config registry to list the new recipe as an opt-in aggressive
effect-validation route. No existing registry row is changed.

## Non-Goals

- No 2x2 or sweep experiment.
- No automatic statistical test inside the training runner.
- No W&B metric-schema or aggregation change.
- No modification of existing experiment YAML files.
- No new shell script.
- No change to checkpoint, replay, resume, or evaluation runner behavior.
