# Minimal Aggressive Dreamer Experiment Design

## Objective

Add one opt-in Hydra experiment for a short, aggressive Dreamer effectiveness
test on eight H100 GPUs. The experiment must be selected explicitly and must not
change the existing `openvla_libero` recipe or any runtime implementation.

## Scope

The minimal change contains:

1. `configs/experiment/openvla_libero_aggressive.yaml`;
2. a focused Hydra composition and isolation test;
3. one new config-registry entry.

There are no Actor, PPO, runner, worker, channel, replay, checkpoint, evaluation,
or shell-launcher code changes. Packed samples, async execution, adaptive sampling,
new metrics, and automatic early stopping are deferred until a short configuration
experiment demonstrates that stronger updates improve real success.

## Experiment Configuration

The new experiment composes the same task, Dreamer route, world model, classifier,
and cotrain launcher as `openvla_libero`, then owns only these overrides:

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

The experiment inherits 1,024 imagined trajectories, global batch 16,384,
microbatch 8, gradient clip 1, replay capacity 80,000, task balancing, PPO clip
bounds, reward coefficient, reward filtering, gamma, GAE, KL coefficient, frozen
WM/classifier behavior, and per-step policy synchronization.

Both actor learning-rate paths are explicit because the resolved runtime config
contains a runner-level actor learning rate and a concrete policy optimizer learning
rate. The test must prove that both resolve to `1.0e-6`.

## Launch Contract

The new experiment is selected explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero_aggressive \
  --wm_ckpt /path/to/wm/checkpoints/latest.ckpt \
  --cls_ckpt /path/to/classifier/checkpoints/latest.ckpt
```

The existing command with `--config openvla_libero` keeps its current resolved
configuration and behavior.

Resident evaluations use 10 episodes per task at steps 0, 5, 10, 15, and 20. A
selected checkpoint can be confirmed without another config file:

```bash
bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/aggressive-run \
  eval.num_episodes_per_task=30
```

## Validation

The focused test must prove:

1. the new experiment resolves to `DreamerRunner`;
2. every value in the override table resolves exactly;
3. group size and entropy propagate into `actor.train_cfg.algorithm_cfg`;
4. policy optimizer LR and actor LR agree;
5. the new resolved configuration passes `validate_cfg`;
6. `experiment=openvla_libero` retains its existing run name, 20,000-step budget,
   checkpoint/eval cadence, group size 8, policy LR `5.0e-7`, PPO epoch count 1,
   entropy coefficient 0, 32 real trajectories, and KL limit 0.1.

Update `configs/README.md` by adding a new registry row only. No existing row or
command is edited.

## Non-Goals

- No 2x2 experiment or sweep.
- No architecture or algorithm implementation change.
- No new shell script.
- No modification of existing experiment YAML files.
- No automated statistical test or early-stop controller.
