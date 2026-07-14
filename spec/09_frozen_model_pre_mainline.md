# Frozen WM/CLS Pre-Mainline RL Feasibility Test

## Status and Scope

This document defines a feasibility test that runs **before** the formal
cold-start/cotrain mainline. It does not replace the mainline and does not claim
online cotraining performance. The first implementation is deliberately scoped
to the canonical `libero_goal` suite and all ten of its tasks.

The test answers one question:

> Can a world model and success classifier trained from official LIBERO data,
> when held completely fixed, provide a useful imagined-RL signal that improves
> the real-environment success rate of the DreamerVLA policy?

The experiment uses the canonical OpenVLA-OFT observation sidecar
`hidden_token [T,256,4096]`. No alternative observation interface is
allowed.

## Hydra Composition Rule

Experiment names are configuration assemblies, not Python implementations.
`wm_official_upper_bound` reuses `OnlineCotrainPipelineRunner`, and
`classifier_official_upper_bound` reuses `LatentClassifierRunner`; trajectory
splitting, validation selection, data roots, and budgets are Hydra knobs with
backward-compatible defaults. `FrozenModelPolicyRunner` is the single new role
boundary because policy-only optimization with immutable dependencies has a
different lifecycle, not because it represents one LIBERO recipe. Component
construction remains entirely `_target_`-driven.
The shared `pre_mainline=libero_goal_official` group is the single Hydra source
for the ten required task IDs and ten official reward-shard filenames.

## Experimental Chain

The complete chain is:

```text
official LIBERO reward HDF5 + official hidden-token sidecars
    ├── WM upper-bound training  ──> WM checkpoint
    └── CLS upper-bound training ──> CLS checkpoint + threshold
                                      │
                                      ├── both artifacts validated
                                      └── both modules frozen
                                                │
official replay sequences ──> LUMOS imagined rollouts ──> policy-only PPO updates
                                                │
                                      DreamerVLA policy checkpoint
                                                │
                       real LIBERO baseline/RL evaluation and comparison
```

WM and CLS training are independent and may run concurrently, but policy
training cannot start until both checkpoints exist and pass their contracts.

## Stage A: Official-Data World Model

The experiment name is `wm_official_upper_bound`. It reuses the existing
WM-only replay warm-up implementation with these non-negotiable overrides:

- `offline_warmup.data_dir == task.hdf5_reward_dir`;
- `offline_warmup.hidden_dir == task.openvla_oft.hidden_token_dir`;
- `training.classifier_warmup_steps == 0`;
- `online_rollout.total_env_steps == 0`;
- the complete official replay is seeded before the first optimizer step.

The stage writes the normal split WM checkpoint and loss-ranked progress
checkpoints. The launcher first requires a valid, complete `wm_warmup.ckpt`.
Only after that completion gate passes may it select the lowest-loss ranked
checkpoint; every ranked artifact must carry matching step, loss, component,
and Hydra construction metadata. It uses the complete checkpoint when ranked
checkpoints are disabled.

## Stage B: Official-Data Success Classifier

The experiment name is `classifier_official_upper_bound`. It reuses
`LatentClassifierRunner` with:

- `data.success_dir_raw == task.hdf5_reward_dir`;
- `data.success_dir_hidden == task.openvla_oft.hidden_token_dir`;
- deterministic trajectory-level train/validation splitting;
- validation threshold sweeping;
- held-out window-level F1 as the checkpoint selector (official demonstrations
  provide no failure-episode class for a valid episode-F1 selector).

The launcher reads `summary.json` and requires a valid
`best_window_ckpt_path`. The selected checkpoint must contain `model`,
`threshold`, `f1`, `step`, the held-out `val_window` metrics, and the classifier
construction config; the summary F1/path/step must bind the same artifact.

## Stage C: Frozen-Model Policy RL

The experiment name is `dreamervla_frozen_models_rl`; its public runner is
`FrozenModelPolicyRunner`.

For the standalone eight-GPU handoff after independently training WM and CLS,
`dreamervla_frozen_models_rl_ray` reuses `ManualCotrainRayRunner`. It creates
eight WMEnv workers, eight no-grad Rollout workers, and an eight-rank FSDP
ActorGroup. RealEnv and LearnerGroup are absent, so the only optimizer is the
ActorGroup policy optimizer. This route shares the same WMEnv -> Rollout ->
Actor RL implementation as non-frozen manual cotrain; disabling LearnerGroup
updates and loading immutable WM/CLS checkpoints are the model-trainability
differences. Every shared boundary carries canonical
`hidden_token [256,4096]`; the WMEnv may store its state flat internally but
restores the token grid before WM, classifier, rollout-policy, and actor-training
calls. The single-process experiment remains the reference implementation used
by the complete proof launcher.

Each WM lease samples one aligned official replay condition (hidden token,
language embedding, proprioception, and task identity), repeats it across the
worker's 16 slots, and produces two contiguous `group_size=8` comparison
groups. The replay cursor rotates across all ten `libero_goal` tasks and is
stored as lightweight resume state. Actor FSDP synchronizes initial module
state across all eight ranks before the first update.

The Ray experiment uses the same RLinf-aligned PPO geometry as the non-frozen
manual mainline: one manual global step records exactly 1024 trajectories of
512 physical steps, or 65536 flattened OFT chunk samples at `chunk_size=8`.
`actor.train_cfg.global_batch_size=16384` and `micro_batch_size=32` therefore
produce four policy optimizer steps per manual global step. These values live
in the shared mainline Hydra config; the frozen experiment inherits them and
only disables LearnerGroup plus loads immutable WM/CLS checkpoints.

### Construction

The runner:

1. requires explicit WM and CLS checkpoint paths;
2. instantiates all three components through Hydra targets;
3. requires checkpoint construction metadata to exactly match the active
   Hydra WM/CLS configs, then loads both states strictly;
4. obtains the classifier threshold from the classifier checkpoint unless an
   explicit identical override is supplied;
5. calls `eval()` and sets `requires_grad=False` on every WM/CLS parameter;
6. constructs exactly one optimizer, for the policy;
7. optionally creates a frozen reference-policy copy for KL/BC regularization;
8. seeds `OnlineReplay` from the official reward/sidecar pair.

The seeded replay must contain at least one sampleable sequence for every task
ID `0..9`; partial suite coverage aborts before policy training.

The runner does not construct a real LIBERO environment, an encoder, a WM
optimizer, or a classifier optimizer.

### Imagined RL Loop

Each update samples a sequence from official replay and passes its
`obs_embedding`, action, terminal, language, and proprioception tensors to the
registered `LUMOS` actor-update route. LUMOS observes the real replay prefix,
rolls the frozen WM forward, scores imagined trajectories with the frozen CLS,
and updates only the policy.

Official replay is read-only after seeding. Imagined trajectories are not
written back into replay. No real-environment rollout is performed during this
stage.

### Immutability Proof

Before the first policy update, the runner computes deterministic SHA-256 hashes
over the complete WM state dict and complete CLS state dict. It recomputes and
compares these hashes before every policy checkpoint and at the end of training.
Any mismatch aborts the run.

The runner also records the policy state hash before and after training. A
nonzero update budget is considered unsuccessful when the policy hash does not
change or when no PPO optimizer step is applied.

### Checkpoints and Summary

The runner writes:

- `checkpoints/baseline.ckpt`: initial policy plus frozen WM/CLS;
- `checkpoints/latest.ckpt`: resumable policy state;
- `checkpoints/final.ckpt`: final policy plus frozen WM/CLS;
- `frozen_rl_summary.json`: source checkpoint paths, data paths, replay counts,
  initial/final hashes, applied PPO steps, and final training metrics.

All checkpoints contain the resolved training config, so the existing
`EmbodiedEvalRunner` can rebuild the modules. Resume is fail-closed: the runner
requires all component/optimizer states, binds the objective and construction
through a resume-contract hash, verifies the reconstructed reference-policy
hash, and restores the captured Python/Torch RNG state.

The standalone Ray realization uses its manual-cotrain checkpoint layout:
policy + policy-optimizer state, frozen source paths/hashes, threshold, and
causal counters. It deliberately does not duplicate the frozen WM/CLS tensors
into every policy checkpoint; resume reloads those two
explicit immutable source checkpoints and verifies their hashes again.

## Stage D: Real-LIBERO Proof

Training and evaluation are deliberately separated. The launcher evaluates:

1. the unmodified one-trajectory OpenVLA-OFT checkpoint (`ckpt_kind=vla`);
2. the trained `checkpoints/final.ckpt` (`ckpt_kind=dreamer`).

Both evaluations use identical suite, task IDs, initial-state enumeration,
seeds, episode counts, action horizon, and step budget. A comparison utility
rejects mismatched evaluation metadata and writes
`feasibility_summary.json`.

The proof route enables strict component loading. Evaluation records hashes of
the WM, classifier, and policy states from the checkpoint; the comparator also
reopens the final checkpoint and recomputes those hashes. The verdict therefore
requires `ckpt_kind=vla` for the baseline, `ckpt_kind=dreamer` for RL, and exact
agreement between the evaluated path/state and `frozen_rl_summary.json`.

The feasibility gate passes only when:

- the RL real-LIBERO success rate is strictly greater than the base VLA rate;
- the policy state hash changed;
- both frozen hashes are identical before and after RL;
- at least one PPO optimizer step was applied.

The result is a feasibility observation, not a statistical mainline claim. A
paper claim still requires multiple seeds and confidence intervals.

## Launcher

For independent component training, the normal entrypoints are one-command
Hydra launchers:

```bash
bash scripts/experiments/world_model_training/train.sh
bash scripts/experiments/classifier_training/train.sh
```

They create separate timestamped run directories. The manual policy-only frozen
Ray cotrain handoff accepts those run directories or explicit compatible
checkpoint files. Unlike the complete proof-chain selector, its WM handoff does
not require training completion: a run directory resolves current top-k, final,
or latest progress state in that order. Its classifier handoff accepts any
compatible `best_window_*.ckpt`, `final.ckpt`, or `latest.ckpt`; a run directory
prefers the highest held-out window F1, then final, then latest. BaseRunner
classifier checkpoints load `state_dicts.model` plus `cfg.classifier`. Because old
final/latest files do not embed the calibrated threshold, the manual launcher uses
the highest-F1 sibling threshold when present and otherwise emits an explicit
Hydra threshold of `0.5`.

```bash
WORLD_MODEL_CKPT=/path/to/world_model/run \
CLASSIFIER_CKPT=/path/to/classifier/run \
  python -m dreamervla.launchers.frozen_model_cotrain_ray \
  experiment=dreamervla_frozen_models_rl_ray
```

Resume requires the same two immutable sources plus the Ray policy checkpoint:

```bash
WORLD_MODEL_CKPT=/path/to/world_model/run \
CLASSIFIER_CKPT=/path/to/classifier/run \
COTRAIN_RESUME_CKPT=/path/to/frozen_cotrain_run/checkpoints/manual_cotrain_step_500/manual_cotrain.ckpt \
  python -m dreamervla.launchers.frozen_model_cotrain_ray \
  experiment=dreamervla_frozen_models_rl_ray
```

The launcher infers the original run root from the checkpoint path; assign
`COTRAIN_RUN_ROOT=/path/to/run` only when the checkpoint was relocated.

This gate is not a `configs/scripts` workflow. Its retained WM and classifier
stages enter through `configs/experiment/wm_official_upper_bound.yaml` and
`configs/experiment/classifier_official_upper_bound.yaml`; any top-level
orchestration must compose experiment recipes rather than add a script wrapper
config.

Supported stages are `all`, `wm`, `classifier`, `rl`, and `eval`. `dry_run=true`
prints fully resolved commands without running training. Stage guards fail
before subprocess launch when a required upstream checkpoint or summary is
missing.

The shell wrapper remains one command and delegates all iteration and dispatch
to Python.

## Failure Contracts

The route fails closed when:

- an official reward/sidecar shard pair is incomplete;
- either selected checkpoint is absent or has the wrong component schema;
- checkpoint tensor names or shapes do not strictly match the configured model;
- official replay contains no sampleable sequences;
- any of the ten `libero_goal` task IDs is absent from sampleable replay;
- a real-env or collected-rollout path is supplied to the RL experiment;
- WM/CLS optimizer configuration is present in the RL experiment;
- WM/CLS state changes during policy training;
- the policy receives no applied optimizer step;
- evaluation metadata differ between baseline and RL results.

## Verification Strategy

Implementation tests use importable tiny components and a one-step fake actor
route. They verify checkpoint loading, complete freezing, hash stability,
policy-only mutation, resume, config composition, launcher dry-run commands, and
evaluation comparison. No full WM, CLS, RL, Ray, GPU, or LIBERO job is run as
part of repository verification.
