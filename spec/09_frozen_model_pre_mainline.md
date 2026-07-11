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
`input_token_embedding [T,256,4096]`. No alternative observation interface is
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
official LIBERO reward HDF5 + official input-token sidecars
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
- `offline_warmup.hidden_dir == task.openvla_oft.input_token_dir`;
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
- `data.success_dir_hidden == task.openvla_oft.input_token_dir`;
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

They create separate timestamped run directories. Once checkpoints are chosen,
policy-only frozen cotrain takes the two explicit paths:

```bash
WORLD_MODEL_CKPT=/path/to/wm.ckpt \
CLASSIFIER_CKPT=/path/to/classifier.ckpt \
  bash scripts/e2e_frozen_model_cotrain.sh
```

`python -m dreamervla.launchers.frozen_model_pre_mainline` composes
`configs/scripts/frozen_model_pre_mainline.yaml`. It owns one run root with
subdirectories `wm/`, `classifier/`, `rl/`, `eval_baseline/`, `eval_rl/`, and
`summary/`.

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
