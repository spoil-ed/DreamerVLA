# Hydra-Centered Experiment and Resume Design

## Goal

Make Hydra the single source of truth for every active DreamerVLA experiment while
preserving the shared Ray implementation. A composed config must completely describe
the effective run before the Runner is constructed. Checkpoint, TensorBoard, W&B, and
local logs must resume in the checkpoint-owning run root without replay-buffer state.

This design supersedes the replay-checkpoint portions of the earlier run-layout and
Dreamer designs. Replay may remain an in-memory training mechanism, but it is not a
checkpoint or resume contract.

## Experiment Boundary

`configs/train.yaml` remains the only native training entrypoint, but it no longer
selects an implicit experiment. Callers must choose `experiment=<name>` directly or
through the thin launcher. Missing experiment selection fails before Runner setup.

The retained mainline experiment roles are:

- `collect_rollouts`: Ray rollout collection and persisted collection manifest.
- independent WM warmup: `WorldModelTrainingRunner`, producing
  `checkpoints/wm_warmup.ckpt`.
- independent classifier warmup: `SuccessClassifierTrainingRunner`, producing
  `checkpoints/classifier_warmup.ckpt`.
- `openvla_onetraj_libero_cotrain`: full `CotrainRunner` route with real collection,
  encoder SFT, WM/classifier learner updates, imagined rollout, and actor update.
- `openvla_libero`: frozen-WM/classifier `DreamerRunner` route with imagined actor
  update only.
- `eval_cotrain`: read-only evaluation of a selected cotrain checkpoint.

`CotrainRunner` and `DreamerRunner` continue sharing one implementation. Their Hydra
contracts differ explicitly:

```yaml
# CotrainRunner
manual_cotrain:
  training_mode: staged_full_cotrain
  learner_updates_enabled: true
  staged_policy_update: true

# DreamerRunner
manual_cotrain:
  training_mode: failure_imagined_rl
  learner_updates_enabled: false
  staged_policy_update: false
```

The full cotrain recipe must resolve to at least one real-env worker, one WM-env
worker, ActorGroup, RolloutGroup, and LearnerGroup. Configuration validation builds
the same placement plan used by the Runner and rejects impossible topology before
Ray starts.

## Hydra Profiles and Effective Configuration

Production, debug, and smoke budgets live in a Hydra config group such as
`profile=production|debug|smoke`. Profiles own steps, epochs, batch sizes, evaluation
cadence, checkpoint cadence, and reduced worker geometry.

Runners do not rewrite composed configuration. This removes the current runtime
mutation in cotrain, Dreamer, WM warmup, VLA training, and evaluation. Derived runtime
values such as per-rank batch size may be computed into local variables, but they do
not silently replace experimental input. Any derived value needed for reproducibility
is recorded in `run_manifest.json` as runtime metadata.

The Hydra `.hydra/` directory is the canonical configuration snapshot. No additional
`resolved_config.yaml` is written.

## Launcher Contract

The launcher selects and composes a Hydra experiment, supplies process mechanics, and
executes `dreamervla.train`. It does not maintain a parallel experiment schema.

The documented `--wm_ckpt` and `--cls_ckpt` convenience flags remain because they are
part of the public mainline command. They only normalize to the explicit Hydra fields
`init.world_model_state_ckpt` and `init.classifier_state_ckpt`, with both required
together. Resume similarly normalizes one path into `training.resume`,
`training.resume_path`, `training.resume_dir`, and the owning `training.out_dir`.

Launcher aliases that dynamically choose among multiple training fields are removed.
Environment variables that change experiment semantics are removed, including
`WORLD_MODEL_CKPT`, `CLASSIFIER_CKPT`, `WMCLS_COTRAIN_GLOBAL_STEPS`, and
`COTRAIN_DRY_RUN`. Operational environment such as CUDA visibility, Python paths,
allocator behavior, and LIBERO setup remains launcher-owned.

## Resume and Checkpoint Contract

All trainable routes restore model state, optimizer state, loop progress, and RNG
before initializing TensorBoard, W&B, or route-specific append-only logs. This makes
the restored global step the first logger resume step.

TensorBoard creates a new event segment in the existing `tensorboard/` directory with
`purge_step=<restored step>`. Online W&B reuses `wandb/run_id.txt` and resumes from the
restored step. Offline W&B creates another local segment with the same run ID; the
existing one-argument sync launcher uploads all segments as one logical run.

Cotrain checkpoints contain:

- actor/policy, world model, classifier, and optional encoder state;
- all active optimizer states;
- classifier threshold and loop progress;
- controller, actor-rank, and learner-rank RNG state.

They do not contain replay contents, replay cursors, or replay sampling state. Loading
an older checkpoint that contains replay fields ignores those fields. Resume starts a
fresh in-memory replay and follows the epoch-level data flow selected by the active
Hydra experiment.

Independent classifier warmup uses one complete resumable
`classifier_warmup.ckpt`. `latest.ckpt` may be a hard link to the same bytes for the
generic resume path. Metric-named classifier snapshots are created only when an
explicit top-k policy is enabled.

## Output Retention

One invocation owns one run root. `latest.ckpt` is a link/copy pointer and does not
cause a second serialization. Full cotrain step checkpoints use
`checkpoints/global_step_<N>/manual_cotrain.ckpt` and apply a Hydra-configured
keep-last policy. The conservative production default retains the newest two step
checkpoints; top-k and HF exports remain opt-in.

Transient Ray progress uses one current progress location under `diagnostics/` and is
removed or overwritten after the phase completes. It does not create one permanent
directory per global step. Classifier JSONL logs append on resume rather than being
truncated.

## Diagnostics and Script Surface

Files under `scripts/experiments/` must select a Hydra experiment and dispatch through
the unified launcher. Standalone probes and measurement tools belong under a
diagnostics script namespace and are not advertised as experiments. The stale
classifier artifact summarizer is removed rather than preserved because it reads the
retired `log/` and `outputs/classifier/<family>` layout and duplicates evaluation
metadata already available from the Runner and metric backends.

## Validation and Tests

Tests must prove, in red-green order, that:

- `configs/train.yaml` requires explicit experiment selection;
- the full cotrain experiment resolves to `CotrainRunner`, enables learner updates,
  and has both real and WM env placements;
- `openvla_libero` resolves to `DreamerRunner` and keeps WM/classifier updates frozen;
- invalid placement fails in `validate_cfg`, before `_build_groups`;
- debug/smoke behavior is expressed by Hydra composition and no Runner mutates the
  config to choose budgets;
- classifier resume restores progress before any metric logger is created and keeps
  existing JSONL records;
- new cotrain checkpoints contain no replay-related fields and legacy replay fields
  are ignored;
- checkpoint retention and transient-progress cleanup bound the output tree;
- classifier warmup writes one canonical resumable artifact unless top-k is enabled;
- W&B online/offline resume and the one-argument sync command remain intact;
- active experiment shell scripts route through Hydra.

Focused tests run first for each behavior. Completion requires the complete unit test
suite and static checks in the documented `dreamervla` environment.
