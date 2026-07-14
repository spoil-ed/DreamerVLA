# Unified Run Layout, Resume, and Cotrain Progress Design

## Goal

Give every active Hydra experiment one predictable run root, make every trainable
experiment resumable in place, and make cotrain rollout progress use completed
trajectory counts rather than action-chunk counts.

## Run Layout

Hydra owns the default run root:

```text
${OUTPUT_ROOT}/<experiment>/<YYYYMMDD_HHMMSS>/
```

`OUTPUT_ROOT` defaults to `${DVLA_DATA_ROOT}/outputs`; when `DVLA_DATA_ROOT` is
unset it defaults to `${DVLA_ROOT}/data/outputs`. `RUN_ROOT`, when supplied, is the
output base and still receives the experiment and timestamp suffixes. An explicit
`training.out_dir=...` override remains an exact run-root override.

Each BaseRunner run writes a flat, role-based artifact layout:

```text
<run-root>/
  checkpoints/
  wandb/
  tensorboard/
  logs/
  video/
  diagnostics/
  .hydra/
  resolved_config.yaml
  run_manifest.json
```

Component checkpoint names live directly below `checkpoints/`, while step-scoped
runner checkpoints may use one `global_step_<N>/` level. The legacy `ckpt/` write
path is removed. Loaders retain read compatibility for old `ckpt/` runs.

World-model warmup uses:

```text
checkpoints/wm_warmup.ckpt
checkpoints/wm_warmup_hf/
checkpoints/warmup_progress/wm_step_<N>.ckpt
checkpoints/warmup_topk/wm/<metric-name>.ckpt
```

Classifier warmup follows the same convention. DINO-WM writes
`checkpoints/global_step_<N>/model.ckpt`, `checkpoints/latest.ckpt`, and the
canonical `checkpoints/wm_warmup.ckpt` alias.

## Resume Contract

The public Python launchers accept `--resume PATH`; shell scripts remain transparent
one-command entrypoints. `PATH` may name a run root, a canonical checkpoint file,
or a step-checkpoint directory. The launcher resolves the owning run root and
translates the option into Hydra fields. It rejects missing or ambiguous paths.

Resume always reuses the owning run root. It does not create another timestamp.
An explicit `training.out_dir` may be used only when the caller deliberately wants a
forked run; combining that override with `--resume` is rejected by the friendly CLI.

The common training contract is:

```yaml
training:
  resume: false
  resume_dir: null
```

Runner behavior:

- DINO-WM restores model, both AdamW optimizers, epoch, and global step.
- Dreamer-WM restores the latest warmup-progress checkpoint, including AdamW state
  and completed update count, or recognizes a completed `wm_warmup.ckpt`.
- Classifier restores model, AdamW state, epoch, global step, best metrics, best
  checkpoint paths, and threshold-related state.
- Cotrain maps the common resume path to its consolidated manual checkpoint and
  restores actor/encoder optimizers, WM/classifier optimizers, global step, replay
  sampling state, and replay contents when the checkpoint contains them.
- Collection is data-resumable: it reuses the original run root and skips already
  complete task/episode identities from the persisted rollout manifest and shards.
- Evaluation is restartable from its requested model checkpoint but has no optimizer
  or training-loop state, so it is not advertised as a training resume route.

Legacy checkpoint layouts remain readable; all new writes use `checkpoints/`.

## Cotrain Progress

Both real rollout and resident evaluation use the configured target trajectory count
as the progress-bar total. The primary counter is the number of completed trajectories:

```text
cotrain-real-rollout/00000001 ... 12/32 ... completed=12 successes=5 success_rate=0.417 chunks=...
eval/00000001                 ... 37/100 ... completed=37 successes=21 success_rate=0.568 chunks=...
```

Chunk counts remain diagnostic status only. They never become the progress numerator
or denominator. Progress callbacks aggregate totals across all real-rollout workers,
so the bar reports job-level completed trajectories rather than one worker's local
epoch or chunk budget.

## Validation and Compatibility

- Hydra composition tests assert every active experiment uses
  `<experiment>/<timestamp>` without repeated `pre_mainline`, `cotrain`, or model-name
  nesting.
- Launcher tests cover `--resume PATH`, path resolution, run-root inference, exact
  output-dir reuse, duplicate override rejection, and cotrain resume translation.
- Runner tests round-trip loop and optimizer state for WM, classifier, and cotrain.
- Progress tests prove real rollout and eval use completed trajectories over the
  configured job total while retaining chunk status.
- Existing non-resume commands remain valid and create a fresh timestamped run.
