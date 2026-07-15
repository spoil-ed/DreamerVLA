# Training Output, Resume, and W&B Sync Design

## Goal

Make the world-model warmup, success-classifier training, and cotrain routes use one
clean run-root layout and resume model, including checkpoints, optimizers, epoch/global
progress, best metrics, classifier thresholds, RNG state, TensorBoard history, and W&B
identity. Provide one shell command that uploads a local W&B directory to the online
service.

Replay-buffer persistence and restoration are explicitly outside this change. Existing
replay behavior remains untouched.

## Scope and assumptions

- Checkpoints are saved at epoch boundaries. A resumed epoch-based job starts at the
  next epoch, so no dataloader batch cursor or prefetched batch queue is persisted.
- RNG state is persisted so the next epoch's shuffle, dropout, sampling, and random
  augmentation continue from the checkpoint boundary.
- The design applies to the active world-model, classifier, and cotrain training routes.
- Existing canonical and legacy checkpoint paths remain readable. New artifacts use
  only the canonical layout.
- Explicitly configured top-k checkpoints and Hugging Face exports have independent
  semantics and are not treated as accidental duplication.
- Replay-buffer contents, replay sampling state, replay schema migration, and replay
  restore failure handling are not changed or tested by this work.

## Canonical run-root layout

Each invocation owns one shallow run root:

```text
<output_root>/<run.name>/<YYYYMMDD_HHMMSS>/
├── checkpoints/
│   ├── wm_warmup.ckpt                         # WM warmup final/resume artifact
│   ├── classifier_warmup.ckpt                 # classifier warmup final/resume artifact
│   ├── latest.ckpt                            # active route's canonical resume pointer
│   ├── global_step_<N>/manual_cotrain.ckpt    # cotrain point-in-time artifact
│   ├── warmup_progress/                       # at most one latest file per component
│   ├── warmup_topk/                           # only when top-k is explicitly enabled
│   └── *_hf/                                  # only when HF export is explicitly enabled
├── tensorboard/
│   └── events.out.tfevents.*
├── wandb/
│   ├── run_id.txt
│   └── offline-run-*/
├── logs/
├── video/{train,eval}/
├── diagnostics/
├── .hydra/
├── resolved_config.yaml
└── run_manifest.json
```

The logger must not create `<run_root>/wandb/wandb/`. TensorBoard must not copy the
already-canonical resolved configuration into `tensorboard/config.yaml`.

Cotrain keeps the required point-in-time checkpoint and `checkpoints/latest.ckpt`.
The latest path is materialized with an atomic hard link when possible and an atomic
copy only when the filesystem cannot link, so it does not normally consume a second
full checkpoint allocation.

When a warmup recipe spans multiple configured epochs, epoch-boundary progress uses
one atomically replaced latest checkpoint per component instead of accumulating
`*_step_*.ckpt` files. No checkpoint is written from the middle of an epoch. The
completed warmup checkpoint replaces the need for the progress artifact. Explicit
top-k retention remains controlled by Hydra.

## Unified resume contract

`BaseRunner` continues to own run-root discovery, canonical artifact paths, and lazy
metric-logger construction. Resume must infer the checkpoint's owning run root before
any artifact is written.

Each resumable checkpoint stores the state that exists for its route:

- model/module state dictionaries;
- every active optimizer state dictionary;
- completed epoch and global step;
- best metric values and selected checkpoint paths where applicable;
- classifier threshold where applicable;
- Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG state;
- route-specific lightweight progress that is not replay-buffer state.

The restore order is:

1. Resolve the resume checkpoint and original run root.
2. Construct models and optimizers.
3. Restore model and optimizer states.
4. Restore epoch/global step, best metrics, threshold, and RNG state.
5. Set the first valid metric step before the metric logger is constructed.
6. Continue from the next epoch/global unit.

The checkpoint loader remains dual-read for canonical `checkpoints/` and historical
`ckpt/` paths. Writers never create new legacy paths.

## Metric resume semantics

TensorBoard and W&B share one route-provided `metric_resume_step`: the first step whose
old values are invalid and may be replaced after restore.

TensorBoard constructs `SummaryWriter` with `purge_step=metric_resume_step`. This
preserves valid history before the checkpoint and hides any events written after the
checkpoint before a crash.

The metric step must use the same global axis as the values being logged:

- standalone classifier: restored global step;
- WM warmup: restored WM progress step;
- classifier warmup after WM: `wm_total_steps + restored_classifier_step`;
- cotrain: restored cotrain global step.

This fixes the current cases where cotrain falls back to step zero and classifier
warmup passes a component-local step onto a combined WM-plus-classifier axis.

## W&B local identity and online resume

The canonical W&B directory is `<run_root>/wandb`. DreamerVLA persists one validated
run ID in `<run_root>/wandb/run_id.txt`. Every offline process segment created after a
resume uses this ID, while each segment remains a separate W&B binary stream because
offline streams cannot be reopened in place.

Online-mode resume uses the same run ID. When the installed SDK supports
`resume_from`, DreamerVLA passes `<run_id>?_step=<metric_resume_step>` so server history
after the checkpoint is truncated before new values are logged. Otherwise it uses
`resume="allow"`, which resumes the existing identity without rewind support.

Legacy discovery supports both:

- `<run_root>/wandb/{offline-run,run}-*/run-*.wandb`;
- `<run_root>/wandb/wandb/{offline-run,run}-*/run-*.wandb`.

New runs only write the first form.

## One-command offline upload

The repository provides:

```bash
bash scripts/utils/wandb_sync.sh /path/to/run_root/wandb
```

The script accepts exactly one W&B directory. It:

1. Validates the directory and the `wandb` CLI.
2. Discovers canonical and legacy offline segments.
3. Reads `run_id.txt`, or derives the canonical ID from the earliest segment for old
   runs.
4. Sorts segments chronologically.
5. Uploads the first unsynced segment to the canonical ID.
6. Uploads later segments with `wandb sync --append --id <run_id>` so they extend the
   same online run.
7. Relies on W&B sync markers to skip segments already uploaded, making repeated
   invocation safe.
8. Leaves every local file in place.

The normal prerequisite is a one-time `wandb login` or an available `WANDB_API_KEY`.
Entity and project come from the offline run metadata. The script does not add required
entity/project arguments; historical metadata gaps can be handled later with optional
overrides if a real case appears.

The script exits nonzero with a specific message for a missing CLI, invalid directory,
missing segments, invalid/mismatched run IDs, or a failed upload. It never silently
splits one local logical run into multiple online runs.

## Artifact lifecycle

- A fresh run creates the canonical directories once.
- Resume reuses the checkpoint's owning run root and appends logs there.
- `resolved_config.yaml` and `run_manifest.json` remain at the run root rather than
  being copied into backend directories.
- TensorBoard creates a new event file in the same directory and purges the invalid
  tail logically.
- W&B creates a new offline segment under the same canonical W&B directory using the
  stable run ID.
- Completed epoch checkpoints supersede temporary warmup progress files.

## Error handling

- A resume checkpoint missing an active optimizer, progress field, or required model
  state fails loudly instead of silently becoming a weight-only warm start.
- Legacy checkpoints that predate RNG persistence remain readable. Their resume emits
  one compatibility warning and starts from freshly seeded RNG; new checkpoints must
  contain RNG state.
- Metric logging must not initialize before resume state restoration. Existing guards
  remain and are extended to all three routes.
- Invalid W&B IDs, ambiguous IDs across local segments, and upload failures are fatal
  for the upload script and do not delete or rename local data.

## Verification

Tests cover:

- canonical shallow directory creation with no `wandb/wandb` and no duplicated
  TensorBoard config;
- run-root reuse from canonical and legacy resume paths;
- checkpoint round trips for WM, classifier, and cotrain model/optimizer/progress
  state, excluding replay-buffer assertions;
- RNG round trips for Python, NumPy, PyTorch CPU, and available CUDA devices;
- epoch-boundary continuation without a dataloader cursor;
- correct TensorBoard purge steps for standalone classifier, WM-to-classifier warmup,
  and cotrain;
- stable W&B identity across offline segments and online resume arguments;
- upload-script behavior for one segment, multiple resumed segments, legacy nested
  directories, already-synced segments, mismatched IDs, and CLI failure;
- shell syntax, Ruff, formatting, targeted unit tests, and `git diff --check`.

Replay-buffer save/load behavior is intentionally absent from the acceptance criteria.
