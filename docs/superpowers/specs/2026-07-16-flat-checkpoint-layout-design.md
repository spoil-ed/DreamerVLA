# Flat Checkpoint Layout Design

## Goal

Unify every DreamerVLA training and evaluation route around one predictable run
layout and one checkpoint discovery contract. Remove route-specific checkpoint
aliases and nested checkpoint directories while preserving read compatibility for
explicit historical checkpoint files.

## Training run layout

Every new training invocation owns this layout:

```text
${OUTPUTS_ROOT}/<experiment-name>/<timestamp>/
├── .hydra/
├── checkpoint_hf/          # optional; created only when HF saving is enabled
├── checkpoints/
│   ├── latest.ckpt
│   ├── epoch=0003-loss=0.182400.ckpt
│   └── epoch=0005-accuracy=0.846000.ckpt
├── diagnostics/
├── logs/
├── tensorboard/
├── video/
├── wandb/
└── run_manifest.json
```

The directory directly below `outputs/` is the selected experiment name. Hydra
represents that name as `run.name`; every experiment recipe must set it to the
recipe's public name rather than inheriting the generic `run` default. The next
directory is the invocation timestamp; that timestamp directory is the run root.
Resume reuses the checkpoint-owning run root and never creates a new timestamp.

## Evaluation run layout

Evaluation output is separated from training output and does not add a timestamp
directory:

```text
${OUTPUTS_ROOT}/eval/<task-name>/
├── .hydra/
├── diagnostics/
├── logs/
├── tensorboard/
├── video/
├── wandb/
└── run_manifest.json
```

`<task-name>/` is itself the evaluation run root. The task name comes from the
active Hydra evaluation/task configuration (for LIBERO routes, the suite name
such as `libero_goal`), not from the evaluation experiment name. Evaluation does
not create a timestamp layer beneath it. Checkpoints are evaluation inputs and
are not copied into this output directory.

## Torch checkpoint contract

`<run-root>/checkpoints/` is the only Torch checkpoint directory. It contains
files only; new code must not create checkpoint subdirectories.

The authoritative resumable checkpoint is always:

```text
checkpoints/latest.ckpt
```

Metric-selected retained checkpoints use:

```text
epoch=<zero-padded-completed-epoch>-<metric-name>=<metric-value>.ckpt
```

The epoch is the route's count of completed training epochs and is padded to at
least four digits; it is not relabeled global-step state. The monitor key and
comparison direction come from Hydra. Metric names are made filesystem-safe by
replacing `/` with `_`. Examples include `loss` for a world model and `accuracy`
or `success_rate` for classifier, Dreamer, cotrain, or VLA evaluation. The naming
layer must not hardcode which metric a runner uses.

There is no `topk-` prefix. Top-k is a retention policy, not a filename family.
When a checkpoint event occurs, `latest.ckpt` is updated atomically. If the
current metric enters the configured top-k, the same serialized bytes are
materialized at the metric filename by atomic hard link when supported and by
atomic copy otherwise.

New writes remove these names and layouts:

- `manual_cotrain.ckpt`
- `wm_warmup.ckpt`
- `classifier_warmup.ckpt`
- `model.ckpt`
- `global_step_*/`
- `manual_cotrain_step_*/`
- `warmup_progress/`
- `warmup_topk/`
- component checkpoint subdirectories
- per-checkpoint cotrain manifest files

Checkpoint metadata that is required for resume or inspection stays inside the
Torch payload. The run-level `run_manifest.json` remains at the run root.

## Route behavior

All trainable runners follow the same rules:

1. Save resumable state to `checkpoints/latest.ckpt` at the configured cadence
   and at normal completion.
2. Retain metric-selected files directly under `checkpoints/` when top-k is
   enabled.
3. Use the route's Hydra-selected monitor metric and mode. WM commonly monitors
   loss with `min`; classifier/Dreamer/cotrain/VLA commonly monitor an evaluation
   accuracy or success metric with `max`.
4. If no monitor metric is available at a checkpoint event, update only
   `latest.ckpt`; do not invent a ranking value or create a metric checkpoint.

Warmup routes are ordinary independent experiments. Their handoff checkpoint is
their run's `checkpoints/latest.ckpt`, so cotrain receives explicit WM and
classifier latest paths from the corresponding run roots.

## HF export contract

HF export is optional and disabled unless the active Hydra experiment explicitly
selects an HF-capable checkpoint format. When enabled, the latest export is
written to:

```text
<run-root>/checkpoint_hf/
```

`checkpoint_hf/` is a sibling of `checkpoints/`, not a child. No route-specific
HF directory names such as `wm_warmup_hf`, `classifier_warmup_hf`, or
`latest_hf` are written. Torch top-k retention does not duplicate HF exports.

## Resume and evaluation discovery

Training resume and evaluation use one shared resolver:

1. A concrete checkpoint file is used directly.
2. A `checkpoints/` directory resolves to `<directory>/latest.ckpt`.
3. A run root resolves to `<run-root>/checkpoints/latest.ckpt`.
4. A `checkpoint_hf/` directory with valid HF metadata is used as an HF export.
5. A missing `latest.ckpt` is an explicit error; new code does not guess among
   metric checkpoint filenames.

Explicit historical `.ckpt` files remain readable. Existing legacy run roots and
nested checkpoint directories may be discovered by a read-only compatibility
fallback, but no legacy path is produced by new runs.

Evaluation applies the same directory resolution before inspecting checkpoint
kind. Therefore `eval.ckpt_path` accepts a run root, its `checkpoints/` directory,
`checkpoint_hf/`, or a concrete checkpoint file.

## Configuration

Hydra remains the source of truth for checkpoint cadence, top-k count, monitor
metric, comparison mode, and Torch/HF format. Defaults select Torch-only saving;
HF output requires an explicit experiment setting.

Route-specific configuration keys may remain where a runner needs a distinct
cadence or metric, but filesystem paths and filenames are constructed only by
shared checkpoint helpers.

## Error handling and atomicity

- `latest.ckpt` uses temporary-file serialization followed by atomic replace.
- Metric checkpoint files reuse the already serialized latest payload.
- Top-k pruning deletes only known metric checkpoint files and never deletes
  `latest.ckpt` or `checkpoint_hf/`.
- Directory resolution reports the requested directory and expected
  `latest.ckpt` path when discovery fails.
- Resume continues to require complete model, optimizer, progress, and RNG state
  according to the active route's payload contract.

## Verification

Regression coverage must prove:

- training run roots compose as `outputs/<experiment>/<timestamp>`;
- evaluation run roots compose as `outputs/eval/<task-name>` with no timestamp;
- every runner writes flat `checkpoints/latest.ckpt`;
- metric checkpoint names follow `epoch=...-metric=value.ckpt`;
- top-k pruning is flat and preserves `latest.ckpt`;
- no new nested checkpoint directory or retired filename is produced;
- resume resolves a run root, `checkpoints/`, and a concrete `.ckpt`;
- eval resolves the same inputs before loading;
- `checkpoint_hf/` is absent by default and appears only under explicit HF
  configuration;
- historical concrete checkpoint files remain readable;
- full unit, lint, format, and shell gates pass.
