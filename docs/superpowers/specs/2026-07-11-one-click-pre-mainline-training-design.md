# One-Click Pre-Mainline Component Training

## Goal

Expose the official-data world-model and classifier stages as two independent,
one-command jobs. Fresh training must not require artifact paths. Paths are
required only when resuming a job or when frozen-policy RL consumes explicit
component sources.

## Entrypoints

The existing entrypoints remain the only component-training scripts:

```bash
bash scripts/experiments/world_model_training/train.sh
bash scripts/experiments/classifier_training/train.sh
```

They default to `GPU_COUNT=8` and select the Hydra experiments
`wm_official_upper_bound` and `classifier_official_upper_bound`. Shell code owns
environment setup and `torchrun`; Hydra remains the source of truth for data,
models, batch sizes, learning rates, budgets, logging, and checkpoint behavior.

The established 8xH100 parameters remain unchanged:

- world model: per-rank batch 16, global batch 128, learning rate `3e-5`;
- classifier: per-rank batch 4, global batch 32, learning rate `3e-5`.

Each fresh run creates and prints its own timestamped output directory. The two
jobs have no scheduler, node, shared-root, or cross-process coordination in the
repository.

## Resume

Resume preserves the existing explicit-path contract:

```bash
WORLD_MODEL_RESUME=true WORLD_MODEL_RUN_ROOT=/path/to/wm/run \
  bash scripts/experiments/world_model_training/train.sh

CLASSIFIER_RESUME=true CLASSIFIER_RUN_ROOT=/path/to/classifier/run \
  bash scripts/experiments/classifier_training/train.sh
```

Fresh training ignores no hidden previous run. Resume fails before launch when
the requested run has no compatible progress/latest checkpoint.

## Frozen-Policy RL Handoff

A thin handoff script accepts the two selected component checkpoint paths. It
starts the existing `dreamervla_frozen_models_rl` Hydra experiment without
adding another Runner or orchestration layer:

```bash
WORLD_MODEL_CKPT=/path/to/wm.ckpt \
CLASSIFIER_CKPT=/path/to/classifier.ckpt \
  bash scripts/e2e_frozen_model_cotrain.sh
```

The WM path may be any compatible training checkpoint; when a run directory is
given, the manual handoff chooses current top-k, final, or latest progress state
in that order. The classifier can be a compatible checkpoint or completed run
selection. Component schema, construction config, classifier threshold, and
frozen hashes remain validated by the active frozen runner.

## Scope and Verification

No Runner, model, dataset, optimizer, or distributed-training implementation is
added. The change is limited to three thin shell entrypoints, documentation,
and contract tests.
`third_party/` remains ignored and untouched.

Verification is static and lightweight: Hydra composition, generated command
arguments, selector source roots, fresh/resume shell contracts, shell syntax,
Ruff, and focused unit tests. No WM, classifier, RL, GPU, Ray, or LIBERO job is
started by the implementation work.
