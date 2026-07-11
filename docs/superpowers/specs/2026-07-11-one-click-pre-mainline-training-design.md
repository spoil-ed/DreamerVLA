# One-Click Pre-Mainline Component Training

## Goal

Expose the official-data world-model and classifier stages as two independent,
one-command jobs. Fresh training must not require artifact paths. Paths are
required only when resuming a job or when frozen-policy RL consumes the two
completed runs.

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

The pre-mainline launcher accepts two optional Hydra script-config values:
`wm_run_root` and `classifier_run_root`. They default to the existing integrated
layout (`${run_root}/wm` and `${run_root}/classifier`) so `stage=all` remains
backward compatible. Independent component jobs are consumed with:

```bash
bash scripts/e2e_frozen_model_pre_mainline.sh \
  stage=rl \
  wm_run_root=/path/to/wm/run \
  classifier_run_root=/path/to/classifier/run
```

The launcher reuses the existing fail-closed selectors. It requires a completed
WM run before selecting its lowest-loss valid checkpoint and reads the
classifier run's `summary.json` before selecting the held-out-window-F1
checkpoint. It then materializes the usual selection links under the new RL run
root and starts policy-only RL with immutable WM/CLS.

## Scope and Verification

No Runner, model, dataset, optimizer, or distributed-training implementation is
added. The change is limited to the two existing shell entrypoints, launcher
path composition, Hydra script config, documentation, and contract tests.
`third_party/` remains ignored and untouched.

Verification is static and lightweight: Hydra composition, generated command
arguments, selector source roots, fresh/resume shell contracts, shell syntax,
Ruff, and focused unit tests. No WM, classifier, RL, GPU, Ray, or LIBERO job is
started by the implementation work.
