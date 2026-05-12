# Repository Structure

This document is the publishable map of the DreamerVLA repository. The rule is:
source code lives in `src/`, launch recipes live in `scripts/`, experiment
configuration lives in `configs/`, and all generated artifacts live under
`data/`.

## Top-Level Layout

```text
DreamerVLA/
├── README.md                 # Main project overview and setup
├── install.md                # Installation notes
├── pyproject.toml            # Editable package metadata
├── requirements.txt          # Runtime dependencies
├── configs/                  # Hydra configs
├── scripts/                  # Public launch / eval / diagnostic entry points
├── src/                      # Python package source
├── docs/                     # Architecture and experiment documentation
├── data/                     # Ignored runtime artifacts: ckpts, datasets, outputs
├── LIBERO/                   # Ignored local LIBERO checkout
└── dependencies/             # Ignored third-party local checkouts / wheels
```

`data/`, `LIBERO/`, and `dependencies/` are intentionally ignored by git. A
release should document how to obtain them, not commit them.

## Source Package

```text
src/
├── algorithms/               # Dreamer actor-critic / imagination algorithms
├── cli/                      # Hydra CLI entry points
├── dataloader/               # LIBERO, token, pixel, and Rynn-hidden datasets
├── env/                      # Online LIBERO environment wrappers
├── models/                   # VLA actor, encoders, critics, world models
├── preprocess/               # Data preprocessing logic
├── trainer/                  # Distributed training helpers
├── utils/                    # Checkpointing, logging, optimization utilities
├── workspace/                # Experiment orchestration classes
└── xllmx/                    # Chameleon / XLLM integration
```

The stable execution path is:

```text
scripts/*.sh
  -> python -m src.cli.train --config-name <config>
  -> src/workspace/<workspace>.py
  -> src/models/, src/algorithms/, src/dataloader/
```

## Current Research Mainline

The current active mainline is LIBERO-goal with RynnVLA hidden observations and
a DreamerV3-style RSSM:

```text
VLA goal checkpoint
  -> precomputed RynnVLA hidden sidecar
  -> Rynn-pixel DreamerV3 world model
  -> DreamerVLA actor/critic training
  -> LIBERO rollout eval
```

Canonical active assets:

```text
VLA base:
  data/ckpts/VLA_model_256/libero_goal

VLA action head / encoder state:
  data/outputs/vla/pretokenize_vla/pretokenize_vla_libero_goal_libero_goal_h5_20260508_060320/checkpoints/goal_h5_epoch000_train_vla_loss_1p323.ckpt

Precomputed hidden sidecar:
  data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000

Action horizon:
  5
```

Keep those three aligned. Do not mix goal action heads with `libero_10` hidden
sidecars.

## Public Entry Points

For current work, prefer these scripts:

```text
scripts/download_hf.sh
scripts/env_libero_goal.sh
scripts/prepare_data.sh
scripts/prepare_latent_data.sh
scripts/pretokenize_train_vla.sh
scripts/train_rynn_backbone_dreamerv3_wm.sh
scripts/train_dreamer_vla_rynn_pixel.sh
scripts/eval_libero_vla.sh
scripts/analyze_rynn_hidden_action_metrics.py
```

See `scripts/README.md` for details.

## Config Policy

Configs are grouped by experiment family:

```text
configs/pretokenize_vla_*.yaml              # VLA SFT
configs/dreamerv3_pixel_*.yaml              # pixel WM baseline
configs/dreamerv3_token_*.yaml              # token WM baseline
configs/rynn_backbone_dreamerv3_*.yaml      # Rynn hidden / pixel WM
configs/dreamer_vla_*.yaml                  # DreamerVLA actor-critic runs
configs/eval_libero_vla.yaml                # rollout eval
```

See `configs/README.md` for the active registry and historical configs.

## Data And Output Policy

All runtime artifacts belong under `data/`:

```text
data/ckpts/                  # downloaded model checkpoints
data/libero/                 # raw benchmark data
data/processed_data/         # preprocessed HDF5 / token / hidden sidecars
data/configs/                # generated data configs
data/outputs/vla/            # VLA training outputs
data/outputs/worldmodel/     # WM training outputs
data/outputs/dreamervla/     # DreamerVLA training outputs
data/outputs/eval/           # evaluation and diagnostic outputs
```

Generated outputs are not source. Keep only small summaries in `docs/` when an
experiment result matters for the code design.

## Release Checklist

Before publishing:

- `git status --short` shows only intentional source/doc/script changes.
- No checkpoints, HDF5 files, logs, videos, or cache files are staged.
- New configs are documented in `configs/README.md`.
- New scripts are documented in `scripts/README.md`.
- Any experiment conclusion needed by readers is summarized in `docs/`.
- Root README points to the current mainline and not only historical LIBERO-10
  experiments.
