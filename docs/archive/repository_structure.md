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

The current active mainline is LIBERO-goal with pi0 action-query hidden
observations and a DreamerV3-style RSSM:

```text
LIBERO obs + language + state
  -> frozen RynnVLA/Chameleon backbone
  -> pi0 action-query block
  -> action_hidden [H, 1024]
  -> flattened action-hidden sidecar [H*1024]
  -> DreamerV3 RSSM posterior / transition
  -> hidden reconstruction + reward + continue + optional image reconstruction
```

The first implemented version freezes the shared VLA backbone and trains the WM
from precomputed action-hidden sidecars. Joint finetuning and action-hidden
DreamerVLA actor training are follow-up work, not the current mainline.

Canonical active assets:

```text
VLA base:
  data/ckpts/VLA_model_256/libero_goal

pi0 VLA action head / encoder state:
  data/ckpts/pi0_query_vla_libero_goal/epoch003_train_vla_loss1.255_success8of10.ckpt

Precomputed action-hidden sidecar:
  data/processed_data/libero_goal_no_noops_t_256_pi0_action_hidden_latest_fullseq

Action horizon:
  5
```

Keep the VLA base, pi0 action-head checkpoint, sidecar metadata,
`action_head_type=pi0_query`, and `time_horizon` aligned. Do not mix pooled
hidden sidecars with action-hidden WM configs.

## Public Entry Points

For current work, prefer these scripts:

```text
scripts/download_hf.sh
scripts/env_libero_goal.sh
scripts/env_libero_goal_pi0_query.sh
scripts/prepare_data.sh
scripts/pretokenize_train_vla.sh
scripts/run_pi0_query_hidden_pipeline.sh
scripts/train_pi0_action_hidden_dreamerv3_wm.sh
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
configs/rynn_backbone_dreamerv3_action_hidden_*.yaml  # current pi0 action-hidden WM
configs/chameleon_latent_action_wm_*.yaml             # Chameleon / LaDiWM-style baseline
configs/dreamer_vla_libero_goal_pi0_action_hidden_head_actor.yaml
configs/eval_libero_vla.yaml                # rollout eval
```

Active configs should target the public workspace API in `src.workspace`:

```text
ActionHiddenWMWorkspace      # current pi0 action-hidden WM
PixelWMWorkspace             # pixel DreamerV3 baseline
TokenWMWorkspace             # token DreamerV3 baseline
ChameleonLatentWMWorkspace   # Chameleon / LaDiWM-style baseline
VLASFTWorkspace              # VLA action-head SFT
JointDreamerVLAWorkspace     # current action-hidden actor route
LiberoEvalWorkspace          # LIBERO rollout eval
```

Each config points directly at a route-specific workspace class.

Physical layout follows the same boundary:

```text
src/workspace/__init__.py        # public workspace exports
src/workspace/base_workspace.py  # shared lifecycle/checkpoint helpers
src/workspace/*_workspace.py     # route-specific workspace implementations
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
