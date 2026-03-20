# Dreamer-VLA

Research scaffold for combining:

- `RynnVLA-001` as the VLA encoder / action prior
- a bottleneck module for extracting compact physical state
- `dreamerv3-main` as the world model backbone
- a controller or planner that selects actions from imagined rollouts

## Current Status

This repository is still a scaffold. The main goal of the current layout is to
separate configuration, architecture notes, training entrypoints, and model
submodules before the full implementation is filled in.

The codebase currently references two external roots in
[`configs/base.yaml`](configs/base.yaml):

- `RynnVLA-001`
- `dreamerv3-main`

## Layout

```text
Dreamer-VLA/
├── configs/
│   ├── base.yaml
│   └── ppo_trainer.yaml
├── docs/
│   └── architecture.md
├── scripts/
├── src/
│   ├── models/
│   │   ├── actor_critic/
│   │   ├── bottleneck/
│   │   ├── vla_encoder/
│   │   └── world_model/
│   ├── single_controller/
│   ├── trainer/
│   │   ├── main.py
│   │   └── main_ray.py
│   └── utils/
└── tests/
    └── test_smoke.py
```

## Directory Roles

- `configs/`: experiment, model, trainer, and external dependency paths
- `docs/`: high-level design notes for the Dreamer-VLA pipeline
- `scripts/`: helper scripts for training, evaluation, or data preparation
- `src/models/`: model-side components, split by responsibility
- `src/single_controller/`: controller or planner logic that uses imagined rollouts
- `src/trainer/`: local and Ray-based training entrypoints
- `src/utils/`: shared utility code
- `tests/`: smoke tests for the intended end-to-end pipeline

## Planned Pipeline

1. Encode `(image, proprio, text)` into a latent state.
2. Compress that latent into a bottleneck representation.
3. Run imagined rollouts with the world model.
4. Score candidate behaviors with actor / critic style modules.
5. Select or refine the final action with the controller.

## Notes

- Several directories are placeholders and do not yet contain the full
  implementation.
- The README reflects the current on-disk structure under `src/`, rather than a
  future package layout.
