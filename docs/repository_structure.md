# Repository Structure

This is the current source-of-truth map for Dreamer-VLA after the package
move from `src/` to `dreamer_vla/`.

## Top Level

```text
DreamerVLA/
├── dreamer_vla/          # Python package imported as dreamer_vla
├── configs/              # Hydra train/experiment/module configs and LIBERO tasks
├── scripts/              # Shell launchers for install, data prep, train, eval
├── tests/                # Unit and e2e tests
├── docs/                 # Repository, install, data-layout, and paper notes
├── data/                 # Runtime datasets, checkpoints, outputs
├── third_party/          # Vendored/local upstream checkouts and wheels
│   ├── LIBERO/           # Local LIBERO checkout
│   ├── openvla-oft/      # Official OpenVLA-OFT checkout for eval
│   └── openvla-oft-lightweight/
│                         # Lightweight OpenVLA-OFT compatibility tree
├── pyproject.toml        # Editable install metadata
└── requirements.txt      # Runtime dependencies
```

`data/` and `third_party/` are runtime inputs. Keep generated
artifacts out of source commits unless a small summary belongs in `docs/`.
Dot-prefixed local tool folders are ignored by this main structure map.

## Package Layout

```text
dreamer_vla/
├── algorithms/           # PPO, GRPO, DINO-WMPO, TD-MPC, actor-critic steps
├── train.py              # Canonical Hydra train/eval entrypoint
├── dataset/              # Offline datasets and online rollout dumpers
├── diagnostics/          # Diagnostics, eval CLIs, smoke checks
├── envs/                 # LIBERO sim and online env wrappers
├── models/               # Encoders, actors, critics, rewards, world models
│   ├── actor/            # BaseActor, VLAPolicy, RynnVLAActionHiddenActor, VLAActionHeadActor
│   ├── critic/           # Critic modules
│   ├── encoder/          # BaseEncoder plus encoder input protocol helpers
│   ├── reward/           # Latent success classifier
│   └── world_model/      # BaseWorldModel and retained WM architectures
├── preprocess/           # Dataset preprocessing, xllmx helpers, hidden extraction
├── legacy/               # Isolated non-mainline utilities for old artifacts
├── utils/                # Checkpoints, logging, optim, EMA, visualization
└── runners/              # Public route runners, distributed and online-training helpers
```

There is no active `src/`, `workspace/`, or Ray-style worker tree. The training
unit is a runner.

## Execution Path

```text
scripts/*.sh
  -> python -m dreamer_vla.train --config-name <route>
  -> configs/<route>.yaml with _target_: dreamer_vla.runners.<Runner>
  -> runner.setup() -> runner.execute() -> runner.teardown()
```

Public runner classes are exported from `dreamer_vla.runners`. Route
configs should target those public names rather than implementation classes.

## Active Routes

```text
VLA SFT:
  vla_rynnvla_action_head
  vla_sft_one_trajectory
  openvla_oft_hdf5
  openvla_oft_hdf5_one_trajectory
  openvla_oft_hdf5_one_trajectory_l1

World model / classifier:
  world_model_dinowm_step
  world_model_dinowm_chunk
  oft_world_model_dinowm_chunk
  latent_classifier_libero_goal_chunk
  oft_latent_classifier_chunk

DreamerVLA:
  dreamervla_rynn_dino_wm_actor_critic
  dreamervla_rynn_dino_wm_wmpo_outcome
  dreamervla_oft_dino_wm_wmpo_outcome

Evaluation:
  eval_libero_vla
```

Release launchers stay in `scripts/`; route experiments should graduate to a
top-level config only when they have a runner, defaults, and tests.

## Interface Boundaries

Runners own orchestration: datasets, encoders, world models, actors,
critics, optimizers, logging, and checkpoints. Shared lifecycle and checkpoint
plumbing belongs in `dreamer_vla/runners/base_runner.py`.

Models stay behind focused public interfaces:

- Encoders inherit `BaseEncoder` and use `encoder/protocol.py` helpers for
  structured VLA input batches.
- Actors inherit `BaseActor`; canonical implementations live in
  `dreamer_vla/models/actor/`.
- World models inherit `BaseWorldModel`; Dreamer-style actor adapters live in
  `base_world_model.py`.
- Datasets inherit `BaseDataset` and expose `data_spec` plus
  `get_normalizer()`.

Do not add package-level compatibility shims for moved modules. Update imports
and Hydra targets to the canonical subpackage path.
