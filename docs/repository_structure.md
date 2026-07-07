# Repository Structure

This is the current source-of-truth map for Dreamer-VLA after the package
move from `src/` to `dreamervla/`.

## Top Level

```text
DreamerVLA/
├── dreamervla/          # Python package imported as dreamervla
│   └── models/embodiment # Vendored embodiment model code used at runtime
├── configs/              # Hydra train/experiment/module configs and LIBERO tasks
├── scripts/              # Shell launchers for install, data prep, train, eval
├── tests/                # Unit and e2e tests
├── docs/                 # Docs index, architecture, reference, tutorials, reports, papers
├── data/                 # Runtime datasets, checkpoints, outputs
├── third_party/          # Vendored/local upstream checkouts and wheels
│   ├── LIBERO/           # Local LIBERO checkout
│   └── openvla-oft/      # OpenVLA-OFT upstream checkout used for setup/fallback
├── pyproject.toml        # Editable install metadata
└── requirements.txt      # Runtime dependencies
```

`data/` and `third_party/` are runtime inputs. Embodiment model code that
DreamerVLA imports directly lives under `dreamervla/models/embodiment`.
Keep generated
artifacts out of source commits unless a small summary belongs in `docs/`.
Dot-prefixed local tool folders are ignored by this main structure map.

## Package Layout

```text
dreamervla/
├── algorithms/           # PPO/GRPO/LUMOS, actor modules, critics/classifiers
│   ├── actor/            # BaseActor, VLAPolicy, OpenVLA/RynnVLA actor adapters
│   ├── critic/           # Critic and success-classifier/verifier modules
│   ├── reward/           # Algorithmic reward model protocols and registries
│   └── ...
├── train.py              # Canonical Hydra train/eval entrypoint
├── dataset/              # Offline datasets and online rollout dumpers
├── diagnostics/          # Diagnostics, eval CLIs, smoke checks
├── envs/                 # LIBERO three-file env surface plus world-model env
├── models/               # Embodiment models only
│   └── embodiment/       # VLA/encoder code plus retained world-model architectures
│       ├── openvla_oft/  # Vendored OpenVLA-OFT model/runtime components
│       ├── chameleon_model/ # Chameleon/RynnVLA components
│       └── world_model/  # BaseWorldModel and retained WM architectures
├── preprocess/           # Dataset preprocessing, xllmx helpers, hidden extraction
├── scheduler/            # Optional Ray backend scheduling primitives
├── workers/              # Optional Ray backend workers
├── hybrid_engines/       # Optional Ray backend object-store / weight-sync helpers
├── legacy/               # Isolated non-mainline utilities for old artifacts
├── utils/                # Checkpoints, logging, optim, EMA, visualization
└── runners/              # Public route runners, distributed and online-training helpers
```

There is no active `src/` or `workspace/` tree. The training unit is a runner.
Ray-specific scheduler / worker modules are optional backend internals and
should not define a separate model, dataset, checkpoint, or logging contract.

## Execution Path

```text
scripts/*.sh
  -> python -m dreamervla.train --config-name <route>
  -> configs/<route>.yaml with _target_: dreamervla.runners.<Runner>
  -> runner.setup() -> runner.execute() -> runner.teardown()
```

Public runner classes are exported from `dreamervla.runners`. Route
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
  world_model_step
  world_model_chunk
  oft_world_model_chunk
  latent_classifier_libero_goal_chunk
  oft_latent_classifier_chunk

DreamerVLA:
  dreamervla_rynn_wm_actor_critic
  dreamervla_rynn_wm_lumos
  dreamervla_oft_wm_lumos

Evaluation:
  eval_libero_vla
```

Release launchers stay in `scripts/`; route experiments should graduate to a
top-level config only when they have a runner, defaults, and tests.

## Interface Boundaries

Runners own orchestration: datasets, embodiment encoders, world models, actors,
critics/classifiers, optimizers, logging, and checkpoints. Shared lifecycle and checkpoint
plumbing belongs in `dreamervla/runners/base_runner.py`.

Models stay behind focused public interfaces:

- VLA/encoder code is one embodiment boundary. Encoders inherit `BaseEncoder`
  and use `models/embodiment/protocol.py` helpers for structured VLA input
  batches.
- World models inherit `BaseWorldModel`; canonical implementations live under
  `dreamervla/models/embodiment/world_model/`.
- Actors inherit `BaseActor`; canonical implementations live in
  `dreamervla/algorithms/actor/`.
- Critic and classifier code is one verifier/value boundary. Canonical
  implementations live in `dreamervla/algorithms/critic/`.
- Datasets inherit `BaseDataset` and expose `data_spec` plus
  `get_normalizer()`.

Do not add package-level compatibility shims for moved modules. Update imports
and Hydra targets to the canonical subpackage path.
