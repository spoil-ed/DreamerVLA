# Repository Structure

This is the current source-of-truth map for Dreamer-VLA after the package
move from `src/` to `dreamervla/`.

## Top Level

```text
DreamerVLA/
├── dreamervla/          # Python package imported as dreamervla
│   └── models/embodiment # Vendored embodiment model code used at runtime
├── configs/              # Hydra train/experiment/module configs and LIBERO tasks
│   └── environments/     # Isolated Python dependency profiles and container variables
├── scripts/              # Shell launchers for install, data prep, train, eval
├── tests/                # Unit and e2e tests
├── docs/                 # Usage, architecture, references, tutorials, papers, licenses
│   ├── architecture/     # Current contracts and original manual notes
│   ├── reference/        # Model, dataset, metrics, and classifier sampling references
│   └── licenses/         # Retained upstream license notices
├── data/                 # Runtime datasets, checkpoints, outputs
├── third_party/          # Ignored upstream runtime dependencies
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

Environment profiles in `configs/environments/` describe Python dependencies and
container variables; Hydra training configuration stays in its existing groups.
The seven retained `third_party/` checkouts are LIBERO, OpenVLA-OFT, egl_probe,
mimicgen, robomimic, robosuite, and robosuite-task-zoo. Install scripts and package
sources still use them, so keep their upstream code read-only and in place.

## Package Layout

```text
dreamervla/
├── algorithms/           # PPO/GRPO/LUMOS, actor modules, critics/classifiers
│   ├── actor/            # BaseActor, VLAPolicy, and action adapters
│   ├── critic/           # Critic and success-classifier/verifier modules
│   ├── reward/           # Algorithmic reward model protocols and registries
│   └── ...
├── train.py              # Canonical Hydra train/eval entrypoint
├── dataset/              # Local data readers and LIBERO adapters
│   ├── base/             # Base, HDF5, LeRobot v3, mixture, and latent-token loaders
│   ├── libero.py         # LIBERO feature mapping and loader factories
│   ├── classifier_dataset.py # Classifier windows, labels, and balancing
│   └── storage/          # Rollout HDF5 writer and collection manifest
├── diagnostics/          # Executable install, eval, smoke, and measurement CLIs
│   ├── checks/           # Installation, data, runtime, and training-signal checks
│   ├── evaluation/       # Policy/WM evaluation and collected-rollout inspection
│   ├── benchmarks/       # Worker/render performance and single-trajectory overfit
│   └── fixtures/         # Importable synthetic models, envs, replay, and workers
├── envs/                 # LIBERO three-file env surface plus world-model env
├── models/               # Embodiment models only
│   └── embodiment/       # VLA/encoder code plus retained world-model architectures
│       ├── openvla_oft/  # Vendored OpenVLA-OFT model/runtime components
│       ├── chameleon_model/ # Chameleon model components
│       └── world_model/  # BaseWorldModel and retained WM architectures
├── preprocess/           # Canonical reward and OpenVLA hidden-token preprocessing
├── runtime/              # Shared runner metrics, warmup, collection, and eval support
│   ├── rollout/          # Collection, action chunks, and observation extraction
│   ├── replay/           # Online replay, offline seeding, and trajectory caching
│   ├── training/         # WM construction/training bases and classifier updates
│   ├── evaluation/       # VLA/WM/classifier evaluation and metric aggregation
│   ├── envs/             # Subprocess envs, rendering, and episode-end semantics
│   └── common/           # Observation metadata and reproduction workflow state
├── scheduler/            # Ray backend scheduling primitives
├── workers/              # Ray backend workers
├── hybrid_engines/       # Ray backend object-store / weight-sync helpers
├── utils/                # Checkpoints, logging, optim, EMA, visualization
│   ├── checkpoint/       # Persistence, compatibility, and run artifact discovery
│   ├── config/           # Source/data paths and Hydra script helpers
│   ├── logging/          # Backends, console, progress, metrics, and timing
│   ├── training/         # Tensor ops, optimizers, averaging, RNG, and distributed helpers
│   ├── integrations/     # Optional OpenPI/OpenVLA import setup
│   └── visualization/    # Image-token decoding and WM visualization
└── runners/              # Public route runners, distributed and online-training helpers
```

There is no active `src/` or `workspace/` tree. The training unit is a runner.
Ray-specific scheduler / worker modules are mainline backend internals and
should not define a separate model, dataset, checkpoint, or logging contract.

`runtime/` owns workflow and environment behavior; `utils/` provides shared
utilities. Render-device parsing and EGL setup share
`runtime/envs/render_device.py`. Generic distributed helpers live in
`utils/training/distributed.py`; EMA and Polyak updates share `training/ema.py`.
`utils/logging/metrics.py` contains resource measurements and success-rate tracking.
Run roots, resume checkpoints, and persisted Hydra configs are discovered through
`utils/checkpoint/run_artifacts.py`.

Model package exports load on demand, so using the registry or a world model
does not import every VLA family. The low-level `get_model` registry retains
the OpenVLA-OFT loader; other embodiment models use their Hydra `_target_` paths.
Synthetic diagnostic components live in `diagnostics/fixtures/`, which remains
part of the installed package so Ray subprocesses can import them.

The dataset base modules are `base_dataloader.py`, `hdf5_dataloader.py`,
`lerobot_v3_dataloader.py`, `multi_dataloader.py`, and
`latent_token_dataloader.py`. `LeRobotV3DataLoader` reads local Parquet and camera
data; `LiberoDataset` maps their native features. OpenPI and OpenVLA tokenization,
normalization, and model batch transforms live in each model's `sft_data.py`
under `models/embodiment/`.

`MultiDataset` concatenates compatible sources. Its optional
`DistributedMixtureSampler` samples sources by configured weights with
replacement, then samples uniformly within each source. `num_samples` describes
the global epoch length and must divide evenly across replicas; `set_epoch`
controls reproducible sampling.

## Execution Path

```text
scripts/*.sh
  -> python -m dreamervla.launchers.train --config <experiment>
  -> configs/train.yaml + configs/experiment/<experiment>.yaml
  -> component groups (classifier/worldmodel/dreamervla/task)
  -> runner.setup() -> runner.execute() -> runner.teardown()
```

Public runner classes are exported from `dreamervla.runners`. Route
configs should target those public names rather than implementation classes.

## Release Routes

```text
Collection:
  collect_rollouts

Cotrain:
  openvla_onetraj_libero_cotrain
  openvla_libero

Eval:
  eval_cotrain
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
