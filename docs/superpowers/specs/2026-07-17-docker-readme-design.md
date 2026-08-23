# Docker Reproduction README Design

## Goal

Replace the current project README with a minimal reproduction guide. Docker remains
the default path, followed by a native Conda path for users who do not want Docker.
A new user should be able to reproduce the published DreamerVLA workflow without
first understanding Hydra, Ray, or the repository layout.

## Language layout

- `README.md` is the default English page and links to `README.zh-CN.md` at the top.
- `README.zh-CN.md` is the matching Chinese page and links back to `README.md`.
- Both files use the same section order, commands, paths, requirements, and claims.

## Content

Each README contains only:

1. A one-paragraph description of DreamerVLA.
2. Host requirements: Docker, NVIDIA Container Toolkit, 8 H100 80 GB GPUs,
   300 GiB free disk space, and network access for downloads.
3. Three copyable steps:
   - pull `spoil/dreamervla:cu124-h100-v1`;
   - run `scripts/reproduce/01_prepare_assets.sh` with a persistent data mount;
   - run `scripts/reproduce/02_train_dreamer.sh` with the same mount.
4. The fixed training order: WM for 30 epochs, classifier for 8 epochs, then
   Dreamer for 20,000 steps with WM and classifier frozen.
5. Output locations and automatic resume behavior.
6. A short explanation that source and pinned third-party dependencies are inside
   the image, while weights, datasets, checkpoints, and logs live in the mounted
   host directory.
7. Minimal commands for viewing progress and stopping a container.
8. A native reproduction section that uses the same public asset and training
   launchers after installing the repository environment.

Development internals, configuration-field catalogs, repository layout, W&B details,
and alternative experimental routes are excluded from the main README. The native
installation section contains only the commands required for reproduction. Existing
detailed documentation remains available under `docs/`.

## Command contract

The Docker commands use:

- `--gpus all`;
- `--ipc=host`;
- `--network=host`;
- `--shm-size=100g`;
- `--ulimit memlock=-1`;
- the same `dreamervla-data:/data` bind mount for both stages.

The guide does not present the single-GPU smoke route as the supported full
reproduction profile. The published full profile remains 8 H100 GPUs.

## Native reproduction

The native path reuses the same reproduction entrypoints as Docker so asset checks,
training budgets, checkpoint selection, and resume behavior cannot drift between
the two methods.

The README presents these steps:

1. Install Ubuntu 22.04 host prerequisites, Miniconda or Anaconda, and an NVIDIA
   driver compatible with the CUDA 12.4 PyTorch wheels.
2. Clone the repository and set `DVLA_ROOT` to the checkout and `DVLA_DATA_ROOT` to
   a persistent data directory with at least 300 GiB free space.
3. Run `bash scripts/install_env.sh`, activate the `dreamervla` Conda environment,
   and run `bash scripts/install/60_verify.sh`.
4. Run `bash scripts/reproduce/01_prepare_assets.sh` to download, verify, and
   preprocess the pinned OpenVLA-OFT weights and LIBERO Goal dataset.
5. Run `bash scripts/reproduce/02_train_dreamer.sh` to execute WM 30 epochs,
   classifier 8 epochs, and frozen-WM/classifier Dreamer for 20,000 steps.
6. Re-run the same command after interruption to resume from the owning run root's
   `checkpoints/latest.ckpt`.

The README does not expand the underlying download and Hydra training commands;
that would duplicate the public launchers and create a second reproduction contract.

## Runtime verification scope

Implementation verification must prove that the native environment check succeeds,
the asset workflow can resolve and start its pinned download/preprocess commands,
the complete training workflow resolves the expected budgets and checkpoint paths,
and the resume path recognizes existing run roots/checkpoints. Verification does not
wait for the complete 20,000-step production experiment to finish.

## Verification

After editing, verify that:

- language links are reciprocal;
- both files contain the same Docker image tag and commands;
- WM is 30 epochs, classifier is 8 epochs, and Dreamer is 20,000 steps;
- source/third-party inclusion and external weight/data storage are stated clearly;
- the native path uses the same two reproduction scripts as Docker;
- native install verification, asset startup, training startup, and resume behavior
  are exercised without claiming that the complete production run finished;
- repository documentation tests pass.
