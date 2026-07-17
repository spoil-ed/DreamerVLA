# Docker Reproduction README Design

## Goal

Replace the current project README with a minimal Docker-first reproduction guide.
A new user should be able to reproduce the published DreamerVLA workflow by copying
three commands without first understanding Hydra, Ray, or the repository layout.

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

Development internals, local installation, configuration-field catalogs, repository
layout, W&B details, and alternative experimental routes are excluded from the main
README. Existing detailed documentation remains available under `docs/`.

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

## Verification

After editing, verify that:

- language links are reciprocal;
- both files contain the same Docker image tag and commands;
- WM is 30 epochs, classifier is 8 epochs, and Dreamer is 20,000 steps;
- source/third-party inclusion and external weight/data storage are stated clearly;
- repository documentation tests pass.
