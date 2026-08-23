# Docker Layer Caching Design

## Goal

Keep the published DreamerVLA image self-contained while preventing README and
ordinary source changes from invalidating the expensive Conda, CUDA, Python, and
third-party installation layer.

## Current problem

`docker/Dockerfile` currently copies the entire repository before running
`scripts/install_env.sh`. Docker therefore includes the whole build-context checksum
in the dependency-layer cache key. Any tracked source or documentation edit forces a
clean environment installation and creates a new multi-gigabyte layer for Docker Hub.

## Selected architecture

Use one Dockerfile with three ordered boundaries:

1. **System layer:** install Miniconda and the operating-system bootstrap packages.
2. **Dependency layer:** copy only dependency declarations, install configuration,
   install shell scripts, and the minimum Python workflow modules required to run
   `scripts/install_env.sh`. Create a minimal temporary `README.md` because the package
   metadata references it. Install PyTorch, project requirements, pinned
   OpenVLA-OFT, and all required `third_party` repositories.
3. **Source layer:** copy the full repository only after dependency installation,
   write `.dreamervla-image.json`, and run the final import/environment check.

The final image layout and public commands do not change. `/opt/dreamervla` still
contains the complete source and pinned `third_party`; `/data` remains the external
runtime volume.

## Dependency cache inputs

The expensive layer may depend on:

- `pyproject.toml` and `requirements.txt`;
- `scripts/install_env.sh` and `scripts/install/**`;
- `configs/scripts/install/**`;
- `dreamervla/__init__.py`, `dreamervla/config_resolvers.py`,
  `dreamervla/launchers/__init__.py`, and `dreamervla/launchers/workflow.py`;
- Docker build arguments and the CUDA base image.

It must not depend on the public README files, tests, documentation, training code,
or other ordinary repository source. Changes to an explicit dependency input
correctly invalidate the environment layer.

## Build and release behavior

The Docker Hub tags remain:

- `spoil/dreamervla:cu124-h100-v1`;
- `spoil/dreamervla:v1`;
- `spoil/dreamervla:sha-` followed by the first 12 characters of the commit.

The GitHub workflow will retain BuildKit cache metadata so a new runner can reuse the
dependency layer when its inputs are unchanged. The final OCI labels and image
metadata continue to identify the exact source commit and build time.

## Verification

Automated tests will assert that:

- the dependency bootstrap inputs appear before `scripts/install_env.sh`;
- the full `COPY . /opt/dreamervla` occurs after dependency installation;
- public README files are not dependency-layer inputs;
- final metadata generation and install verification occur after the full source
  copy;
- the publish workflow configures BuildKit cache import and export.

An actual Docker build will then be run twice. Between builds, a documentation-only
context change will be introduced temporarily. The second build must report the
environment installation layer as cached while rebuilding only the final source and
verification layers. Finally, the optimized image will be checked and pushed under
all three release tags.

## Non-goals

- Splitting dependencies into a separately published base image.
- Embedding model weights or datasets in the image.
- Changing package versions, third-party revisions, training commands, or runtime
  behavior.
