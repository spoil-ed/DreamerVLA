# Simulation system containers

The `mujoco-system` and `sapien-system` images contain native system
dependencies only. They do not install simulator Python packages, datasets,
checkpoints, or simulator assets. The optional `mujoco-vla` image extends
`mujoco-system` with the repository's complete uv-installed LIBERO training
environment. None of the images install Isaac Sim or Isaac Lab.

The two final targets share a CUDA 12.8 / Ubuntu 22.04 development base:

- `mujoco-system:cu128-ubuntu2204-v1`: CUDA, EGL, OpenGL, OSMesa, and media/XML libraries.
- `sapien-system:cu128-ubuntu2204-v1`: CUDA, Vulkan, NVIDIA ICD metadata, and native build tools.
- `mujoco-vla:cu124-h100-v1`: the complete project environment, including
  PyTorch, Ray, LIBERO, robosuite, OpenVLA-OFT, its pinned Transformers fork,
  FlashAttention, and all dependencies installed through uv.

SAPIEN renders headlessly through Vulkan. EGL is the headless GPU backend for
the MuJoCo image. Both targets include NVIDIA GLVND vendor metadata because
older NVIDIA Container Toolkit releases may inject the driver libraries but
omit that metadata.

## Host requirements

- Linux x86_64
- Docker with the Compose plugin
- NVIDIA Container Toolkit
- An NVIDIA driver compatible with CUDA 12.8; R570 or newer is recommended

The NVIDIA runtime belongs on the host. It is not installed inside either
image.

## Network policy

Builds explicitly clear all common HTTP, HTTPS, and SOCKS proxy variables. The
default CUDA image comes from the DaoCloud mirror of NVIDIA's official Docker
Hub image, Ubuntu packages use the Tsinghua mirror, and Miniconda uses the
Nanjing University mirror. Python packages use the same university's PyPI
mirror. The complete environment uses direct `codeload.github.com` commit
archives for the two small uv-installed source packages. It uses `gh-proxy.com`
for Git checkouts and `ghfast.top` for the large FlashAttention wheel because
large GitHub transfers may time out on this host; these are download mirrors,
not configured network proxies. The wheel download uses a persistent BuildKit
cache and supports resuming an interrupted partial file.

To use the upstream registries directly:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  CUDA_BASE_IMAGE=docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 \
  UBUNTU_APT_MIRROR=http://archive.ubuntu.com/ubuntu \
  MINICONDA_BASE_URL=https://repo.anaconda.com/miniconda \
  PYPI_INDEX_URL=https://pypi.org/simple \
  docker compose -f docker/system/compose.yaml build
```

No proxy build arguments are forwarded by the compose file. Do not add
`--build-arg HTTP_PROXY=...` when building these images.

## Build

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  docker compose -f docker/system/compose.yaml build \
    mujoco-system sapien-system
```

Build the complete MuJoCo VLA environment after `mujoco-system` exists locally:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  DVLA_GIT_COMMIT="$(git rev-parse HEAD)" \
  DVLA_BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  docker compose -f docker/system/compose.yaml build mujoco-vla
```

The compose build supplies the repository's existing, pinned `third_party/`
checkouts as a read-only BuildKit context. This avoids downloading the large
LIBERO and robosuite repositories again; the copied checkouts are still
validated and checked out at the revisions required by the install scripts.
Plain `docker build` and CI use the Dockerfile's online fallback stage instead.

The full environment uses the repository's official Python 3.11, PyTorch
2.5.1/cu124, and FlashAttention 2.7.1.post1 pins. Its supported reproduction
profile is H100. The CUDA 12.8 system base supports newer host drivers, but it
does not make the pinned cu124 PyTorch wheel compatible with RTX 5090/sm_120.
Weights, datasets, checkpoints, and logs remain outside the image under the
`/data` mount.

## Smoke tests

The smoke tests validate CUDA visibility, compiler/tool availability, absence
of Isaac packages, real EGL initialization, and real Vulkan device discovery.
They do not import any simulator Python package.

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  docker compose -f docker/system/compose.yaml \
    --profile smoke run --rm mujoco-smoke

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  docker compose -f docker/system/compose.yaml \
    --profile smoke run --rm sapien-smoke
```

After building the full environment, verify its complete uv dependency set,
pinned versions, third-party import paths, CUDA visibility, and EGL rendering:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  docker compose -f docker/system/compose.yaml \
    --profile full-smoke run --rm mujoco-vla-smoke
```

## Interactive shells

```bash
docker compose -f docker/system/compose.yaml run --rm mujoco-system
docker compose -f docker/system/compose.yaml run --rm sapien-system
docker compose -f docker/system/compose.yaml run --rm mujoco-vla
```

The repository is mounted at `/workspace`; `${DVLA_DATA_ROOT:-../../data}` is
mounted at `/data`.
