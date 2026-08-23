# `rlinf-sysm` system-only image

This image contains only the native CUDA, rendering, media, and build
dependencies required by RLinf embodied environments. It deliberately excludes
uv, uv-managed Python installations, virtual environments, RLinf, OpenPI,
LIBERO Python packages, assets, datasets, and checkpoints.

The default CUDA base is fetched directly from DaoCloud's Docker Hub mirror and
Ubuntu packages are fetched directly from Tsinghua's apt mirror. These are
download origins, not HTTP/SOCKS proxies; all proxy variables are cleared for
the build and runtime.

The sibling RLinf checkout remains the source of truth for system packages. Build
from the DreamerVLA repository root with a BuildKit named context:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  docker build \
    --build-arg HTTP_PROXY= \
    --build-arg HTTPS_PROXY= \
    --build-arg ALL_PROXY= \
    --build-arg http_proxy= \
    --build-arg https_proxy= \
    --build-arg all_proxy= \
    --build-arg RLINF_GIT_SHA="$(git -C ../RLinf rev-parse HEAD)" \
    --build-arg SYSTEM_UID="$(id -u)" \
    --build-arg SYSTEM_GID="$(id -g)" \
    --build-context rlinf_source=../RLinf \
    -f docker/rlinf-system/Dockerfile \
    -t rlinf-sysm:cu128-ubuntu2204-v1 \
    .
```

Keep the complete uv/application layer outside the image. A typical job bind
mounts DreamerVLA at `/workspace/DreamerVLA`, the sibling RLinf checkout at
`/workspace/RLinf`, the current checkout's ignored `.rlinf-runtime/` directory
at `/runtime`, and its persistent home at `/home/sim`. The uv executable is
`/runtime/bin/uv`; its cache, managed Python, and virtual environment live below
`/runtime` as well.

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  docker run --rm --gpus all --ipc host --network host \
    --env-file environments/rlinf-libero-pi05/runtime.env \
    -v "$PWD:/workspace/DreamerVLA" \
    -v "$PWD/../RLinf:/workspace/RLinf:ro" \
    -v "$PWD/.rlinf-runtime:/runtime" \
    -v "$PWD/.rlinf-runtime/home:/home/sim" \
    -w /workspace/DreamerVLA \
    rlinf-sysm:cu128-ubuntu2204-v1
```

Run the system-only smoke test with a GPU visible:

```bash
docker run --rm --gpus all \
  --entrypoint /usr/local/bin/smoke-rlinf-system \
  rlinf-sysm:cu128-ubuntu2204-v1
```
