# syntax=m.daocloud.io/docker.io/docker/dockerfile:1.7

ARG CUDA_BASE_IMAGE=m.daocloud.io/docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG MINICONDA_VERSION=py311_25.1.1-2
ARG MINICONDA_BASE_URL=https://mirror.nju.edu.cn/anaconda/miniconda
ARG MINICONDA_SHA256=d8c1645776c0758214e4191c605abe5878002051316bd423f2b14b22d6cb4251
ARG UV_VERSION=0.8.17
ARG PYPI_INDEX_URL=https://mirror.nju.edu.cn/pypi/web/simple

# Never inherit Docker-client proxy settings. Downloads use direct connections
# or the explicitly selected public mirrors above.
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG ALL_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG all_proxy=

LABEL org.opencontainers.image.title="Embodied simulation system base" \
      org.opencontainers.image.description="CUDA system dependencies shared by MuJoCo and SAPIEN containers"

ENV HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    NO_PROXY=* \
    no_proxy=* \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [[ -n "${UBUNTU_APT_MIRROR}" ]]; then \
      sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        /etc/apt/sources.list; \
    fi \
    && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
           -u http_proxy -u https_proxy -u all_proxy \
       apt-get update \
    && apt-get install -y --no-install-recommends \
      bash bash-completion build-essential ca-certificates cmake curl ffmpeg \
      git git-lfs libglib2.0-0 libjpeg-dev libpng-dev locales ninja-build \
      pkg-config rsync tmux unzip vim wget \
    && rm -rf /var/lib/apt/lists/*

RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
      curl --noproxy '*' --fail --show-error --location \
        "${MINICONDA_BASE_URL}/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
        --output /tmp/miniconda.sh \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check - \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && conda config --system --set auto_update_conda false \
    && python -m pip --isolated install \
      --index-url "${PYPI_INDEX_URL}" "uv==${UV_VERSION}" \
    && conda clean -afy

RUN groupadd --gid 1000 sim \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash sim \
    && mkdir -p /data /workspace \
    && chown -R sim:sim /data /workspace

COPY docker/system/smoke/common.sh /usr/local/bin/smoke-system-common
RUN chmod 0755 /usr/local/bin/smoke-system-common

WORKDIR /workspace
CMD ["/bin/bash"]


FROM base AS mujoco

LABEL org.opencontainers.image.title="MuJoCo system environment" \
      org.opencontainers.image.description="CUDA, EGL, OpenGL, OSMesa, and native libraries for MuJoCo simulators"

RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
      apt-get update \
    && apt-get install -y --no-install-recommends \
      libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
      libavutil-dev libegl-dev libegl1 libexpat1 libexpat1-dev \
      libfontconfig1-dev libgl-dev libgl1 libgles-dev libgles2 libglfw3 \
      libglfw3-dev libglvnd-dev libglvnd0 libhidapi-dev libmagickwand-dev \
      libopengl-dev libopengl0 libosmesa6 libosmesa6-dev libswresample-dev \
      libswscale-dev libx11-6 libxext6 libxml2-dev libxrender1 libxslt1-dev \
      mesa-utils mesa-utils-extra \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' \
      '{' \
      '  "file_format_version": "1.0.0",' \
      '  "ICD": {' \
      '    "library_path": "libEGL_nvidia.so.0"' \
      '  }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY docker/system/smoke/egl_probe.c /tmp/egl_probe.c
RUN cc -O2 -Wall -Wextra /tmp/egl_probe.c -o /usr/local/bin/egl-device-smoke -lEGL \
    && rm /tmp/egl_probe.c

COPY docker/system/smoke/mujoco.sh /usr/local/bin/smoke-mujoco-system
RUN chmod 0755 /usr/local/bin/smoke-mujoco-system

USER sim


FROM base AS sapien

LABEL org.opencontainers.image.title="SAPIEN system environment" \
      org.opencontainers.image.description="CUDA and headless Vulkan native libraries for SAPIEN and PhysX simulators"

RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
      apt-get update \
    && apt-get install -y --no-install-recommends \
      libegl-dev libegl1 libgl-dev libgl1 libglvnd-dev libglvnd0 \
      libopengl-dev libopengl0 libvulkan-dev libvulkan1 libx11-6 libxext6 \
      libxrender1 mesa-vulkan-drivers vulkan-tools \
    && mkdir -p /usr/share/vulkan/icd.d \
    && printf '%s\n' \
      '{' \
      '  "file_format_version": "1.0.0",' \
      '  "ICD": {' \
      '    "library_path": "libGLX_nvidia.so.0",' \
      '    "api_version": "1.3.0"' \
      '  }' \
      '}' > /usr/share/vulkan/icd.d/nvidia_icd.json \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' \
      '{' \
      '  "file_format_version": "1.0.0",' \
      '  "ICD": {' \
      '    "library_path": "libEGL_nvidia.so.0"' \
      '  }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    XDG_RUNTIME_DIR=/tmp/runtime-sim

RUN mkdir -p "${XDG_RUNTIME_DIR}" \
    && chmod 1777 "${XDG_RUNTIME_DIR}"

COPY docker/system/smoke/sapien.sh /usr/local/bin/smoke-sapien-system
RUN chmod 0755 /usr/local/bin/smoke-sapien-system

USER sim
