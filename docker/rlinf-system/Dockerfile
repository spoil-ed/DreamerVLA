ARG CUDA_SYSTEM_IMAGE=m.daocloud.io/docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_SYSTEM_IMAGE}

ARG RLINF_GIT_SHA=unknown
ARG UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG SYSTEM_USER=sim
ARG SYSTEM_UID=1000
ARG SYSTEM_GID=1000

LABEL org.opencontainers.image.title="rlinf-sysm" \
      org.opencontainers.image.description="Native CUDA, EGL, OSMesa, media, and build dependencies for externally mounted RLinf uv environments" \
      org.opencontainers.image.revision="${RLINF_GIT_SHA}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Downloads must go directly to their upstream endpoints. In particular, do not
# inherit the workstation's loopback VPN proxy into either image builds or jobs.
ENV HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    NO_PROXY="*" \
    no_proxy="*" \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# rlinf_source is a BuildKit named context pointing at the sibling RLinf
# checkout. Its system-dependency script remains the source of truth; the image
# deliberately does not copy RLinf itself or duplicate the apt package list.
COPY --from=rlinf_source requirements/sys_deps.sh /tmp/rlinf-sys-deps.sh
RUN sed -i \
      -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
      -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
      /etc/apt/sources.list \
    && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
      bash /tmp/rlinf-sys-deps.sh nvidia \
    && rm /tmp/rlinf-sys-deps.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RLinf's system script installs the FFmpeg development libraries used by the
# Python stack but not the CLI used for rollout video encoding and diagnostics.
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
      apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' \
      '{' \
      '  "file_format_version": "1.0.0",' \
      '  "ICD": {' \
      '    "library_path": "libEGL_nvidia.so.0"' \
      '  }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY docker/rlinf-system/egl_probe.c /tmp/egl_probe.c
RUN cc -O2 -Wall -Wextra /tmp/egl_probe.c -o /usr/local/bin/egl-device-smoke -lEGL \
    && rm /tmp/egl_probe.c

COPY docker/rlinf-system/smoke.sh /usr/local/bin/smoke-rlinf-system
RUN chmod 0755 /usr/local/bin/smoke-rlinf-system \
    && groupadd --gid "${SYSTEM_GID}" "${SYSTEM_USER}" \
    && useradd --uid "${SYSTEM_UID}" --gid "${SYSTEM_GID}" \
      --create-home --shell /bin/bash "${SYSTEM_USER}" \
    && mkdir -p /runtime /workspace \
    && chown "${SYSTEM_UID}:${SYSTEM_GID}" /runtime /workspace

VOLUME ["/runtime"]
WORKDIR /workspace
USER ${SYSTEM_USER}
ENV HOME=/home/${SYSTEM_USER}
CMD ["/bin/bash"]
