# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MINICONDA_VERSION=py311_25.1.1-2
ARG MINICONDA_SHA256=d8c1645776c0758214e4191c605abe5878002051316bd423f2b14b22d6cb4251

LABEL org.opencontainers.image.title="DreamerVLA" \
      org.opencontainers.image.description="Reproducible DreamerVLA training environment for LIBERO" \
      org.opencontainers.image.source="https://github.com/spoil-ed/DreamerVLA"

ENV CONDA_DIR=/opt/conda \
    CONDA_ENV_NAME=dreamervla \
    DVLA_ROOT=/opt/dreamervla \
    DVLA_DATA_ROOT=/data \
    DVLA_IMAGE_PROFILE=cu124-h100-libero-goal-v1 \
    LIBERO_CONFIG_PATH=/tmp/dreamervla-libero \
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONFAULTHANDLER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl bzip2 bash git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL \
      "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
      -o /tmp/miniconda.sh \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum -c - \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false \
    && "${CONDA_DIR}/bin/conda" clean -afy

ENV PATH=/opt/conda/bin:${PATH}

WORKDIR /opt/dreamervla

# Keep the expensive environment layer independent of ordinary source and
# documentation changes. These are the complete inputs required to run the Hydra
# install workflow and build the editable package metadata.
COPY pyproject.toml requirements.txt /opt/dreamervla/
COPY scripts/install_env.sh /opt/dreamervla/scripts/install_env.sh
COPY scripts/install/ /opt/dreamervla/scripts/install/
COPY configs/scripts/install/ /opt/dreamervla/configs/scripts/install/
COPY dreamervla/__init__.py dreamervla/config_resolvers.py /opt/dreamervla/dreamervla/
COPY dreamervla/launchers/__init__.py dreamervla/launchers/workflow.py /opt/dreamervla/dreamervla/launchers/
COPY dreamervla/diagnostics/__init__.py dreamervla/diagnostics/verify_install.py /opt/dreamervla/dreamervla/diagnostics/
RUN touch README.md

# The repository install workflow pins and installs every required third-party tree:
# LIBERO, robosuite, robosuite-task-zoo, robomimic, mimicgen, OpenVLA-OFT, its
# Transformers fork, and egl_probe. They remain under /opt/dreamervla/third_party.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /opt/dreamervla-build-data \
    && DVLA_DATA_ROOT=/opt/dreamervla-build-data \
       INSTALL_OPENVLA_OFT_THIRD_PARTY=true \
       bash scripts/install_env.sh \
    && /opt/conda/bin/conda clean -afy \
    && rm -rf /opt/dreamervla-build-data

ENV PATH=/opt/conda/envs/dreamervla/bin:/opt/conda/bin:${PATH} \
    PYTHONPATH=/opt/dreamervla

# The complete application is a small, late layer. Changes outside the explicit
# dependency inputs above do not invalidate the installed environment.
COPY . /opt/dreamervla

ARG DVLA_GIT_COMMIT=unknown
ARG DVLA_IMAGE_VERSION=dev
ARG DVLA_BUILD_TIME=unknown
LABEL org.opencontainers.image.revision="${DVLA_GIT_COMMIT}" \
      org.opencontainers.image.version="${DVLA_IMAGE_VERSION}" \
      org.opencontainers.image.created="${DVLA_BUILD_TIME}"

RUN python -c 'import json, pathlib; pathlib.Path(".dreamervla-image.json").write_text(json.dumps({"schema_version": 1, "source": "https://github.com/spoil-ed/DreamerVLA", "commit": "'"${DVLA_GIT_COMMIT}"'", "version": "'"${DVLA_IMAGE_VERSION}"'", "profile": "cu124-h100-libero-goal-v1", "built_at": "'"${DVLA_BUILD_TIME}"'"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")' \
    && mkdir -p /data \
    && python -m dreamervla.diagnostics.verify_install

VOLUME ["/data"]
CMD ["/bin/bash"]
