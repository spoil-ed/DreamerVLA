# syntax=m.daocloud.io/docker.io/docker/dockerfile:1.7

ARG MUJOCO_SYSTEM_IMAGE=simat-acr-bj-registry.cn-beijing.cr.aliyuncs.com/cloud-platform/mujoco-system:cu128-ubuntu2204-v1
ARG DEPENDENCY_SOURCE_STAGE=third-party-online
FROM ${MUJOCO_SYSTEM_IMAGE} AS uv-bootstrap

USER root

ARG PYPI_INDEX_URL=https://mirror.nju.edu.cn/pypi/web/simple
ARG GITHUB_MIRROR_PREFIX=https://gh-proxy.com/
ARG FLASH_ATTN_WHEEL_URL=https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
ARG DLIMP_OPENVLA_SRC=https://github.com/moojink/dlimp_openvla/archive/040105d256bd28866cc6620621a3d5f7b6b91b46.tar.gz
ARG TRANSFORMERS_OFT_FORK_SRC=https://github.com/moojink/transformers-openvla-oft/archive/bc339d9ad707454c0c115970db43c260067c61ab.tar.gz

LABEL org.opencontainers.image.title="MuJoCo VLA training environment" \
      org.opencontainers.image.description="Complete uv-installed VLA training environment for LIBERO"

ENV CONDA_ENV_NAME=base \
    DVLA_ROOT=/opt/dreamervla \
    DVLA_DATA_ROOT=/data \
    DVLA_IMAGE_PROFILE=cu124-h100-libero-goal-v1 \
    LIBERO_CONFIG_PATH=/tmp/libero \
    TOKENIZERS_PARALLELISM=false \
    PYTHONFAULTHANDLER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_DEFAULT_INDEX=${PYPI_INDEX_URL} \
    PIP_INDEX_URL=${PYPI_INDEX_URL} \
    GIT_HTTP_VERSION=HTTP/1.1

WORKDIR /opt/dreamervla

# Keep the expensive uv environment independent of ordinary source and
# documentation changes. uv.lock participates in the cache key even though the
# repository's complete runtime contract is split between requirements.txt and
# the pinned third-party install scripts.
COPY pyproject.toml requirements.txt uv.lock /opt/dreamervla/
COPY scripts/install_env.sh /opt/dreamervla/scripts/install_env.sh
COPY scripts/install/ /opt/dreamervla/scripts/install/
COPY configs/scripts/install/ /opt/dreamervla/configs/scripts/install/
COPY dreamervla/__init__.py dreamervla/config_resolvers.py /opt/dreamervla/dreamervla/
COPY dreamervla/launchers/__init__.py dreamervla/launchers/workflow.py dreamervla/launchers/task_cli.py /opt/dreamervla/dreamervla/launchers/
COPY dreamervla/diagnostics/__init__.py dreamervla/diagnostics/verify_install.py /opt/dreamervla/dreamervla/diagnostics/
RUN touch README.md \
    && git config --global http.version HTTP/1.1 \
    && if [[ -n "${GITHUB_MIRROR_PREFIX}" ]]; then \
         git config --global \
           url."${GITHUB_MIRROR_PREFIX}https://github.com/".insteadOf \
           "https://github.com/"; \
       fi

# Normal CI builds populate the pinned third-party repositories from the
# network. Local compose builds select third-party-local and reuse the existing
# read-only repository checkouts supplied as a named BuildKit context.
FROM uv-bootstrap AS third-party-online
ENV USE_PRESEEDED_THIRD_PARTY=0

FROM uv-bootstrap AS third-party-local
COPY --from=third_party_source . /opt/dreamervla/third_party/
ENV USE_PRESEEDED_THIRD_PARTY=1

FROM ${DEPENDENCY_SOURCE_STAGE} AS uv-dependencies

# The install workflow uses uv for PyTorch, the pinned requirements, the editable
# project, Ray/dev groups, all third-party checkouts, the OpenVLA-OFT Transformers
# fork, FlashAttention, and egl_probe. Miniconda only supplies the isolated
# Python 3.11 base interpreter; uv owns the complete Python package set. System
# packages are already present in mujoco-system.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,id=mujoco-vla-third-party,target=/opt/third-party-cache,sharing=locked \
    --mount=type=cache,id=mujoco-vla-wheels,target=/opt/dreamervla-build-data/wheels,sharing=locked \
    mkdir -p /opt/dreamervla-build-data \
    && if [[ "${USE_PRESEEDED_THIRD_PARTY}" != "1" ]]; then \
         ln -s /opt/third-party-cache /opt/dreamervla/third_party; \
       fi \
    && DVLA_DATA_ROOT=/opt/dreamervla-build-data \
       bash scripts/install_env.sh \
         env.CONDA_ENV_NAME=base \
         env.INSTALL_APT_TOOLS=false \
         env.INSTALL_OPENVLA_OFT_THIRD_PARTY=true \
         env.FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL}" \
         env.DLIMP_OPENVLA_SRC="${DLIMP_OPENVLA_SRC}" \
         env.TRANSFORMERS_OFT_FORK_SRC="${TRANSFORMERS_OFT_FORK_SRC}" \
    && if [[ "${USE_PRESEEDED_THIRD_PARTY}" != "1" ]]; then \
         unlink /opt/dreamervla/third_party \
         && cp -a /opt/third-party-cache /opt/dreamervla/third_party; \
       fi \
    && eval "$(conda shell.bash hook)" \
    && conda activate "${CONDA_ENV_NAME}" \
    && uv pip install --no-build-isolation \
         -e /opt/dreamervla/third_party/robosuite \
         -e /opt/dreamervla/third_party/robosuite-task-zoo \
         -e /opt/dreamervla/third_party/robomimic \
         -e /opt/dreamervla/third_party/mimicgen \
    && uv pip check \
    && conda clean -afy \
    && rm -rf /opt/dreamervla-build-data/install_state

FROM uv-dependencies AS final

ENV PYTHONPATH=/opt/dreamervla

# The complete application is a small, late layer. Source changes do not
# invalidate the complete uv dependency layer above.
COPY . /opt/dreamervla
COPY docker/system/smoke/vla.sh /usr/local/bin/smoke-mujoco-vla
RUN chmod 0755 /usr/local/bin/smoke-mujoco-vla

ARG DVLA_GIT_COMMIT=unknown
ARG DVLA_IMAGE_VERSION=dev
ARG DVLA_BUILD_TIME=unknown
LABEL org.opencontainers.image.revision="${DVLA_GIT_COMMIT}" \
      org.opencontainers.image.version="${DVLA_IMAGE_VERSION}" \
      org.opencontainers.image.created="${DVLA_BUILD_TIME}"

RUN python -c 'import json, pathlib; pathlib.Path(".vla-image.json").write_text(json.dumps({"schema_version": 1, "commit": "'"${DVLA_GIT_COMMIT}"'", "version": "'"${DVLA_IMAGE_VERSION}"'", "profile": "cu124-h100-libero-goal-v1", "built_at": "'"${DVLA_BUILD_TIME}"'"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")' \
    && mkdir -p /data \
    && python -m dreamervla.diagnostics.verify_install \
    && uv pip check

VOLUME ["/data"]
USER sim
CMD ["/bin/bash"]
