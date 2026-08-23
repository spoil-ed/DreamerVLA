from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.skipif(
    os.environ.get("DVLA_SYSTEM_DOCKER_SMOKE") != "1",
    reason="set DVLA_SYSTEM_DOCKER_SMOKE=1 after building the system images",
)
@pytest.mark.parametrize(
    ("image_env", "default_image", "smoke_command"),
    (
        (
            "MUJOCO_SYSTEM_IMAGE",
            "mujoco-system:cu128-ubuntu2204-v1",
            "/usr/local/bin/smoke-mujoco-system",
        ),
        (
            "SAPIEN_SYSTEM_IMAGE",
            "sapien-system:cu128-ubuntu2204-v1",
            "/usr/local/bin/smoke-sapien-system",
        ),
    ),
)
def test_system_image_gpu_rendering_contract(
    image_env: str,
    default_image: str,
    smoke_command: str,
) -> None:
    image = os.environ.get(image_env, default_image)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--ipc=host",
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video",
            image,
            smoke_command,
        ],
        check=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
        },
    )
