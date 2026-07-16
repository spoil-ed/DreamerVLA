from __future__ import annotations

import json
import os
import subprocess

import pytest


@pytest.mark.skipif(
    os.environ.get("DVLA_DOCKER_SMOKE") != "1",
    reason="set DVLA_DOCKER_SMOKE=1 after building the release image",
)
def test_release_image_contains_source_third_party_and_dry_run_contract() -> None:
    image = os.environ.get("DVLA_DOCKER_IMAGE", "spoil/dreamervla:cu124-h100-v1")
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--network=host",
        "--shm-size=100g",
        image,
        "bash",
        "-lc",
        " && ".join(
            (
                "python -m dreamervla.diagnostics.verify_install",
                "test -d third_party/LIBERO/.git",
                "test -d third_party/openvla-oft/.git",
                "python -m dreamervla.launchers.reproduce "
                "--config-name reproduce/train_dreamer dry_run=true",
                "cat .dreamervla-image.json",
            )
        ),
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "training.warmup_replay_epochs=30" in result.stdout
    assert "training.num_epochs=8" in result.stdout
    assert "manual_cotrain.global_steps=20000" in result.stdout
    metadata_line = result.stdout[result.stdout.rfind("{") :]
    metadata = json.loads(metadata_line)
    assert metadata["profile"] == "cu124-h100-libero-goal-v1"
