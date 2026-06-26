from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_world_model_env_ray_smoke(tmp_path: Path) -> None:
    if os.environ.get("DVLA_WORLD_MODEL_ENV_SMOKE") != "1":
        pytest.skip("set DVLA_WORLD_MODEL_ENV_SMOKE=1 to run the Ray WM env smoke")

    repo = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "dvla_world_model_env_smoke"
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONPATH"] = str(repo)
    cmd = [
        sys.executable,
        "-m",
        "dreamervla.train",
        "experiment=online_cotrain_ray_world_model_env_tiny",
        "logger=tensorboard",
        f"training.out_dir={out_dir}",
        "rollout.steps=9",
    ]

    result = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0, output
    assert (out_dir / "resolved_config.yaml").is_file()
    assert (out_dir / "run_manifest.json").is_file()
    resolved = (out_dir / "resolved_config.yaml").read_text()
    assert "emit_hidden_sidecar: false" in resolved
    for metric in (
        "sync/policy_version=",
        "sync/wm_version=",
        "sync/classifier_version=",
        "rollout/steps=9",
    ):
        assert metric in output
