from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_manual_cotrain_ray_tiny_completes_full_global_step(tmp_path: Path) -> None:
    if os.environ.get("DVLA_MANUAL_COTRAIN_RAY_SMOKE") != "1":
        pytest.skip("set DVLA_MANUAL_COTRAIN_RAY_SMOKE=1 to run the manual Ray smoke")

    repo = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "dvla_manual_cotrain_ray_smoke"
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONPATH"] = str(repo)
    cmd = [
        sys.executable,
        str(repo / "tests" / "helpers" / "run_manual_cotrain_fixture.py"),
        str(repo / "tests" / "fixtures" / "manual_cotrain_ray_tiny.yaml"),
        "manual_cotrain.learner_update_step=1",
        f"training.out_dir={out_dir}",
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
    assert "[manual-cotrain] groups=LearnerGroup,ActorGroup,RolloutGroup,EnvGroup" in output
    assert (out_dir / "resolved_config.yaml").is_file()
    assert (out_dir / "run_manifest.json").is_file()
    resolved = (out_dir / "resolved_config.yaml").read_text()
    assert "global_steps: 1" in resolved
    assert "learner_update_step: 1" in resolved
