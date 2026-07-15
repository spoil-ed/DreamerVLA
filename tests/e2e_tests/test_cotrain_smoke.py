from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.helpers.cotrain_smoke import cotrain_smoke_command


def test_cotrain_tiny_completes_full_global_step(tmp_path: Path) -> None:
    if os.environ.get("DVLA_MANUAL_COTRAIN_RAY_SMOKE") != "1":
        pytest.skip("set DVLA_MANUAL_COTRAIN_RAY_SMOKE=1 to run the manual Ray smoke")

    repo = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "dvla_manual_cotrain_ray_smoke"
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONPATH"] = str(repo)
    cmd = cotrain_smoke_command(repo, out_dir)

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
    assert (out_dir / "run_manifest.json").is_file()
    hydra_dir = out_dir / ".hydra"
    for name in ("config.yaml", "overrides.yaml", "hydra.yaml"):
        assert (hydra_dir / name).is_file()
    assert not (out_dir / "resolved_config.yaml").exists()
    hydra_config = OmegaConf.load(hydra_dir / "config.yaml")
    assert hydra_config.manual_cotrain.global_steps == 1
    assert hydra_config.manual_cotrain.learner_update_step == 1
