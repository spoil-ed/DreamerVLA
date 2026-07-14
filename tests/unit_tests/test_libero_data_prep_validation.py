from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_validation_workflow_registers_only_exact_hidden_token_check() -> None:
    root = _root()
    with initialize_config_dir(config_dir=str(root / "configs" / "scripts"), version_base=None):
        cfg = compose(config_name="preprocess/validate_libero_data")

    assert cfg.name == "validate_libero_data"
    assert len(cfg.steps) == 1
    assert cfg.steps[0].id == "validate_suite"
    assert cfg.steps[0].script == "scripts/preprocess/20_validate.sh"
    assert cfg.steps[0].env.TASK == "{item}"
    assert not (root / "dreamervla" / "preprocess" / "validate_libero_data_prep.py").exists()
    assert not (
        root
        / "configs"
        / "scripts"
        / "preprocess"
        / "validate_libero_data_prep.yaml"
    ).exists()


def test_validation_wrapper_fans_out_suites_through_hydra(tmp_path: Path) -> None:
    root = _root()
    env = os.environ.copy()
    env["DVLA_DATA_ROOT"] = str(tmp_path / "data")
    result = subprocess.run(
        [
            "bash",
            "scripts/preprocess/validate_libero_data.sh",
            "dry_run=true",
            "tasks=[libero_goal,libero_object]",
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("run validate_suite") == 2
    assert "20_validate.sh" in result.stdout
