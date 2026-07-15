from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from tests.helpers.cotrain_smoke import cotrain_smoke_command


def test_native_hydra_writes_its_metadata_under_training_out_dir(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    run_root = tmp_path / "native-hydra-run"
    command = [
        sys.executable,
        "-c",
        (
            "import dreamervla.train as train; "
            "train.run = lambda cfg: None; "
            "train.main()"
        ),
        f"training.out_dir={run_root}",
        "hydra.job.chdir=false",
    ]

    result = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    hydra_dir = run_root / ".hydra"
    assert (hydra_dir / "config.yaml").is_file()
    assert (hydra_dir / "overrides.yaml").is_file()
    assert (hydra_dir / "hydra.yaml").is_file()
    assert not (run_root / "resolved_config.yaml").exists()

    saved_config = yaml.safe_load((hydra_dir / "config.yaml").read_text())
    assert saved_config["training"]["out_dir"] == str(run_root)


def test_cotrain_smoke_command_composes_fixture_through_native_hydra(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    run_root = tmp_path / "cotrain-fixture-hydra"
    command = cotrain_smoke_command(project_root, run_root)
    assert command[:3] == [sys.executable, "-m", "dreamervla.train"]
    assert command[3:7] == [
        "--config-path",
        str(project_root / "tests" / "fixtures"),
        "--config-name",
        "cotrain_tiny",
    ]
    assert f"hydra.run.dir={run_root}" in command
    assert f"training.out_dir={run_root}" in command

    no_op_command = [
        sys.executable,
        "-c",
        (
            "import dreamervla.train as train; "
            "train.run = lambda cfg: None; "
            "train.main()"
        ),
        *command[3:],
    ]
    result = subprocess.run(
        no_op_command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    hydra_dir = run_root / ".hydra"
    for name in ("config.yaml", "overrides.yaml", "hydra.yaml"):
        assert (hydra_dir / name).is_file()
    saved_config = yaml.safe_load((hydra_dir / "config.yaml").read_text())
    assert saved_config["manual_cotrain"]["global_steps"] == 1
    assert saved_config["manual_cotrain"]["learner_update_step"] == 1
