from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_hydra_configs_and_runs_help_in_isolation(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(unpacked)
        wheel_files = set(archive.namelist())

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(unpacked)
    env["PYTHONNOUSERSITE"] = "1"
    help_result = subprocess.run(
        [sys.executable, "-m", "dreamervla.train", "--help"],
        cwd=runtime_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "train is powered by Hydra" in help_result.stdout
    source_configs = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "configs").rglob("*.yaml")
    }
    assert source_configs <= wheel_files

    config_origin = subprocess.run(
        [sys.executable, "-c", "import configs; print(configs.__file__)"],
        cwd=runtime_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert config_origin.returncode == 0, config_origin.stdout + config_origin.stderr
    assert Path(config_origin.stdout.strip()).is_relative_to(unpacked)
