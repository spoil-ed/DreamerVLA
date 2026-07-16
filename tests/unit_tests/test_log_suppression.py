from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_sitecustomize_suppresses_gym_deprecation_notice() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    proc = subprocess.run(
        [sys.executable, "-c", "import gym; print('imports_ok')"],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )

    assert proc.stdout.strip() == "imports_ok"
    assert "Gym has been unmaintained since 2022" not in proc.stderr
