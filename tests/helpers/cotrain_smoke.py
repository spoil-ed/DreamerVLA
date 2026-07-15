"""Command construction shared by cotrain smoke coverage."""

from __future__ import annotations

import sys
from pathlib import Path


def cotrain_smoke_command(repo: Path, out_dir: Path) -> list[str]:
    """Build the native Hydra command for the test-only cotrain fixture."""
    return [
        sys.executable,
        "-m",
        "dreamervla.train",
        "--config-path",
        str(repo / "tests" / "fixtures"),
        "--config-name",
        "cotrain_tiny",
        "manual_cotrain.learner_update_step=1",
        f"hydra.run.dir={out_dir}",
        f"training.out_dir={out_dir}",
        "hydra.job.chdir=false",
    ]
