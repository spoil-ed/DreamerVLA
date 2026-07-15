from __future__ import annotations

import subprocess
from pathlib import Path


def test_dreamer_train_script_delegates_checkpoint_and_resume_to_launcher() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "experiments" / "dreamer" / "train.sh"

    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "python -m dreamervla.launchers.cotrain" in text
    assert "--config openvla_libero" in text
    assert '"$@"' in text
    assert "training.resume_path=" not in text
    assert "manual_cotrain.resume_ckpt=" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)
