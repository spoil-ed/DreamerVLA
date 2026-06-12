from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from dreamer_vla.runners.base_runner import BaseRunner


class _ConcreteRunner(BaseRunner):
    def setup(self) -> None:
        return None

    def execute(self) -> None:
        return None

    def run(self) -> object:
        return None

    def teardown(self) -> None:
        return None


def test_base_runner_uses_rlinf_style_run_artifact_dirs(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    runner = _ConcreteRunner(OmegaConf.create({"training": {"out_dir": str(out_dir)}}))

    assert runner.get_run_dir() == out_dir.resolve()
    assert runner.get_log_dir() == out_dir.resolve() / "log"
    assert runner.get_checkpoint_dir() == out_dir.resolve() / "checkpoints"
    assert runner.get_tensorboard_dir() == out_dir.resolve() / "log" / "tensorboard"
    assert runner.get_wandb_dir() == out_dir.resolve() / "log" / "wandb"
    assert runner.get_video_dir("eval") == out_dir.resolve() / "video" / "eval"
    assert (
        runner.get_global_step_checkpoint_dir(12)
        == out_dir.resolve() / "checkpoints" / "global_step_12"
    )
    assert (
        runner.get_component_checkpoint_dir("actor", step=12)
        == out_dir.resolve() / "checkpoints" / "global_step_12" / "actor"
    )


def test_base_runner_prefers_new_checkpoint_dir_but_resumes_legacy_latest(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    legacy_latest = out_dir / "ckpt" / "latest.ckpt"
    legacy_latest.parent.mkdir(parents=True)
    legacy_latest.write_bytes(b"legacy")
    runner = _ConcreteRunner(OmegaConf.create({"training": {"out_dir": str(out_dir)}}))

    assert runner.get_checkpoint_path("latest") == out_dir / "checkpoints" / "latest.ckpt"
    assert (
        runner.get_checkpoint_path("latest", prefer_existing=True)
        == legacy_latest.resolve()
    )

    new_latest = out_dir / "checkpoints" / "latest.ckpt"
    new_latest.parent.mkdir(parents=True)
    new_latest.write_bytes(b"new")
    assert runner.get_checkpoint_path("latest", prefer_existing=True) == new_latest


def test_base_runner_writes_reproducible_run_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "training": {
                "out_dir": str(out_dir),
                "distributed_strategy": "ddp",
            },
            "runner": {
                "logger": {
                    "logger_backends": ["tensorboard", "wandb"],
                }
            },
        }
    )
    runner = _ConcreteRunner(cfg)

    runner.write_run_artifacts()

    resolved_config = out_dir / "resolved_config.yaml"
    manifest_path = out_dir / "run_manifest.json"
    assert resolved_config.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["runner"]["class"] == "_ConcreteRunner"
    assert manifest["runner"]["name"] == "base"
    assert manifest["run_dir"] == str(out_dir.resolve())
    assert manifest["artifact_dirs"]["checkpoints"] == str(
        out_dir.resolve() / "checkpoints"
    )
    assert manifest["artifact_dirs"]["tensorboard"] == str(
        out_dir.resolve() / "log" / "tensorboard"
    )
    assert manifest["logging"]["backends"] == ["tensorboard", "wandb"]
    assert manifest["distributed"]["strategy"] == "ddp"
    assert "git" in manifest
