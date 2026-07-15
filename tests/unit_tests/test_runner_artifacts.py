from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner


class _ConcreteRunner(BaseRunner):
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
    assert runner.get_log_dir() == out_dir.resolve() / "logs"
    assert runner.get_checkpoint_dir() == out_dir.resolve() / "checkpoints"
    assert runner.get_tensorboard_dir() == out_dir.resolve() / "tensorboard"
    assert runner.get_wandb_dir() == out_dir.resolve() / "wandb"
    assert runner.get_video_dir("eval") == out_dir.resolve() / "video" / "eval"
    assert (
        runner.get_global_step_checkpoint_dir(12)
        == out_dir.resolve() / "checkpoints" / "global_step_12"
    )
    assert (
        runner.get_component_checkpoint_dir("actor", step=12)
        == out_dir.resolve() / "checkpoints" / "global_step_12" / "actor"
    )


def test_base_runner_prefers_new_checkpoint_dir_but_resumes_compat_latest(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    compat_latest = out_dir / "ckpt" / "latest.ckpt"
    compat_latest.parent.mkdir(parents=True)
    compat_latest.write_bytes(b"legacy_payload")
    runner = _ConcreteRunner(OmegaConf.create({"training": {"out_dir": str(out_dir)}}))

    assert runner.get_checkpoint_path("latest") == out_dir / "checkpoints" / "latest.ckpt"
    assert runner.get_checkpoint_path("latest", prefer_existing=True) == compat_latest.resolve()

    new_latest = out_dir / "checkpoints" / "latest.ckpt"
    new_latest.parent.mkdir(parents=True)
    new_latest.write_bytes(b"new")
    assert runner.get_checkpoint_path("latest", prefer_existing=True) == new_latest


def test_base_runner_resume_reuses_checkpoint_owning_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "dreamer-wm" / "20260714_120000"
    checkpoint = run_dir / "checkpoints" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    cfg = OmegaConf.create(
        {
            "training": {
                "out_dir": str(tmp_path / "fresh"),
                "resume": True,
                "resume_dir": str(checkpoint),
                "resume_path": str(checkpoint),
            }
        }
    )

    runner = _ConcreteRunner(cfg)

    assert runner.get_run_dir() == run_dir.resolve()


def test_base_runner_setup_writes_only_shallow_run_artifacts(tmp_path: Path) -> None:
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

    runner.setup()

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.is_file()
    assert (out_dir / "checkpoints").is_dir()
    for absent_artifact in (
        "resolved_config.yaml",
        "logs",
        "tensorboard",
        "wandb",
        "video",
        "diagnostics",
    ):
        assert not (out_dir / absent_artifact).exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "created_at_utc",
        "runner",
        "distributed",
        "logging",
        "git",
    }
    assert manifest["schema_version"] == 2
    assert manifest["runner"]["class"] == "_ConcreteRunner"
    assert manifest["runner"]["name"] == "base"
    assert manifest["runner"]["family"] == "runner"
    assert manifest["runner"]["status"] == "abstract"
    assert manifest["logging"]["backends"] == ["tensorboard", "wandb"]
    assert manifest["distributed"]["strategy"] == "ddp"
    assert "git" in manifest


def test_base_runner_metric_logger_keeps_tensorboard_artifacts_shallow(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(out_dir)},
            "runner": {"logger": {"logger_backends": ["tensorboard"]}},
        }
    )
    runner = _ConcreteRunner(cfg)

    runner.log_metrics({"train/loss": 1.0}, step=0)
    runner.finish_metric_logger()

    tensorboard_dir = out_dir / "tensorboard"
    assert any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir())
    assert not (tensorboard_dir / "config.yaml").exists()
