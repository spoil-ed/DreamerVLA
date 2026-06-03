from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dreamer_vla.runners.base_runner import BaseRunner
from dreamer_vla.utils.metric_logger import MetricLogger


class _ConcreteRunner(BaseRunner):
    def run(self) -> object:
        return None


class _CaptureMetricLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int, Any]] = []
        self.finished = False

    def log(
        self,
        data: dict[str, float],
        step: int,
        backend: Any = None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ) -> None:
        self.calls.append((dict(data), int(step), backend))

    def finish(self) -> None:
        self.finished = True


def test_metric_logger_writes_tensorboard_run_and_config(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(log_root),
                    "project_name": "dreamer_vla",
                    "experiment_name": "unit",
                    "logger_backends": ["tensorboard"],
                }
            },
            "training": {"out_dir": str(tmp_path / "out")},
        }
    )

    logger = MetricLogger(cfg)
    logger.log({"train/loss": 1.25, "train/count": 2.0}, step=3)
    logger.finish()

    tensorboard_dir = log_root / "tensorboard"
    assert (tensorboard_dir / "config.yaml").is_file()
    assert any(
        path.name.startswith("events.out.tfevents")
        for path in tensorboard_dir.iterdir()
    )


def test_metric_logger_defaults_to_training_output_log_dir(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    cfg = OmegaConf.create({"training": {"out_dir": str(out_dir)}})

    logger = MetricLogger(cfg)
    logger.log({"train/loss": 1.0}, step=0)
    logger.finish()

    assert (out_dir / "log" / "tensorboard" / "config.yaml").is_file()


def test_base_runner_log_metrics_normalizes_current_metric_names(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create({"training": {"out_dir": str(tmp_path / "out")}})
    runner = _ConcreteRunner(cfg)
    capture = _CaptureMetricLogger()
    runner._metric_logger = capture

    runner.log_metrics(
        {
            "train_loss": 1,
            "val_accuracy": 0.5,
            "eval_return": 2.0,
            "time_data": 3.0,
            "lr": 1.0e-4,
            "global_step": 8,
            "epoch": 1,
            "note": "ignored",
            "bad": float("nan"),
            "enabled": True,
        }
    )

    assert capture.calls == [
        (
            {
                "train/loss": 1.0,
                "eval/accuracy": 0.5,
                "eval/return": 2.0,
                "time/data": 3.0,
                "train/lr": 1.0e-4,
            },
            8,
            None,
        )
    ]

    runner.teardown()
    assert capture.finished is True
