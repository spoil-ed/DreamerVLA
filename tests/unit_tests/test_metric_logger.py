from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.utils.metric_logger import MetricLogger


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
                    "project_name": "dreamervla",
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


def test_metric_logger_passes_online_mode_to_wandb_init(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []
    settings_calls: list[dict[str, Any]] = []

    class FakeWandb:
        @staticmethod
        def Settings(**kwargs):
            settings_calls.append(dict(kwargs))
            return {"settings": dict(kwargs)}

        @staticmethod
        def init(**kwargs) -> None:
            calls.append(dict(kwargs))

        @staticmethod
        def log(*args, **kwargs) -> None:
            return None

        @staticmethod
        def finish() -> None:
            return None

    log_root = tmp_path / "logs"
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(log_root),
                    "project_name": "dreamervla",
                    "experiment_name": "unit-online",
                    "logger_backends": ["wandb"],
                    "wandb_mode": "online",
                    "wandb_proxy": "http://proxy.local:8080",
                }
            },
            "training": {"out_dir": str(tmp_path / "out")},
        }
    )
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    logger = MetricLogger(cfg)
    logger.finish()

    assert settings_calls == [{"https_proxy": "http://proxy.local:8080"}]
    assert calls == [
        {
            "project": "dreamervla",
            "name": "unit-online",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "settings": {"settings": {"https_proxy": "http://proxy.local:8080"}},
            "dir": str(log_root / "wandb"),
            "mode": "online",
            "reinit": True,
        }
    ]


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
            "env_success_rate": 0.75,
            "rollout_episode_len": 25,
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
                "env/success_rate": 0.75,
                "rollout/episode_len": 25.0,
                "time/data": 3.0,
                "train/lr": 1.0e-4,
            },
            8,
            None,
        )
    ]

    runner.teardown()
    assert capture.finished is True
