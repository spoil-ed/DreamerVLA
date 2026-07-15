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
    assert any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir())


def test_metric_logger_defaults_to_training_output_root(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    cfg = OmegaConf.create({"training": {"out_dir": str(out_dir)}})

    logger = MetricLogger(cfg)
    logger.log({"train/loss": 1.0}, step=0)
    logger.finish()

    assert (out_dir / "tensorboard" / "config.yaml").is_file()


def test_metric_logger_passes_online_mode_to_wandb_init(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []
    settings_calls: list[dict[str, Any]] = []

    class FakeWandb:
        class util:
            @staticmethod
            def generate_id() -> str:
                return "newrun01"

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
            "id": "newrun01",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "settings": {"settings": {"https_proxy": "http://proxy.local:8080"}},
            "dir": str(log_root / "wandb"),
            "mode": "online",
            "reinit": True,
        }
    ]
    assert (log_root / "wandb" / "run_id.txt").read_text().strip() == "newrun01"


def test_metric_logger_resumes_existing_online_wandb_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeWandb:
        @staticmethod
        def init(**kwargs) -> None:
            calls.append(dict(kwargs))

        @staticmethod
        def finish() -> None:
            return None

    log_root = tmp_path / "logs"
    identity = log_root / "wandb" / "run_id.txt"
    identity.parent.mkdir(parents=True)
    identity.write_text("same1234\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(log_root),
                    "project_name": "dreamervla",
                    "experiment_name": "resume-online",
                    "logger_backends": ["wandb"],
                    "wandb_mode": "online",
                }
            },
            "training": {"out_dir": str(tmp_path / "out"), "resume": True},
        }
    )
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    logger = MetricLogger(cfg, resume=True, resume_step=17)
    logger.finish()

    assert calls[0]["id"] == "same1234"
    assert calls[0]["resume"] == "allow"


def test_metric_logger_truncates_online_wandb_tail_when_sdk_supports_resume_from(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeWandb:
        @staticmethod
        def init(*, resume_from=None, **kwargs) -> None:
            calls.append({**kwargs, "resume_from": resume_from})

        @staticmethod
        def finish() -> None:
            return None

    log_root = tmp_path / "logs"
    identity = log_root / "wandb" / "run_id.txt"
    identity.parent.mkdir(parents=True)
    identity.write_text("same1234\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(log_root),
                    "project_name": "dreamervla",
                    "experiment_name": "resume-online",
                    "logger_backends": ["wandb"],
                    "wandb_mode": "online",
                }
            },
            "training": {"out_dir": str(tmp_path / "out"), "resume": True},
        }
    )
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    logger = MetricLogger(cfg, resume=True, resume_step=6000)
    logger.finish()

    assert calls[0]["id"] == "same1234"
    assert calls[0]["resume_from"] == "same1234?_step=6000"
    assert "resume" not in calls[0]


def test_metric_logger_offline_resume_reuses_legacy_id_without_sdk_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeWandb:
        @staticmethod
        def init(**kwargs) -> None:
            calls.append(dict(kwargs))

        @staticmethod
        def finish() -> None:
            return None

    log_root = tmp_path / "logs"
    legacy = log_root / "wandb" / "wandb" / "offline-run-20260714_120000-oldrun42"
    legacy.mkdir(parents=True)
    (legacy / "run-oldrun42.wandb").touch()
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(log_root),
                    "project_name": "dreamervla",
                    "experiment_name": "resume-offline",
                    "logger_backends": ["wandb"],
                    "wandb_mode": "offline",
                }
            },
            "training": {"out_dir": str(tmp_path / "out"), "resume": True},
        }
    )
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    logger = MetricLogger(cfg, resume=True, resume_step=17)
    logger.finish()

    assert calls[0]["id"] == "oldrun42"
    assert "resume" not in calls[0]
    assert (log_root / "wandb" / "run_id.txt").read_text().strip() == "oldrun42"


def test_metric_logger_tensorboard_resume_purges_post_checkpoint_tail(tmp_path: Path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_root = tmp_path / "logs"
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(log_root),
                    "logger_backends": ["tensorboard"],
                }
            },
            "training": {"out_dir": str(tmp_path / "out")},
        }
    )
    first = MetricLogger(cfg)
    for step in range(4):
        first.log({"train/loss": float(step)}, step=step)
    first.finish()

    resumed = MetricLogger(cfg, resume=True, resume_step=2)
    resumed.log({"train/loss": 20.0}, step=2)
    resumed.log({"train/loss": 30.0}, step=3)
    resumed.finish()

    events = EventAccumulator(str(log_root / "tensorboard")).Reload().Scalars("train/loss")
    assert [(event.step, event.value) for event in events] == [
        (0, 0.0),
        (1, 1.0),
        (2, 20.0),
        (3, 30.0),
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
