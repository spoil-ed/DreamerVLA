from __future__ import annotations

import math
import numbers
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


class _TensorboardLogger:
    def __init__(self, log_path: str | os.PathLike[str]) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(log_path))

    def log(self, data: Mapping[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(str(key), float(value), int(step))

    def finish(self) -> None:
        self.writer.close()


class NullMetricLogger:
    """Metric logger used when logging is disabled or this rank should not log."""

    logger_backends: list[str] = []

    def log(
        self,
        data: Mapping[str, float],
        step: int,
        backend: str | list[str] | tuple[str, ...] | None = None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ) -> None:
        return None

    def log_table(self, df_data: Any, name: str, step: int) -> None:
        return None

    def finish(self) -> None:
        return None


class MetricLogger:
    """RLinf-style metric logger for TensorBoard, W&B, and SwanLab.

    Preferred config shape mirrors RLinf:

    ```
    runner:
      logger:
        log_path: ...
        project_name: dreamer_vla
        experiment_name: ...
        logger_backends: [tensorboard]
    ```

    Existing DreamerVLA configs do not need a `runner` section; when it is
    absent, logs default to `${training.out_dir}/log`.
    """

    supported_logger = ["wandb", "swanlab", "tensorboard"]

    def __init__(
        self,
        cfg: DictConfig,
        *,
        default_log_path: str | os.PathLike[str] | None = None,
        default_project_name: str = "dreamer_vla",
        default_experiment_name: str | None = None,
    ) -> None:
        self.cfg = cfg
        logger_cfg = _select_logger_cfg(cfg)

        training_out_dir = OmegaConf.select(cfg, "training.out_dir", default=None)
        fallback_log_path = (
            Path(str(training_out_dir)).expanduser() / "log"
            if training_out_dir is not None
            else Path("log")
        )
        log_path = _cfg_get(logger_cfg, "log_path", default_log_path)
        if log_path is None:
            log_path = fallback_log_path
        self.log_path = str(Path(str(log_path)).expanduser())

        self.project_name = str(
            _cfg_get(logger_cfg, "project_name", default_project_name)
        )
        experiment_default = default_experiment_name
        if experiment_default is None:
            if training_out_dir is not None:
                experiment_default = Path(str(training_out_dir)).expanduser().name
            else:
                experiment_default = "default"
        self.experiment_name = str(
            _cfg_get(logger_cfg, "experiment_name", experiment_default)
        )

        self.per_worker_log = bool(
            OmegaConf.select(cfg, "runner.per_worker_log", default=False)
        )
        self.per_worker_log_root = str(
            OmegaConf.select(
                cfg,
                "runner.per_worker_log_path",
                default=os.path.join(self.log_path, "worker_logs"),
            )
        )

        logger_backends = _cfg_get(logger_cfg, "logger_backends", ["tensorboard"])
        self.logger_backends = _normalize_backends(logger_backends)
        unsupported = [
            backend
            for backend in self.logger_backends
            if backend not in self.supported_logger
        ]
        if unsupported:
            raise ValueError(f"Unsupported logger backend: {unsupported}")

        self.wandb_proxy = _cfg_get(logger_cfg, "wandb_proxy", None)
        self.wandb_mode = str(_cfg_get(logger_cfg, "wandb_mode", "online"))
        self.swanlab_mode = str(_cfg_get(logger_cfg, "swanlab_mode", "cloud"))
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self._all_loggers: list[dict[str, Any]] = []
        self._worker_loggers: dict[tuple[str, int], dict[str, Any]] = {}
        self._finished = False
        self.logger = self._create_logger_bundle(
            log_path=self.log_path,
            experiment_name=self.experiment_name,
            log_path_suffix="all" if self.per_worker_log else "",
        )

    def _create_logger_bundle(
        self,
        *,
        log_path: str,
        experiment_name: str,
        log_path_suffix: str = "",
    ) -> dict[str, Any]:
        bundle: dict[str, Any] = {}
        if "wandb" in self.logger_backends:
            import wandb

            wandb_log_path = Path(log_path) / "wandb" / log_path_suffix
            wandb_log_path.mkdir(parents=True, exist_ok=True)

            settings = None
            if self.wandb_proxy:
                settings = wandb.Settings(https_proxy=self.wandb_proxy)
            wandb.init(
                project=self.project_name,
                name=experiment_name,
                config=self.config,
                settings=settings,
                dir=str(wandb_log_path),
                mode=self.wandb_mode,
                reinit=True,
            )
            bundle["wandb"] = wandb

        if "swanlab" in self.logger_backends:
            import swanlab

            swanlab_log_path = Path(log_path) / "swanlab" / log_path_suffix
            swanlab_log_path.mkdir(parents=True, exist_ok=True)
            swanlab.init(
                project=self.project_name,
                experiment_name=experiment_name,
                config=self.config,
                logdir=str(swanlab_log_path),
                mode=self.swanlab_mode,
            )
            bundle["swanlab"] = swanlab

        if "tensorboard" in self.logger_backends:
            tensorboard_log_path = Path(log_path) / "tensorboard" / log_path_suffix
            tensorboard_log_path.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(
                config=self.cfg,
                f=str(tensorboard_log_path / "config.yaml"),
                resolve=True,
            )
            bundle["tensorboard"] = _TensorboardLogger(tensorboard_log_path)

        self._all_loggers.append(bundle)
        return bundle

    def _get_scoped_logger(
        self,
        *,
        worker_group_name: str,
        rank: int,
    ) -> dict[str, Any]:
        key = (str(worker_group_name), int(rank))
        if key in self._worker_loggers:
            return self._worker_loggers[key]

        scoped_log_path = os.path.join(
            self.per_worker_log_root,
            str(worker_group_name),
            f"rank_{int(rank)}",
        )
        scoped_experiment_name = (
            f"{self.experiment_name}-{worker_group_name}-rank_{int(rank)}"
        )
        scoped_logger = self._create_logger_bundle(
            log_path=scoped_log_path,
            experiment_name=scoped_experiment_name,
        )
        self._worker_loggers[key] = scoped_logger
        return scoped_logger

    def log(
        self,
        data: Mapping[str, Any],
        step: int,
        backend: str | list[str] | tuple[str, ...] | None = None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ) -> None:
        metrics = _coerce_scalar_metrics(data)
        if not metrics:
            return
        target_logger = self.logger
        if self.per_worker_log and worker_group_name is not None and rank is not None:
            target_logger = self._get_scoped_logger(
                worker_group_name=worker_group_name,
                rank=rank,
            )
        backend_filter = _normalize_backend_filter(backend)
        for default_backend, logger_instance in target_logger.items():
            if backend_filter is None or default_backend in backend_filter:
                logger_instance.log(data=metrics, step=int(step))

    def log_table(self, df_data: Any, name: str, step: int) -> None:
        if "wandb" not in self.logger:
            raise ValueError(f"Unsupported log table for {self.logger_backends}")
        table = self.logger["wandb"].Table(dataframe=df_data)
        self.logger["wandb"].log({name: table}, step=int(step))

    def finish(self) -> None:
        if self._finished:
            return
        for logger_bundle in self._all_loggers:
            for logger_instance in logger_bundle.values():
                finish = getattr(logger_instance, "finish", None)
                if callable(finish):
                    finish()
        self._finished = True

    def __del__(self) -> None:
        try:
            self.finish()
        except Exception:
            pass


def _select_logger_cfg(cfg: DictConfig) -> Any:
    logger_cfg = OmegaConf.select(cfg, "runner.logger", default=None)
    if logger_cfg is not None:
        return logger_cfg
    return OmegaConf.select(cfg, "logging", default=None)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_backends(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_backends = [value]
    elif isinstance(value, (list, tuple, ListConfig)):
        raw_backends = list(value)
    else:
        raw_backends = [value]

    backends: list[str] = []
    for backend in raw_backends:
        normalized = str(backend).strip().lower()
        if normalized in {"", "none", "null", "false", "off", "disabled"}:
            continue
        backends.append(normalized)
    return backends


def _normalize_backend_filter(
    value: str | list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    backends = _normalize_backends(value)
    if not backends:
        return None if value is None else set()
    return set(backends)


def _coerce_scalar_metrics(data: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        scalar = _coerce_scalar(value)
        if scalar is None:
            continue
        metrics[str(key)] = scalar
    return metrics


def _coerce_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Number):
        scalar = float(value)
    elif hasattr(value, "detach") and hasattr(value, "numel"):
        try:
            if int(value.numel()) != 1:
                return None
            scalar = float(value.detach().item())
        except Exception:
            return None
    else:
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


__all__ = ["MetricLogger", "NullMetricLogger"]
