from __future__ import annotations

import copy
import inspect
import json
import math
import numbers
import os
import pathlib
import pickle
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime
from pprint import pprint
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.runners.online_utils import SuccessTracker
from dreamervla.utils.console import fmt_value, metric_box, phase_banner
from dreamervla.utils.hf_checkpoint import (
    is_hf_checkpoint,
    load_runner_payload,
    resolve_hf_checkpoint_dir,
)
from dreamervla.utils.metric_logger import MetricLogger, NullMetricLogger
from dreamervla.utils.progress import ProgressReporter

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _group_metric_rows(metrics: dict, *, skip_success: bool = False) -> list[str]:
    """Group namespaced metrics into one row per prefix, dropping meta keys."""
    meta = {"global_step", "step", "epoch", "ts", "phase"}
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for k, v in metrics.items():
        if k in meta or isinstance(v, str):
            continue
        if skip_success and k.startswith("rollout/success_rate"):
            continue
        prefix, _, name = k.partition("/")
        if not name:
            prefix, name = "metrics", k
        if prefix not in groups:
            groups[prefix] = []
            order.append(prefix)
        groups[prefix].append(f"{name}={fmt_value(v)}")
    return [f"{p:<7} " + "  ".join(groups[p]) for p in order]


class BaseRunner(ABC):
    runner_name = "base"
    runner_status = "abstract"
    runner_family = "runner"
    include_keys = tuple()
    exclude_keys = tuple()
    checkpoint_restore_output_dir = False

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        # Runtime config
        self.config = config
        self.cfg = config
        self._output_dir = output_dir

        # Resolved config
        OmegaConf.resolve(self.config)

        # Loop state
        self.global_step = 0
        self.epoch = 0
        self._metric_logger: Any | None = None
        self._run_artifacts_written = False

    @property
    def output_dir(self) -> str:
        # Output dir
        if self._output_dir is not None:
            return str(pathlib.Path(self._output_dir).expanduser().resolve())

        configured_output_dir = OmegaConf.select(self.config, "training.out_dir")
        if configured_output_dir is not None:
            output_dir = pathlib.Path(str(configured_output_dir)).expanduser()
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir
            return str(output_dir.resolve())
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            return str(pathlib.Path(".").resolve())

    def get_checkpoint_dir(self) -> pathlib.Path:
        # RLinf-style canonical checkpoint directory.
        return self.get_run_dir().joinpath("checkpoints")

    def get_legacy_checkpoint_dir(self) -> pathlib.Path:
        # Compatibility with older DreamerVLA latest.ckpt checkpoints.
        return self.get_run_dir().joinpath("ckpt")

    def get_run_dir(self) -> pathlib.Path:
        return pathlib.Path(self.output_dir)

    def get_artifact_dir(self, *parts: str) -> pathlib.Path:
        return self.get_run_dir().joinpath(*parts)

    def get_log_dir(self) -> pathlib.Path:
        # Log dir
        return self.get_artifact_dir("log")

    def get_tensorboard_dir(self) -> pathlib.Path:
        return self.get_log_dir().joinpath("tensorboard")

    def get_wandb_dir(self) -> pathlib.Path:
        return self.get_log_dir().joinpath("wandb")

    def get_video_dir(self, split: str = "eval") -> pathlib.Path:
        return self.get_artifact_dir("video", str(split))

    def get_diagnostics_dir(self) -> pathlib.Path:
        return self.get_artifact_dir("diagnostics")

    def get_resolved_config_path(self) -> pathlib.Path:
        return self.get_artifact_dir("resolved_config.yaml")

    def get_run_manifest_path(self) -> pathlib.Path:
        return self.get_artifact_dir("run_manifest.json")

    def get_global_step_checkpoint_dir(self, step: int) -> pathlib.Path:
        return self.get_checkpoint_dir().joinpath(f"global_step_{int(step)}")

    def get_component_checkpoint_dir(
        self,
        component: str,
        *,
        step: int | None = None,
    ) -> pathlib.Path:
        root = (
            self.get_checkpoint_dir()
            if step is None
            else self.get_global_step_checkpoint_dir(int(step))
        )
        return root.joinpath(str(component))

    def get_log_path(self, name: str = "logs.json.txt") -> pathlib.Path:
        # Log file
        return self.get_log_dir().joinpath(name)

    def print_config(self) -> None:
        # Config dump — suppressed by default; the resolved config is always
        # persisted to resolved_config.yaml + .hydra/, so nothing is lost.
        if not bool(OmegaConf.select(self.cfg, "training.print_config", default=False)):
            return
        pprint(OmegaConf.to_container(self.config, resolve=True))

    def _checkpoint_format(self) -> str:
        return str(OmegaConf.select(self.cfg, "training.checkpoint_format", default="both")).lower()

    def checkpoint_save_torch(self) -> bool:
        return self._checkpoint_format() in ("torch", "both")

    def checkpoint_save_hf(self) -> bool:
        return self._checkpoint_format() in ("hf", "both")

    def write_run_artifacts(self) -> dict[str, Any] | None:
        """Write reproducibility artifacts for the current run."""

        if not self.is_main_process:
            return None

        self.get_run_dir().mkdir(parents=True, exist_ok=True)
        self.get_log_dir().mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            config=self.cfg,
            f=str(self.get_resolved_config_path()),
            resolve=True,
        )
        manifest = self.build_run_manifest()
        self.get_run_manifest_path().write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._run_artifacts_written = True
        return manifest

    def append_model_summary(self, summary: dict[str, Any]) -> None:
        """Write runtime-derived model info (param counts, freeze flags) into
        the existing run manifest. These are not in any config because they are
        computed after model instantiation."""
        if not self.is_main_process:
            return
        path = self.get_run_manifest_path()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        manifest["model"] = summary
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def build_run_manifest(self) -> dict[str, Any]:
        """Build a compact RLinf-style run manifest."""

        logger_cfg = OmegaConf.select(self.cfg, "runner.logger", default=None)
        if logger_cfg is None:
            logger_cfg = OmegaConf.select(self.cfg, "logging", default=None)
        backends = _normalise_logger_backends(
            _mapping_get(logger_cfg, "logger_backends", ["tensorboard"])
        )

        distributed = getattr(self, "distributed", None)
        return {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "runner": {
                "class": type(self).__name__,
                "name": str(self.runner_name),
                "family": str(self.runner_family),
                "status": str(self.runner_status),
            },
            "run_dir": str(self.get_run_dir()),
            "artifact_dirs": {
                "checkpoints": str(self.get_checkpoint_dir()),
                "diagnostics": str(self.get_diagnostics_dir()),
                "log": str(self.get_log_dir()),
                "tensorboard": str(self.get_tensorboard_dir()),
                "wandb": str(self.get_wandb_dir()),
                "video_eval": str(self.get_video_dir("eval")),
            },
            "state": {
                "global_step": int(self.global_step),
                "epoch": int(self.epoch),
            },
            "distributed": {
                "strategy": str(
                    OmegaConf.select(
                        self.cfg,
                        "training.distributed_strategy",
                        default="ddp",
                    )
                ),
                "rank": int(getattr(distributed, "rank", 0) or 0),
                "local_rank": int(getattr(distributed, "local_rank", 0) or 0),
                "world_size": int(getattr(distributed, "world_size", 1) or 1),
            },
            "logging": {
                "backends": backends,
                "log_path": str(self.get_log_dir()),
            },
            "config": {
                "resolved_config_path": str(self.get_resolved_config_path()),
            },
            "git": self._git_metadata(),
        }

    @staticmethod
    def _git_metadata() -> dict[str, Any]:
        def capture(*args: str) -> str | None:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=str(PROJECT_ROOT),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
            except Exception:
                return None
            return result.stdout.strip()

        status = capture("status", "--short")
        return {
            "commit": capture("rev-parse", "HEAD"),
            "branch": capture("rev-parse", "--abbrev-ref", "HEAD"),
            "is_dirty": None if status is None else bool(status),
        }

    def build_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        # Encoder config
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if (
            init_model_path is not None
            and OmegaConf.select(encoder_cfg, "model_path") is None
        ):
            encoder_cfg.model_path = str(init_model_path)
        return self._target_compatible_cfg(encoder_cfg)

    @staticmethod
    def _target_compatible_cfg(component_cfg: DictConfig) -> DictConfig:
        """Drop merge-leftover keys that the selected Hydra target cannot accept."""
        target = OmegaConf.select(component_cfg, "_target_")
        if target is None:
            return component_cfg
        try:
            target_obj = hydra.utils.get_class(str(target))
        except Exception:
            return component_cfg
        signature = inspect.signature(target_obj.__init__)
        parameters = signature.parameters.values()
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
            return component_cfg
        allowed = {
            p.name
            for p in signature.parameters.values()
            if p.name != "self"
            and p.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        reserved = {"_target_", "_recursive_", "_convert_", "_partial_"}
        raw = OmegaConf.to_container(component_cfg, resolve=False)
        if not isinstance(raw, dict):
            return component_cfg
        filtered = {
            key: value
            for key, value in raw.items()
            if key in reserved or key in allowed
        }
        return OmegaConf.create(filtered)

    def _resolve_vla_init_path(self) -> str:
        configured = OmegaConf.select(self.cfg, "init.vla_ckpt_path")
        default_dir = getattr(self, "default_vla_init_dir", None)
        candidate = (
            pathlib.Path(str(configured)).expanduser().resolve()
            if configured is not None
            else pathlib.Path(str(default_dir)).expanduser().resolve()
        )
        if is_hf_checkpoint(candidate):
            return str(resolve_hf_checkpoint_dir(candidate))
        if candidate.is_dir():
            if (candidate / "config.json").is_file():
                return str(candidate)
            for subdir in sorted(path for path in candidate.iterdir() if path.is_dir()):
                if (subdir / "config.json").is_file():
                    return str(subdir.resolve())
        return str(candidate.resolve())

    def _build_frozen_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = self.build_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            if OmegaConf.select(encoder_cfg, "freeze_vla_backbone", default=None) is not None:
                encoder_cfg.freeze_vla_backbone = True
            else:
                encoder_cfg.freeze_backbone = True
        return encoder_cfg

    @staticmethod
    def _dataloader_kwargs(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
        if isinstance(config, DictConfig):
            return dict(OmegaConf.to_container(config, resolve=True))
        return dict(config)

    @staticmethod
    def _sanitize_worker_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        if int(kwargs.get("num_workers", 0) or 0) <= 0:
            kwargs.pop("prefetch_factor", None)
            kwargs["persistent_workers"] = False
        return kwargs

    def make_distributed_dataloader(
        self,
        dataset: Any,
        dataloader_cfg: Mapping[str, Any] | DictConfig,
        *,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
        sanitize_worker_kwargs: bool = False,
    ) -> DataLoader:
        dataloader_kwargs = self._dataloader_kwargs(dataloader_cfg)
        if sanitize_worker_kwargs:
            dataloader_kwargs = self._sanitize_worker_kwargs(dataloader_kwargs)

        effective_shuffle = (
            bool(dataloader_kwargs.get("shuffle", True))
            if shuffle is None
            else bool(shuffle)
        )
        effective_drop_last = (
            bool(dataloader_kwargs.get("drop_last", False))
            if drop_last is None
            else bool(drop_last)
        )
        dataloader_kwargs["shuffle"] = effective_shuffle
        dataloader_kwargs["drop_last"] = effective_drop_last

        distributed = getattr(self, "distributed", None)
        if distributed is not None and hasattr(distributed, "maybe_make_sampler"):
            sampler = distributed.maybe_make_sampler(
                dataset,
                shuffle=effective_shuffle,
                drop_last=effective_drop_last,
            )
            if sampler is not None:
                dataloader_kwargs["shuffle"] = False
                dataloader_kwargs["sampler"] = sampler

        collate_fn = getattr(dataset, "collate_fn", None)
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        return DataLoader(dataset, **dataloader_kwargs)

    def make_val_dataloaders(
        self,
        cfg: DictConfig,
        *,
        split_names: tuple[str, ...] = ("val_ind", "val_ood"),
        sanitize_worker_kwargs: bool = False,
    ) -> dict[str, DataLoader]:
        val_dataloaders: dict[str, DataLoader] = {}
        for split_name in split_names:
            val_ds_cfg = OmegaConf.select(cfg, f"dataset_{split_name}", default=None)
            if val_ds_cfg is None:
                continue
            val_ds = hydra.utils.instantiate(val_ds_cfg)
            val_dataloaders[split_name] = self.make_distributed_dataloader(
                val_ds,
                cfg.dataloader,
                shuffle=False,
                drop_last=False,
                sanitize_worker_kwargs=sanitize_worker_kwargs,
            )
        return val_dataloaders

    @staticmethod
    def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
        set_epoch = getattr(getattr(dataloader, "sampler", None), "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def freeze_module_in_place(self, module: Any) -> Any:
        # Freeze module
        if module is None:
            return None
        if hasattr(module, "eval"):
            module.eval()
        if hasattr(module, "parameters"):
            for parameter in module.parameters():
                parameter.requires_grad = False
        return module

    def infer_hidden_dim_from_encoder(self, encoder: Any | None) -> int | None:
        # Encoder hidden dim
        if encoder is None:
            return None
        backbone = getattr(encoder, "backbone", None)
        config = getattr(backbone, "config", None)
        for attr_name in ("hidden_size", "d_model"):
            value = getattr(config, attr_name, None)
            if value is not None:
                return int(value)
        return None

    def infer_hidden_dim_from_dataset(self, dataset: Any) -> int | None:
        # Dataset hidden dim
        data_spec = getattr(dataset, "data_spec", None)
        value = getattr(data_spec, "hidden_dim", None)
        if value is not None:
            return int(value)
        return None

    def extract_hidden_from_obs(
        self,
        obs: Mapping[str, object],
        device: torch.device,
        fallback_hidden_dim: int | None = None,
    ) -> torch.Tensor:
        # Fallback hidden extraction
        state = obs.get("state")
        if isinstance(state, torch.Tensor) and state.numel() > 0:
            return state.to(device)

        proprio = obs.get("proprio")
        if isinstance(proprio, torch.Tensor) and proprio.numel() > 0:
            return proprio.to(device)

        image = obs.get("image")
        if isinstance(image, torch.Tensor) and image.numel() > 0:
            return image.flatten(start_dim=1).to(device)

        batch_size = 1
        task_id = obs.get("task_id")
        if isinstance(task_id, torch.Tensor) and task_id.ndim >= 1:
            batch_size = int(task_id.shape[0])

        hidden_dim = fallback_hidden_dim
        if hidden_dim is None:
            hidden_dim = int(
                OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1)
            )
        return torch.zeros(
            batch_size, int(hidden_dim), device=device, dtype=torch.float32
        )

    def attach_encoder_outputs(
        self,
        batch: dict[str, object],
        *,
        encoder: Any | None,
        device: torch.device,
        fallback_hidden_dim: int | None = None,
        detach: bool = True,
    ) -> dict[str, object]:
        # Encoder bridge
        obs = batch.get("obs")
        next_obs = batch.get("next_obs")

        if encoder is None:
            if isinstance(obs, Mapping):
                batch["obs_embedding"] = self.extract_hidden_from_obs(
                    obs,
                    device=device,
                    fallback_hidden_dim=fallback_hidden_dim,
                )
            if isinstance(next_obs, Mapping):
                try:
                    batch["next_obs_embedding"] = self.extract_hidden_from_obs(
                        next_obs,
                        device=device,
                        fallback_hidden_dim=fallback_hidden_dim,
                    )
                except ValueError:
                    if "obs_embedding" in batch and isinstance(
                        batch["obs_embedding"], torch.Tensor
                    ):
                        batch["next_obs_embedding"] = (
                            batch["obs_embedding"].detach().clone()
                        )
            return batch

        with torch.no_grad():
            if isinstance(obs, Mapping):
                obs_embedding = encoder.encode(obs)
                batch["obs_embedding"] = (
                    obs_embedding.detach() if detach else obs_embedding
                )
            if isinstance(next_obs, Mapping):
                next_obs_embedding = encoder.encode(next_obs)
                batch["next_obs_embedding"] = (
                    next_obs_embedding.detach() if detach else next_obs_embedding
                )
        return batch

    @staticmethod
    def slice_batch_mapping(
        mapping: Mapping[str, Any], indices: torch.Tensor
    ) -> dict[str, Any]:
        # Indexed slice for one mapping level plus one nested mapping level.
        sliced: dict[str, Any] = {}
        index_list = indices.tolist()
        for key, value in mapping.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value.index_select(0, indices)
                continue
            if isinstance(value, list):
                sliced[key] = [value[int(idx)] for idx in index_list]
                continue
            if isinstance(value, tuple):
                sliced[key] = tuple(value[int(idx)] for idx in index_list)
                continue
            if isinstance(value, Mapping):
                nested: dict[str, Any] = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, torch.Tensor):
                        nested[nested_key] = nested_value.index_select(0, indices)
                    elif isinstance(nested_value, list):
                        nested[nested_key] = [
                            nested_value[int(idx)] for idx in index_list
                        ]
                    elif isinstance(nested_value, tuple):
                        nested[nested_key] = tuple(
                            nested_value[int(idx)] for idx in index_list
                        )
                    else:
                        nested[nested_key] = nested_value
                sliced[key] = nested
                continue
            sliced[key] = value
        return sliced

    def make_history_entry(
        self,
        stage: str,
        step: int,
        metrics: Mapping[str, float],
    ) -> dict[str, float | str | int]:
        # History entry
        entry = {
            "stage": stage,
            "step": step,
            "global_step": self.global_step,
            "epoch": self.epoch,
            **metrics,
        }
        self.global_step += 1
        return entry

    def finish_epoch(self) -> None:
        # Epoch update
        self.epoch += 1

    @property
    def is_main_process(self) -> bool:
        # Distributed-aware main-process guard; returns True for non-distributed runners.
        distributed = getattr(self, "distributed", None)
        if distributed is not None:
            return bool(distributed.is_main_process)
        return True

    def resume(self, cfg: DictConfig | None = None) -> None:
        """Load latest checkpoint when training.resume=True."""
        if cfg is None:
            cfg = self.cfg
        if cfg.training.resume:
            explicit_resume_dir = OmegaConf.select(
                cfg, "training.resume_dir", default=None
            )
            if explicit_resume_dir is not None:
                resume_path = (
                    pathlib.Path(str(explicit_resume_dir)).expanduser().resolve()
                )
                if is_hf_checkpoint(resume_path):
                    self.load_hf_checkpoint(resume_path)
                    return
                if resume_path.is_dir():
                    hf_candidates = (
                        resume_path / "checkpoints" / "latest_hf",
                        resume_path / "latest_hf",
                    )
                    for hf_path in hf_candidates:
                        if is_hf_checkpoint(hf_path):
                            self.load_hf_checkpoint(hf_path)
                            return
                    ckpt_candidates = (
                        resume_path / "checkpoints" / "latest.ckpt",
                        resume_path / "ckpt" / "latest.ckpt",
                        resume_path / "latest.ckpt",
                    )
                    resume_path = next(
                        (candidate for candidate in ckpt_candidates if candidate.is_file()),
                        ckpt_candidates[0],
                    )
                if resume_path.is_file():
                    if self.is_main_process:
                        print(f"Resuming from checkpoint {resume_path}")
                    self.load_checkpoint(path=resume_path)
                    return
            lastest_ckpt_path = self.get_checkpoint_path(prefer_existing=True)
            if lastest_ckpt_path.is_file():
                if self.is_main_process:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                return
            latest_hf_path = self.get_hf_checkpoint_path(prefer_existing=True)
            if latest_hf_path.is_dir():
                self.load_hf_checkpoint(latest_hf_path)

    def print_history(self, history: list[dict[str, float | str | int]]) -> None:
        # Metric print
        for entry in history:
            stage = str(entry["stage"])
            step = int(entry["step"])
            global_step = int(entry["global_step"])
            epoch = int(entry["epoch"])
            metrics = " ".join(
                f"{key}={value:.4f}"
                for key, value in entry.items()
                if key not in {"stage", "step", "global_step", "epoch"}
            )
            print(
                f"{stage}_step={step} global_step={global_step} epoch={epoch} {metrics}"
            )

    def _ensure_metric_logger(self) -> Any:
        if self._metric_logger is not None:
            return self._metric_logger
        if not self.is_main_process:
            self._metric_logger = NullMetricLogger()
            return self._metric_logger

        output_name = pathlib.Path(self.output_dir).name or str(self.runner_name)
        self._metric_logger = MetricLogger(
            self.cfg,
            default_log_path=str(self.get_log_dir()),
            default_project_name="dreamervla",
            default_experiment_name=output_name,
        )
        return self._metric_logger

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
        prefix: str | None = None,
        backend: str | list[str] | tuple[str, ...] | None = None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ) -> None:
        """Send scalar metrics to the configured external metric backends."""
        prepared = self._prepare_metric_payload(metrics, prefix=prefix)
        if not prepared:
            return
        metric_step = self._resolve_metric_step(metrics, explicit_step=step)
        self._ensure_metric_logger().log(
            prepared,
            step=metric_step,
            backend=backend,
            worker_group_name=worker_group_name,
            rank=rank,
        )

    def finish_metric_logger(self) -> None:
        if self._metric_logger is None:
            return
        finish = getattr(self._metric_logger, "finish", None)
        if callable(finish):
            finish()

    def _prepare_metric_payload(
        self,
        metrics: Mapping[str, Any],
        *,
        prefix: str | None = None,
    ) -> dict[str, float]:
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            key_str = str(key)
            if key_str in {"global_step", "step", "epoch", "ts"}:
                continue
            scalar = self._coerce_metric_scalar(value)
            if scalar is None:
                continue
            metric_name = self._normalize_metric_name(key_str, prefix=prefix)
            if metric_name:
                payload[metric_name] = scalar
        return payload

    def _resolve_metric_step(
        self,
        metrics: Mapping[str, Any],
        *,
        explicit_step: int | None,
    ) -> int:
        if explicit_step is not None:
            return int(explicit_step)
        for key in ("global_step", "step", "epoch"):
            value = metrics.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, numbers.Number):
                return int(value)
        return int(self.global_step)

    @staticmethod
    def _coerce_metric_scalar(value: Any) -> float | None:
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

    @staticmethod
    def _normalize_metric_name(key: str, *, prefix: str | None = None) -> str:
        if "/" in key:
            return key
        if key.startswith("train_"):
            return f"train/{key[len('train_'):]}"
        if key.startswith("val_"):
            return f"eval/{key[len('val_'):]}"
        if key.startswith("eval_"):
            return f"eval/{key[len('eval_'):]}"
        if key.startswith("env_"):
            return f"env/{key[len('env_'):]}"
        if key.startswith("rollout_"):
            return f"rollout/{key[len('rollout_'):]}"
        if key.startswith("time_"):
            return f"time/{key[len('time_'):]}"
        if key.startswith("wall_"):
            return f"time/{key}"
        if prefix:
            return f"{prefix.strip('/')}/{key}"
        return f"train/{key}"

    def save_checkpoint(
        self,
        path: str | pathlib.Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        extra_paths: tuple[str | pathlib.Path, ...] = (),
    ) -> str:
        # Checkpoint path
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)
        extra = tuple(pathlib.Path(p) for p in extra_paths)

        # Payload config
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        distributed = getattr(self, "distributed", None)
        is_main_process = (
            True if distributed is None else bool(distributed.is_main_process)
        )
        requires_collective = (
            False
            if distributed is None
            else bool(getattr(distributed, "requires_collective_checkpointing", False))
        )
        if distributed is not None and not requires_collective and not is_main_process:
            return str(path.absolute())

        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "cfg": self.cfg,
            "state_dicts": {},
            "pickles": {},
        }

        # State dicts
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    state_dict = self._state_dict_for_checkpoint(key, value)
                    if is_main_process and state_dict is not None:
                        payload["state_dicts"][key] = _copy_to_cpu(state_dict)
            elif key in include_keys and is_main_process:
                payload["pickles"][key] = pickle.dumps(value)

        if is_main_process:
            # Serialize the payload ONCE (atomic temp-then-rename), then
            # materialize it at any extra destinations (top-k) by linking the
            # already-written file instead of re-running torch.save on the same
            # bytes. Sidecars are emitted per destination from the in-memory
            # payload.
            _atomic_torch_save(payload, path)
            self._save_checkpoint_sidecars(path, payload)
            for dst in extra:
                _materialize_checkpoint_copy(path, dst)
                self._save_checkpoint_sidecars(dst, payload)
        return str(path.absolute())

    def get_checkpoint_path(
        self,
        tag: str = "latest",
        *,
        prefer_existing: bool = False,
    ) -> pathlib.Path:
        # Checkpoint file
        canonical_path = self.get_checkpoint_dir().joinpath(f"{tag}.ckpt")
        if not prefer_existing:
            return canonical_path
        if canonical_path.is_file():
            return canonical_path
        legacy_path = self.get_legacy_checkpoint_dir().joinpath(f"{tag}.ckpt")
        if legacy_path.is_file():
            return legacy_path
        return canonical_path

    def get_hf_checkpoint_path(
        self,
        tag: str = "latest",
        *,
        prefer_existing: bool = False,
    ) -> pathlib.Path:
        # HF sidecar directory for VLA-compatible checkpoints.
        canonical_path = self.get_checkpoint_dir().joinpath(f"{tag}_hf")
        if not prefer_existing:
            return canonical_path
        if canonical_path.is_dir():
            return canonical_path
        legacy_path = self.get_legacy_checkpoint_dir().joinpath(f"{tag}_hf")
        if legacy_path.is_dir():
            return legacy_path
        return canonical_path

    def _save_checkpoint_sidecars(
        self, path: pathlib.Path, payload: dict[str, Any]
    ) -> None:
        return None

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        # Key filters
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        pickles = payload.get("pickles", {})
        state_dicts = payload.get("state_dicts", {})
        if include_keys is None:
            include_keys = tuple(pickles.keys())
            if not bool(getattr(self, "checkpoint_restore_output_dir", False)):
                include_keys = tuple(
                    key for key in include_keys if key != "_output_dir"
                )

        # State restore
        for key, value in state_dicts.items():
            if (
                key not in exclude_keys
                and key in self.__dict__
                and self.__dict__[key] is not None
            ):
                self._load_state_dict_from_checkpoint(
                    key, self.__dict__[key], value, **kwargs
                )

        # Pickle restore
        for key in include_keys:
            if key in pickles:
                self.__dict__[key] = pickle.loads(pickles[key])

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        return value.state_dict()

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        value.load_state_dict(state_dict, **kwargs)

    def load_checkpoint(
        self,
        path: str | pathlib.Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Checkpoint path
        if path is None:
            path = self.get_checkpoint_path(tag=tag, prefer_existing=True)
        path = pathlib.Path(path)

        if is_hf_checkpoint(path):
            payload = self.load_hf_checkpoint(path, **kwargs)
            return payload

        payload = load_runner_payload(path, **kwargs)
        self.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
        )
        return payload

    def load_hf_checkpoint(
        self,
        path: str | pathlib.Path,
        **_: Any,
    ) -> dict[str, Any]:
        raise RuntimeError(
            f"{type(self).__name__} does not support loading Hugging Face checkpoints: {path}"
        )

    def setup(self) -> None:
        """Optional lifecycle hook before execution.

        Existing runners build most state inside ``run``.  The hook gives
        new runners a common interface without forcing a large rewrite of
        the current training loops.
        """
        self.write_run_artifacts()
        return None

    def execute(self) -> object:
        """Run through the public lifecycle interface."""
        return self.run()

    def teardown(self) -> None:
        """Optional lifecycle hook after execution."""
        st = getattr(self, "_console_state", None)
        if st is not None:
            for rep in st.get("progress", {}).values():
                rep.close()
        self.finish_metric_logger()

    def _console_state_get(self) -> dict:
        # Lazily build + cache the console state. All knobs are Hydra overrides
        # (no config file required; read via OmegaConf.select with defaults):
        #   console.banner_width   (int, default 65)  — width of === banners / metric boxes
        #   console.log_every      (int, default 1)   — print the metric box every N console_metrics()
        #                                               calls (floored at 1); pass force=True to bypass
        #                                               for one-shot summary boxes
        #   console.success_window (int, default 50)  — episodes in the VLA success-rate window
        #   console.progress_every_s (float, default 5.0) — min wall-time between progress lines
        #                                               (0 prints every call); gates console_progress()
        # Override per run, e.g. `console.log_every=50 console.success_window=100`.
        st = getattr(self, "_console_state", None)
        if st is None:
            st = {
                "width": int(OmegaConf.select(self.cfg, "console.banner_width", default=65)),
                "log_every": max(1, int(OmegaConf.select(self.cfg, "console.log_every", default=1))),
                "window": int(OmegaConf.select(self.cfg, "console.success_window", default=50)),
                "progress_every_s": float(
                    OmegaConf.select(self.cfg, "console.progress_every_s", default=5.0)
                ),
                "counter": 0,
                "tracker": None,
                "progress": {},
            }
            self._console_state = st
        return st

    def console_progress(
        self,
        current: int,
        total: int | None,
        desc: str,
        *,
        unit: str = "it",
        status: str | None = None,
    ) -> None:
        # One uniform progress line per flow. Caches a wall-time-throttled
        # ProgressReporter per ``desc`` so every loop reports identically; the
        # reporter is main-process-guarded and closed in ``teardown``.
        st = self._console_state_get()
        reporters = st["progress"]
        rep = reporters.get(desc)
        if rep is None:
            rep = ProgressReporter(
                total,
                desc,
                enabled=self.is_main_process,
                min_interval_s=st["progress_every_s"],
                unit=unit,
                status=status,
            )
            reporters[desc] = rep
        else:
            rep.set_status(status)
        rep.set(current)

    def console_banner(self, title: str, *, subtitle: str | None = None, done: bool = False) -> None:
        if not self.is_main_process:
            return
        st = self._console_state_get()
        print(phase_banner(title, subtitle=subtitle, done=done, width=st["width"]), flush=True)

    def console_record_success(self, success: bool) -> None:
        st = self._console_state_get()
        if st["tracker"] is None:
            st["tracker"] = SuccessTracker(window=st["window"])
        st["tracker"].update(bool(success))

    def console_rollout_episode(
        self,
        *,
        episode: int,
        success: bool,
        avg_success_rate: float,
        window_success_rate: float,
    ) -> None:
        if not self.is_main_process:
            return
        print(
            f"[rollout] episode={int(episode)} success={int(bool(success))} "
            f"avg_success_rate={float(avg_success_rate):.3f} "
            f"window_success_rate={float(window_success_rate):.3f}",
            flush=True,
        )

    def console_metrics(self, header: str, metrics: dict, *, force: bool = False) -> None:
        if not self.is_main_process:
            return
        st = self._console_state_get()
        st["counter"] += 1
        if not force and st["counter"] % st["log_every"] != 0:
            return
        tr = st["tracker"]
        rows: list[str] = []
        if tr is not None and len(tr) > 0:
            rows.append(
                f"VLA     succ@{st['window']}={fmt_value(tr.rate())} "
                f"(Δ {tr.delta():+.3f} · best {tr.best:.3f})"
            )
        rows.extend(_group_metric_rows(metrics, skip_success=tr is not None))
        print(metric_box(header, rows, width=st["width"]), flush=True)
        if tr is not None:
            tr.mark_printed()

    def console_success_rate(self) -> float:
        st = self._console_state_get()
        tr = st["tracker"]
        return tr.rate() if tr is not None else 0.0

    @abstractmethod
    def run(self) -> object:
        # Runner API
        raise NotImplementedError


def _atomic_torch_save(payload: Any, path: pathlib.Path) -> None:
    # Temp-then-rename write so a crash mid-write never leaves a half-written
    # checkpoint at the destination (mirrors _dreamer_runner_common._save_ckpt).
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _materialize_checkpoint_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    # Place the already-serialized checkpoint at an additional destination
    # without re-serializing: hardlink when possible, fall back to a file copy
    # (e.g. cross-filesystem). Atomic via a temp-then-rename either way.
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        os.link(src, tmp_path)
    except OSError:
        shutil.copyfile(src, tmp_path)
    tmp_path.replace(dst)


def _copy_to_cpu(value: Any) -> Any:
    # CPU copy
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    return copy.deepcopy(value)


def _mapping_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalise_logger_backends(value: Any) -> list[str]:
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
        backend_name = str(backend).strip().lower()
        if backend_name in {"", "none", "null", "false", "off", "disabled"}:
            continue
        backends.append(backend_name)
    return backends
