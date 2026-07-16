"""LatentSuccessClassifier training runner (LUMOS-aligned).

Launch path:
    python -m dreamervla.train \
        experiment=wmpo_token_classifier_openvla_onetraj_libero_goal_h1 \
        task=openvla_onetraj_libero
        → dreamervla.runners.SuccessClassifierTrainingRunner.run()
            → dreamervla.dataset.lumos_aligned_latent_dataset
            → dreamervla.algorithms.critic.LatentSuccessClassifier

Why a dedicated runner, not another standalone script:
  * The existing v3 / wm_replay classifier scripts are 500+ lines each and bypass
    BaseRunner, so resume / checkpoint / Hydra-override semantics don't
    transfer. A runner fixes that.
  * Decouples sampling protocol (dataset), model (LatentSuccessClassifier head_type),
    and training loop (this runner) — so head_type ablation is a 1-line config
    override, not a script fork.

The training loop is epoch-based:
  * Resampled train loader → ``cfg.training.num_epochs`` passes, default 20
  * Eval every ``cfg.training.eval_every`` steps; window F1 + (optional) episode F1
  * Best ckpt saved by val window F1 (softmax + threshold sweep, LUMOS protocol)
  * Final ckpt saved after the last epoch

Window-level F1 uses softmax + threshold sweep to mirror ``predict_success``
(note: LUMOS sweep is [0.3, 1.0]; we expose the bounds via cfg).
Episode-level F1 mirrors ``predict_success`` (stride-1 sliding window +
``any-positive`` aggregation).

The runner owns resume, checkpointing, logging, and Hydra override behavior so
classifier training follows the same contract as WM and DreamerVLA routes.
"""

from __future__ import annotations

import json
import pathlib
import time
from itertools import islice
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from dreamervla.algorithms.critic import (
    LatentSuccessClassifier,
    LatentSuccessClassifierConfig,
)
from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.preprocess.sidecar_schema import validate_hidden_token_sidecar_dir
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runtime.classifier_metrics import sweep_threshold_metrics as _sweep_metrics
from dreamervla.runtime.distributed import NopretokenizeSFTDistributedHelper
from dreamervla.utils.checkpoint_util import TopKCheckpointManager
from dreamervla.utils.torch_utils import autocast_context
from dreamervla.utils.update_timing import GradientUpdateTimer

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _unpack_classifier_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        xs, ys, extra = batch
        return xs, ys, dict(extra or {})
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        xs, ys = batch
        return xs, ys, {}
    raise TypeError(f"unexpected classifier batch type: {type(batch).__name__}")


def _classifier_forward_kwargs(
    model: LatentSuccessClassifier,
    extra: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    if bool(getattr(model, "supports_proprio_conditioning", False)):
        proprio = extra.get("proprio")
        if not isinstance(proprio, torch.Tensor):
            raise ValueError("classifier requires proprio conditioning, but batch has no proprio")
        kwargs["proprio"] = proprio.to(device, non_blocking=True)
    if bool(getattr(model, "supports_language_conditioning", False)):
        lang_emb = extra.get("lang_emb")
        if not isinstance(lang_emb, torch.Tensor):
            raise ValueError("classifier requires language conditioning, but batch has no lang_emb")
        kwargs["lang_emb"] = lang_emb.to(device, non_blocking=True)
    return kwargs


def _success_probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"classifier logits must be [B,C], got {tuple(logits.shape)}")
    if int(logits.shape[-1]) == 1:
        return torch.sigmoid(logits.squeeze(-1))
    if int(logits.shape[-1]) == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    raise ValueError(f"classifier logits last dim must be 1 or 2, got {logits.shape[-1]}")


def _classifier_loss_and_predictions(
    logits: torch.Tensor,
    ys: torch.Tensor,
    *,
    loss_type: str,
    label_smoothing: float,
    class_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_type = str(loss_type)
    if loss_type == "bce":
        if int(logits.shape[-1]) != 1:
            raise ValueError("BCE classifier loss requires output_dim=1")
        targets = ys.to(dtype=logits.dtype)
        if label_smoothing:
            eps = float(label_smoothing)
            targets = targets * (1.0 - eps) + 0.5 * eps
        pos_weight = None
        if class_weight is not None:
            if int(class_weight.numel()) == 2:
                pos_weight = class_weight[1].reshape(())
            elif int(class_weight.numel()) == 1:
                pos_weight = class_weight.reshape(())
            else:
                raise ValueError("BCE class_weight must have one or two elements")
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            targets,
            pos_weight=pos_weight,
        )
        pred = (_success_probabilities_from_logits(logits) >= 0.5).long()
        return loss, pred
    if loss_type == "ce":
        if int(logits.shape[-1]) != 2:
            raise ValueError("CE classifier loss requires output_dim=2")
        loss = F.cross_entropy(
            logits,
            ys.long(),
            weight=class_weight,
            label_smoothing=float(label_smoothing),
        )
        return loss, logits.argmax(dim=-1)
    raise ValueError(f"unknown classifier loss_type {loss_type!r} (bce|ce)")


class SuccessClassifierTrainingRunner(BaseRunner):
    """Epoch-based trainer for LatentSuccessClassifier.

    Lifecycle: setup() → run() → teardown() (via BaseRunner.execute).
    """

    runner_name = "latent_classifier"
    runner_status = "current"
    runner_family = "reward"
    include_keys = (
        "global_step",
        "epoch",
        "best_window_f1",
        "best_episode_f1",
        "best_window_ckpt_path",
        "best_episode_ckpt_path",
        "best_window_threshold",
        "best_episode_threshold",
        "classifier_threshold",
    )

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        strategy = str(
            OmegaConf.select(
                config,
                "training.distributed_strategy",
                default="ddp",
            )
            or "ddp"
        ).lower()
        if strategy == "single":
            strategy = "ddp"
        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=strategy,
            fsdp_mixed_precision=str(
                OmegaConf.select(
                    config,
                    "training.fsdp_mixed_precision",
                    default="bf16",
                )
                or "bf16"
            ),
            enable_activation_checkpointing=bool(
                OmegaConf.select(
                    config,
                    "training.activation_checkpointing",
                    default=True,
                )
            ),
            nccl_timeout_seconds=OmegaConf.select(
                config,
                "training.nccl_timeout_seconds",
                default=None,
            ),
        )
        self.rank = self.distributed.rank
        self.local_rank = self.distributed.local_rank
        self.world_size = self.distributed.world_size
        self.device = self.distributed.resolve_device(
            str(OmegaConf.select(self.cfg, "training.device") or "cuda")
        )
        self.train_ds: object | None = None
        self.val_ds: object | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.model: LatentSuccessClassifier | None = None
        self.optim: torch.optim.Optimizer | None = None
        self.best_window_f1: float = -1.0
        self.best_episode_f1: float = -1.0
        self.best_window_ckpt_path: str | None = None
        self.best_episode_ckpt_path: str | None = None
        self.best_window_threshold = 0.5
        self.best_episode_threshold = 0.5
        self.classifier_threshold = 0.5
        self._log_path: pathlib.Path | None = None
        self._pending_setup_logs: list[dict[str, Any]] = []

    def _make_classifier_loader(
        self,
        dataset: object,
        *,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        drop_last: bool,
        use_distributed_sampler: bool,
    ) -> DataLoader:
        """Build a classifier dataloader without attaching samplers to streams."""

        collate_fn = getattr(dataset, "collate_fn", None)
        dataloader_kwargs: dict[str, Any] = {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": True,
            "drop_last": bool(drop_last),
        }
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        if not isinstance(dataset, IterableDataset):
            dataloader_kwargs["shuffle"] = bool(shuffle)
            if use_distributed_sampler:
                sampler = self.distributed.maybe_make_sampler(
                    dataset,
                    shuffle=bool(shuffle),
                    drop_last=bool(drop_last),
                )
                if sampler is not None:
                    dataloader_kwargs["shuffle"] = False
                    dataloader_kwargs["sampler"] = sampler
        return DataLoader(dataset, **dataloader_kwargs)

    def _prepare_train_dataset_for_distributed(self, dataset: object) -> None:
        if not isinstance(dataset, IterableDataset):
            return
        dataset.distributed_rank = int(self.distributed.rank)
        dataset.distributed_world_size = int(self.distributed.world_size)

    def _classifier_module(self) -> LatentSuccessClassifier:
        assert self.model is not None
        distributed = getattr(self, "distributed", None)
        if distributed is None:
            return self.model
        return distributed.unwrap_module(self.model)

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "model":
            return self.distributed.unwrap_module(value).state_dict()
        return super()._state_dict_for_checkpoint(key, value)

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key == "model":
            self.distributed.unwrap_module(value).load_state_dict(state_dict, **kwargs)
            return
        super()._load_state_dict_from_checkpoint(key, value, state_dict, **kwargs)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "classifier_threshold": float(getattr(self, "classifier_threshold", 0.5)),
            "best_window_f1": float(getattr(self, "best_window_f1", -1.0)),
            "best_episode_f1": float(getattr(self, "best_episode_f1", -1.0)),
        }

    def load_payload(
        self,
        payload: dict[str, Any],
        *,
        restore_rng: bool = False,
        **kwargs: Any,
    ) -> None:
        version = payload.get("format_version")
        if isinstance(version, int) and version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint payload has format_version={version}, but this build supports "
                f"up to {CHECKPOINT_FORMAT_VERSION}; upgrade DreamerVLA to load it."
            )
        if isinstance(version, int) and version >= 2:
            state_dicts = payload.get("state_dicts")
            pickles = payload.get("pickles")
            if not isinstance(state_dicts, dict) or "model" not in state_dicts:
                raise RuntimeError("format v2 classifier checkpoint is missing model state")
            if "optim" not in state_dicts:
                raise RuntimeError("format v2 classifier checkpoint is missing optim state")
            if (
                not isinstance(pickles, dict)
                or "global_step" not in pickles
                or "epoch" not in pickles
            ):
                raise RuntimeError(
                    "format v2 classifier checkpoint is missing global_step/epoch progress"
                )
        super().load_payload(payload, restore_rng=restore_rng, **kwargs)

    # --------------------------- setup ---------------------------------

    def setup(self) -> None:
        super().setup()
        torch.manual_seed(int(OmegaConf.select(self.cfg, "training.seed") or 0))
        torch.backends.cudnn.benchmark = True

        ckpt_dir = self.get_checkpoint_dir()
        log_dir = self.get_log_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "train_log.jsonl"
        self._pending_setup_logs = []

        self._pending_setup_logs.append(
            {
                "event": "setup_begin",
                "output_dir": str(self.output_dir),
                "device": str(self.device),
                "head_type": str(OmegaConf.select(self.cfg, "classifier.head_type") or "linear"),
            }
        )

        # ----- datasets ---------------------------------------------------
        d = self.cfg.data
        require_reference_complete = bool(
            OmegaConf.select(d, "require_reference_complete", default=True)
        )
        validate_hidden_token_sidecar_dir(
            d.success_dir_hidden,
            reference_dir=d.success_dir_raw,
            require_reference_complete=require_reference_complete,
        )
        failure_hidden_dir = OmegaConf.select(d, "failure_dir_hidden")
        if failure_hidden_dir is not None:
            failure_raw_dir = OmegaConf.select(d, "failure_dir_raw")
            if failure_raw_dir is None:
                raise ValueError("failure_dir_hidden requires failure_dir_raw")
            validate_hidden_token_sidecar_dir(
                failure_hidden_dir,
                reference_dir=failure_raw_dir,
                require_reference_complete=require_reference_complete,
            )
        # Chunk-level (Type B) classifier: each window frame is pooled from
        # K = chunk_subsample env-step frames. Defaults reduce to env-step
        # (action) granularity for backwards compatibility.
        chunk_subsample = int(OmegaConf.select(d, "chunk_subsample") or 1)
        chunk_pool = str(OmegaConf.select(d, "chunk_pool") or "last")
        sampling_protocol = str(OmegaConf.select(d, "sampling_protocol") or "lumos")
        balance_batches = bool(OmegaConf.select(d, "balance_batches") or False)
        if bool(
            OmegaConf.select(
                d,
                "require_sidecar_contract",
                default=False,
            )
        ):
            validate_hidden_token_sidecar_dir(
                d.success_dir_hidden,
                expected_filenames=OmegaConf.select(
                    d,
                    "required_filenames",
                    default=None,
                ),
                reference_dir=d.success_dir_raw,
                require_reference_complete=require_reference_complete,
                require_sparse_rewards=True,
            )
            failure_raw = OmegaConf.select(d, "failure_dir_raw", default=None)
            failure_hidden = OmegaConf.select(d, "failure_dir_hidden", default=None)
            if failure_raw is not None or failure_hidden is not None:
                if failure_raw is None or failure_hidden is None:
                    raise ValueError("failure sidecar validation requires both raw and hidden dirs")
                validate_hidden_token_sidecar_dir(
                    failure_hidden,
                    reference_dir=failure_raw,
                    require_reference_complete=require_reference_complete,
                    require_sparse_rewards=True,
                )
        val_fraction = float(OmegaConf.select(d, "val_fraction", default=0.2))
        split_seed = int(
            OmegaConf.select(
                d,
                "split_seed",
                default=OmegaConf.select(self.cfg, "training.seed", default=0),
            )
        )
        train_dataset_cfg = OmegaConf.select(self.cfg, "task.classifier.dataset.train")
        validation_dataset_cfg = OmegaConf.select(self.cfg, "task.classifier.dataset.validation")
        if train_dataset_cfg is None or validation_dataset_cfg is None:
            raise ValueError(
                "classifier training requires Hydra-selected "
                "task.classifier.dataset.train and task.classifier.dataset.validation"
            )
        self.train_ds = hydra.utils.instantiate(
            train_dataset_cfg,
            success_dir_raw=d.success_dir_raw,
            success_dir_hidden=d.success_dir_hidden,
            failure_dir_raw=OmegaConf.select(d, "failure_dir_raw"),
            failure_dir_hidden=OmegaConf.select(d, "failure_dir_hidden"),
            window=int(d.window),
            stride=int(d.stride_train),
            seed=int(OmegaConf.select(self.cfg, "training.seed") or 0),
            chunk_subsample=chunk_subsample,
            chunk_pool=chunk_pool,
            proprio_keys=OmegaConf.select(d, "proprio_keys", default=None),
            lang_emb_dir=OmegaConf.select(d, "lang_emb_dir", default=None),
            lang_emb_key=str(OmegaConf.select(d, "lang_emb_key", default="lang_emb")),
            sampling_protocol=sampling_protocol,
            balance_batches=balance_batches,
            demo_split=str(OmegaConf.select(d, "train_split", default="all")),
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
        self.val_ds = hydra.utils.instantiate(
            validation_dataset_cfg,
            success_dir_raw=d.success_dir_raw,
            success_dir_hidden=d.success_dir_hidden,
            failure_dir_raw=OmegaConf.select(d, "failure_dir_raw"),
            failure_dir_hidden=OmegaConf.select(d, "failure_dir_hidden"),
            window=int(d.window),
            stride=int(d.stride_val),
            chunk_subsample=chunk_subsample,
            chunk_pool=chunk_pool,
            proprio_keys=OmegaConf.select(d, "proprio_keys", default=None),
            lang_emb_dir=OmegaConf.select(d, "lang_emb_dir", default=None),
            lang_emb_key=str(OmegaConf.select(d, "lang_emb_key", default="lang_emb")),
            sampling_protocol=sampling_protocol,
            demo_split=str(OmegaConf.select(d, "val_split", default="all")),
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
        self._prepare_train_dataset_for_distributed(self.train_ds)
        self._pending_setup_logs.append(self._dataset_summary_payload("train", self.train_ds))
        self._pending_setup_logs.append(self._dataset_summary_payload("val", self.val_ds))

        # ----- dataloaders -----------------------------------------------
        tr = self.cfg.training
        if sampling_protocol == "wmpo" and balance_batches and int(tr.batch_size) % 2:
            raise ValueError("WMPO batch-balanced sampling requires an even training.batch_size")
        if (
            sampling_protocol == "wmpo"
            and balance_batches
            and int(OmegaConf.select(tr, "num_workers") or 0) != 0
        ):
            raise ValueError(
                "WMPO batch-balanced sampling requires training.num_workers=0 "
                "so consecutive positive/negative samples form exact batches"
            )
        self.train_loader = self._make_classifier_loader(
            self.train_ds,
            batch_size=int(tr.batch_size),
            num_workers=int(OmegaConf.select(tr, "num_workers") or 0),
            shuffle=False,
            drop_last=True,
            use_distributed_sampler=False,
        )
        self.val_loader = self._make_classifier_loader(
            self.val_ds,
            batch_size=int(OmegaConf.select(tr, "val_batch_size") or 256),
            num_workers=0,  # val data is in-RAM; workers add overhead
            shuffle=False,
            drop_last=False,
            use_distributed_sampler=False,
        )

        # ----- model -----------------------------------------------------
        cfg_dict = OmegaConf.to_container(self.cfg.classifier, resolve=True)
        # only keep keys LatentSuccessClassifierConfig accepts
        valid_keys = LatentSuccessClassifierConfig.__dataclass_fields__.keys()
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cls_cfg = LatentSuccessClassifierConfig(**cfg_dict)
        loss_type_for_cfg = str(OmegaConf.select(self.cfg, "training.loss_type") or "ce")
        if loss_type_for_cfg == "bce" and int(cls_cfg.output_dim) != 1:
            raise ValueError("training.loss_type=bce requires classifier.output_dim=1")
        if loss_type_for_cfg == "ce" and int(cls_cfg.output_dim) != 2:
            raise ValueError("training.loss_type=ce requires classifier.output_dim=2")
        if int(cls_cfg.window) != int(d.window):
            raise ValueError(f"classifier.window ({cls_cfg.window}) != data.window ({d.window})")
        # Chunk granularity consistency: classifier.cfg.chunk_size must match
        # data.chunk_subsample, otherwise the windows produced by the dataset
        # have a different time-coverage than what the classifier expects at
        # inference time (predict_success internally subsamples by chunk_size).
        if str(getattr(cls_cfg, "granularity", "action")) == "chunk":
            if int(cls_cfg.chunk_size) != chunk_subsample:
                raise ValueError(
                    f"classifier.chunk_size ({cls_cfg.chunk_size}) != "
                    f"data.chunk_subsample ({chunk_subsample})"
                )
            if str(cls_cfg.chunk_pool) != chunk_pool:
                raise ValueError(
                    f"classifier.chunk_pool ({cls_cfg.chunk_pool!r}) != "
                    f"data.chunk_pool ({chunk_pool!r})"
                )
        elif chunk_subsample != 1:
            raise ValueError(
                f"data.chunk_subsample={chunk_subsample} requires classifier.granularity='chunk'"
            )
        classifier_target = OmegaConf.select(self.cfg, "classifier._target_", default=None)
        if classifier_target:
            self.model = hydra.utils.instantiate(self.cfg.classifier).to(self.device)
        else:
            self.model = LatentSuccessClassifier(cls_cfg).to(self.device)
        self.model = self.distributed.wrap_trainable_module(
            self.model,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._pending_setup_logs.append(
            {
                "event": "model_built",
                "head_type": str(cls_cfg.head_type),
                "n_params": int(n_params),
                "n_trainable": int(n_trainable),
            }
        )

        # ----- optimizer + (optional) scheduler --------------------------
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tr.lr),
            weight_decay=float(OmegaConf.select(tr, "weight_decay") or 1e-4),
        )

        self._finish_setup_after_optimizer()

    def _prepare_train_log(self, *, resume: bool) -> None:
        """Prepare classifier JSONL without truncating a resumed run."""
        if not self.is_main_process or self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self._log_path.write_text("", encoding="utf-8")

    def _finish_setup_after_optimizer(self) -> None:
        """Restore progress before initializing any metric or JSON logger."""
        is_resume = bool(OmegaConf.select(self.cfg, "training.resume", default=False))
        if is_resume:
            self.resume(self.cfg)
            self.set_metric_resume_step(int(self.global_step))
        self._prepare_train_log(resume=is_resume)
        for payload in list(getattr(self, "_pending_setup_logs", [])):
            self._log(payload)
        self._pending_setup_logs = []
        self._log({"event": "setup_done"})

    # --------------------------- run -----------------------------------

    @staticmethod
    def _dataset_summary_payload(split: str, dataset: object) -> dict[str, object]:
        summary_fn = getattr(dataset, "summary", None)
        summary = summary_fn() if callable(summary_fn) else {}
        return {"event": "dataset_summary", "split": str(split), **dict(summary)}

    def _should_profile_update(self) -> bool:
        profile_steps = int(
            OmegaConf.select(
                self.cfg,
                "training.update_profile_steps",
                default=0,
            )
            or 0
        )
        return profile_steps < 0 or int(self.global_step) < profile_steps

    def _record_update_profile(self, timings: dict[str, float]) -> None:
        device_stages = ("h2d", "forward", "backward", "grad_clip", "optimizer")
        device_active = sum(float(timings.get(name, 0.0)) for name in device_stages)
        total = float(timings.get("total", 0.0))
        enriched = dict(timings)
        enriched["device_active"] = device_active
        enriched["host_or_wait"] = max(0.0, total - device_active)
        enriched["device_active_fraction"] = device_active / max(total, 1.0e-12)
        metrics = {
            f"time/classifier_update_{name}_ms": float(value) * 1000.0
            for name, value in enriched.items()
            if name != "device_active_fraction"
        }
        metrics["time/classifier_update_device_active_fraction"] = enriched[
            "device_active_fraction"
        ]
        self.log_metrics(metrics, step=int(self.global_step))
        if not self.is_main_process:
            return
        order = (
            "data_wait",
            "h2d",
            "forward",
            "backward",
            "grad_clip",
            "optimizer",
            "metrics",
            "total",
        )
        parts = [
            f"{name}={float(enriched[name]) * 1000.0:.1f}ms" for name in order if name in enriched
        ]
        parts.append(f"device_active={float(enriched['device_active_fraction']) * 100.0:.1f}%")
        print(
            f"[classifier-profile] step={int(self.global_step)} " + " ".join(parts),
            flush=True,
        )

    def _finalize_validation_checkpoints(self) -> dict[str, dict[str, Any]]:
        """Apply the Hydra-selected final checkpoint selection protocol."""

        selection = str(
            OmegaConf.select(
                self.cfg,
                "training.final_selection_metric",
                default="none",
            )
        ).lower()
        if selection == "none":
            return {}
        if selection == "window_f1":
            metrics = self._evaluate_window_level()
            if float(metrics["best_f1"]) > float(self.best_window_f1):
                self.best_window_f1 = float(metrics["best_f1"])
            self._maybe_save_named(
                "best_window_"
                f"f1{float(metrics['best_f1']):.4f}_"
                f"th{float(metrics['best_thresh']):.2f}",
                extra={"val_window": metrics},
            )
            return {"window": metrics}
        if selection == "episode_f1":
            if not bool(
                OmegaConf.select(
                    self.cfg,
                    "training.episode_eval_enabled",
                    default=False,
                )
            ):
                raise ValueError(
                    "final_selection_metric=episode_f1 requires training.episode_eval_enabled=true"
                )
            metrics = self._evaluate_episode_level()
            if float(metrics["best_f1"]) > float(self.best_episode_f1):
                self.best_episode_f1 = float(metrics["best_f1"])
            self._maybe_save_named(
                "best_episode_"
                f"f1{float(metrics['best_f1']):.4f}_"
                f"th{float(metrics['best_thresh']):.2f}",
                extra={"val_episode": metrics},
            )
            return {"episode": metrics}
        raise ValueError(
            "training.final_selection_metric must be one of: none, window_f1, episode_f1"
        )

    def run(self) -> dict[str, float]:
        assert self.model is not None and self.optim is not None
        assert self.train_loader is not None and self.val_loader is not None

        tr = self.cfg.training
        num_epochs_cfg = OmegaConf.select(tr, "num_epochs", default=20)
        num_epochs = 20 if num_epochs_cfg is None else int(num_epochs_cfg)
        eval_every = int(OmegaConf.select(tr, "eval_every") or 500)
        checkpoint_every_epochs = int(
            OmegaConf.select(tr, "checkpoint_every_epochs", default=1) or 0
        )
        log_every = int(OmegaConf.select(tr, "log_every") or 50)
        steps_per_epoch_cfg = int(OmegaConf.select(tr, "steps_per_epoch") or 0)
        steps_per_epoch = (
            steps_per_epoch_cfg if steps_per_epoch_cfg > 0 else max(1, len(self.train_loader))
        )
        label_smoothing = float(OmegaConf.select(tr, "label_smoothing") or 0.0)
        loss_type = str(OmegaConf.select(tr, "loss_type") or "ce")

        # class-balanced CE (matches LUMOS `nn.CrossEntropyLoss()` *unweighted* by
        # default; user can flip via cfg.training.class_balanced)
        class_balanced = bool(OmegaConf.select(tr, "class_balanced") or False)
        if class_balanced:
            n_succ = sum(1 for d in self.train_ds._demos if d.complete)
            n_fail = len(self.train_ds._demos) - n_succ
            n_pos = n_succ
            n_neg = n_succ + 2 * n_fail
            cw = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=self.device)
            self._log(
                {
                    "event": "class_balanced",
                    "class_weight": [1.0, float(n_neg / max(n_pos, 1))],
                }
            )
        else:
            cw = None
        running_loss = torch.zeros((), device=self.device)
        running_correct = torch.zeros((), device=self.device)
        running_total = 0
        precision = str(OmegaConf.select(tr, "precision", default="fp32") or "fp32")

        t0 = time.time()
        self.console_banner("TRAINING", subtitle=f"{num_epochs} epochs")
        while self.epoch < num_epochs:
            self.set_dataloader_epoch(self.train_loader, self.epoch)
            data_wait_started_at = time.perf_counter()
            for batch in islice(self.train_loader, steps_per_epoch):
                update_started_at = data_wait_started_at
                data_wait_s = time.perf_counter() - data_wait_started_at
                profile = self._should_profile_update()
                timer = GradientUpdateTimer(self.device, enabled=profile)
                profile_timings = {"data_wait": data_wait_s} if profile else {}

                with timer.device_stage("h2d"):
                    xs, ys, extra = _unpack_classifier_batch(batch)
                    xs = xs.to(self.device, non_blocking=True)
                    ys = ys.to(self.device, non_blocking=True)
                    forward_kwargs = _classifier_forward_kwargs(
                        self._classifier_module(), extra, self.device
                    )

                self.optim.zero_grad(set_to_none=True)
                self.model.train()
                with timer.device_stage("forward"):
                    with autocast_context(self.device, precision):
                        logits = self.model(xs, **forward_kwargs)
                        loss, pred = _classifier_loss_and_predictions(
                            logits,
                            ys,
                            loss_type=loss_type,
                            label_smoothing=label_smoothing,
                            class_weight=cw,
                        )
                with timer.device_stage("backward"):
                    loss.backward()
                with timer.device_stage("grad_clip"):
                    clip_tensor = getattr(
                        self.distributed,
                        "clip_grad_norm_tensor",
                        None,
                    )
                    if callable(clip_tensor):
                        grad_norm = clip_tensor(self.model, 5.0)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                with timer.device_stage("optimizer"):
                    self.optim.step()

                timer.synchronize_device()
                with timer.wall_stage("metrics"), torch.no_grad():
                    running_correct += (pred.detach() == ys).sum()
                    running_total += int(ys.numel())
                    running_loss += loss.detach()

                self.global_step += 1
                if profile:
                    profile_timings.update(timer.finish())
                    profile_timings["total"] = time.perf_counter() - update_started_at
                    self._record_update_profile(profile_timings)
                self.console_progress(self.global_step, num_epochs * steps_per_epoch, "train")

                if self.global_step % log_every == 0:
                    step_loss = running_loss / log_every
                    step_acc = running_correct / max(running_total, 1)
                    reduced = self.distributed.reduce_mean_dict(
                        {
                            "loss": step_loss,
                            "acc": step_acc,
                            "grad_norm": grad_norm,
                        }
                    )
                    self._log(
                        {
                            "event": "train_step",
                            "step": self.global_step,
                            "epoch": self.epoch,
                            "loss": reduced["loss"],
                            "acc": reduced["acc"],
                            "grad_norm": reduced["grad_norm"],
                            "wall_s": time.time() - t0,
                        }
                    )
                    self.console_metrics(
                        f"train · epoch {self.epoch}",
                        {
                            "train/loss": float(reduced["loss"]),
                            "train/acc": float(reduced["acc"]),
                        },
                    )
                    running_loss = torch.zeros((), device=self.device)
                    running_correct = torch.zeros((), device=self.device)
                    running_total = 0

                # ---- periodic eval ---------------------------------------
                maintenance_metrics: dict[str, float] = {}
                if self.global_step % eval_every == 0:
                    eval_started_at = time.perf_counter()
                    w_metrics = self._evaluate_window_level()
                    self._log({"event": "val_window", "step": self.global_step, **w_metrics})
                    if w_metrics["best_f1"] > self.best_window_f1:
                        self.best_window_f1 = float(w_metrics["best_f1"])
                        self.best_window_threshold = float(w_metrics["best_thresh"])

                    if bool(OmegaConf.select(tr, "episode_eval_enabled") or False):
                        e_metrics = self._evaluate_episode_level()
                        self._log(
                            {
                                "event": "val_episode",
                                "step": self.global_step,
                                **e_metrics,
                            }
                        )
                        if e_metrics["best_f1"] > self.best_episode_f1:
                            self.best_episode_f1 = float(e_metrics["best_f1"])
                            self.best_episode_threshold = float(e_metrics["best_thresh"])
                    maintenance_metrics["time/classifier_eval_s"] = (
                        time.perf_counter() - eval_started_at
                    )

                if maintenance_metrics:
                    self.log_metrics(maintenance_metrics, step=int(self.global_step))
                data_wait_started_at = time.perf_counter()
            self.finish_epoch()
            if (
                checkpoint_every_epochs > 0
                and self.epoch < num_epochs
                and self.epoch % checkpoint_every_epochs == 0
            ):
                checkpoint_started_at = time.perf_counter()
                self.save_checkpoint(tag="latest")
                self.log_metrics(
                    {"time/classifier_checkpoint_s": (time.perf_counter() - checkpoint_started_at)},
                    step=int(self.global_step),
                )

        self.console_banner("TRAINING", done=True)
        final_validation = self._finalize_validation_checkpoints()
        if final_validation:
            self._log(
                {
                    "event": "val_final",
                    "step": self.global_step,
                    "metrics": final_validation,
                }
            )
        # ---- final ckpt + summary ------------------------------------
        self._save_final_checkpoint()
        summary = {
            "best_window_f1": self.best_window_f1,
            "best_episode_f1": self.best_episode_f1,
            "best_window_ckpt_path": self.best_window_ckpt_path,
            "best_episode_ckpt_path": self.best_episode_ckpt_path,
            "total_steps": int(self.global_step),
            "total_epochs": int(self.epoch),
            "wall_s": time.time() - t0,
        }
        self._log({"event": "done", **summary})
        if self.is_main_process:
            with open(pathlib.Path(self.output_dir) / "summary.json", "w") as fh:
                json.dump(summary, fh, indent=2)
        return summary

    # --------------------------- evaluation ----------------------------

    @torch.no_grad()
    def _evaluate_window_level(self) -> dict[str, Any]:
        """Softmax + threshold sweep over the positive class.

        Keep this aligned with ``LatentSuccessClassifier.predict_success`` and
        the two-class CE objective used during training.
        """
        assert self.model is not None and self.val_loader is not None
        self.model.eval()
        probs_l: list[float] = []
        ys_l: list[int] = []
        for batch in self.val_loader:
            xs, ys, extra_or_meta = batch
            xs = xs.to(self.device, non_blocking=True)
            extra: dict[str, torch.Tensor] = {}
            if isinstance(extra_or_meta, list):
                if extra_or_meta and all("proprio" in meta for meta in extra_or_meta):
                    extra["proprio"] = torch.stack([meta["proprio"] for meta in extra_or_meta])
                if extra_or_meta and all("lang_emb" in meta for meta in extra_or_meta):
                    extra["lang_emb"] = torch.stack([meta["lang_emb"] for meta in extra_or_meta])
            forward_kwargs = _classifier_forward_kwargs(
                self._classifier_module(), extra, self.device
            )
            logits = self.model(xs, **forward_kwargs)
            probs = _success_probabilities_from_logits(logits).detach().cpu().numpy()
            probs_l.extend(probs.tolist())
            ys_l.extend(ys.tolist())
        probs = np.asarray(probs_l, dtype=np.float32)
        ys = np.asarray(ys_l, dtype=np.int64)

        tr = self.cfg.training
        thresholds = np.linspace(
            float(OmegaConf.select(tr, "thresh_min") or 0.3),
            float(OmegaConf.select(tr, "thresh_max") or 1.0),
            int(OmegaConf.select(tr, "thresh_steps") or 20),
        )
        return _sweep_metrics(probs, ys, thresholds, tag="window")

    @torch.no_grad()
    def _evaluate_episode_level(self) -> dict[str, Any]:
        """LUMOS predict_success protocol — stride-1 sliding + any-positive.

        For each demo, scan stride-1 windows over the full trajectory (from
        ``min_steps + W`` to ``finish_step``). Use ``max`` over windows as the
        episode-level score, sweep thresholds, return best F1.

        Unit convention: ``episode_eval_min_steps`` and
        ``episode_eval_stride`` are in the classifier's NATIVE unit
        (env-step for action granularity, chunk for chunk granularity).
        The dataset's pooling K only affects how the env-step obs is folded
        to chunks for the sliding window; the gate values themselves are
        already chunk-unit in chunk configs.
        """
        assert self.model is not None and self.val_ds is not None
        tr = self.cfg.training
        W = int(self.cfg.data.window)
        min_steps = int(OmegaConf.select(tr, "episode_eval_min_steps") or 0)
        stride = int(OmegaConf.select(tr, "episode_eval_stride") or 1)
        ep_batch = max(1, int(OmegaConf.select(tr, "episode_eval_batch") or 256))

        K = int(getattr(self.val_ds, "K", 1))
        chunk_pool = str(getattr(self.val_ds, "chunk_pool", "last"))

        self.model.eval()
        ep_max_prob: list[float] = []
        ep_true: list[int] = []

        # Stream windows through the classifier in small batches. Token-grid
        # episodes are large, so materializing every episode window before the
        # first forward pass makes episode eval CPU-bound and memory-hungry.
        flat_xs: list[np.ndarray] = []
        flat_proprio: list[np.ndarray] = []
        flat_lang: list[np.ndarray] = []
        flat_ep: list[int] = []

        def flush_windows() -> None:
            if not flat_xs:
                return
            if flat_proprio and len(flat_proprio) != len(flat_xs):
                raise ValueError("episode eval proprio windows are incomplete")
            if flat_lang and len(flat_lang) != len(flat_xs):
                raise ValueError("episode eval language windows are incomplete")
            chunk = np.stack(flat_xs)
            extra: dict[str, torch.Tensor] = {}
            if flat_proprio:
                extra["proprio"] = torch.from_numpy(np.stack(flat_proprio)).float()
            if flat_lang:
                extra["lang_emb"] = torch.from_numpy(np.stack(flat_lang)).float()
            forward_kwargs = _classifier_forward_kwargs(
                self._classifier_module(), extra, self.device
            )
            logits = self.model(
                torch.from_numpy(chunk).float().to(self.device),
                **forward_kwargs,
            )
            p = _success_probabilities_from_logits(logits).detach().cpu().numpy()
            for eid, pj in zip(flat_ep, p, strict=True):
                if pj > ep_max_prob[eid]:
                    ep_max_prob[eid] = float(pj)
            flat_xs.clear()
            flat_proprio.clear()
            flat_lang.clear()
            flat_ep.clear()

        for ep_idx, trajectory in enumerate(self.val_ds.trajectories()):
            if len(trajectory) == 5:
                obs, complete, finish_step, _eid, extra = trajectory
            else:
                obs, complete, finish_step, _eid = trajectory
                extra = {}
            T_env = int(min(finish_step, obs.shape[0]))
            proprio = extra.get("proprio") if isinstance(extra, dict) else None
            lang_emb = extra.get("lang_emb") if isinstance(extra, dict) else None
            if K > 1:
                T_chunk = T_env // K
                if T_chunk < 1:
                    obs_pooled = None
                    proprio_pooled = None
                    T = 0
                else:
                    trailing_shape = obs.shape[1:]
                    reshaped = obs[: T_chunk * K].reshape(T_chunk, K, *trailing_shape)
                    if chunk_pool == "last":
                        obs_pooled = reshaped[:, -1]
                    elif chunk_pool == "first":
                        obs_pooled = reshaped[:, 0]
                    else:
                        obs_pooled = reshaped.mean(axis=1)
                    if isinstance(proprio, np.ndarray):
                        reshaped_proprio = proprio[: T_chunk * K].reshape(
                            T_chunk, K, proprio.shape[-1]
                        )
                        if chunk_pool == "last":
                            proprio_pooled = reshaped_proprio[:, -1]
                        elif chunk_pool == "first":
                            proprio_pooled = reshaped_proprio[:, 0]
                        else:
                            proprio_pooled = reshaped_proprio.mean(axis=1)
                    else:
                        proprio_pooled = None
                    T = T_chunk
            else:
                obs_pooled = obs
                proprio_pooled = proprio if isinstance(proprio, np.ndarray) else None
                T = T_env
            ep_true.append(int(bool(complete)))
            first_end = max(W, min_steps + W)
            if T < first_end or obs_pooled is None:
                ep_max_prob.append(0.0)
                continue
            ep_max_prob.append(-1.0)  # placeholder; updated below
            for end in range(first_end, T + 1, stride):
                flat_xs.append(obs_pooled[end - W : end])
                if proprio_pooled is not None:
                    flat_proprio.append(proprio_pooled[end - W : end])
                if isinstance(lang_emb, np.ndarray):
                    flat_lang.append(lang_emb)
                flat_ep.append(ep_idx)
                if len(flat_xs) >= ep_batch:
                    flush_windows()
        flush_windows()

        # placeholder -1.0 → 0.0 (too-short episodes)
        ep_max_prob = [max(0.0, p) for p in ep_max_prob]

        probs = np.asarray(ep_max_prob, dtype=np.float32)
        ys = np.asarray(ep_true, dtype=np.int64)
        thresholds = np.linspace(
            float(OmegaConf.select(tr, "thresh_min") or 0.3),
            float(OmegaConf.select(tr, "thresh_max") or 1.0),
            int(OmegaConf.select(tr, "thresh_steps") or 20),
        )
        return _sweep_metrics(probs, ys, thresholds, tag="episode")

    # --------------------------- io helpers ----------------------------

    def _maybe_save_named(self, name: str, *, extra: dict | None = None) -> None:
        """Write a metric-named snapshot only when top-k is explicitly enabled."""
        if self._classifier_topk_manager() is None:
            return
        self._save_named(name, extra=extra)

    def _save_final_checkpoint(self) -> str:
        """Write the canonical resumable classifier checkpoint."""
        return self.save_checkpoint(path=self.get_checkpoint_path())

    def _classifier_topk_manager(self) -> TopKCheckpointManager | None:
        cached = getattr(self, "_topk_checkpoint_manager", None)
        if cached is not None:
            return cached
        topk_cfg = OmegaConf.select(self.cfg, "checkpoint.topk", default=None)
        if topk_cfg is None:
            k = int(OmegaConf.select(self.cfg, "training.topk_k", default=0) or 0)
            values = {
                "monitor_key": "f1",
                "metric_name": "f1",
                "mode": "max",
                "k": k,
            }
        else:
            values = dict(OmegaConf.to_container(topk_cfg, resolve=True))
        if int(values.get("k", 0) or 0) <= 0:
            return None
        manager = TopKCheckpointManager(
            save_dir=self.get_checkpoint_dir(),
            **values,
        )
        self._topk_checkpoint_manager = manager
        return manager

    def _save_named(self, name: str, *, extra: dict | None = None) -> None:
        """Save latest and a full metric-selected payload from one serialization."""

        del name
        f1 = 0.0
        threshold = 0.5
        selection = "window"
        if isinstance(extra, dict):
            for k in ("val_episode", "val_window"):
                v = extra.get(k)
                if isinstance(v, dict):
                    f1 = float(v.get("best_f1", f1))
                    threshold = float(v.get("best_thresh", threshold))
                    if k == "val_episode":
                        selection = "episode"
                    else:
                        selection = "window"
                    break
        manager = self._classifier_topk_manager()
        if manager is None:
            return
        path = None
        if self.is_main_process:
            path = manager.get_ckpt_path(
                {
                    "epoch": int(self.epoch),
                    manager.monitor_key: f1,
                }
            )
        if path is None:
            return
        self.classifier_threshold = threshold
        if selection == "episode":
            self.best_episode_ckpt_path = str(path)
            self.best_episode_threshold = threshold
        else:
            self.best_window_ckpt_path = str(path)
            self.best_window_threshold = threshold
        self.save_checkpoint(
            path=self.get_checkpoint_path(),
            extra_paths=(path,),
        )
        self._log({"event": "ckpt_named", "path": str(path), "f1": f1, "threshold": threshold})

    def _log(self, payload: dict) -> None:
        if not self.is_main_process:
            return
        payload = {"ts": time.strftime("%H:%M:%S"), **payload}
        event = str(payload.get("event", ""))
        metric_prefix = "eval" if event.startswith("val_") else None
        if event == "train_step":
            metric_prefix = "train"
        step_value = payload.get("step", payload.get("global_step", self.global_step))
        self.log_metrics(payload, step=int(step_value), prefix=metric_prefix)
        print(json.dumps(payload), flush=True)
        if self._log_path is not None:
            with open(self._log_path, "a") as fh:
                fh.write(json.dumps(payload) + "\n")

    def teardown(self) -> None:
        super().teardown()
        self.distributed.barrier()
        self.distributed.cleanup()

    def teardown_after_setup_failure(self) -> None:
        """Clean up a partial setup without entering a cross-rank barrier."""
        try:
            BaseRunner.teardown(self)
        finally:
            self.distributed.cleanup()

    # ----------------------- BaseRunner exclusions ------------------

    # Datasets carry the in-memory demo cache (multi-GB); keep them out of
    # the BaseRunner state_dict serializer so save_checkpoint doesn't try
    # to pickle them.
    exclude_keys = ("train_ds", "val_ds", "train_loader", "val_loader")


# Threshold sweep lives in dreamervla.runtime.classifier_metrics; `_sweep_metrics`
# is re-exported above for existing importers.
__all__ = ["SuccessClassifierTrainingRunner", "_sweep_metrics"]
