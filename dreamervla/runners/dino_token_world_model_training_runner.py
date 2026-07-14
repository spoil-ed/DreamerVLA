"""Exact DINO-WM dynamics training over DreamerVLA token sidecars."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dreamervla.models.embodiment.world_model import DinoTokenWorldModel
from dreamervla.runtime.distributed import unwrap_module
from dreamervla.runtime.world_model_training_base import WorldModelTrainingBase
from dreamervla.utils.torch_utils import precision_dtype


def _adamw_from_cfg(
    parameters: Any,
    cfg: DictConfig,
) -> torch.optim.AdamW:
    """Build the upstream DINO-WM AdamW optimizer from Hydra values."""

    if str(cfg.name).lower() != "adamw":
        raise ValueError("DINO-WM reproduction requires AdamW")
    return torch.optim.AdamW(
        parameters,
        lr=float(cfg.lr),
        betas=tuple(float(value) for value in cfg.betas),
        eps=float(cfg.eps),
        weight_decay=float(cfg.weight_decay),
    )


class DinoTokenWorldModelTrainingRunner(WorldModelTrainingBase):
    """Train DINO-WM with its original epoch and trajectory-slice protocol.

    The visual encoder is replaced by persisted OpenVLA-OFT token grids. Every
    other dynamics-training boundary is retained: trajectory-level split,
    fixed shuffled slice order, frame skipping with concatenated actions,
    normalized actions/proprio, FP32, separate predictor and conditioning
    AdamW optimizers, full train/valid epochs, and per-epoch checkpoints.
    """

    runner_name = "dino_token_world_model_training"
    runner_family = "world_model"
    include_keys = (*WorldModelTrainingBase.include_keys,)

    @staticmethod
    def _per_rank_batch_size(
        *,
        configured_batch_size: int,
        global_batch_size: int | None,
        world_size: int,
    ) -> int:
        configured = int(configured_batch_size)
        if configured < 1:
            raise ValueError("dataloader.batch_size must be positive")
        if global_batch_size is None:
            return configured
        global_batch = int(global_batch_size)
        ranks = max(1, int(world_size))
        if global_batch < 1 or global_batch % ranks:
            raise ValueError(
                "training.global_batch_size must be positive and divisible by "
                f"world_size ({global_batch} % {ranks} != 0)"
            )
        return global_batch // ranks

    def _build_model_and_optimizers(self, cfg: DictConfig) -> None:
        model_cfg = OmegaConf.select(cfg, "world_model")
        if model_cfg is None:
            raise ValueError("world_model config is required")
        dtype = precision_dtype(str(OmegaConf.select(cfg, "optim.precision")))
        if dtype != torch.float32:
            raise ValueError("DINO-WM reproduction requires optim.precision=fp32")
        model = hydra.utils.instantiate(model_cfg)
        if not isinstance(model, DinoTokenWorldModel):
            raise TypeError(
                "DinoTokenWorldModelTrainingRunner requires DinoTokenWorldModel, "
                f"got {type(model).__name__}"
            )
        self._unwrapped_world_model = model.to(device=self.device, dtype=dtype)
        self.world_model = self.distributed.wrap_trainable_module(
            self._unwrapped_world_model,
            find_unused_parameters=False,
            broadcast_buffers=True,
        )
        raw_model = unwrap_module(self.world_model)
        predictor_cfg = OmegaConf.select(cfg, "optim.predictor")
        conditioning_cfg = OmegaConf.select(cfg, "optim.conditioning")
        if predictor_cfg is None or conditioning_cfg is None:
            raise ValueError("optim.predictor and optim.conditioning are required")
        self.predictor_optimizer = _adamw_from_cfg(
            raw_model.predictor.parameters(), predictor_cfg
        )
        self.conditioning_optimizer = _adamw_from_cfg(
            itertools.chain(
                raw_model.action_encoder.parameters(),
                raw_model.proprio_encoder.parameters(),
            ),
            conditioning_cfg,
        )

    def _state_dict_for_checkpoint(
        self,
        key: str,
        value: Any,
    ) -> dict[str, Any] | None:
        if key == "world_model":
            return unwrap_module(self.world_model).state_dict()
        return super()._state_dict_for_checkpoint(key, value)

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key == "world_model":
            unwrap_module(self.world_model).load_state_dict(state_dict, **kwargs)
            return
        super()._load_state_dict_from_checkpoint(
            key,
            value,
            state_dict,
            **kwargs,
        )

    def _move_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: (
                value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }

    @staticmethod
    def _loss_metrics(losses: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        names = (
            "loss",
            "z_loss",
            "z_visual_loss",
            "z_proprio_loss",
            "hidden_cosine_loss",
            "hidden_cosine_similarity",
            "one_step_cosine_similarity",
            "persistence_cosine_similarity",
            "chunk_cosine_similarity",
            "hidden_pred_norm",
            "hidden_target_norm",
        )
        return {
            name: value.detach().float()
            for name in names
            if isinstance((value := losses.get(name)), torch.Tensor)
        }

    def _train_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        self.world_model.train()
        self.predictor_optimizer.zero_grad(set_to_none=True)
        self.conditioning_optimizer.zero_grad(set_to_none=True)
        losses = self.world_model(self._move_batch(batch))
        loss = losses.get("_loss", losses.get("loss"))
        if not isinstance(loss, torch.Tensor):
            raise KeyError("DINO world model must return Tensor '_loss' or 'loss'")
        loss.backward()
        self.predictor_optimizer.step()
        self.conditioning_optimizer.step()
        return self._loss_metrics(losses)

    @staticmethod
    def _progress_status(
        metrics: Mapping[str, float],
        *,
        global_step: int,
    ) -> str:
        """Format detached, diagnostic-only values for the epoch progress bar."""

        loss = float(metrics.get("loss", float("nan")))
        cosine = float(metrics.get("one_step_cosine_similarity", float("nan")))
        return f"global_step={int(global_step)} loss={loss:.6f} cos={cosine:.6f}"

    @torch.no_grad()
    def _evaluate(self, dataloader: Any) -> dict[str, float]:
        self.world_model.eval()
        totals: dict[str, torch.Tensor] = {}
        examples = torch.zeros((), device=self.device, dtype=torch.float64)
        for batch in dataloader:
            losses = self.world_model(self._move_batch(batch))
            batch_size = int(batch["obs_embedding"].shape[0])
            examples += batch_size
            for name, value in self._loss_metrics(losses).items():
                totals[name] = totals.get(
                    name,
                    torch.zeros((), device=self.device, dtype=torch.float64),
                ) + value.to(dtype=torch.float64) * batch_size
        count = self.distributed.reduce_sum(examples)
        if count <= 0:
            return {}
        return {
            f"eval/{name}": self.distributed.reduce_sum(total) / count
            for name, total in totals.items()
        }

    def _save_epoch_checkpoint(self) -> None:
        checkpoint = self.get_global_step_checkpoint_dir(self.global_step) / "model.ckpt"
        latest = self.get_checkpoint_path()
        warmup = self.get_compat_checkpoint_dir() / "wm_warmup.ckpt"
        self.save_checkpoint(
            path=checkpoint,
            extra_paths=(latest, warmup),
        )

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        cfg = self.cfg
        self._build_model_and_optimizers(cfg)
        self.resume(cfg)

        train_dataset = hydra.utils.instantiate(cfg.dataset.train)
        valid_dataset = hydra.utils.instantiate(cfg.dataset.valid)
        raw_model = unwrap_module(self.world_model)
        for name, actual, expected in (
            ("action_dim", train_dataset.action_dim, raw_model.action_dim),
            ("proprio_dim", train_dataset.proprio_dim, raw_model.proprio_dim),
        ):
            if int(actual) != int(expected):
                raise ValueError(
                    f"DINO dataset/model {name} mismatch: {actual} != {expected}"
                )

        dataloader_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.dataloader, resolve=True)
        )
        dataloader_cfg.batch_size = self._per_rank_batch_size(
            configured_batch_size=int(dataloader_cfg.batch_size),
            global_batch_size=OmegaConf.select(
                cfg,
                "training.global_batch_size",
                default=None,
            ),
            world_size=self.world_size,
        )
        effective_global_batch_size = int(dataloader_cfg.batch_size) * max(
            1, int(self.world_size)
        )
        train_dataloader = self.make_distributed_dataloader(
            train_dataset,
            dataloader_cfg,
            shuffle=False,
            drop_last=False,
            sanitize_worker_kwargs=True,
        )
        valid_dataloader = self.make_distributed_dataloader(
            valid_dataset,
            dataloader_cfg,
            shuffle=False,
            drop_last=False,
            sanitize_worker_kwargs=True,
        )
        if self.is_main_process:
            self.append_model_summary(
                {
                    "trainable_params": sum(
                        parameter.numel()
                        for parameter in raw_model.parameters()
                        if parameter.requires_grad
                    ),
                    "train_windows": len(train_dataset),
                    "valid_windows": len(valid_dataset),
                    "frameskip": int(train_dataset.frameskip),
                    "model_step_frames": int(train_dataset.frameskip),
                    "batch_size_per_rank": int(dataloader_cfg.batch_size),
                    "global_batch_size": effective_global_batch_size,
                }
            )

        num_epochs = int(cfg.training.num_epochs)
        max_steps = int(OmegaConf.select(cfg, "training.max_steps", default=0) or 0)
        eval_every = int(OmegaConf.select(cfg, "training.eval_every", default=1))
        checkpoint_every = int(
            OmegaConf.select(cfg, "training.checkpoint_every", default=1)
        )
        try:
            stop = False
            for epoch in range(int(self.epoch) + 1, num_epochs + 1):
                self.epoch = epoch
                self.set_dataloader_epoch(train_dataloader, epoch)
                sums: dict[str, float] = {}
                batches = 0
                epoch_steps = len(train_dataloader)
                progress_desc = f"dino-wm epoch {epoch}/{num_epochs}"
                for batch in train_dataloader:
                    metrics = self._train_batch(batch)
                    reduced = self.distributed.reduce_mean_dict(metrics)
                    for name, value in reduced.items():
                        sums[name] = sums.get(name, 0.0) + float(value)
                    batches += 1
                    self.global_step += 1
                    if max_steps > 0 and self.global_step >= max_steps:
                        stop = True
                    self.console_progress(
                        batches,
                        epoch_steps,
                        progress_desc,
                        unit="step",
                        status=self._progress_status(
                            reduced,
                            global_step=self.global_step,
                        ),
                        force=stop,
                    )
                    if stop:
                        break
                train_metrics = {
                    f"train/{name}": total / max(1, batches)
                    for name, total in sums.items()
                }
                epoch_metrics: dict[str, float | str | int] = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    **train_metrics,
                }
                if eval_every > 0 and epoch % eval_every == 0:
                    epoch_metrics.update(self._evaluate(valid_dataloader))
                self.log_metrics(epoch_metrics, step=self.global_step)
                history.append(epoch_metrics)
                if self.is_main_process:
                    train_loss = float(train_metrics.get("train/loss", float("nan")))
                    train_cos = float(
                        train_metrics.get(
                            "train/one_step_cosine_similarity",
                            float("nan"),
                        )
                    )
                    val_loss = float(epoch_metrics.get("eval/loss", float("nan")))
                    val_cos = float(
                        epoch_metrics.get(
                            "eval/one_step_cosine_similarity",
                            float("nan"),
                        )
                    )
                    print(
                        f"[dino-wm] epoch={epoch}/{num_epochs} "
                        f"step={self.global_step} train_loss={train_loss:.6f} "
                        f"train_cos={train_cos:.6f} val_loss={val_loss:.6f} "
                        f"val_cos={val_cos:.6f}",
                        flush=True,
                    )
                if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                    self._save_epoch_checkpoint()
                if stop:
                    break
        finally:
            self.distributed.barrier()
            self.distributed.cleanup()
        return history


__all__ = ["DinoTokenWorldModelTrainingRunner"]
