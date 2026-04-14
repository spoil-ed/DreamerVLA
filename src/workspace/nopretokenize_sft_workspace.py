from __future__ import annotations

import copy
import os
import pathlib
import pickle
from pathlib import Path
from typing import Any, Mapping

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.dataloader import BaseDataset
from src.trainer import NopretokenizeSFTDistributedHelper
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.workspace.base_workspace import BaseWorkspace


class NopretokenizeSFTWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")
    exclude_keys = tuple()
    default_vla_init_dir = "/home/user01/yuxinglei/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_10"
    default_output_dir = "/home/user01/yuxinglei/workspace/DreamerVLA/data/outputs/debug_nopretokenize_sft"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        if output_dir is None:
            output_dir = str(OmegaConf.select(config, "training.out_dir", default=self.default_output_dir))
        super().__init__(config, output_dir=output_dir)

        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=str(OmegaConf.select(config, "training.distributed_strategy", default="ddp")),
            fsdp_mixed_precision=str(OmegaConf.select(config, "training.fsdp_mixed_precision", default="bf16")),
            enable_activation_checkpointing=bool(
                OmegaConf.select(config, "training.enable_activation_checkpointing", default=True)
            ),
        )
        self.rank = self.distributed.rank
        self.local_rank = self.distributed.local_rank
        self.world_size = self.distributed.world_size
        self.device = self.distributed.resolve_device(str(self.config.trainer.device))
        if self.distributed.is_main_process:
            self.print_config()
        set_seed(int(self.config.seed) + self.rank)
        self.world_model = None
        self.world_model_optimizer = None
        self.encoder = None
        self.vla_optimizer = None

    def _resolve_vla_init_path(self) -> str:
        configured = OmegaConf.select(self.cfg, "init.vla_ckpt_path")
        candidate = Path(str(configured)).expanduser().resolve() if configured is not None else Path(self.default_vla_init_dir)
        if candidate.is_dir():
            config_path = candidate / "config.json"
            if config_path.is_file():
                return str(candidate)
            subdirs = sorted(path for path in candidate.iterdir() if path.is_dir())
            for subdir in subdirs:
                if (subdir / "config.json").is_file():
                    return str(subdir.resolve())
        return str(candidate.resolve())

    def build_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            encoder_cfg.model_path = str(init_model_path)
        return encoder_cfg

    def infer_hidden_dim_from_encoder(self, encoder: Any | None) -> int | None:
        if encoder is None:
            return None
        backbone = getattr(encoder, "backbone", None)
        config = getattr(backbone, "config", None)
        for attr_name in ("hidden_size", "d_model"):
            value = getattr(config, attr_name, None)
            if value is not None:
                return int(value)
        return None

    def extract_hidden_from_obs(
        self,
        obs: Mapping[str, object],
        device: torch.device,
        fallback_hidden_dim: int | None = None,
    ) -> torch.Tensor:
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
            hidden_dim = int(OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1))
        return torch.zeros(batch_size, int(hidden_dim), device=device, dtype=torch.float32)

    def attach_encoder_outputs(
        self,
        batch: dict[str, object],
        *,
        encoder: Any | None,
        device: torch.device,
        fallback_hidden_dim: int | None = None,
        detach: bool = True,
    ) -> dict[str, object]:
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
                    if "obs_embedding" in batch and isinstance(batch["obs_embedding"], torch.Tensor):
                        batch["next_obs_embedding"] = batch["obs_embedding"].detach().clone()
            return batch

        with torch.no_grad():
            if isinstance(obs, Mapping):
                obs_embedding = encoder.encode(obs)
                batch["obs_embedding"] = obs_embedding.detach() if detach else obs_embedding
            if isinstance(next_obs, Mapping):
                next_obs_embedding = encoder.encode(next_obs)
                batch["next_obs_embedding"] = next_obs_embedding.detach() if detach else next_obs_embedding
        return batch

    def _build_trainable_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = self.build_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            encoder_cfg.freeze_backbone = False
        return encoder_cfg

    def _build_action_records(self, batch: dict[str, object]) -> list[dict[str, Any]]:
        records = batch.get("record")
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def save_checkpoint(
        self,
        path: str | pathlib.Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
    ) -> str:
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)

        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        if not self.distributed.requires_collective_checkpointing and not self.distributed.is_main_process:
            return str(path.absolute())

        payload = {
            "cfg": self.cfg,
            "state_dicts": {},
            "pickles": {},
        }

        for key, value in self.__dict__.items():
            if key in exclude_keys:
                continue
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                state_dict = self._state_dict_for_checkpoint(key, value)
                if self.distributed.is_main_process and state_dict is not None:
                    payload["state_dicts"][key] = _copy_to_cpu(state_dict)
            elif key in include_keys and self.distributed.is_main_process:
                payload["pickles"][key] = pickle.dumps(value)

        if self.distributed.is_main_process:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
        return str(path.absolute())

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = tuple(payload["pickles"].keys())

        for key, value in payload["state_dicts"].items():
            if key in exclude_keys or key not in self.__dict__ or self.__dict__[key] is None:
                continue
            self._load_state_dict_from_checkpoint(key, self.__dict__[key], value, **kwargs)

        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = pickle.loads(payload["pickles"][key])

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "encoder" and self.encoder is not None:
            with self.distributed.model_state_dict_context(self.encoder.backbone):
                return self.encoder.state_dict()
        if key == "vla_optimizer" and self.vla_optimizer is not None and self.encoder is not None:
            return self.distributed.optimizer_state_dict(self.encoder.backbone, self.vla_optimizer)
        return value.state_dict()

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key == "encoder" and self.encoder is not None:
            with self.distributed.model_state_dict_context(self.encoder.backbone):
                value.load_state_dict(state_dict, **kwargs)
            return
        if key == "vla_optimizer" and self.vla_optimizer is not None and self.encoder is not None:
            self.distributed.load_optimizer_state_dict(self.encoder.backbone, self.vla_optimizer, state_dict)
            return
        value.load_state_dict(state_dict, **kwargs)

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        self.resume(cfg)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset), "Dataset must be an instance of BaseDataset"

        dataloader_kwargs = dict(cfg.dataloader)
        sampler = self.distributed.maybe_make_sampler(
            dataset,
            shuffle=bool(dataloader_kwargs.get("shuffle", True)),
            drop_last=bool(dataloader_kwargs.get("drop_last", False)),
        )
        if sampler is not None:
            dataloader_kwargs["shuffle"] = False
            dataloader_kwargs["sampler"] = sampler
        collate_fn = getattr(dataset, "collate_fn", None)
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)

        encoder_cfg = OmegaConf.select(cfg, "encoder")
        if encoder_cfg is not None:
            encoder_cfg = self._build_trainable_encoder_cfg(cfg)
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
            self.distributed.wrap_encoder(self.encoder)
            vla_optim_cfg = OmegaConf.select(cfg, "optim.vla")
            if vla_optim_cfg is None:
                raise ValueError("`optim.vla` must be configured for nopretokenize VLA SFT.")
            self.vla_optimizer = build_optimizer(self.encoder, vla_optim_cfg)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 2
            cfg.training.checkpoint_every = 1

        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        self.distributed.barrier()
        vla_log_path = os.path.join(self.output_dir, "vla_logs.json.txt")
        vla_logger_cm = self.distributed.logger_context(vla_log_path)

        try:
            with vla_logger_cm as vla_json_logger:
                for _local_epoch_idx in range(cfg.training.num_epochs):
                    if sampler is not None:
                        sampler.set_epoch(self.epoch)
                    step_log: dict[str, float | str | int] = {}
                    train_vla_losses: list[float] = []
                    train_vla_token_losses: list[float] = []
                    train_vla_action_losses: list[float] = []
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        disable=not self.distributed.is_main_process,
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            action_records = self._build_action_records(batch)
                            if self.encoder is None or self.vla_optimizer is None or not action_records:
                                if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                                    break
                                continue

                            self.encoder.train()
                            vla_loss_dict = self.encoder.compute_action_sft_loss(
                                action_records,
                                token_loss_coef=float(OmegaConf.select(cfg, "training.vla_token_loss_coef", default=1.0)),
                                action_loss_coef=float(OmegaConf.select(cfg, "training.vla_action_loss_coef", default=1.0)),
                            )
                            vla_raw_loss = vla_loss_dict["loss"]
                            vla_loss = vla_raw_loss / cfg.training.gradient_accumulate_every
                            vla_loss.backward()

                            grad_clip_norm = cfg.optim.get("grad_clip_norm")
                            if grad_clip_norm is not None:
                                self.distributed.clip_grad_norm(self.encoder.backbone, float(grad_clip_norm))

                            self.vla_optimizer.step()
                            self.vla_optimizer.zero_grad(
                                set_to_none=bool(cfg.optim.get("zero_grad_set_to_none", True))
                            )

                            train_vla_losses.append(float(vla_raw_loss.item()))
                            train_vla_token_losses.append(float(vla_loss_dict["token_loss"].item()))
                            train_vla_action_losses.append(float(vla_loss_dict["action_loss"].item()))
                            step_log = {
                                "train_vla_loss": float(vla_raw_loss.item()),
                                "train_vla_token_loss": float(vla_loss_dict["token_loss"].item()),
                                "train_vla_action_loss": float(vla_loss_dict["action_loss"].item()),
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                            }
                            reduced_step_metrics = self.distributed.reduce_mean_dict(
                                {
                                    key: value
                                    for key, value in step_log.items()
                                    if key not in {"global_step", "epoch"}
                                }
                            )
                            step_log = {
                                **reduced_step_metrics,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                            }

                            tepoch.set_postfix(
                                vla=float(step_log["train_vla_loss"]),
                                token=float(step_log["train_vla_token_loss"]),
                                action=float(step_log["train_vla_action_loss"]),
                                refresh=False,
                            )

                            is_last_batch = batch_idx == (len(train_dataloader) - 1)
                            if not is_last_batch:
                                vla_json_logger.log(
                                    {
                                        "train_vla_loss": float(step_log["train_vla_loss"]),
                                        "train_vla_token_loss": float(step_log["train_vla_token_loss"]),
                                        "train_vla_action_loss": float(step_log["train_vla_action_loss"]),
                                        "global_step": self.global_step,
                                        "epoch": self.epoch,
                                    }
                                )
                                self.global_step += 1

                            if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                                break

                    if not train_vla_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    vla_count = max(self.distributed.reduce_sum(len(train_vla_losses)), 1.0)
                    step_log["train_vla_loss"] = self.distributed.reduce_sum(sum(train_vla_losses)) / vla_count
                    step_log["train_vla_token_loss"] = (
                        self.distributed.reduce_sum(sum(train_vla_token_losses)) / vla_count
                    )
                    step_log["train_vla_action_loss"] = (
                        self.distributed.reduce_sum(sum(train_vla_action_losses)) / vla_count
                    )
                    vla_json_logger.log(
                        {
                            "train_vla_loss": float(step_log["train_vla_loss"]),
                            "train_vla_token_loss": float(step_log["train_vla_token_loss"]),
                            "train_vla_action_loss": float(step_log["train_vla_action_loss"]),
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }
                    )

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if (
                            cfg.checkpoint.save_last_snapshot
                            and hasattr(self, "save_snapshot")
                            and not self.distributed.requires_collective_checkpointing
                        ):
                            self.save_snapshot()

                        metric_dict = {}
                        for key, value in step_log.items():
                            metric_dict[key.replace("/", "_")] = value
                        topk_ckpt_path = None
                        if self.distributed.is_main_process:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        topk_ckpt_path = self.distributed.broadcast_object(topk_ckpt_path)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    self.global_step += 1
                    self.epoch += 1
        finally:
            self.distributed.barrier()
            self.distributed.cleanup()

        return history


__all__ = ["NopretokenizeSFTWorkspace"]


def _copy_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    return copy.deepcopy(value)
