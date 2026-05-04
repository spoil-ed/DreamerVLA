from __future__ import annotations

import copy
import os
import pathlib
import pickle
from typing import Any

import hydra
import torch
import tqdm
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.dataloader import BaseDataset
from src.trainer import NopretokenizeSFTDistributedHelper
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.ema import EMAHelper
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.workspace.base_workspace import BaseWorkspace


class PretokenizeSFTWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")
    exclude_keys = tuple()
    default_vla_init_dir = "/home/user01/liops/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_10"
    default_output_dir = "/home/user01/liops/workspace/DreamerVLA/data/outputs/vla/debug_pretokenize_sft"

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
        self.encoder = None
        self.vla_optimizer = None
        self.vla_ema: EMAHelper | None = None
        self.world_model = None
        self.world_model_optimizer = None
        self.world_model_ema: EMAHelper | None = None

    def _resolve_vla_init_path(self) -> str:
        configured = OmegaConf.select(self.cfg, "init.vla_ckpt_path")
        candidate = pathlib.Path(str(configured)).expanduser().resolve() if configured is not None else pathlib.Path(
            self.default_vla_init_dir
        )
        if candidate.is_dir():
            if (candidate / "config.json").is_file():
                return str(candidate)
            for subdir in sorted(path for path in candidate.iterdir() if path.is_dir()):
                if (subdir / "config.json").is_file():
                    return str(subdir.resolve())
        return str(candidate.resolve())

    def build_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            encoder_cfg.model_path = str(init_model_path)
        return encoder_cfg

    def _build_trainable_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = self.build_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            train_encoder_backbone = bool(OmegaConf.select(cfg, "training.train_encoder_backbone", default=True))
            encoder_cfg.freeze_backbone = not train_encoder_backbone
        return encoder_cfg

    @staticmethod
    def _set_trainable_encoder_parameters(encoder: Any, patterns: list[str]) -> int:
        if not hasattr(encoder, "named_parameters"):
            return 0
        for _, parameter in encoder.named_parameters():
            parameter.requires_grad = False
        matched = 0
        for name, parameter in encoder.named_parameters():
            if any(pattern in name for pattern in patterns):
                parameter.requires_grad = True
                matched += 1
        return matched

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

        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}
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
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                return self.world_model.state_dict()
        if key == "world_model_optimizer" and self.world_model_optimizer is not None and self.world_model is not None:
            return self.distributed.optimizer_state_dict(self.world_model, self.world_model_optimizer)
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
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                value.load_state_dict(state_dict, **kwargs)
            return
        if key == "world_model_optimizer" and self.world_model_optimizer is not None and self.world_model is not None:
            self.distributed.load_optimizer_state_dict(self.world_model, self.world_model_optimizer, state_dict)
            return
        value.load_state_dict(state_dict, **kwargs)

    def _build_world_model_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        if self.world_model is None:
            return None

        if (
            self.encoder is not None
            and "obs_embedding" not in batch
            and isinstance(batch.get("wm_obs_input_ids"), list)
            and isinstance(batch.get("wm_next_obs_input_ids"), list)
        ):
            obs_embedding = self._encode_hidden_from_tokenized(batch["wm_obs_input_ids"])
            next_obs_embedding = self._encode_hidden_from_tokenized(batch["wm_next_obs_input_ids"])
            batch["obs_embedding"] = obs_embedding
            batch["next_obs_embedding"] = next_obs_embedding

        if "obs_embedding" not in batch and isinstance(batch.get("obs"), dict):
            batch = self.attach_encoder_outputs(
                batch,
                encoder=self.encoder,
                device=self.device,
                fallback_hidden_dim=getattr(self.world_model, "obs_dim", None),
                detach=True,
            )

        wm_batch: dict[str, Any] = {}
        for key in ("obs_embedding", "next_obs_embedding", "action", "action_mask", "reward"):
            value = batch.get(key)
            if value is not None:
                wm_batch[key] = value

        if "action" not in wm_batch:
            conditioning_action = batch.get("conditioning_action")
            if isinstance(conditioning_action, torch.Tensor):
                wm_batch["action"] = conditioning_action
        if "action_mask" not in wm_batch:
            conditioning_action_mask = batch.get("conditioning_action_mask")
            if isinstance(conditioning_action_mask, torch.Tensor):
                wm_batch["action_mask"] = conditioning_action_mask

        required = ("obs_embedding", "next_obs_embedding", "action")
        if not all(isinstance(wm_batch.get(key), torch.Tensor) for key in required):
            return None

        for key in ("obs_embedding", "next_obs_embedding", "action", "action_mask", "reward"):
            value = wm_batch.get(key)
            if isinstance(value, torch.Tensor):
                wm_batch[key] = value.to(self.device)
        return wm_batch

    def _encode_hidden_from_tokenized(self, input_ids_list: list[list[int]]) -> torch.Tensor:
        if self.encoder is None:
            raise ValueError("Encoder is required for token-level world-model conditioning.")
        if not input_ids_list:
            hidden_dim = self.infer_hidden_dim_from_encoder(self.encoder) or int(
                OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1)
            )
            return torch.zeros((0, hidden_dim), device=self.device, dtype=torch.float32)
        labels_list = [[-100] * len(example) for example in input_ids_list]
        lengths = [len(example) for example in input_ids_list]
        with torch.no_grad():
            _, _, _, hidden_states, _, _, _ = self.encoder.backbone(
                input_ids=input_ids_list,
                labels=labels_list,
                training=True,
                output_hidden_states=True,
                att_mask=False,
            )
        attention_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        for idx, length in enumerate(lengths):
            if length > 0:
                attention_mask[idx, :length] = True
        weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled.float().detach()

    @torch.no_grad()
    def evaluate_val_loss(self, val_dataloader: DataLoader, split_name: str) -> dict[str, float]:
        if self.encoder is not None:
            self.encoder.eval()
        if self.world_model is not None:
            self.world_model.eval()

        vla_losses: list[float] = []
        wm_losses: list[float] = []
        for batch in val_dataloader:
            has_tokenized = isinstance(batch.get("input_ids"), list) and isinstance(batch.get("labels"), list)
            if has_tokenized and self.encoder is not None:
                vla_loss_dict = self.encoder.compute_action_sft_loss_from_tokenized(
                    input_ids_list=batch["input_ids"],
                    labels_list=batch["labels"],
                )
                vla_losses.append(float(vla_loss_dict["loss"].item()))
            wm_batch = self._build_world_model_batch(batch)
            if wm_batch is not None and self.world_model is not None:
                wm_loss_dict = self.world_model.compute_loss_dict(wm_batch)
                wm_losses.append(float(wm_loss_dict["loss"].item()))

        if self.encoder is not None:
            self.encoder.train()
        if self.world_model is not None:
            self.world_model.train()

        metrics: dict[str, float] = {}
        if vla_losses:
            count = max(self.distributed.reduce_sum(len(vla_losses)), 1.0)
            metrics[f"val_{split_name}_vla_loss"] = self.distributed.reduce_sum(sum(vla_losses)) / count
        if wm_losses:
            count = max(self.distributed.reduce_sum(len(wm_losses)), 1.0)
            metrics[f"val_{split_name}_wm_loss"] = self.distributed.reduce_sum(sum(wm_losses)) / count
        if metrics and self.distributed.is_main_process:
            print(f"  [Val {split_name}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

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

        # configure validation dataset
        val_dataloaders: dict[str, DataLoader] = {}
        for split_name in ("val_ind", "val_ood"):
            val_ds_cfg = OmegaConf.select(cfg, f"dataset_{split_name}", default=None)
            if val_ds_cfg is None:
                continue
            val_ds = hydra.utils.instantiate(val_ds_cfg)
            val_dl_kwargs = dict(cfg.dataloader)
            val_dl_kwargs["shuffle"] = False
            val_dl_kwargs["drop_last"] = False
            val_sampler = self.distributed.maybe_make_sampler(val_ds, shuffle=False, drop_last=False)
            if val_sampler is not None:
                val_dl_kwargs["sampler"] = val_sampler
            val_collate = getattr(val_ds, "collate_fn", None)
            if callable(val_collate):
                val_dl_kwargs["collate_fn"] = val_collate
            val_dataloaders[split_name] = DataLoader(val_ds, **val_dl_kwargs)

        encoder_cfg = OmegaConf.select(cfg, "encoder")
        if encoder_cfg is not None:
            encoder_cfg = self._build_trainable_encoder_cfg(cfg)
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
            if bool(OmegaConf.select(cfg, "training.vla_train_action_head_only", default=False)):
                matched = self._set_trainable_encoder_parameters(self.encoder, patterns=["action_head"])
                if matched == 0:
                    raise ValueError("No trainable parameters matched pattern `action_head` for VLA training.")
            self.distributed.wrap_encoder(self.encoder)
            vla_optim_cfg = OmegaConf.select(cfg, "optim.vla")
            if vla_optim_cfg is None:
                raise ValueError("`optim.vla` must be configured for pretokenize VLA SFT.")
            self.vla_optimizer = build_optimizer(self.encoder, vla_optim_cfg)

        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is not None:
            world_model_hidden_dim = self.infer_hidden_dim_from_dataset(dataset)
            if world_model_hidden_dim is None:
                world_model_hidden_dim = self.infer_hidden_dim_from_encoder(self.encoder)
            if world_model_hidden_dim is None:
                self.world_model = hydra.utils.instantiate(world_model_cfg).to(self.device)
            else:
                self.world_model = hydra.utils.instantiate(
                    world_model_cfg,
                    hidden_dim=world_model_hidden_dim,
                ).to(self.device)
            world_optim_cfg = OmegaConf.select(cfg, "optim.world_model")
            if world_optim_cfg is None:
                raise ValueError("`optim.world_model` must be configured when `world_model` is enabled.")
            self.world_model = self.distributed.wrap_trainable_module(self.world_model)
            self.world_model_optimizer = build_optimizer(self.world_model, world_optim_cfg)

        if self.encoder is None and self.world_model is None:
            raise ValueError("No trainable module configured. Set at least one of `encoder` or `world_model`.")

        # configure ema
        if bool(OmegaConf.select(cfg, "training.use_ema", default=False)):
            ema_decay = float(OmegaConf.select(cfg, "ema.decay", default=0.9999))
            ema_update_after = int(OmegaConf.select(cfg, "ema.update_after_step", default=0))
            if self.encoder is not None and self.vla_ema is None:
                self.vla_ema = EMAHelper(self.encoder, decay=ema_decay, update_after_step=ema_update_after)
            if self.world_model is not None and self.world_model_ema is None:
                self.world_model_ema = EMAHelper(
                    self.world_model, decay=ema_decay, update_after_step=ema_update_after
                )

        # resume training
        self.resume(cfg)

        # configure lr scheduler
        total_training_steps = (
            len(train_dataloader) * int(cfg.training.num_epochs)
        ) // int(cfg.training.gradient_accumulate_every)
        lr_scheduler_name = str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant"))
        lr_warmup_steps = int(OmegaConf.select(cfg, "training.lr_warmup_steps", default=0))
        vla_lr_scheduler = None
        if self.vla_optimizer is not None:
            vla_lr_scheduler = get_scheduler(
                lr_scheduler_name,
                optimizer=self.vla_optimizer,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=total_training_steps,
                last_epoch=self.global_step - 1,
            )
        wm_lr_scheduler = None
        if self.world_model_optimizer is not None:
            wm_lr_scheduler = get_scheduler(
                lr_scheduler_name,
                optimizer=self.world_model_optimizer,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=total_training_steps,
                last_epoch=self.global_step - 1,
            )

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
        train_log_path = os.path.join(self.output_dir, "train_logs.json.txt")
        train_logger_cm = self.distributed.logger_context(train_log_path)

        try:
            with train_logger_cm as train_json_logger:
                reached_max_steps = False
                for _local_epoch_idx in range(cfg.training.num_epochs):
                    if sampler is not None:
                        sampler.set_epoch(self.epoch)
                    step_log: dict[str, float | str | int] = {}
                    train_vla_losses: list[float] = []
                    train_vla_token_losses: list[float] = []
                    train_vla_action_losses: list[float] = []
                    train_wm_losses: list[float] = []
                    train_wm_transition_losses: list[float] = []
                    train_wm_kl_losses: list[float] = []
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        disable=not self.distributed.is_main_process,
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            local_step_metrics: dict[str, float] = {}
                            step_had_update = False

                            has_tokenized_batch = (
                                isinstance(batch.get("input_ids"), list) and isinstance(batch.get("labels"), list)
                            )
                            if has_tokenized_batch and self.encoder is not None and self.vla_optimizer is not None:
                                self.encoder.train()
                                vla_loss_dict = self.encoder.compute_action_sft_loss_from_tokenized(
                                    input_ids_list=batch["input_ids"],
                                    labels_list=batch["labels"],
                                    token_loss_coef=float(
                                        OmegaConf.select(cfg, "training.vla_token_loss_coef", default=1.0)
                                    ),
                                    action_loss_coef=float(
                                        OmegaConf.select(cfg, "training.vla_action_loss_coef", default=1.0)
                                    ),
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
                                if vla_lr_scheduler is not None:
                                    vla_lr_scheduler.step()

                                # update ema
                                if self.vla_ema is not None:
                                    self.vla_ema.step(self.encoder)

                                train_vla_losses.append(float(vla_raw_loss.item()))
                                train_vla_token_losses.append(float(vla_loss_dict["token_loss"].item()))
                                train_vla_action_losses.append(float(vla_loss_dict["action_loss"].item()))
                                local_step_metrics["train_vla_loss"] = float(vla_raw_loss.item())
                                local_step_metrics["train_vla_token_loss"] = float(vla_loss_dict["token_loss"].item())
                                local_step_metrics["train_vla_action_loss"] = float(vla_loss_dict["action_loss"].item())
                                if vla_lr_scheduler is not None:
                                    local_step_metrics["vla_lr"] = float(vla_lr_scheduler.get_last_lr()[0])
                                step_had_update = True

                            wm_batch = self._build_world_model_batch(batch)
                            if wm_batch is not None and self.world_model is not None and self.world_model_optimizer is not None:
                                self.world_model.train()
                                wm_loss_dict = self.world_model.compute_loss_dict(wm_batch)
                                wm_raw_loss = wm_loss_dict["loss"]
                                wm_loss = wm_raw_loss / cfg.training.gradient_accumulate_every
                                wm_loss.backward()

                                grad_clip_norm = cfg.optim.get("grad_clip_norm")
                                if grad_clip_norm is not None:
                                    self.distributed.clip_grad_norm(self.world_model, float(grad_clip_norm))

                                self.world_model_optimizer.step()
                                self.world_model_optimizer.zero_grad(
                                    set_to_none=bool(cfg.optim.get("zero_grad_set_to_none", True))
                                )
                                if wm_lr_scheduler is not None:
                                    wm_lr_scheduler.step()

                                # update ema
                                if self.world_model_ema is not None:
                                    self.world_model_ema.step(self.world_model)

                                train_wm_losses.append(float(wm_raw_loss.item()))
                                train_wm_transition_losses.append(float(wm_loss_dict["transition_loss"].item()))
                                train_wm_kl_losses.append(float(wm_loss_dict["kl_loss"].item()))
                                local_step_metrics["train_wm_loss"] = float(wm_raw_loss.item())
                                local_step_metrics["train_wm_transition_loss"] = float(
                                    wm_loss_dict["transition_loss"].item()
                                )
                                local_step_metrics["train_wm_kl_loss"] = float(wm_loss_dict["kl_loss"].item())
                                if wm_lr_scheduler is not None:
                                    local_step_metrics["wm_lr"] = float(wm_lr_scheduler.get_last_lr()[0])
                                step_had_update = True

                            if not step_had_update:
                                continue

                            reduced_step_metrics = self.distributed.reduce_mean_dict(local_step_metrics)
                            step_log = {
                                **reduced_step_metrics,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                            }

                            tepoch_postfix: dict[str, float] = {}
                            if "train_vla_loss" in step_log:
                                tepoch_postfix["vla"] = float(step_log["train_vla_loss"])
                            if "train_wm_loss" in step_log:
                                tepoch_postfix["wm"] = float(step_log["train_wm_loss"])
                            if "train_wm_kl_loss" in step_log:
                                tepoch_postfix["wm_kl"] = float(step_log["train_wm_kl_loss"])
                            if tepoch_postfix:
                                tepoch.set_postfix(refresh=False, **tepoch_postfix)

                            is_last_batch = batch_idx == (len(train_dataloader) - 1)
                            if not is_last_batch:
                                train_json_logger.log(step_log)
                                self.global_step += 1

                            if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                                reached_max_steps = True
                                break

                    if not train_vla_losses and not train_wm_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    if train_vla_losses:
                        vla_count = max(self.distributed.reduce_sum(len(train_vla_losses)), 1.0)
                        step_log["train_vla_loss"] = self.distributed.reduce_sum(sum(train_vla_losses)) / vla_count
                        step_log["train_vla_token_loss"] = (
                            self.distributed.reduce_sum(sum(train_vla_token_losses)) / vla_count
                        )
                        step_log["train_vla_action_loss"] = (
                            self.distributed.reduce_sum(sum(train_vla_action_losses)) / vla_count
                        )
                    if train_wm_losses:
                        wm_count = max(self.distributed.reduce_sum(len(train_wm_losses)), 1.0)
                        step_log["train_wm_loss"] = self.distributed.reduce_sum(sum(train_wm_losses)) / wm_count
                        step_log["train_wm_transition_loss"] = (
                            self.distributed.reduce_sum(sum(train_wm_transition_losses)) / wm_count
                        )
                        step_log["train_wm_kl_loss"] = (
                            self.distributed.reduce_sum(sum(train_wm_kl_losses)) / wm_count
                        )
                    step_log.setdefault("train_vla_loss", float("inf"))
                    step_log.setdefault("train_wm_loss", float("inf"))

                    # run validation
                    eval_every = int(OmegaConf.select(cfg, "eval.eval_every", default=1))
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            step_log.update(self.evaluate_val_loss(val_dl, split_name))

                    train_json_logger.log(step_log)

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()

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
                    if reached_max_steps:
                        break
        finally:
            self.distributed.barrier()
            self.distributed.cleanup()

        return history


__all__ = ["PretokenizeSFTWorkspace"]


def _copy_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    return copy.deepcopy(value)
