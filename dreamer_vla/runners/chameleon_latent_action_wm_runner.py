"""Frozen-Chameleon-latent action dynamics runner.

This runner trains a z-only model:

    image_t, image_{t+k} --frozen Chameleon backbone--> z_t, z_{t+k}
    (z_t, action_t:t+k) --trainable model--------------> z_hat_{t+k}

No Dreamer h, no posterior/prior KL, no image decoder.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import hydra
import torch
import tqdm
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from dreamer_vla.dataset import BaseDataset
from dreamer_vla.trainer import NopretokenizeSFTDistributedHelper
from dreamer_vla.utils.checkpoint_util import TopKCheckpointManager
from dreamer_vla.utils.optim import build_optimizer
from dreamer_vla.utils.seed import set_seed
from dreamer_vla.runners.base_runner import BaseRunner


class ChameleonLatentActionWMRunner(BaseRunner):
    runner_name = "chameleon_latent_wm_compat"
    runner_status = "compatibility"
    runner_family = "world_model"
    include_keys = ("global_step", "epoch")
    exclude_keys = ("encoder",)
    checkpoint_restore_output_dir = True
    default_vla_init_dir = (
        "/mnt/data/spoil/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_goal"
    )
    default_output_dir = "/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/chameleon_latent_action_wm/debug"

    @staticmethod
    def _first_finite_metric(metrics: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key not in metrics:
                continue
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        if output_dir is None:
            output_dir = str(
                OmegaConf.select(
                    config, "training.out_dir", default=self.default_output_dir
                )
            )
        super().__init__(config, output_dir=output_dir)
        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=str(
                OmegaConf.select(config, "training.distributed_strategy", default="ddp")
            ),
            fsdp_mixed_precision=str(
                OmegaConf.select(
                    config, "training.fsdp_mixed_precision", default="bf16"
                )
            ),
            enable_activation_checkpointing=bool(
                OmegaConf.select(
                    config, "training.enable_activation_checkpointing", default=False
                )
            ),
        )
        self.rank = self.distributed.rank
        self.local_rank = self.distributed.local_rank
        self.world_size = self.distributed.world_size
        self.device = self.distributed.resolve_device(
            str(OmegaConf.select(config, "trainer.device", default="auto"))
        )
        if self.distributed.is_main_process and bool(
            OmegaConf.select(config, "training.print_config", default=True)
        ):
            self.print_config()
        set_seed(int(OmegaConf.select(config, "seed", default=7)) + self.rank)

        self.encoder = None
        self.world_model = None
        self.world_model_optimizer = None
        self._image_bpe_set_cache: set[int] | None = None

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                return self.world_model.state_dict()
        if (
            key == "world_model_optimizer"
            and self.world_model_optimizer is not None
            and self.world_model is not None
        ):
            return self.distributed.optimizer_state_dict(
                self.world_model, self.world_model_optimizer
            )
        return value.state_dict()

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(
                self.world_model, rank0_only=False
            ):
                value.load_state_dict(state_dict, strict=False)
            return
        if (
            key == "world_model_optimizer"
            and self.world_model_optimizer is not None
            and self.world_model is not None
        ):
            self.distributed.load_optimizer_state_dict(
                self.world_model, self.world_model_optimizer, state_dict
            )
            return
        value.load_state_dict(state_dict, **kwargs)

    # ---- latent extraction ----

    def _get_image_bpe_set(self) -> set[int]:
        if self._image_bpe_set_cache is not None:
            return self._image_bpe_set_cache
        if self.encoder is None:
            raise RuntimeError(
                "encoder must be built before extracting image-token positions"
            )
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        self._image_bpe_set_cache = set(int(x) for x in vocab_mapping.bpe2img.keys())
        return self._image_bpe_set_cache

    @torch.no_grad()
    def _encode_chameleon_latents(
        self, input_ids_list: list[list[int]]
    ) -> torch.Tensor:
        """Return frozen Chameleon image-block latents.

        latent_mode:
          - image_pooled: [B, hidden=4096], mean over selected image block(s).
          - image_tokens: [B, N_img * N_views, hidden=4096], concatenating
            selected image blocks along the token dimension.
        """
        if self.encoder is None:
            raise RuntimeError("encoder is required")
        if not input_ids_list:
            hidden_dim = int(
                OmegaConf.select(self.cfg, "world_model.latent_dim", default=4096)
            )
            return torch.zeros((0, hidden_dim), device=self.device)

        labels_list = [[-100] * len(example) for example in input_ids_list]
        _, _, _, hidden_states, _, _, _ = self.encoder.backbone(
            input_ids=input_ids_list,
            labels=labels_list,
            training=True,
            output_hidden_states=True,
            att_mask=False,
        )

        from dreamer_vla.utils.wm_image_viz import extract_image_blocks

        img_bpe = self._get_image_bpe_set()
        which_blocks_cfg = OmegaConf.select(
            self.cfg, "latent.which_blocks", default=None
        )
        if which_blocks_cfg is None:
            which_blocks = [
                int(OmegaConf.select(self.cfg, "latent.which_block", default=-2))
            ]
        else:
            which_blocks = [int(block_idx) for block_idx in which_blocks_cfg]
        if not which_blocks:
            raise ValueError(
                "latent.which_blocks must contain at least one image block index"
            )
        n_img_tok = int(
            OmegaConf.select(self.cfg, "latent.n_image_tokens", default=256)
        )
        mode = str(OmegaConf.select(self.cfg, "latent.mode", default="image_pooled"))
        samples: list[torch.Tensor] = []
        for idx, seq in enumerate(input_ids_list):
            blocks = extract_image_blocks(list(seq))
            if not blocks:
                raise ValueError(f"sample {idx}: no image block found")
            block_hiddens: list[torch.Tensor] = []
            for which_block in which_blocks:
                bidx = which_block if which_block >= 0 else len(blocks) + which_block
                if not (0 <= bidx < len(blocks)):
                    raise ValueError(
                        f"sample {idx}: which_block={which_block} out of range for {len(blocks)} blocks"
                    )
                start, _end, block_ids = blocks[bidx]
                positions = [
                    start + off
                    for off, tok in enumerate(block_ids)
                    if int(tok) in img_bpe
                ]
                if len(positions) != n_img_tok:
                    raise ValueError(
                        f"sample {idx}: selected image block {which_block} has "
                        f"{len(positions)} image tokens, expected {n_img_tok}"
                    )
                pos_t = torch.tensor(
                    positions, device=hidden_states.device, dtype=torch.long
                )
                block_hiddens.append(hidden_states[idx].index_select(0, pos_t).float())
            image_h = torch.cat(block_hiddens, dim=0)
            if mode == "image_pooled":
                image_h = image_h.mean(dim=0)
            elif mode != "image_tokens":
                raise ValueError("latent.mode must be 'image_pooled' or 'image_tokens'")
            samples.append(image_h)
        return torch.stack(samples, dim=0).detach()

    def _build_world_model_batch(
        self, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        horizon = int(OmegaConf.select(self.cfg, "latent.horizon", default=-1))
        if isinstance(batch.get("wm_obs_input_ids_seq"), list) and isinstance(
            batch.get("action_seq"), torch.Tensor
        ):
            seq_ids = batch["wm_obs_input_ids_seq"]
            if not seq_ids:
                return None
            T = len(seq_ids[0])
            target_idx = horizon if horizon >= 0 else T + horizon
            if not (1 <= target_idx < T):
                raise ValueError(
                    f"latent.horizon resolves to target_idx={target_idx}, but T={T}"
                )
            context_frames = int(
                OmegaConf.select(
                    self.cfg,
                    "latent.context_frames",
                    default=OmegaConf.select(
                        self.cfg, "world_model.context_frames", default=1
                    ),
                )
            )
            if not (1 <= context_frames <= target_idx):
                raise ValueError(
                    f"latent.context_frames={context_frames} must be in [1,{target_idx}]"
                )
            selected_ids = [
                list(step_ids)
                for sample in seq_ids
                for step_ids in sample[: target_idx + 1]
            ]
            latents = self._encode_chameleon_latents(selected_ids)
            batch_size = len(seq_ids)
            latent_seq = latents.view(batch_size, target_idx + 1, *latents.shape[1:])
            action_seq = batch["action_seq"][:, context_frames : target_idx + 1]
            latent = latent_seq[:, context_frames - 1]
            target_latent = latent_seq[:, -1]
        elif isinstance(batch.get("wm_obs_input_ids"), list) and isinstance(
            batch.get("wm_next_obs_input_ids"), list
        ):
            cur_ids = batch["wm_obs_input_ids"]
            target_ids = batch["wm_next_obs_input_ids"]
            action = batch.get("action")
            if (
                not isinstance(action, torch.Tensor)
                or action.ndim != 3
                or action.shape[1] < 1
            ):
                return None
            action_seq = action[:, :1]
            latents = self._encode_chameleon_latents(cur_ids + target_ids)
            batch_size = len(cur_ids)
            latent = latents[:batch_size]
            target_latent = latents[batch_size:]
            latent_seq = torch.stack([latent, target_latent], dim=1)
            context_frames = 1
        else:
            return None

        model_dtype = (
            next(self.world_model.parameters()).dtype
            if self.world_model is not None
            else latent.dtype
        )
        latent = latent.to(self.device, dtype=model_dtype)
        target_latent = target_latent.to(self.device, dtype=model_dtype)
        latent_seq = latent_seq.to(self.device, dtype=model_dtype)
        action_seq = action_seq.to(self.device, dtype=model_dtype)
        return {
            "latent": latent,
            "target_latent": target_latent,
            "latent_seq": latent_seq,
            "action_seq": action_seq,
            "context_frames": int(context_frames),
        }

    # ---- validation ----

    @torch.no_grad()
    def evaluate_val_loss(
        self, val_dataloader: DataLoader, split_name: str
    ) -> dict[str, float]:
        self.world_model.eval()
        metrics_sum: dict[str, float] = {}
        count = 0
        for batch in val_dataloader:
            wm_batch = self._build_world_model_batch(batch)
            if wm_batch is None:
                continue
            out = self.world_model(wm_batch)
            for key, value in out.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value.item())
            count += 1
        self.world_model.train()
        count_global = max(self.distributed.reduce_sum(count), 1.0)
        reduced: dict[str, float] = {}
        for key, value in metrics_sum.items():
            reduced[f"val_{split_name}_{key}"] = (
                self.distributed.reduce_sum(value) / count_global
            )
        if self.distributed.is_main_process and reduced:
            display: list[str] = []
            for name, keys, fmt in (
                ("loss", (f"val_{split_name}_loss",), ".4f"),
                ("rec", (f"val_{split_name}_rec_loss",), ".4f"),
                (
                    "mse",
                    (
                        f"val_{split_name}_image_mse",
                        f"val_{split_name}_mse_loss",
                        f"val_{split_name}_latent_mse_loss",
                    ),
                    ".4f",
                ),
                ("psnr", (f"val_{split_name}_image_psnr",), ".2f"),
                ("cos", (f"val_{split_name}_pred_target_cos",), ".4f"),
            ):
                value = self._first_finite_metric(reduced, *keys)
                if value is not None:
                    display.append(f"{name}={value:{fmt}}")
            if display:
                print(f"  [Val {split_name}] " + " ".join(display))
        return reduced

    # ---- run ----

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        cfg = copy.deepcopy(self.cfg)
        if self.distributed.is_main_process:
            print(f"{self.__class__.__name__} begin.")

        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset)
        train_dataloader = self.make_distributed_dataloader(dataset, cfg.dataloader)

        val_dataloaders = self.make_val_dataloaders(cfg)

        encoder_cfg = self._build_frozen_encoder_cfg(cfg)
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is None:
            raise ValueError("world_model config is required")
        if OmegaConf.select(world_model_cfg, "latent_dim", default=None) is None:
            with open_dict(world_model_cfg):
                world_model_cfg.latent_dim = int(
                    self.infer_hidden_dim_from_encoder(self.encoder) or 4096
                )
        self.world_model = hydra.utils.instantiate(world_model_cfg).to(self.device)

        if bool(OmegaConf.select(cfg, "training.debug", default=False)):
            cfg.training.num_epochs = 1
            cfg.training.max_train_steps = 3
            cfg.training.checkpoint_every = 1

        fsdp_precision = str(
            OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16")
        )
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(fsdp_precision, torch.bfloat16)
        self.world_model = self.world_model.to(dtype=dtype)
        self.world_model = self.distributed.wrap_trainable_module(self.world_model)
        self.world_model_optimizer = build_optimizer(
            self.world_model, cfg.optim.world_model
        )
        self.resume(cfg)
        for pg in self.world_model_optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
        lr_scheduler = get_scheduler(
            str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant")),
            optimizer=self.world_model_optimizer,
            num_warmup_steps=int(
                OmegaConf.select(cfg, "training.lr_warmup_steps", default=0)
            ),
            num_training_steps=max(
                1,
                (len(train_dataloader) * int(cfg.training.num_epochs))
                // int(cfg.training.gradient_accumulate_every),
            ),
            last_epoch=self.global_step - 1,
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        self.distributed.barrier()
        log_name = str(
            getattr(self, "log_filename", "chameleon_latent_wm_logs.json.txt")
        )
        log_path = os.path.join(self.output_dir, log_name)
        try:
            with self.distributed.logger_context(log_path) as logger:
                reached_max_steps = False
                for _ in range(int(cfg.training.num_epochs)):
                    self.set_dataloader_epoch(train_dataloader, self.epoch)
                    epoch_metrics: dict[str, list[float]] = {}
                    self.world_model.train()
                    accum_steps = max(1, int(cfg.training.gradient_accumulate_every))
                    log_every = int(
                        OmegaConf.select(cfg, "training.log_every", default=1)
                    )
                    micro_batches = 0
                    self.world_model_optimizer.zero_grad(
                        set_to_none=bool(cfg.optim.get("zero_grad_set_to_none", True))
                    )
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        disable=not self.distributed.is_main_process,
                        leave=False,
                        mininterval=float(
                            OmegaConf.select(
                                cfg, "training.tqdm_interval_sec", default=1.0
                            )
                        ),
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            wm_batch = self._build_world_model_batch(batch)
                            if wm_batch is None:
                                continue
                            out = self.world_model(wm_batch)
                            raw_loss = out["loss"]
                            loss = raw_loss / accum_steps
                            loss.backward()
                            micro_batches += 1

                            reached_max_steps = (
                                cfg.training.max_train_steps is not None
                                and batch_idx >= (int(cfg.training.max_train_steps) - 1)
                            )
                            do_optimizer_step = (
                                micro_batches % accum_steps
                            ) == 0 or reached_max_steps
                            if do_optimizer_step:
                                grad_clip_norm = cfg.optim.get("grad_clip_norm")
                                if grad_clip_norm is not None:
                                    grad_norm = self.distributed.clip_grad_norm(
                                        self.world_model, float(grad_clip_norm)
                                    )
                                else:
                                    grad_norm = float("nan")
                                self.world_model_optimizer.step()
                                self.world_model_optimizer.zero_grad(
                                    set_to_none=bool(
                                        cfg.optim.get("zero_grad_set_to_none", True)
                                    )
                                )
                                lr_scheduler.step()
                            else:
                                grad_norm = float("nan")

                            local_metrics: dict[str, float] = {}
                            for key, value in out.items():
                                if key == "_loss":
                                    continue
                                if isinstance(value, torch.Tensor) and value.ndim == 0:
                                    local_metrics[f"train_{key}"] = float(value.item())
                                    epoch_metrics.setdefault(key, []).append(
                                        float(value.item())
                                    )
                            diag_every = int(
                                OmegaConf.select(
                                    cfg,
                                    "diagnostics.action_sensitivity_every",
                                    default=100,
                                )
                            )
                            if diag_every > 0 and (batch_idx % diag_every) == 0:
                                diag_fn = getattr(
                                    self.distributed.unwrap_module(self.world_model),
                                    "action_sensitivity_metrics",
                                    None,
                                )
                                if callable(diag_fn):
                                    diag = diag_fn(wm_batch)
                                    for key, value in diag.items():
                                        if (
                                            isinstance(value, torch.Tensor)
                                            and value.ndim == 0
                                        ):
                                            local_metrics[f"train_{key}"] = float(
                                                value.item()
                                            )
                            if math.isfinite(float(grad_norm)):
                                local_metrics["train_grad_norm"] = float(grad_norm)
                            local_metrics["lr"] = float(lr_scheduler.get_last_lr()[0])
                            local_metrics["optimizer_step"] = float(do_optimizer_step)
                            reduced = self.distributed.reduce_mean_dict(local_metrics)
                            step_log = {
                                **reduced,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                            }
                            if (
                                do_optimizer_step
                                and log_every > 0
                                and (self.global_step % log_every) == 0
                            ):
                                logger.log(step_log)
                                self.log_metrics(step_log, step=self.global_step)
                            postfix: dict[str, float] = {}
                            for name, keys in (
                                ("loss", ("train_loss",)),
                                ("rec", ("train_rec_loss",)),
                                (
                                    "mse",
                                    (
                                        "train_image_mse",
                                        "train_mse_loss",
                                        "train_latent_mse_loss",
                                    ),
                                ),
                                ("psnr", ("train_image_psnr",)),
                                ("flow", ("train_flow_loss",)),
                                ("cos", ("train_pred_target_cos",)),
                            ):
                                value = self._first_finite_metric(step_log, *keys)
                                if value is not None:
                                    postfix[name] = value
                            tepoch.set_postfix(postfix, refresh=False)
                            if do_optimizer_step:
                                self.global_step += 1
                            if reached_max_steps:
                                break

                    if not epoch_metrics:
                        self.epoch += 1
                        continue
                    epoch_log: dict[str, float | int] = {
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                    }
                    for key, values in epoch_metrics.items():
                        epoch_log[f"train_{key}"] = self.distributed.reduce_mean(
                            sum(values) / max(len(values), 1)
                        )
                    eval_every = int(
                        OmegaConf.select(cfg, "eval.eval_every", default=1)
                    )
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            epoch_log.update(self.evaluate_val_loss(val_dl, split_name))
                    logger.log(epoch_log)
                    self.log_metrics(epoch_log, step=self.global_step)

                    if (self.epoch % int(cfg.training.checkpoint_every)) == 0:
                        if bool(
                            OmegaConf.select(
                                cfg, "checkpoint.save_last_ckpt", default=True
                            )
                        ):
                            self.save_checkpoint()
                        metric_dict = {
                            key.replace("/", "_"): value
                            for key, value in epoch_log.items()
                        }
                        topk_path = None
                        if self.distributed.is_main_process:
                            topk_path = topk_manager.get_ckpt_path(metric_dict)
                        topk_path = self.distributed.broadcast_object(topk_path)
                        if topk_path is not None:
                            self.save_checkpoint(path=topk_path)

                    self.epoch += 1
                    if reached_max_steps:
                        break
        finally:
            self.distributed.barrier()
            self.distributed.cleanup()
        return history


__all__ = ["ChameleonLatentActionWMRunner"]
