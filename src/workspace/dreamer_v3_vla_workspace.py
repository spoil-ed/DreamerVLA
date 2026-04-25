"""DreamerV3-style DreamerVLA workspace.

Training loop is identical in structure to `DreamerVLAWorkspace`: each batch
runs a WM pretrain step (Phase-1) followed by an actor-critic imagination
step (Phase-2). The differences live in Phase-2 / critic internals:

  • `TwohotCritic` (symlog-binned categorical) replaces the scalar-MSE critic
  • A Polyak-averaged `target_critic` supplies bootstrap values
  • `ReturnPercentileTracker` normalises actor advantages

See `src/algorithms/dreamer_v3_vla.py` for the step implementation.

The Phase-1 WM update and the TSSM itself are unchanged, so WM checkpoints
from `pretokenize_wm_*` / `dreamer_vla_*` runs remain loadable here. The
critic state is *not* cross-compatible with `dreamer_vla_*` ckpts (different
head shape); use `resume=false` or ignore-missing loading for the critic.
"""
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
from omegaconf import DictConfig, OmegaConf

from src.algorithms.dreamer_vla import world_model_pretrain_step
from src.algorithms.dreamer_v3_vla import imagine_actor_critic_step_v3
from src.dataloader import BaseDataset
from src.models.critic.twohot_critic import ReturnPercentileTracker
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.ema import EMAHelper
from src.utils.optim import build_optimizer
from src.utils.torch_utils import freeze_module
from src.workspace.dreamer_vla_workspace import DreamerVLAWorkspace, _copy_to_cpu
from torch.utils.data import DataLoader


class DreamerV3VLAWorkspace(DreamerVLAWorkspace):
    """Closed-loop DreamerVLA with DreamerV3 actor-critic (twohot + target critic)."""

    default_output_dir = "/home/user01/liops/workspace/DreamerVLA/data/outputs/dreamer_v3_vla"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir=output_dir)
        self.target_critic = None
        self.return_tracker: ReturnPercentileTracker | None = None

    # ── checkpoint plumbing ───────────────────────────────────────────────

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "target_critic" and self.target_critic is not None:
            with self.distributed.model_state_dict_context(self.target_critic):
                return self.target_critic.state_dict()
        if key == "return_tracker" and self.return_tracker is not None:
            return self.return_tracker.state_dict()
        return super()._state_dict_for_checkpoint(key, value)

    def _load_state_dict_from_checkpoint(
        self, key: str, value: Any, state_dict: dict[str, Any], **kwargs: Any
    ) -> None:
        if key == "target_critic" and self.target_critic is not None:
            with self.distributed.model_state_dict_context(self.target_critic):
                value.load_state_dict(state_dict, **kwargs)
            return
        if key == "return_tracker" and self.return_tracker is not None:
            value.load_state_dict(state_dict)
            return
        super()._load_state_dict_from_checkpoint(key, value, state_dict, **kwargs)

    # ── main training loop ────────────────────────────────────────────────

    def run(self) -> list[dict[str, float | str | int]]:  # noqa: C901
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("DreamerV3VLA Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        # ── dataset & dataloader ────────────────────────────────────────
        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset)

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

        # ── encoder (frozen) ───────────────────────────────────────────
        encoder_cfg = self._build_frozen_encoder_cfg(cfg)
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        freeze_module(self.encoder)

        # ── world model ────────────────────────────────────────────────
        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is None:
            raise ValueError("`world_model` config section is required.")
        wm_hidden_dim = (
            self.infer_hidden_dim_from_dataset(dataset)
            or self.infer_hidden_dim_from_encoder(self.encoder)
        )
        if wm_hidden_dim is not None:
            self.world_model = hydra.utils.instantiate(world_model_cfg, hidden_dim=wm_hidden_dim).to(self.device)
        else:
            self.world_model = hydra.utils.instantiate(world_model_cfg).to(self.device)

        fsdp_precision = str(OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16"))
        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.world_model = self.world_model.to(dtype=_dtype_map.get(fsdp_precision, torch.bfloat16))
        self.world_model = self.distributed.wrap_trainable_module(self.world_model)

        wm_optim_cfg = OmegaConf.select(cfg, "optim.world_model")
        if wm_optim_cfg is None:
            raise ValueError("`optim.world_model` must be configured.")
        self.world_model_optimizer = build_optimizer(self.world_model, wm_optim_cfg)

        # ── policy (Dreamer actor) ─────────────────────────────────────
        policy_cfg = OmegaConf.select(cfg, "policy")
        if policy_cfg is None:
            raise ValueError("`policy` config section is required.")
        self.policy = hydra.utils.instantiate(policy_cfg).to(self.device)
        self.policy = self.distributed.wrap_trainable_module(self.policy)

        policy_optim_cfg = OmegaConf.select(cfg, "optim.policy")
        if policy_optim_cfg is None:
            raise ValueError("`optim.policy` must be configured.")
        self.policy_optimizer = build_optimizer(self.policy, policy_optim_cfg)

        # ── twohot critic (online + target copy) ───────────────────────
        critic_cfg = OmegaConf.select(cfg, "critic")
        if critic_cfg is None:
            raise ValueError("`critic` config section is required.")

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.target_critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        freeze_module(self.target_critic)  # updated by Polyak averaging, never by optimiser
        self.critic = self.distributed.wrap_trainable_module(self.critic)

        critic_optim_cfg = OmegaConf.select(cfg, "optim.critic")
        if critic_optim_cfg is None:
            raise ValueError("`optim.critic` must be configured.")
        self.critic_optimizer = build_optimizer(self.critic, critic_optim_cfg)

        # ── return percentile tracker ──────────────────────────────────
        tracker_cfg = OmegaConf.select(cfg, "algorithm.return_tracker", default=None) or {}
        self.return_tracker = ReturnPercentileTracker(
            decay=float(OmegaConf.select(cfg, "algorithm.return_tracker.decay", default=0.99)),
            low=float(OmegaConf.select(cfg, "algorithm.return_tracker.low", default=0.05)),
            high=float(OmegaConf.select(cfg, "algorithm.return_tracker.high", default=0.95)),
        )

        if bool(OmegaConf.select(cfg, "training.use_ema", default=False)) and self.world_model_ema is None:
            self.world_model_ema = EMAHelper(
                self.world_model,
                decay=float(OmegaConf.select(cfg, "ema.decay", default=0.9999)),
                update_after_step=int(OmegaConf.select(cfg, "ema.update_after_step", default=0)),
            )

        self.resume(cfg)

        lr_scheduler_name = str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant"))
        lr_warmup_steps = int(OmegaConf.select(cfg, "training.lr_warmup_steps", default=0))
        total_training_steps = (
            len(train_dataloader) * int(cfg.training.num_epochs)
        ) // int(cfg.training.gradient_accumulate_every)
        wm_lr_scheduler = get_scheduler(
            lr_scheduler_name, optimizer=self.world_model_optimizer,
            num_warmup_steps=lr_warmup_steps, num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )
        policy_lr_scheduler = get_scheduler(
            lr_scheduler_name, optimizer=self.policy_optimizer,
            num_warmup_steps=lr_warmup_steps, num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )
        critic_lr_scheduler = get_scheduler(
            lr_scheduler_name, optimizer=self.critic_optimizer,
            num_warmup_steps=lr_warmup_steps, num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )

        run_wm_phase = bool(OmegaConf.select(cfg, "training.run_wm_phase", default=True))
        run_ac_phase = bool(OmegaConf.select(cfg, "training.run_actor_critic_phase", default=True))
        algorithm_cfg = OmegaConf.select(cfg, "algorithm")
        if algorithm_cfg is None:
            raise ValueError("`algorithm` config section is required.")
        optim_cfg = OmegaConf.select(cfg, "optim")

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

        train_log_path = os.path.join(self.output_dir, "dreamer_v3_vla_logs.json.txt")
        train_logger_cm = self.distributed.logger_context(train_log_path)

        try:
            with train_logger_cm as train_json_logger:
                reached_max_steps = False
                for _local_epoch_idx in range(cfg.training.num_epochs):
                    if sampler is not None:
                        sampler.set_epoch(self.epoch)

                    step_log: dict[str, float | str | int] = {}
                    epoch_wm_losses: list[float] = []
                    epoch_actor_losses: list[float] = []
                    epoch_critic_losses: list[float] = []
                    epoch_returns: list[float] = []
                    epoch_rewards: list[float] = []
                    epoch_scales: list[float] = []

                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Epoch {self.epoch}",
                        disable=not self.distributed.is_main_process,
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            local_metrics: dict[str, float] = {}
                            step_had_update = False

                            # Phase 1 — world-model pretraining
                            if run_wm_phase:
                                wm_batch = self._build_wm_pretrain_batch(batch)
                                if wm_batch is not None:
                                    self.world_model.train()
                                    self.policy.eval()
                                    self.critic.eval()
                                    wm_metrics = world_model_pretrain_step(
                                        policy=self.policy,
                                        world_model=self.world_model,
                                        optimizer=self.world_model_optimizer,
                                        batch=wm_batch,
                                        device=self.device,
                                        optim_cfg=optim_cfg,
                                    )
                                    wm_lr_scheduler.step()
                                    if self.world_model_ema is not None:
                                        self.world_model_ema.step(self.world_model)
                                    epoch_wm_losses.append(wm_metrics["loss"])
                                    local_metrics["train_wm_loss"] = wm_metrics["loss"]
                                    local_metrics["train_wm_transition_loss"] = wm_metrics["transition_loss"]
                                    local_metrics["train_wm_reward_loss"] = wm_metrics["reward_loss"]
                                    local_metrics["train_wm_grad_norm"] = wm_metrics["grad_norm"]
                                    local_metrics["wm_lr"] = float(wm_lr_scheduler.get_last_lr()[0])
                                    step_had_update = True

                            # Phase 2 — DreamerV3 actor-critic imagination
                            if run_ac_phase:
                                ac_batch = self._build_actor_critic_batch(batch)
                                if ac_batch is not None:
                                    self.world_model.eval()
                                    ac_metrics = imagine_actor_critic_step_v3(
                                        policy=self.policy,
                                        world_model=self.world_model,
                                        critic=self.critic,
                                        target_critic=self.target_critic,
                                        actor_optimizer=self.policy_optimizer,
                                        critic_optimizer=self.critic_optimizer,
                                        return_tracker=self.return_tracker,
                                        obs=ac_batch["obs"],
                                        device=self.device,
                                        algorithm_cfg=algorithm_cfg,
                                        optim_cfg=optim_cfg,
                                    )
                                    epoch_actor_losses.append(ac_metrics["actor_loss"])
                                    epoch_critic_losses.append(ac_metrics["critic_loss"])
                                    epoch_returns.append(ac_metrics["returns_mean"])
                                    epoch_rewards.append(ac_metrics["reward_mean"])
                                    epoch_scales.append(ac_metrics["return_scale"])
                                    local_metrics.update({
                                        "train_actor_loss": ac_metrics["actor_loss"],
                                        "train_critic_loss": ac_metrics["critic_loss"],
                                        "train_returns_mean": ac_metrics["returns_mean"],
                                        "train_returns_std": ac_metrics["returns_std"],
                                        "train_return_scale": ac_metrics["return_scale"],
                                        "train_reward_mean": ac_metrics["reward_mean"],
                                        "train_value_mean": ac_metrics["value_mean"],
                                        "train_actor_grad_norm": ac_metrics["actor_grad_norm"],
                                        "train_critic_grad_norm": ac_metrics["critic_grad_norm"],
                                    })
                                    policy_lr_scheduler.step()
                                    critic_lr_scheduler.step()
                                    local_metrics["policy_lr"] = float(policy_lr_scheduler.get_last_lr()[0])
                                    local_metrics["critic_lr"] = float(critic_lr_scheduler.get_last_lr()[0])
                                    step_had_update = True

                            if not step_had_update:
                                continue

                            reduced = self.distributed.reduce_mean_dict(local_metrics)
                            step_log = {**reduced, "global_step": self.global_step, "epoch": self.epoch}

                            pbar_postfix: dict[str, float] = {}
                            if "train_wm_loss" in reduced:
                                pbar_postfix["wm"] = reduced["train_wm_loss"]
                            if "train_actor_loss" in reduced:
                                pbar_postfix["actor"] = reduced["train_actor_loss"]
                            if "train_critic_loss" in reduced:
                                pbar_postfix["critic"] = reduced["train_critic_loss"]
                            if "train_returns_mean" in reduced:
                                pbar_postfix["G"] = reduced["train_returns_mean"]
                            if pbar_postfix:
                                tepoch.set_postfix(refresh=False, **pbar_postfix)

                            is_last_batch = batch_idx == len(train_dataloader) - 1
                            if not is_last_batch:
                                train_json_logger.log(step_log)
                                self.global_step += 1

                            if (
                                cfg.training.max_train_steps is not None
                                and batch_idx >= cfg.training.max_train_steps - 1
                            ):
                                reached_max_steps = True
                                break

                    if not epoch_wm_losses and not epoch_actor_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    if epoch_wm_losses:
                        wm_n = max(self.distributed.reduce_sum(len(epoch_wm_losses)), 1.0)
                        step_log["epoch_wm_loss"] = self.distributed.reduce_sum(sum(epoch_wm_losses)) / wm_n
                    if epoch_actor_losses:
                        ac_n = max(self.distributed.reduce_sum(len(epoch_actor_losses)), 1.0)
                        step_log["epoch_actor_loss"] = self.distributed.reduce_sum(sum(epoch_actor_losses)) / ac_n
                        step_log["epoch_critic_loss"] = self.distributed.reduce_sum(sum(epoch_critic_losses)) / ac_n
                        step_log["epoch_returns_mean"] = self.distributed.reduce_sum(sum(epoch_returns)) / ac_n
                        step_log["epoch_reward_mean"] = self.distributed.reduce_sum(sum(epoch_rewards)) / ac_n
                        step_log["epoch_return_scale"] = self.distributed.reduce_sum(sum(epoch_scales)) / ac_n

                    step_log.setdefault("epoch_wm_loss", float("inf"))
                    step_log.setdefault("epoch_actor_loss", float("inf"))
                    step_log.setdefault("epoch_critic_loss", float("inf"))

                    eval_every = int(OmegaConf.select(cfg, "eval.eval_every", default=1))
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            step_log.update(self.evaluate_val_loss(val_dl, split_name))

                    train_json_logger.log(step_log)
                    history.append(dict(step_log))

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
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


__all__ = ["DreamerV3VLAWorkspace"]
