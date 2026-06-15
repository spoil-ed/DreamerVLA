from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import hydra
import torch
import tqdm
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.distributed import NopretokenizeSFTDistributedHelper
from dreamervla.utils.checkpoint_util import TopKCheckpointManager
from dreamervla.utils.json_logger import JsonLogger
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OpenVLAOFTTrainingRunner(BaseRunner):
    """OpenVLA-OFT VLA continuation training inside DreamerVLA."""

    runner_name = "openvla_oft_impl"
    runner_status = "implementation"
    runner_family = "vla"
    include_keys = ("global_step", "epoch", "dataset_statistics")
    exclude_keys = tuple()
    checkpoint_restore_output_dir = True
    default_output_dir = str(
        PROJECT_ROOT / "data" / "outputs" / "vla" / "openvla_oft_goal"
    )

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
        if self.distributed.uses_fsdp:
            raise ValueError(
                "OpenVLA-OFT runner supports single-process/DDP training; use distributed_strategy=ddp."
            )
        self.rank = self.distributed.rank
        self.local_rank = self.distributed.local_rank
        self.world_size = self.distributed.world_size
        self.device = self.distributed.resolve_device(
            str(OmegaConf.select(config, "trainer.device", default="auto"))
        )
        set_seed(int(config.seed) + self.rank)
        self.policy = None
        self.optimizer = None
        self.dataset_statistics: dict[str, Any] = {}
        if self.distributed.is_main_process:
            self.print_config()

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "policy" and self.policy is not None:
            return self.policy.state_dict()
        return value.state_dict()

    def _save_oft_components(self, step: int) -> None:
        if not self.distributed.is_main_process or self.policy is None:
            return
        save_dir = os.path.join(self.output_dir, f"openvla_oft_components--{int(step)}")
        adapter_dir = os.path.join(save_dir, "lora_adapter")
        os.makedirs(save_dir, exist_ok=True)
        vla = self.distributed.unwrap_module(self.policy).vla
        if hasattr(vla, "save_pretrained"):
            vla.save_pretrained(adapter_dir)
        processor = getattr(
            self.distributed.unwrap_module(self.policy), "processor", None
        )
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(save_dir)
        action_head = getattr(
            self.distributed.unwrap_module(self.policy), "action_head", None
        )
        if action_head is not None:
            torch.save(
                action_head.state_dict(),
                os.path.join(save_dir, f"action_head--{int(step)}_checkpoint.pt"),
            )
        proprio_projector = getattr(
            self.distributed.unwrap_module(self.policy), "proprio_projector", None
        )
        if proprio_projector is not None:
            torch.save(
                proprio_projector.state_dict(),
                os.path.join(save_dir, f"proprio_projector--{int(step)}_checkpoint.pt"),
            )

    def run(self) -> list[dict[str, float | int | str]]:
        history: list[dict[str, float | int | str]] = []
        cfg = copy.deepcopy(self.cfg)
        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            print("OpenVLA-OFT Runner begin.", flush=True)

        self.policy = hydra.utils.instantiate(cfg.policy).to(self.device)
        self.policy = self.distributed.wrap_trainable_module(self.policy)
        self.optimizer = build_optimizer(self.policy, cfg.optim.policy)
        self.resume(cfg)
        for param_group in self.optimizer.param_groups:
            param_group.setdefault("initial_lr", param_group["lr"])

        dataset_factory = hydra.utils.instantiate(cfg.dataset)
        if not hasattr(dataset_factory, "build"):
            raise TypeError(
                "OpenVLA-OFT dataset config must instantiate a factory with build(policy, train=...)."
            )
        bundle = dataset_factory.build(
            self.distributed.unwrap_module(self.policy), train=True
        )
        dataloader = bundle.dataloader
        self.dataset_statistics = bundle.dataset_statistics

        max_train_steps = int(
            OmegaConf.select(cfg, "training.max_train_steps", default=1000)
        )
        grad_accumulation = int(
            OmegaConf.select(cfg, "training.gradient_accumulate_every", default=1)
        )
        lr_scheduler = get_scheduler(
            str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant")),
            optimizer=self.optimizer,
            num_warmup_steps=int(
                OmegaConf.select(cfg, "training.lr_warmup_steps", default=0)
            ),
            num_training_steps=max_train_steps,
            last_epoch=self.global_step - 1,
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        log_path = os.path.join(self.output_dir, "openvla_oft_logs.json.txt")
        checkpoint_every = int(
            OmegaConf.select(cfg, "training.checkpoint_every", default=500)
        )
        save_components_every = int(
            OmegaConf.select(cfg, "training.save_components_every", default=0)
        )

        self.policy.train()
        self.optimizer.zero_grad(
            set_to_none=bool(
                OmegaConf.select(cfg, "optim.zero_grad_set_to_none", default=True)
            )
        )
        with JsonLogger(log_path) as logger:
            progress = tqdm.tqdm(
                total=max_train_steps,
                initial=self.global_step,
                disable=not self.distributed.is_main_process,
                desc="OpenVLA-OFT train",
                mininterval=float(
                    OmegaConf.select(cfg, "training.tqdm_interval_sec", default=1.0)
                ),
            )
            while self.global_step < max_train_steps:
                sampler = getattr(dataloader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(int(self.epoch))
                for batch_idx, batch in enumerate(dataloader):
                    loss_policy = self.distributed.unwrap_module(self.policy)
                    loss, metrics = loss_policy.compute_loss(batch, device=self.device)
                    (loss / grad_accumulation).backward()
                    if (batch_idx + 1) % grad_accumulation != 0:
                        continue
                    grad_clip_norm = OmegaConf.select(
                        cfg, "optim.grad_clip_norm", default=None
                    )
                    if grad_clip_norm is not None:
                        self.distributed.clip_grad_norm(
                            self.policy, float(grad_clip_norm)
                        )
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad(
                        set_to_none=bool(
                            OmegaConf.select(
                                cfg, "optim.zero_grad_set_to_none", default=True
                            )
                        )
                    )

                    reduced = self.distributed.reduce_mean_dict(metrics)
                    step_log: dict[str, float | int | str] = {
                        **{f"train_{key}": value for key, value in reduced.items()},
                        "lr": float(lr_scheduler.get_last_lr()[0]),
                        "global_step": int(self.global_step),
                        "epoch": int(self.epoch),
                    }
                    logger.log(step_log)
                    self.log_metrics(step_log, step=self.global_step)
                    history.append(step_log)
                    progress.set_postfix(
                        refresh=False, loss=float(step_log["train_loss_value"])
                    )
                    progress.update(1)

                    if (
                        self.global_step > 0
                        and self.global_step % checkpoint_every == 0
                    ):
                        self.save_checkpoint()
                        metric_dict = {
                            key.replace("/", "_"): value
                            for key, value in step_log.items()
                        }
                        topk_path = (
                            topk_manager.get_ckpt_path(metric_dict)
                            if self.distributed.is_main_process
                            else None
                        )
                        topk_path = self.distributed.broadcast_object(topk_path)
                        if topk_path is not None:
                            self.save_checkpoint(path=topk_path)
                    if (
                        save_components_every > 0
                        and self.global_step > 0
                        and self.global_step % save_components_every == 0
                    ):
                        self._save_oft_components(self.global_step)

                    self.global_step += 1
                    if self.global_step >= max_train_steps:
                        break
                self.epoch += 1
        self.save_checkpoint()
        if save_components_every > 0:
            self._save_oft_components(self.global_step)
        self.distributed.cleanup()
        return history


__all__ = ["OpenVLAOFTTrainingRunner"]
