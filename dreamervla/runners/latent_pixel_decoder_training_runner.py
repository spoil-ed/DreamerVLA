"""Native torch DDP training for a frozen-VLA latent-to-pixel decoder."""

from __future__ import annotations

import contextlib
import random
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils._pytree import tree_map

from dreamervla.models.embodiment.pi05.pytree import register_pytree_dataclasses
from dreamervla.models.embodiment.pi05.sft_data import (
    configured_download_endpoint,
    get_official_openpi_sft_num_batches,
    openpi_torch_loader,
)
from dreamervla.models.embodiment.world_model.latent_pixel_decoder import (
    latent_pixel_reconstruction_loss,
)
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.vla_sft_training_runner import _policy_hydra_config
from dreamervla.utils.training.distributed import NopretokenizeSFTDistributedHelper


class LatentPixelDecoderTrainingRunner(BaseRunner):
    """Train only a pixel decoder over frozen π0.5 PaliGemma prefix tokens."""

    runner_name = "latent_pixel_decoder"
    runner_status = "current"
    runner_family = "world_model"
    include_keys = (
        "global_step",
        "epoch",
        "_data_epoch",
        "_data_iter_offset",
        "_data_generator_state",
    )
    exclude_keys = ("policy", "data_loader", "data_iterator", "distributed")

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.distributed: NopretokenizeSFTDistributedHelper | None = None
        self.device = torch.device("cpu")
        self.policy: torch.nn.Module | None = None
        self.pixel_decoder: torch.nn.Module | None = None
        self.pixel_decoder_optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.data_loader: Any | None = None
        self.data_iterator: Any | None = None
        self.trajectory_replay: Any | None = None
        self._data_epoch = 0
        self._data_iter_offset = 0
        self._data_generator_state: torch.Tensor | None = None
        self._checkpoint_world_size: int | None = None

    def _checkpoint_metadata(self) -> dict[str, Any]:
        """Persist the DDP size needed to resume the data cursor safely."""

        metadata = super()._checkpoint_metadata()
        world_size = 1 if self.distributed is None else int(self.distributed.world_size)
        return {**metadata, "world_size": world_size}

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        restore_rng: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load a decoder checkpoint, tolerating an intentional DDP resize."""

        checkpoint_world_size = payload.get("world_size")
        if not isinstance(checkpoint_world_size, int) or checkpoint_world_size <= 0:
            rng_by_rank = payload.get("rng_by_rank")
            checkpoint_world_size = (
                len(rng_by_rank) if isinstance(rng_by_rank, (list, tuple)) else None
            )
        self._checkpoint_world_size = checkpoint_world_size
        current_world_size = 1 if self.distributed is None else int(self.distributed.world_size)
        world_size_changed = (
            checkpoint_world_size is not None and checkpoint_world_size != current_world_size
        )
        super().load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            restore_rng=restore_rng and not world_size_changed,
            **kwargs,
        )

    def setup(self) -> None:
        distributed_cfg = self.cfg.decoder_training.distributed
        if str(distributed_cfg.strategy).lower() != "ddp":
            raise ValueError("latent pixel decoder training supports native torch DDP only")
        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy="ddp",
            fsdp_mixed_precision="bf16",
            enable_activation_checkpointing=False,
            nccl_timeout_seconds=int(distributed_cfg.get("nccl_timeout_seconds", 1800)),
        )
        self.device = self.distributed.resolve_device(str(self.cfg.training.device))
        seed = int(self.cfg.seed) + int(self.distributed.rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        policy_cfg = _policy_hydra_config(self.cfg.latent_producer.policy_cfg)
        policy = hydra.utils.instantiate(policy_cfg)
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("latent_producer.policy_cfg must instantiate torch.nn.Module")
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        policy.eval().to(self.device)
        if not callable(getattr(policy, "encode_observation_prefix", None)):
            raise TypeError("latent producer must implement encode_observation_prefix(observation)")
        self.policy = policy

        replay_cfg = OmegaConf.select(self.cfg, "data.replay", default=None)
        if replay_cfg is not None:
            self.trajectory_replay = hydra.utils.instantiate(
                replay_cfg,
                encoder=policy,
                rank=int(self.distributed.rank),
                world_size=int(self.distributed.world_size),
            )
            source = str(getattr(self.trajectory_replay, "data_dir", ""))
            counts = self.trajectory_replay.task_episode_counts()
            expected_episodes = int(
                OmegaConf.select(
                    self.cfg,
                    "decoder_training.expected_episodes",
                    default=0,
                )
                or 0
            )
            if expected_episodes > 0 and sum(counts.values()) != expected_episodes:
                raise RuntimeError(
                    "collected pixel replay episode count mismatch: "
                    f"expected {expected_episodes}, found {sum(counts.values())}"
                )
        else:
            loader_factory = hydra.utils.instantiate(self.cfg.data.loader)
            build_loader = getattr(loader_factory, "build", None)
            if not callable(build_loader):
                raise TypeError("data.loader must instantiate RLinf's OpenPI loader factory")
            loader_bundle = build_loader(
                rank=int(self.distributed.rank),
                world_size=int(self.distributed.world_size),
            )
            self.data_loader = getattr(loader_bundle, "data_loader", None)
            source = getattr(loader_bundle, "source", None)
            if self.data_loader is None or not isinstance(source, str):
                raise TypeError("RLinf OpenPI loader factory returned an invalid bundle")
            generator = getattr(openpi_torch_loader(self.data_loader), "generator", None)
            self._data_generator_state = (
                None if generator is None else generator.get_state().clone()
            )

        decoder = hydra.utils.instantiate(self.cfg.pixel_decoder)
        if not isinstance(decoder, torch.nn.Module):
            raise TypeError("pixel_decoder must instantiate torch.nn.Module")
        decoder.to(self.device)
        self.pixel_decoder = self.distributed.wrap_trainable_module(
            decoder,
            find_unused_parameters=bool(distributed_cfg.get("find_unused_parameters", False)),
            broadcast_buffers=bool(distributed_cfg.get("broadcast_buffers", False)),
            gradient_as_bucket_view=bool(distributed_cfg.get("gradient_as_bucket_view", True)),
        )
        optim_cfg = self.cfg.decoder_training.optim
        self.pixel_decoder_optimizer = torch.optim.AdamW(
            self.pixel_decoder.parameters(),
            lr=float(optim_cfg.lr),
            betas=(float(optim_cfg.adam_beta1), float(optim_cfg.adam_beta2)),
            eps=float(optim_cfg.adam_eps),
            weight_decay=float(optim_cfg.weight_decay),
        )
        self.lr_scheduler = _build_decoder_scheduler(
            self.pixel_decoder_optimizer,
            warmup_steps=int(optim_cfg.lr_warmup_steps),
            total_steps=int(self.cfg.training.max_steps),
        )

        super().setup()
        self.append_model_summary(
            {
                "family": "latent_token_pixel_decoder",
                "parameters": sum(parameter.numel() for parameter in decoder.parameters()),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in decoder.parameters()
                    if parameter.requires_grad
                ),
                "latent_producer": type(policy).__name__,
                "latent_producer_frozen": True,
                "dataset": source,
                "dataset_episodes": (
                    None
                    if self.trajectory_replay is None
                    else sum(self.trajectory_replay.task_episode_counts().values())
                ),
                "dataset_frames": (
                    None
                    if self.trajectory_replay is None
                    else int(self.trajectory_replay.num_transitions)
                ),
                "download_endpoint": (
                    configured_download_endpoint() if self.trajectory_replay is None else None
                ),
                "distributed_backend": "torch.nn.parallel.DistributedDataParallel",
            }
        )
        self.resume(self.cfg)
        self._restore_data_iterator()

    @property
    def gradient_accumulation(self) -> int:
        assert self.distributed is not None
        numerator = int(self.cfg.decoder_training.global_batch_size)
        denominator = int(self.cfg.decoder_training.micro_batch_size) * int(
            self.distributed.world_size
        )
        if numerator % denominator:
            raise ValueError(
                "decoder_training.global_batch_size must be divisible by "
                "micro_batch_size * world_size"
            )
        return numerator // denominator

    def _restore_data_iterator(self) -> None:
        if self.trajectory_replay is not None:
            micro_updates = int(self.global_step) * int(self.gradient_accumulation)
            batch_size = int(self.cfg.decoder_training.micro_batch_size)
            self.trajectory_replay.seek_update(micro_updates, batch_size=batch_size)
            steps_per_epoch = int(self.trajectory_replay.steps_per_epoch(batch_size=batch_size))
            self._data_epoch, self._data_iter_offset = divmod(
                micro_updates,
                steps_per_epoch,
            )
            self.epoch = self._data_epoch
            self.data_iterator = None
            self._data_generator_state = None
            return
        assert self.data_loader is not None
        num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        self._normalize_resume_cursor(num_batches)
        torch_loader = openpi_torch_loader(self.data_loader)
        generator = getattr(torch_loader, "generator", None)
        if generator is not None and self._data_generator_state is not None:
            generator.set_state(self._data_generator_state)
        sampler = getattr(torch_loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(self._data_epoch))
        self.data_iterator = iter(self.data_loader)
        for _ in range(int(self._data_iter_offset)):
            next(self.data_iterator)

    def _normalize_resume_cursor(self, num_batches: int) -> None:
        """Map the logical optimizer step to the active DDP data-loader shape."""

        if num_batches <= 0:
            raise ValueError("decoder data loader must contain at least one batch")
        current_world_size = 1 if self.distributed is None else int(self.distributed.world_size)
        if self._checkpoint_world_size in (None, current_world_size):
            return
        micro_batches_consumed = int(self.global_step) * int(self.gradient_accumulation)
        self._data_epoch, self._data_iter_offset = divmod(micro_batches_consumed, num_batches)
        self.epoch = self._data_epoch
        # Rank-local generator states are topology-specific. The new ranks use
        # setup's deterministic seed and advance to the normalized cursor.
        self._data_generator_state = None

    def _next_batch(self) -> Any:
        if self.trajectory_replay is not None:
            batch_size = int(self.cfg.decoder_training.micro_batch_size)
            batch = self.trajectory_replay.sample(batch_size, include_images=True)
            steps_per_epoch = int(self.trajectory_replay.steps_per_epoch(batch_size=batch_size))
            self._data_iter_offset += 1
            if self._data_iter_offset >= steps_per_epoch:
                self._data_epoch += 1
                self._data_iter_offset = 0
            self.epoch = self._data_epoch
            return batch
        assert self.data_iterator is not None and self.data_loader is not None
        num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        if self._data_iter_offset >= num_batches:
            self._start_next_data_epoch()
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self._start_next_data_epoch()
            assert self.data_iterator is not None
            batch = next(self.data_iterator)
        self._data_iter_offset += 1
        return batch

    def _start_next_data_epoch(self) -> None:
        """Advance OpenPI's infinite wrapper with explicit sampler epoch state."""

        assert self.data_loader is not None
        self._data_epoch += 1
        self.epoch = self._data_epoch
        self._data_iter_offset = 0
        torch_loader = openpi_torch_loader(self.data_loader)
        generator = getattr(torch_loader, "generator", None)
        if generator is not None:
            self._data_generator_state = generator.get_state().clone()
        set_epoch = getattr(getattr(torch_loader, "sampler", None), "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(self._data_epoch))
        self.data_iterator = iter(self.data_loader)

    def _prepare_observation(self, batch: Any) -> tuple[Any, torch.Tensor]:
        if isinstance(batch, tuple):
            observation, _actions = batch
        else:
            observation, _actions = batch["observation"], batch["actions"]
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda value: (
                torch.as_tensor(value).to(device=self.device, non_blocking=True).contiguous()
                if value is not None
                else None
            ),
            observation,
        )
        image_keys = tuple(str(key) for key in self.cfg.decoder_training.target_image_keys)
        missing = [key for key in image_keys if key not in observation.images]
        if missing:
            raise KeyError(f"OpenPI observation is missing target image(s): {missing}")
        target = torch.stack([observation.images[key] for key in image_keys], dim=1)
        # OpenPI's model-space pixels use [-1, 1]. Decoder output is [0, 1].
        return observation, ((target.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def _encode_frozen_prefix(self, observation: Any) -> torch.Tensor:
        """Encode one prefix with the runner's frozen latent producer."""

        assert self.policy is not None
        prefix = self.policy.encode_observation_prefix(observation)
        if not isinstance(prefix, torch.Tensor):
            raise TypeError("latent producer must return a tensor prefix")
        return prefix.detach()

    def _prepare_replay_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Move one single-frame collected replay batch to decoder model space."""

        if not isinstance(batch, dict):
            raise TypeError("collected pixel replay must return a dictionary batch")
        prefix = batch.get("obs_embedding")
        images = batch.get("images")
        if not isinstance(prefix, torch.Tensor) or not isinstance(images, torch.Tensor):
            raise TypeError("collected pixel replay requires obs_embedding and images tensors")
        if prefix.ndim != 4 or int(prefix.shape[1]) != 1:
            raise ValueError(
                f"collected pixel replay prefix must be [B,1,N,D], got {tuple(prefix.shape)}"
            )
        if images.ndim != 6 or int(images.shape[1]) != 1 or int(images.shape[-1]) != 3:
            raise ValueError(
                f"collected pixel replay images must be [B,1,V,H,W,3], got {tuple(images.shape)}"
            )
        prefix = prefix[:, 0].to(device=self.device, non_blocking=True).contiguous()
        target = (
            images[:, 0]
            .to(device=self.device, dtype=torch.float32, non_blocking=True)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .div_(255.0)
        )
        image_size = int(self.cfg.pixel_decoder.image_size)
        if tuple(target.shape[-2:]) != (image_size, image_size):
            target = F.interpolate(
                target.flatten(0, 1),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).unflatten(0, (-1, int(target.shape[1])))
        return prefix.detach(), target.detach()

    def _prepare_training_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve either an official OpenPI batch or an encoded RGB replay batch."""

        if self.trajectory_replay is not None:
            return self._prepare_replay_batch(batch)
        observation, target = self._prepare_observation(batch)
        return self._encode_frozen_prefix(observation), target.detach()

    def run(self) -> list[dict[str, float]]:
        assert self.policy is not None
        assert self.pixel_decoder is not None
        assert self.pixel_decoder_optimizer is not None
        assert self.lr_scheduler is not None
        assert self.distributed is not None
        max_steps = int(self.cfg.training.max_steps)
        log_every = max(1, int(self.cfg.training.log_every))
        checkpoint_every = int(self.cfg.training.checkpoint_every)
        visualize_every = int(self.cfg.decoder_training.visualize_every)
        accumulation = self.gradient_accumulation
        grad_clip = float(self.cfg.decoder_training.optim.clip_grad)
        precision = str(self.cfg.decoder_training.precision).lower()
        if precision not in {"bf16", "fp32"}:
            raise ValueError("decoder_training.precision must be bf16 or fp32")
        autocast_enabled = self.device.type == "cuda" and precision == "bf16"
        history: list[dict[str, float]] = []
        self.console_banner(
            "Latent Pixel Decoder",
            subtitle=(
                f"π0.5 steps={max_steps} global_batch={self.cfg.decoder_training.global_batch_size}"
            ),
        )
        start = time.perf_counter()
        self.pixel_decoder_optimizer.zero_grad(set_to_none=True)
        while self.global_step < max_steps:
            self.pixel_decoder.train()
            metric_sums = {
                key: torch.zeros((), device=self.device) for key in ("loss", "l1", "ssim", "psnr")
            }
            step_start = time.perf_counter()
            latest_prediction: torch.Tensor | None = None
            latest_target: torch.Tensor | None = None
            for micro_step in range(accumulation):
                prefix, target = self._prepare_training_batch(self._next_batch())
                last_micro = micro_step + 1 == accumulation
                no_sync = getattr(self.pixel_decoder, "no_sync", None)
                sync_context = (
                    contextlib.nullcontext() if last_micro or not callable(no_sync) else no_sync()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ):
                        prediction = self.pixel_decoder(prefix)
                    losses = latent_pixel_reconstruction_loss(
                        prediction,
                        target,
                        l1_weight=float(self.cfg.decoder_training.loss.l1_weight),
                        ssim_weight=float(self.cfg.decoder_training.loss.ssim_weight),
                    )
                    (losses["loss"] / float(accumulation)).backward()
                for key in metric_sums:
                    metric_sums[key] += losses[key].detach().float()
                latest_prediction = prediction.detach()
                latest_target = target.detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.pixel_decoder.parameters(), grad_clip)
            self.pixel_decoder_optimizer.step()
            self.pixel_decoder_optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self.global_step += 1
            local_metrics: dict[str, float | torch.Tensor] = {
                f"train/pixel_{key}": value / float(accumulation)
                for key, value in metric_sums.items()
            }
            local_metrics.update(
                {
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": float(
                        self.pixel_decoder_optimizer.param_groups[0]["lr"]
                    ),
                    "time/step_s": time.perf_counter() - step_start,
                }
            )
            metrics = self.distributed.reduce_mean_dict(local_metrics)
            metrics["global_step"] = float(self.global_step)
            history.append(metrics)
            if self.global_step % log_every == 0 or self.global_step == max_steps:
                self.log_metrics(metrics, step=self.global_step)
                self.console_metric_table(
                    step=max(0, self.global_step - 1),
                    total_steps=max_steps,
                    elapsed_s=time.perf_counter() - start,
                    metrics=metrics,
                )
            if (
                visualize_every > 0
                and self.global_step % visualize_every == 0
                and self.distributed.is_main_process
                and latest_prediction is not None
                and latest_target is not None
            ):
                self._save_visualization(latest_target, latest_prediction, self.global_step)
            if checkpoint_every > 0 and self.global_step % checkpoint_every == 0:
                self.save_checkpoint(tag="latest")
            self.console_progress(self.global_step, max_steps, "pixel-decoder", unit="step")

        if bool(self.cfg.training.get("save_at_end", True)) and (
            checkpoint_every <= 0 or self.global_step % checkpoint_every != 0
        ):
            self.save_checkpoint(tag="latest")
        self.console_banner(
            "Latent Pixel Decoder",
            subtitle=f"completed step={self.global_step}",
            done=True,
        )
        return history

    def _save_visualization(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        step: int,
    ) -> Path:
        """Save one target/reconstruction grid (rows) over camera views (columns)."""

        target_np = target[0].detach().float().clamp(0, 1).cpu().numpy()
        prediction_np = prediction[0].detach().float().clamp(0, 1).cpu().numpy()
        rows = []
        for source in (target_np, prediction_np):
            views = [np.moveaxis(view, 0, -1) for view in source]
            rows.append(np.concatenate(views, axis=1))
        grid = (np.concatenate(rows, axis=0) * 255.0).round().astype(np.uint8)
        output_dir = self.get_video_dir("train")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"step={int(step):08d}.png"
        Image.fromarray(grid, mode="RGB").save(path)
        return path

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            if self.distributed is not None:
                self.distributed.cleanup()


def _build_decoder_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    def scale(step: int) -> float:
        if step < int(warmup_steps):
            return float(step) / float(max(1, warmup_steps))
        progress = min(
            1.0,
            (step - int(warmup_steps)) / float(max(1, int(total_steps) - int(warmup_steps))),
        )
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


__all__ = ["LatentPixelDecoderTrainingRunner"]
