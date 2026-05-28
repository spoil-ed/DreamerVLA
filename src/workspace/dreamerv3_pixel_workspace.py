from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import warnings

import hydra
import torch
import torch.distributed as dist
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.workspace.base_workspace import BaseWorkspace


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


class DreamerV3PixelWorkspace(BaseWorkspace):
    """Standalone pixel-level DreamerV3 world-model trainer for LIBERO."""

    workspace_name = "pixel_wm_compat"
    workspace_status = "compatibility"
    workspace_family = "world_model"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.rank = 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed_strategy = str(
            OmegaConf.select(config, "training.distributed_strategy", default="single")
        ).lower()
        self.use_ddp = self.distributed_strategy == "ddp" and self.world_size > 1
        if self.use_ddp:
            if not torch.cuda.is_available():
                raise ValueError("training.distributed_strategy=ddp requires CUDA")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.local_rank)))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(
                OmegaConf.select(config, "training.device", default="cuda:0")
            )
        self.out_dir = Path(self.output_dir)
        self.log_path = self.out_dir / "dreamerv3_pixel_logs.json.txt"
        self.ckpt_dir = self.out_dir / "ckpt"

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def _print(self, message: str) -> None:
        if self.is_main_process:
            print(message)

    def _barrier(self) -> None:
        if self.use_ddp and dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])

    def _make_loader(
        self,
        dataset: Any,
        *,
        shuffle: bool,
    ) -> tuple[DataLoader, DistributedSampler | None]:
        drop_last = bool(
            OmegaConf.select(self.cfg, "dataloader.drop_last", default=True)
        )
        sampler = None
        if self.use_ddp:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        num_workers = int(
            OmegaConf.select(self.cfg, "dataloader.num_workers", default=4)
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": int(
                OmegaConf.select(self.cfg, "dataloader.batch_size", default=8)
            ),
            "shuffle": shuffle if sampler is None else False,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": bool(
                OmegaConf.select(self.cfg, "dataloader.pin_memory", default=True)
            ),
            "drop_last": drop_last,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                OmegaConf.select(
                    self.cfg, "dataloader.persistent_workers", default=True
                )
            )
            prefetch_factor = OmegaConf.select(
                self.cfg, "dataloader.prefetch_factor", default=None
            )
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            multiprocessing_context = OmegaConf.select(
                self.cfg, "dataloader.multiprocessing_context", default=None
            )
            if multiprocessing_context not in (None, "", "null"):
                loader_kwargs["multiprocessing_context"] = str(multiprocessing_context)
        return DataLoader(**loader_kwargs), sampler

    def _setup_auxiliary_modules(self) -> None:
        return None

    def _prepare_batch_for_model(
        self,
        batch: dict[str, Any],
        model_core: torch.nn.Module,
    ) -> dict[str, Any]:
        del model_core
        return batch

    def _move_batch_to_device_before_prepare(self) -> bool:
        return True

    def _save_ckpt(
        self,
        model_core: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        path: Path,
    ) -> None:
        if not self.is_main_process:
            return
        payload = {
            "model": model_core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def _resolve_resume_path(self) -> Path:
        configured = OmegaConf.select(self.cfg, "training.resume_path", default=None)
        if configured is None:
            return self.ckpt_dir / "latest.ckpt"
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_dir():
            if (path / "ckpt" / "latest.ckpt").is_file():
                return path / "ckpt" / "latest.ckpt"
            return path / "latest.ckpt"
        return path

    def _maybe_resume(
        self,
        model_core: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        if not bool(OmegaConf.select(self.cfg, "training.resume", default=False)):
            return False
        path = self._resolve_resume_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"training.resume=true but checkpoint not found: {path}"
            )
        self._print(f"[dreamerv3-pixel] resuming from {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_sd = payload.get("model")
        if model_sd is None and "state_dicts" in payload:
            model_sd = payload["state_dicts"].get("model") or payload[
                "state_dicts"
            ].get("model_core")
        if model_sd is None:
            raise KeyError(f"Checkpoint {path} does not contain a model state_dict")
        strict = bool(
            OmegaConf.select(self.cfg, "training.resume_strict", default=True)
        )
        missing, unexpected = model_core.load_state_dict(model_sd, strict=strict)
        if missing or unexpected:
            self._print(
                f"[dreamerv3-pixel] resume model missing={len(missing)} unexpected={len(unexpected)}"
            )

        skip_optimizer = bool(
            OmegaConf.select(self.cfg, "training.resume_skip_optimizer", default=False)
        )
        if not skip_optimizer and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        elif skip_optimizer:
            self._print(
                "[dreamerv3-pixel] skipping optimizer state (training.resume_skip_optimizer=true)"
            )

        # ``resume_reset_step`` = warm-start a NEW training run from the ckpt's
        # model weights only.  global_step / epoch / RNG stay at their
        # initial (fresh-seed) values so the lr warmup, save-every cadence,
        # and max_steps budget all behave as a fresh run.  Use this when the
        # training objective or model topology has changed (e.g. switching
        # per-step teacher forcing -> chunk_loss with a new mask_obs_token).
        reset_step = bool(
            OmegaConf.select(self.cfg, "training.resume_reset_step", default=False)
        )
        if reset_step:
            self._print(
                "[dreamerv3-pixel] resume_reset_step=true: keeping fresh "
                f"global_step={self.global_step} epoch={self.epoch} and RNG"
            )
        else:
            self.global_step = int(payload.get("global_step", self.global_step))
            self.epoch = int(payload.get("epoch", self.epoch))
            rng = payload.get("rng")
            if isinstance(rng, dict):
                torch_state = rng.get("torch")
                if isinstance(torch_state, torch.Tensor):
                    torch.set_rng_state(torch_state)
                cuda_state = rng.get("cuda")
                if (
                    torch.cuda.is_available()
                    and isinstance(cuda_state, list)
                    and cuda_state
                ):
                    try:
                        torch.cuda.set_rng_state_all(cuda_state)
                    except Exception as exc:
                        self._print(
                            f"[dreamerv3-pixel] warning: could not restore CUDA RNG: {exc}"
                        )
        self._print(
            f"[dreamerv3-pixel] resumed at global_step={self.global_step} epoch={self.epoch}"
        )
        return True

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor):
        from PIL import Image

        image = image.detach().float().cpu()
        if image.max() > 2.0:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)
        if image.ndim != 3:
            raise ValueError(f"Expected [C,H,W] image tensor, got {tuple(image.shape)}")
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if image.shape[0] != 3:
            raise ValueError(f"Expected 3 channels per view, got {image.shape[0]}")
        arr = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _save_viz_strip(
        path: Path, panels: list[tuple[str, Any]], cell_size: int
    ) -> None:
        from PIL import Image, ImageDraw

        header = 22
        canvas = Image.new(
            "RGB", (cell_size * len(panels), cell_size + header), color=(32, 32, 32)
        )
        draw = ImageDraw.Draw(canvas)
        for idx, (label, image) in enumerate(panels):
            x0 = idx * cell_size
            if image is not None:
                canvas.paste(
                    image.convert("RGB").resize((cell_size, cell_size)), (x0, header)
                )
            else:
                draw.rectangle(
                    [x0, header, x0 + cell_size, header + cell_size], fill=(70, 20, 20)
                )
                draw.text(
                    (x0 + 8, header + cell_size // 2), "(missing)", fill=(230, 230, 230)
                )
            draw.text((x0 + 4, 4), str(label), fill=(230, 230, 230))
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)

    @torch.no_grad()
    def _maybe_save_viz(
        self, model_core: torch.nn.Module, batch: dict[str, Any]
    ) -> None:
        if not self.is_main_process:
            return
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or not bool(
            OmegaConf.select(viz_cfg, "enabled", default=False)
        ):
            return
        every = int(OmegaConf.select(viz_cfg, "every_n_steps", default=100))
        if every <= 0 or self.global_step % every != 0:
            return

        images = batch.get("images")
        actions = batch.get("actions")
        is_first = batch.get("is_first")
        if not isinstance(images, torch.Tensor) or not isinstance(
            actions, torch.Tensor
        ):
            return
        if not isinstance(is_first, torch.Tensor):
            return
        if images.ndim != 5 or images.shape[1] < 2:
            return

        was_training = model_core.training
        model_core.eval()
        try:
            enc = model_core.encoder(images)
            seq = model_core.rssm.observe(enc, actions, is_first)
            recon = model_core.decoder(seq["deter"], seq["stoch"])

            deter0 = seq["deter"][:, 0]
            stoch0 = seq["stoch"][:, 0]
            action1 = actions[:, 1].to(device=deter0.device, dtype=deter0.dtype)
            prior_deter1 = model_core.rssm._core(deter0, stoch0, action1)
            prior_logits1 = model_core.rssm._prior(prior_deter1)
            prior_idx1 = prior_logits1.argmax(dim=-1)
            prior_stoch1 = torch.nn.functional.one_hot(
                prior_idx1, model_core.rssm.classes
            ).to(dtype=prior_logits1.dtype)
            prior_recon = model_core.decoder(
                prior_deter1[:, None], prior_stoch1[:, None]
            )[:, 0]
        finally:
            if was_training:
                model_core.train()

        view_labels = list(
            OmegaConf.select(viz_cfg, "view_labels", default=["third", "wrist"])
        )
        channels_per_view = int(
            OmegaConf.select(viz_cfg, "channels_per_view", default=3)
        )
        total_channels = int(images.shape[2])
        num_views = len(view_labels)
        if num_views <= 0 or total_channels != num_views * channels_per_view:
            if total_channels % channels_per_view != 0:
                print(
                    f"[dreamerv3-pixel][viz] skip: channels={total_channels} "
                    f"not divisible by channels_per_view={channels_per_view}"
                )
                return
            num_views = total_channels // channels_per_view
            view_labels = [f"view{idx}" for idx in range(num_views)]

        num_samples = min(
            int(OmegaConf.select(viz_cfg, "num_samples", default=4)),
            int(images.shape[0]),
        )
        cell_size = int(OmegaConf.select(viz_cfg, "cell_size", default=192))
        out_dir = self.out_dir / "viz"
        saved = 0
        for sample_idx in range(num_samples):
            panels: list[tuple[str, Any]] = []
            for view_idx, label in enumerate(view_labels):
                c0 = view_idx * channels_per_view
                c1 = c0 + channels_per_view
                panels.extend(
                    [
                        (
                            f"{label} cur",
                            self._tensor_to_pil(images[sample_idx, 0, c0:c1]),
                        ),
                        (
                            f"{label} recon",
                            self._tensor_to_pil(recon[sample_idx, 0, c0:c1]),
                        ),
                        (
                            f"{label} next",
                            self._tensor_to_pil(images[sample_idx, 1, c0:c1]),
                        ),
                        (
                            f"{label} prior",
                            self._tensor_to_pil(prior_recon[sample_idx, c0:c1]),
                        ),
                    ]
                )
            path = out_dir / f"step_{self.global_step:07d}_sample{sample_idx:02d}.png"
            self._save_viz_strip(path, panels, cell_size=cell_size)
            saved += 1
        if saved:
            print(
                f"[dreamerv3-pixel][viz] step {self.global_step}: wrote {saved} panel(s) under {out_dir}"
            )

    def _reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.use_ddp:
            return metrics
        keys = list(metrics.keys())
        if not keys:
            return metrics
        values = torch.tensor(
            [float(metrics[key]) for key in keys],
            device=self.device,
            dtype=torch.float32,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(self.world_size)
        return {
            key: float(value)
            for key, value in zip(keys, values.detach().cpu().tolist())
        }

    def _progress_postfix(self, row: dict[str, Any], max_steps: int) -> dict[str, Any]:
        return {
            "step": f"{self.global_step}/{max_steps}",
            "wm": float(row["loss"]),
            "rec": float(row.get("rec_loss", 0.0)),
            "dyn": float(row.get("dyn_loss", 0.0)),
            "rep": float(row.get("rep_loss", 0.0)),
            "mse": float(row.get("image_mse", 0.0)),
            "psnr": float(row.get("image_psnr", 0.0)),
        }

    def run(self) -> None:
        if self.is_main_process:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        seed = int(OmegaConf.select(self.cfg, "training.seed", default=7))
        torch.manual_seed(seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + self.rank)

        self._print("[dreamerv3-pixel] building dataset ...")
        dataset = hydra.utils.instantiate(self.cfg.dataset)
        self._print(f"[dreamerv3-pixel]   windows = {len(dataset):,}")
        self._print(f"[dreamerv3-pixel]   spec = {dataset.data_spec}")
        loader, sampler = self._make_loader(dataset, shuffle=True)

        self._setup_auxiliary_modules()

        self._print("[dreamerv3-pixel] building model ...")
        model_core = hydra.utils.instantiate(self.cfg.world_model).to(self.device)
        model: torch.nn.Module = model_core
        if self.use_ddp:
            model = DDP(
                model_core,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            self._print(
                f"[dreamerv3-pixel]   distributed data parallel = {self.world_size} ranks"
            )
        elif bool(OmegaConf.select(self.cfg, "training.data_parallel", default=False)):
            if self.device.type != "cuda":
                raise ValueError("training.data_parallel=true requires CUDA")
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model_core)
                self._print(
                    f"[dreamerv3-pixel]   data parallel = {torch.cuda.device_count()} visible CUDA devices"
                )
        n_params = sum(p.numel() for p in model_core.parameters() if p.requires_grad)
        self._print(f"[dreamerv3-pixel]   trainable params = {n_params:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(OmegaConf.select(self.cfg, "optim.lr", default=4e-5)),
            betas=(
                float(OmegaConf.select(self.cfg, "optim.beta1", default=0.9)),
                float(OmegaConf.select(self.cfg, "optim.beta2", default=0.999)),
            ),
            eps=float(OmegaConf.select(self.cfg, "optim.eps", default=1e-20)),
            weight_decay=float(
                OmegaConf.select(self.cfg, "optim.weight_decay", default=0.0)
            ),
        )

        resumed = self._maybe_resume(model_core, optimizer)

        max_steps_cfg = OmegaConf.select(self.cfg, "training.max_steps", default=10000)
        num_epochs_cfg = OmegaConf.select(self.cfg, "training.num_epochs", default=None)
        if max_steps_cfg is None:
            if num_epochs_cfg is None:
                raise ValueError("Set either training.max_steps or training.num_epochs")
            max_steps = int(num_epochs_cfg) * len(loader)
        else:
            max_steps = int(max_steps_cfg)
        log_every = int(OmegaConf.select(self.cfg, "training.log_every", default=20))
        save_every = int(
            OmegaConf.select(self.cfg, "training.save_every", default=1000)
        )
        tqdm_interval_sec = float(
            OmegaConf.select(self.cfg, "training.tqdm_interval_sec", default=1.0)
        )
        warmup = int(OmegaConf.select(self.cfg, "optim.warmup", default=1000))
        base_lr = float(OmegaConf.select(self.cfg, "optim.lr", default=4e-5))
        grad_clip = float(OmegaConf.select(self.cfg, "optim.grad_clip", default=100.0))

        self._print(
            f"[dreamerv3-pixel] training for {max_steps:,} steps "
            f"({len(loader):,} local batches/epoch, batch_size={int(OmegaConf.select(self.cfg, 'dataloader.batch_size', default=8))}"
            f"{', global_batch=' + str(int(OmegaConf.select(self.cfg, 'dataloader.batch_size', default=8)) * self.world_size) if self.use_ddp else ''}) ..."
        )
        log_mode = "a" if resumed else "w"
        log_handle = open(self.log_path, log_mode) if self.is_main_process else None
        try:
            while self.global_step < max_steps:
                self.epoch += 1
                if sampler is not None:
                    sampler.set_epoch(self.epoch)
                with tqdm.tqdm(
                    loader,
                    desc=f"Training epoch {self.epoch}",
                    disable=not self.is_main_process,
                    leave=False,
                    mininterval=tqdm_interval_sec,
                ) as tepoch:
                    for batch in tepoch:
                        if self.global_step >= max_steps:
                            break
                        if warmup > 0:
                            lr_scale = min(
                                1.0, float(self.global_step + 1) / float(warmup)
                            )
                            for group in optimizer.param_groups:
                                group["lr"] = base_lr * lr_scale

                        if self._move_batch_to_device_before_prepare():
                            batch = _to_device(batch, self.device)
                        batch = self._prepare_batch_for_model(batch, model_core)
                        model.train()
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=r".*Was asked to gather along dimension 0.*",
                                category=UserWarning,
                            )
                            out = model(batch)
                        loss = out["_loss"].mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip
                        )
                        optimizer.step()

                        row = {
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "grad_norm": float(grad_norm),
                            **self._reduce_metrics(
                                {
                                    k: float(v.detach().float().mean().cpu())
                                    for k, v in out.items()
                                    if k != "_loss"
                                }
                            ),
                        }
                        tepoch.set_postfix(
                            self._progress_postfix(row, max_steps), refresh=False
                        )
                        if (
                            self.is_main_process
                            and log_handle is not None
                            and log_every > 0
                            and self.global_step % log_every == 0
                        ):
                            log_handle.write(json.dumps(row) + "\n")
                            log_handle.flush()

                        self._maybe_save_viz(model_core, batch)

                        if (
                            save_every > 0
                            and self.global_step > 0
                            and self.global_step % save_every == 0
                        ):
                            self._save_ckpt(
                                model_core, optimizer, self.ckpt_dir / "latest.ckpt"
                            )
                        self.global_step += 1
                        if self.global_step >= max_steps:
                            break
        finally:
            if log_handle is not None:
                log_handle.close()

        if bool(OmegaConf.select(self.cfg, "training.save_final", default=True)):
            self._save_ckpt(model_core, optimizer, self.ckpt_dir / "latest.ckpt")
            self._save_ckpt(
                model_core,
                optimizer,
                self.ckpt_dir / f"step_{self.global_step:08d}.ckpt",
            )
        self._barrier()
        self._print("[dreamerv3-pixel] done")
        self._print(f"  log  = {self.log_path}")
        if bool(OmegaConf.select(self.cfg, "training.save_final", default=True)):
            self._print(f"  ckpt = {self.ckpt_dir / 'latest.ckpt'}")
        if self.use_ddp and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["DreamerV3PixelWorkspace"]
