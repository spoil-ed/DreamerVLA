from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dreamervla.runners._dreamer_runner_common import (
    DreamerCkptResumeMixin,
    to_device,
)
from dreamervla.runners.base_runner import BaseRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DreamerV3TokenRunner(DreamerCkptResumeMixin, BaseRunner):
    """Standalone DreamerV3-style world-model trainer for image tokens."""

    runner_name = "token_wm"
    runner_status = "secondary"
    runner_family = "world_model"
    _ckpt_log_tag = "dreamerv3-token"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.device = torch.device(
            OmegaConf.select(config, "training.device", default="cuda:0")
        )
        self.out_dir = Path(self.output_dir)
        self.log_path = self.out_dir / "dreamerv3_token_logs.json.txt"
        self.ckpt_dir = self.out_dir / "ckpt"
        self.vq_model = None

    def _make_loader(self, dataset: Any, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(
                OmegaConf.select(self.cfg, "dataloader.batch_size", default=4)
            ),
            shuffle=shuffle,
            num_workers=int(
                OmegaConf.select(self.cfg, "dataloader.num_workers", default=4)
            ),
            pin_memory=bool(
                OmegaConf.select(self.cfg, "dataloader.pin_memory", default=True)
            ),
            drop_last=bool(
                OmegaConf.select(self.cfg, "dataloader.drop_last", default=True)
            ),
            persistent_workers=bool(
                OmegaConf.select(
                    self.cfg, "dataloader.persistent_workers", default=True
                )
            ),
            collate_fn=getattr(dataset, "collate_fn", None),
        )

    def _maybe_build_viz(self) -> None:
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or not bool(
            OmegaConf.select(viz_cfg, "enabled", default=False)
        ):
            return
        cfg_path = OmegaConf.select(
            viz_cfg,
            "vqgan_config_path",
            default=str(
                PROJECT_ROOT
                / "data"
                / "checkpoints"
                / "chameleon"
                / "tokenizer"
                / "vqgan.yaml"
            ),
        )
        ckpt_path = OmegaConf.select(
            viz_cfg,
            "vqgan_ckpt_path",
            default=str(
                PROJECT_ROOT
                / "data"
                / "checkpoints"
                / "chameleon"
                / "tokenizer"
                / "vqgan.ckpt"
            ),
        )
        try:
            from dreamervla.utils.vq_image_decoder import load_vq_model

            viz_device_cfg = OmegaConf.select(viz_cfg, "device", default=None)
            viz_device = (
                self.device
                if viz_device_cfg is None
                else torch.device(str(viz_device_cfg))
            )
            self.vq_model = load_vq_model(
                cfg_path=cfg_path, ckpt_path=ckpt_path, device=viz_device
            )
            print(f"[dreamerv3-token][viz] VQGAN ready on {viz_device}")
        except Exception as exc:
            print(
                f"[dreamerv3-token][viz] failed to build VQGAN visualizer, disabling: {exc}"
            )
            self.vq_model = None

    @torch.no_grad()
    def _decode_token_view(self, token_ids: torch.Tensor, h: int, w: int):
        from dreamervla.utils.vq_image_decoder import tensor_to_pil, vq_tokens_to_pixels

        if self.vq_model is None:
            return None
        token_ids = token_ids.reshape(1, h * w).to(
            device=next(self.vq_model.parameters()).device,
            dtype=torch.long,
        )
        pixels = vq_tokens_to_pixels(token_ids, self.vq_model, h_latent=h, w_latent=w)
        return tensor_to_pil(pixels[0])

    @torch.no_grad()
    def _maybe_save_viz(
        self, model_core: torch.nn.Module, batch: dict[str, Any]
    ) -> None:
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or self.vq_model is None:
            return
        every = int(OmegaConf.select(viz_cfg, "every_n_steps", default=100))
        if every <= 0 or self.global_step % every != 0:
            return

        tokens = batch.get("tokens")
        actions = batch.get("actions")
        is_first = batch.get("is_first")
        if not isinstance(tokens, torch.Tensor) or not isinstance(
            actions, torch.Tensor
        ):
            return
        if not isinstance(is_first, torch.Tensor):
            return
        if tokens.ndim != 4 or tokens.shape[1] < 2:
            return

        was_training = model_core.training
        model_core.eval()
        try:
            enc = model_core.encoder(tokens.long())
            seq = model_core.rssm.observe(enc, actions, is_first)
            post_logits = model_core.decoder(seq["deter"], seq["stoch"])
            post_pred = post_logits.argmax(dim=-1)

            deter0 = seq["deter"][:, 0]
            stoch0 = seq["stoch"][:, 0]
            action1 = actions[:, 1].to(device=deter0.device, dtype=deter0.dtype)
            prior_deter1 = model_core.rssm._core(deter0, stoch0, action1)
            prior_logits1 = model_core.rssm._prior(prior_deter1)
            prior_idx1 = prior_logits1.argmax(dim=-1)
            prior_stoch1 = F.one_hot(prior_idx1, model_core.rssm.classes).to(
                dtype=prior_logits1.dtype
            )
            prior_dec_logits = model_core.decoder(
                prior_deter1[:, None], prior_stoch1[:, None]
            )
            prior_pred = prior_dec_logits.argmax(dim=-1)[:, 0]
        finally:
            if was_training:
                model_core.train()

        b, _t, num_views, tokens_per_view = tokens.shape
        h, w = tuple(int(x) for x in model_core.encoder.spatial_grid)
        if h * w != tokens_per_view:
            print(
                f"[dreamerv3-token][viz] skip: h*w={h * w} != tokens_per_view={tokens_per_view}"
            )
            return
        view_labels = list(
            OmegaConf.select(viz_cfg, "view_labels", default=["third", "wrist"])
        )
        if len(view_labels) != num_views:
            view_labels = [f"view{idx}" for idx in range(num_views)]

        num_samples = min(int(OmegaConf.select(viz_cfg, "num_samples", default=4)), b)
        cell_size = int(OmegaConf.select(viz_cfg, "cell_size", default=192))
        out_dir = self.out_dir / "viz"
        saved = 0
        for sample_idx in range(num_samples):
            panels: list[tuple[str, Any]] = []
            for view_idx, label in enumerate(view_labels):
                panels.extend(
                    [
                        (
                            f"{label} cur",
                            self._decode_token_view(
                                tokens[sample_idx, 0, view_idx], h, w
                            ),
                        ),
                        (
                            f"{label} recon",
                            self._decode_token_view(
                                post_pred[sample_idx, 0, view_idx], h, w
                            ),
                        ),
                        (
                            f"{label} next",
                            self._decode_token_view(
                                tokens[sample_idx, 1, view_idx], h, w
                            ),
                        ),
                        (
                            f"{label} prior",
                            self._decode_token_view(
                                prior_pred[sample_idx, view_idx], h, w
                            ),
                        ),
                    ]
                )
            path = out_dir / f"step_{self.global_step:07d}_sample{sample_idx:02d}.png"
            self._save_viz_strip(path, panels, cell_size=cell_size)
            saved += 1
        if saved:
            print(
                f"[dreamerv3-token][viz] step {self.global_step}: wrote {saved} panel(s) under {out_dir}"
            )

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        seed = int(OmegaConf.select(self.cfg, "training.seed", default=7))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print("[dreamerv3-token] building dataset ...")
        dataset = hydra.utils.instantiate(self.cfg.dataset)
        print(f"[dreamerv3-token]   windows = {len(dataset):,}")
        print(f"[dreamerv3-token]   spec = {dataset.data_spec}")
        loader = self._make_loader(dataset, shuffle=True)

        print("[dreamerv3-token] building model ...")
        model_core = hydra.utils.instantiate(self.cfg.world_model).to(self.device)
        model: torch.nn.Module = model_core
        if bool(OmegaConf.select(self.cfg, "training.data_parallel", default=False)):
            if self.device.type != "cuda":
                raise ValueError("training.data_parallel=true requires CUDA")
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model_core)
                print(
                    f"[dreamerv3-token]   data parallel = {torch.cuda.device_count()} visible CUDA devices"
                )
        n_params = sum(p.numel() for p in model_core.parameters() if p.requires_grad)
        print(f"[dreamerv3-token]   trainable params = {n_params:,}")
        self._maybe_build_viz()

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

        num_epochs_cfg = OmegaConf.select(
            self.cfg, "training.num_epochs", default=20
        )
        num_epochs = 20 if num_epochs_cfg is None else int(num_epochs_cfg)
        max_steps = max(0, num_epochs - self.epoch) * len(loader)
        log_every = int(OmegaConf.select(self.cfg, "training.log_every", default=20))
        save_every = int(
            OmegaConf.select(self.cfg, "training.save_every", default=1000)
        )
        progress_total = max(1, num_epochs * len(loader))
        warmup = int(OmegaConf.select(self.cfg, "optim.warmup", default=1000))
        base_lr = float(OmegaConf.select(self.cfg, "optim.lr", default=4e-5))
        grad_clip = float(OmegaConf.select(self.cfg, "optim.grad_clip", default=100.0))

        print(
            f"[dreamerv3-token] training for {num_epochs:,} epochs "
            f"({len(loader):,} batches/epoch, ~{max_steps:,} remaining steps) ..."
        )
        log_mode = "a" if resumed else "w"
        log_handle = open(self.log_path, log_mode)
        self.console_banner("TRAINING", subtitle=f"{num_epochs} epochs")
        try:
            while self.epoch < num_epochs:
                self.epoch += 1
                for batch in loader:
                    if warmup > 0:
                        lr_scale = min(
                            1.0, float(self.global_step + 1) / float(warmup)
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = base_lr * lr_scale

                    batch = to_device(batch, self.device)
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
                        **{
                            k: float(v.detach().float().mean().cpu())
                            for k, v in out.items()
                            if k != "_loss"
                        },
                    }
                    self.console_progress(self.global_step, progress_total, "train")
                    if log_every > 0 and self.global_step % log_every == 0:
                        log_handle.write(json.dumps(row) + "\n")
                        log_handle.flush()
                        self.log_metrics(row, step=self.global_step)
                        self.console_metrics(f"train · epoch {self.epoch}", {f"train/{k}": v for k, v in row.items()})

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
        finally:
            log_handle.close()

        self._save_ckpt(model_core, optimizer, self.ckpt_dir / "latest.ckpt")
        self._save_ckpt(
            model_core, optimizer, self.ckpt_dir / f"step_{self.global_step:08d}.ckpt"
        )
        self.console_banner("TRAINING", done=True)
        print("[dreamerv3-token] done")
        print(f"  log  = {self.log_path}")
        print(f"  ckpt = {self.ckpt_dir / 'latest.ckpt'}")


__all__ = ["DreamerV3TokenRunner"]
