from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import hydra
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.workspace.base_workspace import BaseWorkspace


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


class DreamerV3TokenWorkspace(BaseWorkspace):
    """Standalone DreamerV3-style world-model trainer for image tokens."""

    workspace_name = "token_wm_compat"
    workspace_status = "compatibility"
    workspace_family = "world_model"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.device = torch.device(OmegaConf.select(config, "training.device", default="cuda:0"))
        self.out_dir = Path(self.output_dir)
        self.log_path = self.out_dir / "dreamerv3_token_logs.json.txt"
        self.ckpt_dir = self.out_dir / "ckpt"
        self.vq_model = None

    def _make_loader(self, dataset: Any, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(OmegaConf.select(self.cfg, "dataloader.batch_size", default=4)),
            shuffle=shuffle,
            num_workers=int(OmegaConf.select(self.cfg, "dataloader.num_workers", default=4)),
            pin_memory=bool(OmegaConf.select(self.cfg, "dataloader.pin_memory", default=True)),
            drop_last=bool(OmegaConf.select(self.cfg, "dataloader.drop_last", default=True)),
            persistent_workers=bool(OmegaConf.select(self.cfg, "dataloader.persistent_workers", default=True)),
            collate_fn=getattr(dataset, "collate_fn", None),
        )

    def _save_ckpt(
        self,
        model_core: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        path: Path,
    ) -> None:
        payload = {
            "model": model_core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
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
            raise FileNotFoundError(f"training.resume=true but checkpoint not found: {path}")
        print(f"[dreamerv3-token] resuming from {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_sd = payload.get("model")
        if model_sd is None and "state_dicts" in payload:
            model_sd = payload["state_dicts"].get("model") or payload["state_dicts"].get("model_core")
        if model_sd is None:
            raise KeyError(f"Checkpoint {path} does not contain a model state_dict")
        strict = bool(OmegaConf.select(self.cfg, "training.resume_strict", default=True))
        missing, unexpected = model_core.load_state_dict(model_sd, strict=strict)
        if missing or unexpected:
            print(f"[dreamerv3-token] resume model missing={len(missing)} unexpected={len(unexpected)}")

        skip_optimizer = bool(OmegaConf.select(self.cfg, "training.resume_skip_optimizer", default=False))
        if not skip_optimizer and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        elif skip_optimizer:
            print("[dreamerv3-token] skipping optimizer state (training.resume_skip_optimizer=true)")

        self.global_step = int(payload.get("global_step", self.global_step))
        self.epoch = int(payload.get("epoch", self.epoch))
        rng = payload.get("rng")
        if isinstance(rng, dict):
            torch_state = rng.get("torch")
            if isinstance(torch_state, torch.Tensor):
                torch.set_rng_state(torch_state)
            cuda_state = rng.get("cuda")
            if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
                try:
                    torch.cuda.set_rng_state_all(cuda_state)
                except Exception as exc:
                    print(f"[dreamerv3-token] warning: could not restore CUDA RNG: {exc}")
        print(f"[dreamerv3-token] resumed at global_step={self.global_step} epoch={self.epoch}")
        return True

    def _maybe_build_viz(self) -> None:
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or not bool(OmegaConf.select(viz_cfg, "enabled", default=False)):
            return
        cfg_path = OmegaConf.select(
            viz_cfg,
            "vqgan_config_path",
            default="/home/user01/liops/workspace/DreamerVLA/data/ckpts/chameleon/tokenizer/vqgan.yaml",
        )
        ckpt_path = OmegaConf.select(
            viz_cfg,
            "vqgan_ckpt_path",
            default="/home/user01/liops/workspace/DreamerVLA/data/ckpts/chameleon/tokenizer/vqgan.ckpt",
        )
        try:
            from src.utils.vq_image_decoder import load_vq_model

            viz_device_cfg = OmegaConf.select(viz_cfg, "device", default=None)
            viz_device = self.device if viz_device_cfg is None else torch.device(str(viz_device_cfg))
            self.vq_model = load_vq_model(cfg_path=cfg_path, ckpt_path=ckpt_path, device=viz_device)
            print(f"[dreamerv3-token][viz] VQGAN ready on {viz_device}")
        except Exception as exc:
            print(f"[dreamerv3-token][viz] failed to build VQGAN visualizer, disabling: {exc}")
            self.vq_model = None

    @torch.no_grad()
    def _decode_token_view(self, token_ids: torch.Tensor, h: int, w: int):
        from src.utils.vq_image_decoder import tensor_to_pil, vq_tokens_to_pixels

        if self.vq_model is None:
            return None
        token_ids = token_ids.reshape(1, h * w).to(
            device=next(self.vq_model.parameters()).device,
            dtype=torch.long,
        )
        pixels = vq_tokens_to_pixels(token_ids, self.vq_model, h_latent=h, w_latent=w)
        return tensor_to_pil(pixels[0])

    @staticmethod
    def _save_viz_strip(path: Path, panels: list[tuple[str, Any]], cell_size: int) -> None:
        from PIL import Image, ImageDraw

        header = 22
        canvas = Image.new("RGB", (cell_size * len(panels), cell_size + header), color=(32, 32, 32))
        draw = ImageDraw.Draw(canvas)
        for idx, (label, image) in enumerate(panels):
            x0 = idx * cell_size
            if image is not None:
                canvas.paste(image.convert("RGB").resize((cell_size, cell_size)), (x0, header))
            else:
                draw.rectangle([x0, header, x0 + cell_size, header + cell_size], fill=(70, 20, 20))
                draw.text((x0 + 8, header + cell_size // 2), "(missing)", fill=(230, 230, 230))
            draw.text((x0 + 4, 4), str(label), fill=(230, 230, 230))
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)

    @torch.no_grad()
    def _maybe_save_viz(self, model_core: torch.nn.Module, batch: dict[str, Any]) -> None:
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or self.vq_model is None:
            return
        every = int(OmegaConf.select(viz_cfg, "every_n_steps", default=100))
        if every <= 0 or self.global_step % every != 0:
            return

        tokens = batch.get("tokens")
        actions = batch.get("actions")
        is_first = batch.get("is_first")
        if not isinstance(tokens, torch.Tensor) or not isinstance(actions, torch.Tensor):
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
            prior_stoch1 = F.one_hot(prior_idx1, model_core.rssm.classes).to(dtype=prior_logits1.dtype)
            prior_dec_logits = model_core.decoder(prior_deter1[:, None], prior_stoch1[:, None])
            prior_pred = prior_dec_logits.argmax(dim=-1)[:, 0]
        finally:
            if was_training:
                model_core.train()

        b, _t, num_views, tokens_per_view = tokens.shape
        h, w = tuple(int(x) for x in model_core.encoder.spatial_grid)
        if h * w != tokens_per_view:
            print(f"[dreamerv3-token][viz] skip: h*w={h*w} != tokens_per_view={tokens_per_view}")
            return
        view_labels = list(OmegaConf.select(viz_cfg, "view_labels", default=["third", "wrist"]))
        if len(view_labels) != num_views:
            view_labels = [f"view{idx}" for idx in range(num_views)]

        num_samples = min(int(OmegaConf.select(viz_cfg, "num_samples", default=4)), b)
        cell_size = int(OmegaConf.select(viz_cfg, "cell_size", default=192))
        out_dir = self.out_dir / "viz"
        saved = 0
        for sample_idx in range(num_samples):
            panels: list[tuple[str, Any]] = []
            for view_idx, label in enumerate(view_labels):
                panels.extend([
                    (f"{label} cur", self._decode_token_view(tokens[sample_idx, 0, view_idx], h, w)),
                    (f"{label} recon", self._decode_token_view(post_pred[sample_idx, 0, view_idx], h, w)),
                    (f"{label} next", self._decode_token_view(tokens[sample_idx, 1, view_idx], h, w)),
                    (f"{label} prior", self._decode_token_view(prior_pred[sample_idx, view_idx], h, w)),
                ])
            path = out_dir / f"step_{self.global_step:07d}_sample{sample_idx:02d}.png"
            self._save_viz_strip(path, panels, cell_size=cell_size)
            saved += 1
        if saved:
            print(f"[dreamerv3-token][viz] step {self.global_step}: wrote {saved} panel(s) under {out_dir}")

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
                print(f"[dreamerv3-token]   data parallel = {torch.cuda.device_count()} visible CUDA devices")
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
            weight_decay=float(OmegaConf.select(self.cfg, "optim.weight_decay", default=0.0)),
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
        save_every = int(OmegaConf.select(self.cfg, "training.save_every", default=1000))
        tqdm_interval_sec = float(OmegaConf.select(self.cfg, "training.tqdm_interval_sec", default=1.0))
        warmup = int(OmegaConf.select(self.cfg, "optim.warmup", default=1000))
        base_lr = float(OmegaConf.select(self.cfg, "optim.lr", default=4e-5))
        grad_clip = float(OmegaConf.select(self.cfg, "optim.grad_clip", default=100.0))

        print(
            f"[dreamerv3-token] training for {max_steps:,} steps "
            f"({len(loader):,} batches/epoch) ..."
        )
        log_mode = "a" if resumed else "w"
        log_handle = open(self.log_path, log_mode)
        try:
            while self.global_step < max_steps:
                self.epoch += 1
                with tqdm.tqdm(
                    loader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=tqdm_interval_sec,
                ) as tepoch:
                    for batch in tepoch:
                        if self.global_step >= max_steps:
                            break
                        if warmup > 0:
                            lr_scale = min(1.0, float(self.global_step + 1) / float(warmup))
                            for group in optimizer.param_groups:
                                group["lr"] = base_lr * lr_scale

                        batch = _to_device(batch, self.device)
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                        tepoch.set_postfix(
                            refresh=False,
                            step=f"{self.global_step}/{max_steps}",
                            wm=float(row["loss"]),
                            rec=float(row["rec_loss"]),
                            ce=float(row["token_ce"]),
                            acc=float(row["token_acc"]),
                            dyn=float(row["dyn_loss"]),
                            rep=float(row["rep_loss"]),
                            uniq=float(row["pred_unique_tokens"]),
                        )
                        if log_every > 0 and self.global_step % log_every == 0:
                            log_handle.write(json.dumps(row) + "\n")
                            log_handle.flush()

                        self._maybe_save_viz(model_core, batch)

                        if save_every > 0 and self.global_step > 0 and self.global_step % save_every == 0:
                            self._save_ckpt(model_core, optimizer, self.ckpt_dir / "latest.ckpt")
                        self.global_step += 1
                        if self.global_step >= max_steps:
                            break
        finally:
            log_handle.close()

        self._save_ckpt(model_core, optimizer, self.ckpt_dir / "latest.ckpt")
        self._save_ckpt(model_core, optimizer, self.ckpt_dir / f"step_{self.global_step:08d}.ckpt")
        print("[dreamerv3-token] done")
        print(f"  log  = {self.log_path}")
        print(f"  ckpt = {self.ckpt_dir / 'latest.ckpt'}")


__all__ = ["DreamerV3TokenWorkspace"]
