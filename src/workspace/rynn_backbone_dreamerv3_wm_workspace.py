from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

from src.models.encoder.protocol import EncoderInputBatch
from src.workspace.dreamerv3_pixel_workspace import DreamerV3PixelWorkspace


class RynnBackboneDreamerV3WMWorkspace(DreamerV3PixelWorkspace):
    """Pixel DreamerV3 trainer with only the observation encoder replaced.

    The training loop, optimizer, checkpoint format, visualizer cadence, and
    tqdm postfix are inherited from ``DreamerV3PixelWorkspace``.  This workspace
    only prepares ``obs_embedding`` by running frozen RynnVLA-002 on the pixel
    observations before the normal DreamerV3 world-model forward pass.
    """

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.log_path = self.out_dir / "dreamerv3_pixel_rynn_backbone_logs.json.txt"
        self.rynn_encoder: torch.nn.Module | None = None

    @staticmethod
    def _task_prompt_from_path(path: str) -> str:
        stem = Path(str(path)).name
        if stem.endswith("_demo.hdf5"):
            stem = stem[: -len("_demo.hdf5")]
        else:
            stem = Path(stem).stem
        return stem.replace("_", " ")

    @staticmethod
    def _tensor_to_pil_rynn(image_chw: torch.Tensor) -> Image.Image:
        arr = (
            image_chw.detach()
            .clamp(0, 255)
            .to(dtype=torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return Image.fromarray(arr)

    def _build_frozen_encoder_cfg(self) -> DictConfig:
        encoder_cfg = copy.deepcopy(self.cfg.encoder)
        init_model_path = OmegaConf.select(self.cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            with open_dict(encoder_cfg):
                encoder_cfg.model_path = str(init_model_path)
        return encoder_cfg

    def _setup_auxiliary_modules(self) -> None:
        if self.rynn_encoder is not None:
            return
        if OmegaConf.select(self.cfg, "dataset.hidden_dir", default=None) is not None:
            self._print("[dreamerv3-pixel] using precomputed Rynn hidden; skipping backbone encoder build.")
            return
        self._print("[dreamerv3-pixel] building frozen RynnVLA backbone encoder ...")
        encoder_cfg = self._build_frozen_encoder_cfg()
        self.rynn_encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        self.rynn_encoder.eval()
        for param in self.rynn_encoder.parameters():
            param.requires_grad = False

    def _move_batch_to_device_before_prepare(self) -> bool:
        return False

    @torch.no_grad()
    def _encode_pixel_observations(
        self,
        images: torch.Tensor,
        file_paths: list[str] | tuple[str, ...] | None,
    ) -> torch.Tensor:
        if self.rynn_encoder is None:
            raise RuntimeError("RynnVLA encoder must be built before encoding pixel observations")
        if images.ndim != 5:
            raise ValueError(f"expected images [B,T,C,H,W], got {tuple(images.shape)}")
        bsz, steps, channels, _height, _width = images.shape
        if channels != 6:
            raise ValueError(f"Rynn backbone pixel WM expects two RGB views concatenated as C=6, got C={channels}")

        prompt_text: list[str] = []
        conversations: list[list[dict[str, str]]] = []
        image_batches: list[list[Any]] = []
        for bidx in range(bsz):
            prompt = self._task_prompt_from_path(file_paths[bidx]) if file_paths else ""
            for tidx in range(steps):
                frame = images[bidx, tidx]
                third = self._tensor_to_pil_rynn(frame[:3])
                wrist = self._tensor_to_pil_rynn(frame[3:6])
                prompt_text.append(prompt)
                conversations.append([])
                image_batches.append([third, wrist])

        chunk_size = max(1, int(OmegaConf.select(self.cfg, "training.encoder_chunk_size", default=8)))
        hidden_chunks: list[torch.Tensor] = []
        for start in range(0, len(prompt_text), chunk_size):
            end = min(start + chunk_size, len(prompt_text))
            encoder_batch = EncoderInputBatch(
                prompt_text=prompt_text[start:end],
                conversations=conversations[start:end],
                images=image_batches[start:end],
                task_type=None,
            )
            hidden_chunks.append(self.rynn_encoder.encode_inputs(encoder_batch).hidden.detach())
        return torch.cat(hidden_chunks, dim=0).view(bsz, steps, -1)

    def _prepare_batch_for_model(
        self,
        batch: dict[str, Any],
        model_core: torch.nn.Module,
    ) -> dict[str, Any]:
        del model_core
        if isinstance(batch.get("obs_embedding"), torch.Tensor):
            prepared = dict(batch)
            for key, value in list(prepared.items()):
                if key == "images":
                    continue
                if isinstance(value, torch.Tensor):
                    prepared[key] = value.to(device=self.device, non_blocking=True)
            prepared["obs_embedding"] = prepared["obs_embedding"].to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            return prepared

        images = batch.get("images")
        if not isinstance(images, torch.Tensor):
            raise ValueError("Rynn backbone pixel WM batch must contain tensor `images`.")
        obs_embedding = self._encode_pixel_observations(images, batch.get("file_path"))
        prepared = dict(batch)
        for key, value in list(prepared.items()):
            if key == "images":
                continue
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device=self.device, non_blocking=True)
        prepared["obs_embedding"] = obs_embedding.to(device=self.device, dtype=torch.float32)
        return prepared

    @torch.no_grad()
    def _maybe_save_viz(self, model_core: torch.nn.Module, batch: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        viz_cfg = OmegaConf.select(self.cfg, "viz", default=None)
        if viz_cfg is None or not bool(OmegaConf.select(viz_cfg, "enabled", default=False)):
            return
        every = int(OmegaConf.select(viz_cfg, "every_n_steps", default=100))
        if every <= 0 or self.global_step % every != 0:
            return

        images = batch.get("images")
        obs_embedding = batch.get("obs_embedding")
        actions = batch.get("actions")
        is_first = batch.get("is_first")
        if not all(isinstance(x, torch.Tensor) for x in (images, obs_embedding, actions, is_first)):
            return
        if images.ndim != 5 or images.shape[1] < 2:
            return
        num_samples = min(int(OmegaConf.select(viz_cfg, "num_samples", default=4)), int(images.shape[0]))
        if num_samples <= 0:
            return

        images_viz = images[:num_samples]
        obs_embedding_viz = obs_embedding[:num_samples]
        actions_viz = actions[:num_samples]
        is_first_viz = is_first[:num_samples]

        was_training = model_core.training
        model_core.eval()
        try:
            enc = model_core.encoder(obs_embedding_viz)
            seq = model_core.rssm.observe(enc, actions_viz, is_first_viz)
            recon = model_core.decoder(seq["deter"], seq["stoch"])

            deter0 = seq["deter"][:, 0]
            stoch0 = seq["stoch"][:, 0]
            action1 = actions_viz[:, 1].to(device=deter0.device, dtype=deter0.dtype)
            prior_deter1 = model_core.rssm._core(deter0, stoch0, action1)
            prior_logits1 = model_core.rssm._prior(prior_deter1)
            prior_idx1 = prior_logits1.argmax(dim=-1)
            prior_stoch1 = torch.nn.functional.one_hot(prior_idx1, model_core.rssm.classes).to(dtype=prior_logits1.dtype)
            prior_recon = model_core.decoder(prior_deter1[:, None], prior_stoch1[:, None])[:, 0]
        finally:
            if was_training:
                model_core.train()

        view_labels = list(OmegaConf.select(viz_cfg, "view_labels", default=["third", "wrist"]))
        channels_per_view = int(OmegaConf.select(viz_cfg, "channels_per_view", default=3))
        total_channels = int(images.shape[2])
        num_views = len(view_labels)
        if num_views <= 0 or total_channels != num_views * channels_per_view:
            if total_channels % channels_per_view != 0:
                return
            num_views = total_channels // channels_per_view
            view_labels = [f"view{idx}" for idx in range(num_views)]

        cell_size = int(OmegaConf.select(viz_cfg, "cell_size", default=192))
        out_dir = self.out_dir / "viz"
        saved = 0
        for sample_idx in range(num_samples):
            panels: list[tuple[str, Any]] = []
            for view_idx, label in enumerate(view_labels):
                c0 = view_idx * channels_per_view
                c1 = c0 + channels_per_view
                panels.extend([
                    (f"{label} cur", self._tensor_to_pil(images_viz[sample_idx, 0, c0:c1])),
                    (f"{label} recon", self._tensor_to_pil(recon[sample_idx, 0, c0:c1])),
                    (f"{label} next", self._tensor_to_pil(images_viz[sample_idx, 1, c0:c1])),
                    (f"{label} prior", self._tensor_to_pil(prior_recon[sample_idx, c0:c1])),
                ])
            path = out_dir / f"step_{self.global_step:07d}_sample{sample_idx:02d}.png"
            self._save_viz_strip(path, panels, cell_size=cell_size)
            saved += 1
        if saved:
            print(f"[dreamerv3-pixel][viz] step {self.global_step}: wrote {saved} panel(s) under {out_dir}")


__all__ = ["RynnBackboneDreamerV3WMWorkspace"]
