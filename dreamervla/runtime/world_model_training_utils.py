"""Shared helpers for the standalone DreamerV3 world-model runners.

These were duplicated verbatim across ``dreamerv3_pixel_runner`` and
``dreamerv3_token_runner`` (and, for ``save_viz_strip``, ``dreamervla_runner``).
The logic here is behaviour-preserving: the per-file differences that remain
(the ``[dreamerv3-pixel]`` / ``[dreamerv3-token]`` log tag and pixel's
``resume_reset_step`` warm-start branch) are parameterised below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from dreamervla.utils.seed import capture_rng_state, restore_rng_state


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def save_viz_strip(path: Path, panels: list[tuple[str, Any]], cell_size: int) -> None:
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


class DreamerCkptResumeMixin:
    """Checkpoint save/resume shared by the standalone DreamerV3 runners.

    Subclasses set ``_ckpt_log_tag`` (e.g. ``"dreamerv3-pixel"``) for the log
    prefix.  Subclasses that gate console output on the main process expose a
    ``_print`` method; otherwise plain ``print`` is used.  The
    ``resume_reset_step`` branch is config-gated (default ``False``), so runners
    whose configs never set it behave exactly as if the branch were absent.
    """

    _ckpt_log_tag: str = "dreamerv3"

    _save_viz_strip = staticmethod(save_viz_strip)

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
            "rng": capture_rng_state(),
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
        log = getattr(self, "_print", print)
        tag = self._ckpt_log_tag
        if not bool(OmegaConf.select(self.cfg, "training.resume", default=False)):
            return False
        path = self._resolve_resume_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"training.resume=true but checkpoint not found: {path}"
            )
        log(f"[{tag}] resuming from {path}")
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
            log(
                f"[{tag}] resume model missing={len(missing)} unexpected={len(unexpected)}"
            )

        skip_optimizer = bool(
            OmegaConf.select(self.cfg, "training.resume_skip_optimizer", default=False)
        )
        if not skip_optimizer and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        elif skip_optimizer:
            log(
                f"[{tag}] skipping optimizer state (training.resume_skip_optimizer=true)"
            )

        # ``resume_reset_step`` = warm-start a NEW training run from the ckpt's
        # model weights only.  global_step / epoch / RNG stay at their
        # initial (fresh-seed) values so the lr warmup, save-every cadence,
        # and epoch budget all behave as a fresh run.  Use this when the
        # training objective or model topology has changed (e.g. switching
        # per-step teacher forcing -> chunk_loss with a new mask_obs_token).
        reset_step = bool(
            OmegaConf.select(self.cfg, "training.resume_reset_step", default=False)
        )
        if reset_step:
            log(
                f"[{tag}] resume_reset_step=true: keeping fresh "
                f"global_step={self.global_step} epoch={self.epoch} and RNG"
            )
        else:
            self.global_step = int(payload.get("global_step", self.global_step))
            self.epoch = int(payload.get("epoch", self.epoch))
            restore_rng_state(payload.get("rng"))
        log(f"[{tag}] resumed at global_step={self.global_step} epoch={self.epoch}")
        return True


__all__ = ["to_device", "save_viz_strip", "DreamerCkptResumeMixin"]
