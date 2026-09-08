"""Frozen, checkpoint-selected visual readout for closed-loop dynamics supervision."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch import nn
from torch.utils.checkpoint import checkpoint

from dreamervla.utils.run_config import load_run_config

logger = logging.getLogger(__name__)


class FrozenDecodedVisualLoss(nn.Module):
    """Compare rollout→decode to encode→decode without training the decoder.

    Dynamic-region reconstruction prevents background-dominated latent MSE from
    hiding visible prediction errors. Signed temporal differences penalize wrong
    motion, not simply too little motion. Targets never enter recurrent history.
    The readout is serialized with the WM but excluded from its optimizer.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        decoder: nn.Module | None = None,
        loss_scale: float = 0.5,
        temporal_scale: float = 1.0,
        frame_stride: int = 5,
        decode_batch_size: int = 2,
        energy_floor: float = 1.0e-5,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if (checkpoint_path is None) == (decoder is None):
            raise ValueError("Provide exactly one of checkpoint_path or decoder")
        for name, value in (("loss_scale", loss_scale), ("temporal_scale", temporal_scale)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if frame_stride < 1 or decode_batch_size < 1:
            raise ValueError("frame_stride and decode_batch_size must be positive")
        if not math.isfinite(energy_floor) or energy_floor <= 0:
            raise ValueError("energy_floor must be finite and positive")
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            path = Path(checkpoint_path).expanduser().resolve(strict=True)
            config = load_run_config(path)
            if config.get("pixel_decoder") is None:
                raise ValueError(f"Missing pixel_decoder construction config: {path}")
            decoder = instantiate(config.pixel_decoder)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload["state_dicts"]["pixel_decoder"]
            if state and all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            decoder.load_state_dict(state, strict=True)
            logger.info("Loaded frozen visual readout strictly: %s (%d tensors)", path, len(state))
        if not isinstance(decoder, nn.Module):
            raise TypeError("decoder must be a torch module")
        self.decoder = decoder.requires_grad_(False).eval()
        self.loss_scale = float(loss_scale)
        self.temporal_scale = float(temporal_scale)
        self.frame_stride = int(frame_stride)
        self.decode_batch_size = int(decode_batch_size)
        self.energy_floor = float(energy_floor)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.train(False)

    def train(self, mode: bool = True) -> FrozenDecodedVisualLoss:
        """Keep dropout/statistics fixed even when the owning world model trains."""
        super().train(False)
        return self

    def _decode(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.flatten(0, 1)
        outputs = []
        for part in flat.split(self.decode_batch_size):
            if self.gradient_checkpointing and torch.is_grad_enabled() and part.requires_grad:
                value = checkpoint(self.decoder, part, use_reentrant=False)
            else:
                value = self.decoder(part)
            outputs.append(value.float())
        return torch.cat(outputs).unflatten(0, tokens.shape[:2])

    def _view_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Exclude absent camera views; never reinterpret padding as an image."""
        per_view = int(self.decoder.tokens_per_view)
        return torch.stack(
            [
                mask[..., i * per_view : (i + 1) * per_view].all(-1)
                for i in self.decoder.view_indices
            ],
            dim=-1,
        )[..., None, None, None]

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Supervise selected future frames from ``[B,T,N,D]`` latent tensors."""
        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError("prediction and target must have equal [B,T,N,D] shape")
        if anchor.shape != (prediction.shape[0], 1, *prediction.shape[2:]):
            raise ValueError("anchor must be the last REAL context frame [B,1,N,D]")
        steps = prediction.shape[1]
        if steps < 1:
            raise ValueError("prediction must contain a future frame")
        selected = list(range(self.frame_stride - 1, steps, self.frame_stride))
        if not selected or selected[-1] != steps - 1:
            selected.append(steps - 1)
        indices = torch.tensor(selected, device=prediction.device, dtype=torch.long)
        with torch.no_grad():
            reference = self._decode(target.detach().index_select(1, indices))
            initial = self._decode(anchor.detach())
            valid = torch.ones_like(reference[..., :1, :1, :1])
            if token_mask is not None:
                if token_mask.shape != target.shape[:-1] or anchor_mask is None:
                    raise ValueError("token_mask must match [B,T,N] and have anchor_mask")
                if anchor_mask.shape != anchor.shape[:-1]:
                    raise ValueError("anchor_mask must match [B,1,N]")
                valid = self._view_mask(token_mask.index_select(1, indices)).to(reference)
                valid = valid * self._view_mask(anchor_mask).to(reference)
            displacement = reference - initial
            weights = displacement.abs().mean(-3, keepdim=True) * valid
            # Normalize within each trajectory, so different batches or moving
            # camera fractions cannot silently change the objective's scale.
            weights = weights / weights.mean((1, 2, 3, 4, 5), keepdim=True).clamp_min(1e-8)
            weights = weights + (weights.sum((1, 2, 3, 4, 5), keepdim=True) == 0) * valid
            energy = (
                (displacement.square() * weights).mean((1, 2, 3, 4, 5)).clamp_min(self.energy_floor)
            )
            true_delta = reference - torch.cat([initial, reference[:, :-1]], dim=1)
            delta_valid = valid * torch.cat([valid[:, :1], valid[:, :-1]], dim=1)
            delta_energy = (
                (true_delta.square() * delta_valid)
                .mean((1, 2, 3, 4, 5))
                .clamp_min(self.energy_floor)
            )
        decoded = self._decode(prediction.index_select(1, indices))
        reconstruction = ((decoded - reference).square() * weights).mean((1, 2, 3, 4, 5)) / energy
        pred_delta = decoded - torch.cat([initial, decoded[:, :-1]], dim=1)
        temporal = ((pred_delta - true_delta).square() * delta_valid).mean(
            (1, 2, 3, 4, 5)
        ) / delta_energy
        rec_loss, temporal_loss = reconstruction.mean(), temporal.mean()
        total = rec_loss + self.temporal_scale * temporal_loss
        with torch.no_grad():
            motion_ratio = (
                (pred_delta.square() * delta_valid).sum()
                / (true_delta.square() * delta_valid).sum().clamp_min(1e-8)
            ).sqrt()
            pixel_mse = ((decoded - reference).square() * valid).sum() / (
                valid.expand_as(decoded).sum().clamp_min(1)
            )
        return {
            "_loss": self.loss_scale * total,
            "decoded_visual_loss": total.detach(),
            "decoded_reconstruction_ratio": rec_loss.detach(),
            "decoded_temporal_error_ratio": temporal_loss.detach(),
            "decoded_motion_ratio": motion_ratio,
            "decoded_pixel_mse": pixel_mse,
        }
