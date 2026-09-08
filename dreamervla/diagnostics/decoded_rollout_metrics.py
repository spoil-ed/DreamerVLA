"""Evaluation-only decoded rollout metrics; never a world-model training loss."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch import nn

from dreamervla.utils.run_config import load_run_config

logger = logging.getLogger(__name__)


class DecodedRolloutMetrics(nn.Module):
    """Compare rollout→decode to encode→decode with an independent frozen readout.

    All decoding is no-grad, even when called inside a gradient-enabled context.
    Do not attach this evaluator to the WM, its optimizer, or its checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        decoder: nn.Module | None = None,
        temporal_include_anchor: bool = True,
        frame_stride: int = 5,
        decode_batch_size: int = 2,
        energy_floor: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if (checkpoint_path is None) == (decoder is None):
            raise ValueError("Provide exactly one of checkpoint_path or decoder")
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
        self.temporal_include_anchor = bool(temporal_include_anchor)
        self.frame_stride = int(frame_stride)
        self.decode_batch_size = int(decode_batch_size)
        self.energy_floor = float(energy_floor)
        self.train(False)

    def train(self, mode: bool = True) -> DecodedRolloutMetrics:
        """Keep dropout/statistics fixed regardless of the caller's mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def _decode(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.flatten(0, 1)
        outputs = []
        for part in flat.split(self.decode_batch_size):
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

    @torch.no_grad()
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Measure selected future frames without creating any gradient path."""
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
        if not self.temporal_include_anchor and len(selected) < 2:
            raise ValueError("Ongoing temporal metrics require at least two decoded frames")
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
            temporal_valid = delta_valid.clone()
            if not self.temporal_include_anchor:
                temporal_valid[:, 0] = 0
            delta_energy = (
                (true_delta.square() * temporal_valid)
                .mean((1, 2, 3, 4, 5))
                .clamp_min(self.energy_floor)
            )
        decoded = self._decode(prediction.index_select(1, indices))
        reconstruction = ((decoded - reference).square() * weights).mean((1, 2, 3, 4, 5)) / energy
        pred_delta = decoded - torch.cat([initial, decoded[:, :-1]], dim=1)
        temporal = ((pred_delta - true_delta).square() * temporal_valid).mean(
            (1, 2, 3, 4, 5)
        ) / delta_energy
        rec_loss, temporal_loss = reconstruction.mean(), temporal.mean()
        with torch.no_grad():
            motion_ratio = (
                (pred_delta.square() * delta_valid).sum()
                / (true_delta.square() * delta_valid).sum().clamp_min(1e-8)
            ).sqrt()
            pixel_mse = ((decoded - reference).square() * valid).sum() / (
                valid.expand_as(decoded).sum().clamp_min(1)
            )
            # The anchor-inclusive RMS can be high after one jump followed by
            # completely static predictions. Report prediction-to-prediction
            # motion separately, including its direction and target energy.
            ongoing_valid = delta_valid[:, 1:]
            ongoing_pred, ongoing_true = pred_delta[:, 1:], true_delta[:, 1:]
            pred_energy = (ongoing_pred.square() * ongoing_valid).sum()
            true_energy = (ongoing_true.square() * ongoing_valid).sum()
            ongoing_error = ((ongoing_pred - ongoing_true).square() * ongoing_valid).sum()
            ongoing_count = ongoing_valid.expand_as(ongoing_true).sum().clamp_min(1)
            late_start = max(1, len(selected) // 2)
            late_valid = delta_valid[:, late_start:]
            late_ratio = (
                (pred_delta[:, late_start:].square() * late_valid).sum()
                / (true_delta[:, late_start:].square() * late_valid).sum().clamp_min(1e-8)
            ).sqrt()
        return {
            "decoded_reconstruction_ratio": rec_loss.detach(),
            "decoded_temporal_error_ratio": temporal_loss.detach(),
            "decoded_motion_ratio": motion_ratio,
            "decoded_pixel_mse": pixel_mse,
            "decoded_ongoing_motion_ratio": (pred_energy / true_energy.clamp_min(1e-8)).sqrt(),
            "decoded_ongoing_temporal_error_ratio": ongoing_error
            / true_energy.clamp_min(self.energy_floor * ongoing_count),
            "decoded_ongoing_motion_cosine": (ongoing_pred * ongoing_true * ongoing_valid).sum()
            / (pred_energy * true_energy).sqrt().clamp_min(1e-8),
            "decoded_ongoing_target_motion_rms": (true_energy / ongoing_count).sqrt(),
            "decoded_ongoing_valid_pairs": ongoing_valid.sum(),
            "decoded_late_motion_ratio": late_ratio,
        }
