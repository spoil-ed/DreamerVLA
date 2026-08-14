"""Decode continuous VLA observation tokens into RGB camera views."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(32, int(channels))
        while int(channels) % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class _UpsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, output_size: int) -> None:
        super().__init__()
        self.output_size = int(output_size)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.residual = _ResidualConvBlock(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.residual(self.proj(value))


class LatentTokenPixelDecoder(nn.Module):
    """Map selected square token slots to pixel-space RGB images.

    The decoder consumes the visual-width prefix of a Chunk-WM hidden tensor.
    It applies the same parameter-free per-token LayerNorm used by
    :class:`ChunkAwareWorldModel`, so both raw sidecars and WM predictions share
    one decoder input distribution.
    """

    def __init__(
        self,
        token_dim: int,
        token_count: int,
        tokens_per_view: int = 256,
        view_indices: Sequence[int] = (0,),
        image_size: int = 224,
        base_channels: int = 256,
        channel_schedule: Sequence[int] = (192, 128, 96, 64),
        upsample_sizes: Sequence[int] = (28, 56, 112, 224),
        token_normalization: str = "layer_norm",
        token_norm_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.token_count = int(token_count)
        self.tokens_per_view = int(tokens_per_view)
        self.view_indices = tuple(int(index) for index in view_indices)
        self.image_size = int(image_size)
        self.grid_size = int(self.tokens_per_view**0.5)
        if self.grid_size * self.grid_size != self.tokens_per_view:
            raise ValueError("tokens_per_view must be a perfect square")
        if not self.view_indices or min(self.view_indices) < 0:
            raise ValueError("view_indices must contain non-negative indices")
        required_tokens = (max(self.view_indices) + 1) * self.tokens_per_view
        if required_tokens > self.token_count:
            raise ValueError(
                f"selected views require {required_tokens} tokens, token_count={self.token_count}"
            )
        if self.token_dim <= 0 or int(base_channels) <= 0:
            raise ValueError("token_dim and base_channels must be positive")
        channels = tuple(int(value) for value in channel_schedule)
        sizes = tuple(int(value) for value in upsample_sizes)
        if len(channels) != len(sizes) or not channels:
            raise ValueError("channel_schedule and upsample_sizes must have equal non-zero length")
        if sizes[-1] != self.image_size or any(value <= 0 for value in sizes):
            raise ValueError("upsample_sizes must be positive and end at image_size")
        normalization = str(token_normalization).strip().lower()
        if normalization not in {"layer_norm", "none"}:
            raise ValueError("token_normalization must be 'layer_norm' or 'none'")
        self.token_norm = (
            nn.LayerNorm(
                self.token_dim,
                eps=float(token_norm_eps),
                elementwise_affine=False,
            )
            if normalization == "layer_norm"
            else nn.Identity()
        )
        self.token_proj = nn.Linear(self.token_dim, int(base_channels))
        self.view_embedding = nn.Parameter(
            torch.zeros(len(self.view_indices), int(base_channels), 1, 1)
        )
        stages: list[nn.Module] = []
        in_channels = int(base_channels)
        for out_channels, output_size in zip(channels, sizes, strict=True):
            stages.append(_UpsampleStage(in_channels, out_channels, output_size))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.output = nn.Sequential(
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @property
    def num_views(self) -> int:
        return len(self.view_indices)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode ``[..., token_count, D>=token_dim]`` to ``[..., V,3,H,W]``."""

        if tokens.ndim < 3:
            raise ValueError(f"tokens must be [...,N,D], got {tuple(tokens.shape)}")
        if int(tokens.shape[-2]) != self.token_count:
            raise ValueError(
                f"token count mismatch: expected {self.token_count}, got {tokens.shape[-2]}"
            )
        if int(tokens.shape[-1]) < self.token_dim:
            raise ValueError(f"token width must be >= {self.token_dim}, got {tokens.shape[-1]}")
        leading_shape = tuple(int(value) for value in tokens.shape[:-2])
        flat = tokens[..., : self.token_dim].reshape(-1, self.token_count, self.token_dim)
        selected = torch.stack(
            [
                flat[
                    :,
                    index * self.tokens_per_view : (index + 1) * self.tokens_per_view,
                ]
                for index in self.view_indices
            ],
            dim=1,
        )
        selected = self.token_norm(selected)
        value = self.token_proj(selected)
        value = value.reshape(
            -1,
            self.num_views,
            self.grid_size,
            self.grid_size,
            value.shape[-1],
        ).permute(0, 1, 4, 2, 3)
        value = value + self.view_embedding.unsqueeze(0)
        value = value.flatten(0, 1)
        value = self.output(self.stages(value))
        return value.reshape(*leading_shape, self.num_views, 3, self.image_size, self.image_size)


def latent_pixel_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float = 0.85,
    ssim_weight: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Return a lightweight L1 + SSIM pixel reconstruction objective."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    if prediction.ndim < 5 or prediction.shape[-3] != 3:
        raise ValueError("pixel tensors must be [...,V,3,H,W]")
    l1_weight = float(l1_weight)
    ssim_weight = float(ssim_weight)
    if l1_weight < 0.0 or ssim_weight < 0.0 or l1_weight + ssim_weight <= 0.0:
        raise ValueError("loss weights must be non-negative with a positive sum")
    prediction_f = prediction.float().clamp(0.0, 1.0)
    target_f = target.float().clamp(0.0, 1.0)
    flat_pred = prediction_f.reshape(-1, *prediction_f.shape[-3:])
    flat_target = target_f.reshape(-1, *target_f.shape[-3:])
    l1 = F.l1_loss(flat_pred, flat_target)
    ssim = _mean_ssim(flat_pred, flat_target)
    loss = l1_weight * l1 + ssim_weight * (1.0 - ssim)
    mse = F.mse_loss(flat_pred, flat_target)
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-10))
    return {"loss": loss, "l1": l1.detach(), "ssim": ssim.detach(), "psnr": psnr.detach()}


def _mean_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel_size = 7
    padding = kernel_size // 2
    mean_pred = F.avg_pool2d(prediction, kernel_size, stride=1, padding=padding)
    mean_target = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    var_pred = F.avg_pool2d(prediction.square(), kernel_size, 1, padding) - mean_pred.square()
    var_target = F.avg_pool2d(target.square(), kernel_size, 1, padding) - mean_target.square()
    covariance = (
        F.avg_pool2d(prediction * target, kernel_size, 1, padding) - mean_pred * mean_target
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_pred * mean_target + c1) * (2.0 * covariance + c2)
    denominator = (mean_pred.square() + mean_target.square() + c1) * (var_pred + var_target + c2)
    return (numerator / denominator.clamp_min(1.0e-8)).mean()


__all__ = ["LatentTokenPixelDecoder", "latent_pixel_reconstruction_loss"]
