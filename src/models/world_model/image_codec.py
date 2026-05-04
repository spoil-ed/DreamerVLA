"""Strided-conv encoder / ConvTranspose decoder for per-image-token hiddens.

Bridges the VLA backbone's per-image-token hidden representation
[B, N_img=256, C_obs=4096] (arranged as a 16×16 spatial grid) to the world
model's scalar state obs_dim (e.g. 1024), and back.

Design:
- Encoder mirrors TransDreamer's ImgEncoder (modules_transformer.py:506):
  strided 2D convs, K=4 S=2, ELU, xavier init, no norm.  But we begin with a
  1×1 projection from 4096 → a compact channel count so the strided convs do
  not blow up in parameters.  Final spatial 4×4 → flatten → Linear to obs_dim.

- Decoder mirrors DreamerV3 with the `bspace` variant (rssm.py:253 / bspace
  branch at 314-327): a BlockLinear (group-wise Linear) unfolds the 1D deter
  state back to a 4×4 spatial grid, a parallel Linear does the same for the
  stoch vector, their sum is LayerNormed + activated, then ConvTranspose
  stages up-sample 4 → 8 → 16, with a final per-token projection (1×1 conv)
  to restore the 4096-d channel dim that `lm_head` expects.

The output [B, 256, 4096] can be fed through the frozen LLM `lm_head` to
produce logits over the vocabulary for CE training, and additionally MSE'd
against the encoder's own per-token hidden target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_linear import BlockLinear


def _conv2d_block(
    in_ch: int, out_ch: int, kernel: int, stride: int, pad: int,
    *, bias: bool = True, act: bool = True, transpose: bool = False,
    output_padding: int = 0,
) -> nn.Module:
    """Conv (or ConvTranspose) + ELU, Xavier-initialised.  Mirrors
    TransDreamer's Conv2DBlock defaults (num_groups=0, elu, xavier)."""
    if transpose:
        conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad,
            output_padding=output_padding, bias=bias,
        )
    else:
        conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=bias,
        )
    nn.init.xavier_uniform_(conv.weight)
    if bias:
        nn.init.zeros_(conv.bias)
    if act:
        return nn.Sequential(conv, nn.ELU(inplace=True))
    return conv


class ConvEncoderStem(nn.Module):
    """Per-image-token hidden → 1D obs vector via strided convs.

    Input:  x  [B, N_img, in_channels]  with N_img == spatial[0] * spatial[1]
    Output: z  [B, obs_dim]
    """

    def __init__(
        self,
        in_channels: int = 4096,
        spatial: tuple[int, int] = (16, 16),
        obs_dim: int = 1024,
        init_proj_channels: int = 384,
        stage_channels: tuple[int, ...] = (96, 192),
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        # Append LayerNorm after the final 1024-d Linear projection so that
        # batch-axis variance is forced > 0; without it the obs vector can
        # collapse to a single direction across samples (observed empirically
        # under transition_loss / delta_latent_loss MSE objectives).
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.spatial = (int(spatial[0]), int(spatial[1]))
        self.obs_dim = int(obs_dim)
        self.init_proj_channels = int(init_proj_channels)
        self.post_norm = bool(post_norm)

        # 1×1 projection to compress 4096 → init_proj_channels before the
        # strided convs (keeps conv parameter count sane).
        self.init_proj = _conv2d_block(
            in_channels, init_proj_channels, kernel=1, stride=1, pad=0, act=True,
        )

        # Strided convs.  K=4 S=2 pad=1 keeps 2× clean downsampling (16→8→4).
        stages: list[nn.Module] = []
        prev = init_proj_channels
        for c in stage_channels:
            stages.append(_conv2d_block(prev, c, kernel=kernel, stride=stride, pad=padding, act=True))
            prev = c
        self.conv_stages = nn.Sequential(*stages)
        self.final_channels = prev

        # Compute final spatial size (assumes symmetric + divisible).
        h, w = self.spatial
        for _ in stage_channels:
            h = (h + 2 * padding - kernel) // stride + 1
            w = (w + 2 * padding - kernel) // stride + 1
        self.final_spatial = (h, w)

        flat_dim = prev * h * w
        self.proj = nn.Linear(flat_dim, obs_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.post_ln = nn.LayerNorm(obs_dim) if self.post_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., N_img, in_channels]  (leading dims can be [B] or [B, T])
        returns: [..., obs_dim]
        """
        *leading, n_img, c_in = x.shape
        assert c_in == self.in_channels, (x.shape, self.in_channels)
        h, w = self.spatial
        assert n_img == h * w, (n_img, self.spatial)

        # [N, C, H, W] for the conv stack
        x = x.reshape(-1, n_img, c_in).permute(0, 2, 1).contiguous()   # [N, C, N_img]
        x = x.view(-1, c_in, h, w)

        x = self.init_proj(x)
        x = self.conv_stages(x)
        x = x.flatten(1)                                                # [N, flat_dim]
        x = self.proj(x)                                                # [N, obs_dim]
        if self.post_ln is not None:
            x = self.post_ln(x)
        x = x.view(*leading, self.obs_dim)
        return x


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name!r}")


class DreamerCNNEncoderStem(nn.Module):
    """Dreamer-style CNN observation encoder for token embedding grids.

    DreamerV3 encodes image observations with a stack of spatial convolutions
    before the RSSM posterior.  In DreamerVLA token mode we do not have RGB
    pixels anymore, but image-token embeddings still form a spatial grid.  This
    stem treats that grid as the observation image and applies a closer
    Dreamer-style Conv/Norm/Act/Downsample tower before projecting to ``obs_dim``.

    Input:  x  [..., N_img, in_channels]
    Output: z  [..., obs_dim]
    """

    def __init__(
        self,
        in_channels: int = 512,
        spatial: tuple[int, int] = (16, 16),
        obs_dim: int = 1024,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        layers: int = 1,
        norm: bool = True,
        act: str = "gelu",
        strided: bool = False,
        post_norm: bool = True,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("DreamerCNNEncoderStem requires layers >= 1")
        self.in_channels = int(in_channels)
        self.spatial = (int(spatial[0]), int(spatial[1]))
        self.obs_dim = int(obs_dim)
        self.depth = int(depth)
        self.mults = tuple(int(m) for m in mults)
        self.kernel = int(kernel)
        self.layers = int(layers)
        self.use_norm = bool(norm)
        self.act_name = str(act)
        self.strided = bool(strided)
        self.post_norm = bool(post_norm)

        padding = self.kernel // 2
        stages: list[nn.Module] = []
        prev = self.in_channels
        for mult in self.mults:
            out_ch = self.depth * mult
            for layer_idx in range(self.layers):
                stride = 2 if (self.strided and layer_idx == 0) else 1
                conv = nn.Conv2d(
                    prev,
                    out_ch,
                    kernel_size=self.kernel,
                    stride=stride,
                    padding=padding,
                    bias=True,
                )
                nn.init.xavier_uniform_(conv.weight)
                nn.init.zeros_(conv.bias)
                block: list[nn.Module] = [conv]
                if self.use_norm:
                    block.append(nn.GroupNorm(1, out_ch))
                block.append(_activation(self.act_name))
                stages.append(nn.Sequential(*block))
                prev = out_ch
            if not self.strided:
                stages.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.cnn = nn.Sequential(*stages)

        h, w = self.spatial
        for _ in self.mults:
            h = max(h // 2, 1)
            w = max(w // 2, 1)
        self.final_spatial = (h, w)
        flat_dim = prev * h * w
        self.proj = nn.Linear(flat_dim, self.obs_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.post_ln = nn.LayerNorm(self.obs_dim) if self.post_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *leading, n_img, c_in = x.shape
        if c_in != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {c_in}")
        h, w = self.spatial
        if n_img != h * w:
            raise ValueError(f"Expected {h * w} image tokens for spatial={self.spatial}, got {n_img}")

        x = x.reshape(-1, n_img, c_in).permute(0, 2, 1).contiguous()
        x = x.view(-1, c_in, h, w)
        x = self.cnn(x)
        # With odd or tiny grids, explicit interpolation keeps the final Linear
        # shape stable while preserving the Dreamer-style downsampling schedule.
        if tuple(x.shape[-2:]) != self.final_spatial:
            x = F.adaptive_avg_pool2d(x, self.final_spatial)
        x = x.flatten(1)
        x = self.proj(x)
        if self.post_ln is not None:
            x = self.post_ln(x)
        return x.view(*leading, self.obs_dim)


class BspaceConvDecoderHead(nn.Module):
    """1D deter (+ optional stoch) → per-image-token hidden [..., N_img, 4096].

    Mirrors DreamerV3 Decoder bspace branch (rssm.py:314-327) + strided-ConvT
    up-sampling (rssm.py:331-344), but ends at `out_channels` (= 4096) so the
    outputs can be fed through the frozen LLM lm_head.
    """

    def __init__(
        self,
        deter_dim: int = 1024,
        stoch_dim: int = 256,
        minres: tuple[int, int] = (4, 4),
        mid_channels: int = 192,              # C_mid at the 4×4 bottleneck
        bspace_groups: int = 8,               # BlockLinear groups
        stage_channels: tuple[int, ...] = (96, 48),
        out_channels: int = 4096,             # lm_head input size
        out_spatial: tuple[int, int] = (16, 16),
        kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        stoch_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)
        self.minres = (int(minres[0]), int(minres[1]))
        self.mid_channels = int(mid_channels)
        self.bspace_groups = int(bspace_groups)
        self.out_channels = int(out_channels)
        self.out_spatial = (int(out_spatial[0]), int(out_spatial[1]))

        # BlockLinear maps 1D deter → spatial grid's worth of channels.
        # Per DreamerV3: output units = h * w * mid_channels, then rearrange
        # (g h w c) → h w (g c).  We enforce that split via group arithmetic.
        h, w = self.minres
        units = h * w * self.mid_channels
        assert units % self.bspace_groups == 0, (units, self.bspace_groups)
        assert self.mid_channels % self.bspace_groups == 0, (self.mid_channels, self.bspace_groups)
        self.deter_block = BlockLinear(
            in_features=self.deter_dim, out_features=units, groups=self.bspace_groups,
        )

        # Stoch branch: 2-layer MLP → h*w*mid_channels → reshape.
        self.stoch_mlp = nn.Sequential(
            nn.Linear(self.stoch_dim, stoch_hidden),
            nn.LayerNorm(stoch_hidden),
            nn.ELU(inplace=True),
            nn.Linear(stoch_hidden, units),
        )
        # Fuse deter+stoch at 4×4 with Norm + activation.
        self.fuse_norm = nn.LayerNorm(self.mid_channels)

        # ConvTranspose chain: 4 → 8 → 16 (spatial doubles per stage).
        deconv: list[nn.Module] = []
        prev = self.mid_channels
        for c in stage_channels:
            deconv.append(_conv2d_block(
                prev, c, kernel=kernel, stride=stride, pad=padding, act=True,
                transpose=True, output_padding=0,
            ))
            prev = c
        self.deconv_stages = nn.Sequential(*deconv)

        # Validate final spatial size (should match out_spatial).
        fh, fw = self.minres
        for _ in stage_channels:
            fh = (fh - 1) * stride - 2 * padding + kernel
            fw = (fw - 1) * stride - 2 * padding + kernel
        assert (fh, fw) == self.out_spatial, (
            f"Spatial after deconv stages is ({fh},{fw}); expected {self.out_spatial}. "
            f"Adjust stage_channels/minres/kernel so they match."
        )

        # Final per-token projection to lm_head's channel dim (4096).  Written
        # as 1×1 conv so it's uniform with the rest of the conv stack.
        self.to_vocab_channels = nn.Conv2d(prev, out_channels, kernel_size=1, stride=1, bias=True)
        nn.init.xavier_uniform_(self.to_vocab_channels.weight)
        nn.init.zeros_(self.to_vocab_channels.bias)

    def forward(
        self, deter: torch.Tensor, stoch: torch.Tensor,
    ) -> torch.Tensor:
        """
        deter: [..., deter_dim]
        stoch: [..., stoch_dim]
        returns: [..., N_img, out_channels]
        """
        *leading, _ = deter.shape
        h, w = self.minres
        g = self.bspace_groups
        c_per_group = self.mid_channels // g

        # Deter → BlockLinear → rearrange '(g h w c) -> h w (g c)'
        x0 = self.deter_block(deter)                                # [..., h*w*C_mid]
        x0 = x0.reshape(*leading, g, h, w, c_per_group)             # [..., g, h, w, c_g]
        x0 = x0.permute(*range(len(leading)), -3, -2, -4, -1)       # [..., h, w, g, c_g]
        x0 = x0.reshape(*leading, h, w, self.mid_channels)          # [..., h, w, C_mid]

        # Stoch → Linear → reshape to same 4D
        x1 = self.stoch_mlp(stoch)                                   # [..., h*w*C_mid]
        x1 = x1.reshape(*leading, h, w, self.mid_channels)

        x = self.fuse_norm(x0 + x1)                                  # LN on last (C_mid) dim
        x = torch.nn.functional.elu(x)

        # [..., h, w, C_mid] → [N, C_mid, h, w] for conv stack
        x = x.reshape(-1, h, w, self.mid_channels).permute(0, 3, 1, 2).contiguous()

        x = self.deconv_stages(x)
        x = self.to_vocab_channels(x)                                # [N, out_ch, H, W]

        # → [..., N_img, out_ch]
        N_img = self.out_spatial[0] * self.out_spatial[1]
        x = x.flatten(2).permute(0, 2, 1).contiguous()               # [N, N_img, out_ch]
        x = x.view(*leading, N_img, self.out_channels)
        return x


__all__ = ["ConvEncoderStem", "DreamerCNNEncoderStem", "BspaceConvDecoderHead"]
