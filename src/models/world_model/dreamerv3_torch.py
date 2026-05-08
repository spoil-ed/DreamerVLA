from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model.block_linear import BlockLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight.to(dtype=dtype)


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.square().mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        weight = self.weight.to(dtype=dtype).view(1, -1, 1, 1)
        return (x32 * rms).to(dtype) * weight


def _act(name: str) -> nn.Module:
    name = str(name).lower()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _module_ref_tensor(module: nn.Module) -> torch.Tensor | None:
    for tensor in module.parameters(recurse=True):
        return tensor
    for tensor in module.buffers(recurse=True):
        return tensor
    # DataParallel replicas can expose copied weights as plain Tensor
    # attributes instead of registered Parameters.
    for child in module.modules():
        for attr in ("weight", "bias"):
            tensor = getattr(child, attr, None)
            if isinstance(tensor, torch.Tensor):
                return tensor
    return None


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    tensor = _module_ref_tensor(module)
    return tensor.dtype if tensor is not None else fallback


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    tensor = _module_ref_tensor(module)
    return tensor.device if tensor is not None else fallback


class DreamerV3PixelEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        image_size: int = 64,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        act: str = "silu",
        strided: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.depths = tuple(int(depth) * int(m) for m in mults)
        self.strided = bool(strided)
        pad = int(kernel) // 2
        layers: list[nn.Module] = []
        prev = self.in_channels
        h = w = self.image_size
        for out_ch in self.depths:
            stride = 2 if self.strided else 1
            layers.append(nn.Conv2d(prev, out_ch, kernel_size=kernel, stride=stride, padding=pad))
            if not self.strided:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(ChannelRMSNorm(out_ch))
            layers.append(_act(act))
            prev = out_ch
            h = math.ceil(h / 2) if self.strided else h // 2
            w = math.ceil(w / 2) if self.strided else w // 2
        if not (3 <= h <= 16 and 3 <= w <= 16):
            raise ValueError(f"DreamerV3 final image grid should be 3..16, got {(h, w)}")
        self.cnn = nn.Sequential(*layers)
        self.final_hw = (h, w)
        self.out_dim = int(prev * h * w)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,T,C,H,W] in uint8 range [0,255].
        b, t, c, h, w = images.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        param_dtype = _module_dtype(self.cnn, images.dtype)
        x = images.reshape(b * t, c, h, w).to(dtype=param_dtype) / 255.0 - 0.5
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        return x.reshape(b, t, -1)


class DreamerV3TokenEncoder(nn.Module):
    """DreamerV3-style CNN encoder over discrete spatial image-token views."""

    def __init__(
        self,
        num_image_tokens_vocab: int = 8192,
        n_image_tokens: int = 512,
        num_views: int = 2,
        spatial_grid: tuple[int, int] = (16, 16),
        token_embed_dim: int = 512,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3),
        kernel: int = 5,
        act: str = "silu",
        strided: bool = False,
    ) -> None:
        super().__init__()
        self.num_image_tokens_vocab = int(num_image_tokens_vocab)
        self.n_image_tokens = int(n_image_tokens)
        self.num_views = int(num_views)
        self.spatial_grid = (int(spatial_grid[0]), int(spatial_grid[1]))
        self.tokens_per_view = self.spatial_grid[0] * self.spatial_grid[1]
        if self.num_views <= 0:
            raise ValueError(f"num_views must be positive, got {self.num_views}")
        if self.tokens_per_view * self.num_views != self.n_image_tokens:
            raise ValueError(
                f"num_views={self.num_views} and spatial_grid={self.spatial_grid} "
                f"produce {self.tokens_per_view * self.num_views} tokens, "
                f"expected n_image_tokens={self.n_image_tokens}"
            )
        self.token_embed_dim = int(token_embed_dim)
        self.depths = tuple(int(depth) * int(m) for m in mults)
        self.strided = bool(strided)
        self.embed = nn.Embedding(self.num_image_tokens_vocab, self.token_embed_dim)

        pad = int(kernel) // 2
        layers: list[nn.Module] = []
        prev = self.token_embed_dim * self.num_views
        h, w = self.spatial_grid
        for out_ch in self.depths:
            stride = 2 if self.strided else 1
            layers.append(nn.Conv2d(prev, out_ch, kernel_size=kernel, stride=stride, padding=pad))
            if not self.strided:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(ChannelRMSNorm(out_ch))
            layers.append(_act(act))
            prev = out_ch
            h = math.ceil(h / 2) if self.strided else h // 2
            w = math.ceil(w / 2) if self.strided else w // 2
            if h < 1 or w < 1:
                raise ValueError(
                    f"Token encoder downsampled spatial_grid={self.spatial_grid} "
                    f"too far with mults={mults}"
                )
        if not (3 <= h <= 16 and 3 <= w <= 16):
            raise ValueError(
                "DreamerV3 token encoder final grid should be 3..16, "
                f"got {(h, w)} from spatial_grid={self.spatial_grid} and mults={mults}"
            )
        self.cnn = nn.Sequential(*layers)
        self.final_hw = (h, w)
        self.out_dim = int(prev * h * w)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B,T,V,N] integer image-token indices. A flat [B,T,V*N]
        # tensor is also accepted for checkpoint/debug compatibility.
        if tokens.ndim == 3:
            b, t, n = tokens.shape
            if n != self.n_image_tokens:
                raise ValueError(f"Expected {self.n_image_tokens} image tokens, got {n}")
            tokens = tokens.reshape(b, t, self.num_views, self.tokens_per_view)
        elif tokens.ndim == 4:
            b, t, v, n = tokens.shape
            if v != self.num_views or n != self.tokens_per_view:
                raise ValueError(
                    f"Expected tokens [B,T,{self.num_views},{self.tokens_per_view}], "
                    f"got {tuple(tokens.shape)}"
                )
        else:
            raise ValueError(f"Expected token tensor [B,T,V,N], got {tuple(tokens.shape)}")
        b, t, v, n = tokens.shape
        x = self.embed(tokens.long())  # [B,T,V,N,C]
        h, w = self.spatial_grid
        x = x.reshape(b * t, v, h, w, self.token_embed_dim)
        x = x.permute(0, 1, 4, 2, 3).reshape(b * t, v * self.token_embed_dim, h, w)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        return x.reshape(b, t, -1)


class DreamerV3RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        imglayers: int = 2,
        obslayers: int = 1,
        dynlayers: int = 1,
        unimix: float = 0.01,
        free_nats: float = 1.0,
        act: str = "silu",
    ) -> None:
        super().__init__()
        if deter % blocks != 0:
            raise ValueError(f"deter={deter} must be divisible by blocks={blocks}")
        self.action_dim = int(action_dim)
        self.deter = int(deter)
        self.hidden = int(hidden)
        self.stoch = int(stoch)
        self.classes = int(classes)
        self.blocks = int(blocks)
        self.unimix = float(unimix)
        self.free_nats = float(free_nats)
        self.flat_stoch = self.stoch * self.classes
        activation = _act(act)

        self.dynin0 = nn.Sequential(nn.Linear(self.deter, self.hidden), RMSNorm(self.hidden), activation)
        self.dynin1 = nn.Sequential(nn.Linear(self.flat_stoch, self.hidden), RMSNorm(self.hidden), _act(act))
        self.dynin2 = nn.Sequential(nn.Linear(self.action_dim, self.hidden), RMSNorm(self.hidden), _act(act))
        core_in = self.deter + self.blocks * 3 * self.hidden
        dyn_layers: list[nn.Module] = []
        for _ in range(int(dynlayers)):
            dyn_layers.extend([BlockLinear(core_in, self.deter, self.blocks), RMSNorm(self.deter), _act(act)])
            core_in = self.deter
        self.dynhid = nn.Sequential(*dyn_layers)
        self.dyngru = BlockLinear(self.deter, 3 * self.deter, self.blocks)

        prior_layers: list[nn.Module] = []
        prior_in = self.deter
        for _ in range(int(imglayers)):
            prior_layers.extend([nn.Linear(prior_in, self.hidden), RMSNorm(self.hidden), _act(act)])
            prior_in = self.hidden
        prior_layers.append(nn.Linear(prior_in, self.flat_stoch))
        self.prior_net = nn.Sequential(*prior_layers)

        self.obslayers = int(obslayers)
        self._posterior: nn.Sequential | None = None

    def build_posterior(self, token_dim: int, act: str = "silu") -> None:
        layers: list[nn.Module] = []
        in_dim = self.deter + int(token_dim)
        for _ in range(self.obslayers):
            layers.extend([nn.Linear(in_dim, self.hidden), RMSNorm(self.hidden), _act(act)])
            in_dim = self.hidden
        layers.append(nn.Linear(in_dim, self.flat_stoch))
        self._posterior = nn.Sequential(*layers)

    def initial(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self.deter, device=device, dtype=dtype),
            "stoch": torch.zeros(batch_size, self.stoch, self.classes, device=device, dtype=dtype),
        }

    def _core(self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        action = action / torch.maximum(torch.ones_like(action), action.abs()).detach()
        x0 = self.dynin0(deter)
        x1 = self.dynin1(stoch_flat)
        x2 = self.dynin2(action)
        x = torch.cat([x0, x1, x2], dim=-1)
        x = x[:, None, :].expand(-1, self.blocks, -1)
        deter_group = deter.reshape(deter.shape[0], self.blocks, self.deter // self.blocks)
        x = torch.cat([deter_group, x], dim=-1).reshape(deter.shape[0], -1)
        x = self.dynhid(x)
        gates = self.dyngru(x).reshape(deter.shape[0], self.blocks, 3 * (self.deter // self.blocks))
        reset, cand, update = [g.reshape(deter.shape[0], self.deter) for g in gates.chunk(3, dim=-1)]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter

    def _prior(self, deter: torch.Tensor) -> torch.Tensor:
        return self.prior_net(deter).reshape(deter.shape[0], self.stoch, self.classes)

    def _posterior_logits(self, deter: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        if self._posterior is None:
            self.build_posterior(token.shape[-1])
            self._posterior.to(device=token.device, dtype=token.dtype)
        x = torch.cat([deter, token], dim=-1)
        return self._posterior(x).reshape(deter.shape[0], self.stoch, self.classes)

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=-1)
        if self.unimix > 0:
            uniform = torch.full_like(probs, 1.0 / self.classes)
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        return probs.to(dtype=logits.dtype)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self._probs(logits)
        sample_probs = probs.float()
        sample_probs = sample_probs / sample_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        flat = sample_probs.reshape(-1, self.classes)
        idx = torch.distributions.Categorical(probs=flat).sample()
        hard = F.one_hot(idx, self.classes).to(dtype=probs.dtype).reshape_as(probs)
        return hard + probs - probs.detach()

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self._probs(logits).float().clamp_min(1e-8)
        return -(probs * probs.log()).sum(dim=-1).sum(dim=-1)

    def _kl(self, lhs_logits: torch.Tensor, rhs_logits: torch.Tensor) -> torch.Tensor:
        lhs = self._probs(lhs_logits).float().clamp_min(1e-8)
        rhs = self._probs(rhs_logits).float().clamp_min(1e-8)
        return (lhs * (lhs.log() - rhs.log())).sum(dim=-1).sum(dim=-1)

    def observe(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, _ = tokens.shape
        carry = self.initial(b, tokens.device, tokens.dtype)
        deter_seq: list[torch.Tensor] = []
        stoch_seq: list[torch.Tensor] = []
        prior_logits_seq: list[torch.Tensor] = []
        post_logits_seq: list[torch.Tensor] = []
        for step in range(t):
            reset = is_first[:, step].to(device=tokens.device).view(b, 1)
            keep = (~reset.bool()).to(dtype=tokens.dtype)
            deter = carry["deter"] * keep
            stoch = carry["stoch"] * keep.view(b, 1, 1)
            action = actions[:, step].to(device=tokens.device, dtype=tokens.dtype) * keep
            deter = self._core(deter, stoch, action)
            prior_logits = self._prior(deter)
            post_logits = self._posterior_logits(deter, tokens[:, step])
            post_stoch = self._sample(post_logits)
            carry = {"deter": deter, "stoch": post_stoch}
            deter_seq.append(deter)
            stoch_seq.append(post_stoch)
            prior_logits_seq.append(prior_logits)
            post_logits_seq.append(post_logits)
        return {
            "deter": torch.stack(deter_seq, dim=1),
            "stoch": torch.stack(stoch_seq, dim=1),
            "prior_logits": torch.stack(prior_logits_seq, dim=1),
            "post_logits": torch.stack(post_logits_seq, dim=1),
        }

    def kl_loss(self, post_logits: torch.Tensor, prior_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        dyn = self._kl(post_logits.detach(), prior_logits)
        rep = self._kl(post_logits, prior_logits.detach())
        if self.free_nats > 0:
            dyn = torch.maximum(dyn, dyn.new_full((), self.free_nats))
            rep = torch.maximum(rep, rep.new_full((), self.free_nats))
        return {
            "dyn": dyn.mean(),
            "rep": rep.mean(),
            "dyn_entropy": self._entropy(prior_logits).mean(),
            "rep_entropy": self._entropy(post_logits).mean(),
        }


class DreamerV3PixelDecoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 6,
        image_size: int = 64,
        deter: int = 8192,
        stoch: int = 32,
        classes: int = 64,
        units: int = 1024,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        bspace: int = 8,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.depths = tuple(int(depth) * int(m) for m in mults)
        self.minres = self.image_size // (2 ** len(self.depths))
        if not (3 <= self.minres <= 16):
            raise ValueError(f"DreamerV3 decoder minres should be 3..16, got {self.minres}")
        self.shape = (self.depths[-1], self.minres, self.minres)
        self.flat_shape = int(math.prod(self.shape))
        self.sp0 = BlockLinear(int(deter), self.flat_shape, int(bspace))
        self.sp1 = nn.Sequential(nn.Linear(int(stoch) * int(classes), 2 * int(units)), RMSNorm(2 * int(units)), _act(act))
        self.sp2 = nn.Linear(2 * int(units), self.flat_shape)
        self.spnorm = ChannelRMSNorm(self.depths[-1])
        pad = int(kernel) // 2
        convs: list[nn.Module] = []
        prev = self.depths[-1]
        for out_ch in reversed(self.depths[:-1]):
            convs.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(prev, out_ch, kernel_size=kernel, padding=pad),
                ChannelRMSNorm(out_ch),
                _act(act),
            ])
            prev = out_ch
        convs.extend([
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(prev, self.image_channels, kernel_size=kernel, padding=pad),
        ])
        self.net = nn.Sequential(*convs)

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        b, t = deter.shape[:2]
        deter_flat = deter.reshape(b * t, -1)
        stoch_flat = stoch.reshape(b * t, -1)
        x0 = self.sp0(deter_flat).reshape(b * t, *self.shape)
        x1 = self.sp2(self.sp1(stoch_flat)).reshape(b * t, *self.shape)
        x = F.silu(self.spnorm(x0 + x1))
        x = torch.sigmoid(self.net(x))
        return x.reshape(b, t, self.image_channels, self.image_size, self.image_size)


class DreamerV3TokenDecoder(nn.Module):
    """Categorical spatial image-token decoder from DreamerV3 feature (h, z)."""

    def __init__(
        self,
        num_image_tokens_vocab: int = 8192,
        n_image_tokens: int = 512,
        num_views: int = 2,
        spatial_grid: tuple[int, int] = (16, 16),
        deter: int = 8192,
        stoch: int = 32,
        classes: int = 64,
        units: int = 1024,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3),
        kernel: int = 5,
        bspace: int = 8,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.num_image_tokens_vocab = int(num_image_tokens_vocab)
        self.n_image_tokens = int(n_image_tokens)
        self.num_views = int(num_views)
        self.spatial_grid = (int(spatial_grid[0]), int(spatial_grid[1]))
        self.tokens_per_view = self.spatial_grid[0] * self.spatial_grid[1]
        if self.num_views <= 0:
            raise ValueError(f"num_views must be positive, got {self.num_views}")
        if self.tokens_per_view * self.num_views != self.n_image_tokens:
            raise ValueError(
                f"num_views={self.num_views} and spatial_grid={self.spatial_grid} "
                f"produce {self.tokens_per_view * self.num_views} tokens, "
                f"expected n_image_tokens={self.n_image_tokens}"
            )
        self.depths = tuple(int(depth) * int(m) for m in mults)
        factor = 2 ** len(self.depths)
        if self.spatial_grid[0] % factor or self.spatial_grid[1] % factor:
            raise ValueError(
                f"spatial_grid={self.spatial_grid} must be divisible by {factor} "
                f"for decoder mults={mults}"
            )
        self.minres = (self.spatial_grid[0] // factor, self.spatial_grid[1] // factor)
        if not (3 <= self.minres[0] <= 16 and 3 <= self.minres[1] <= 16):
            raise ValueError(
                "DreamerV3 token decoder minres should be 3..16, "
                f"got {self.minres} from spatial_grid={self.spatial_grid} and mults={mults}"
            )
        self.shape = (self.depths[-1], self.minres[0], self.minres[1])
        self.flat_shape = int(math.prod(self.shape))
        self.sp0 = BlockLinear(int(deter), self.flat_shape, int(bspace))
        self.sp1 = nn.Sequential(
            nn.Linear(int(stoch) * int(classes), 2 * int(units)),
            RMSNorm(2 * int(units)),
            _act(act),
        )
        self.sp2 = nn.Linear(2 * int(units), self.flat_shape)
        self.spnorm = ChannelRMSNorm(self.depths[-1])

        pad = int(kernel) // 2
        convs: list[nn.Module] = []
        prev = self.depths[-1]
        for out_ch in reversed(self.depths[:-1]):
            convs.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(prev, out_ch, kernel_size=kernel, padding=pad),
                ChannelRMSNorm(out_ch),
                _act(act),
            ])
            prev = out_ch
        convs.extend([
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(prev, self.num_views * self.num_image_tokens_vocab, kernel_size=kernel, padding=pad),
        ])
        self.net = nn.Sequential(*convs)

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        b, t = deter.shape[:2]
        deter_flat = deter.reshape(b * t, -1)
        stoch_flat = stoch.reshape(b * t, -1)
        x0 = self.sp0(deter_flat).reshape(b * t, *self.shape)
        x1 = self.sp2(self.sp1(stoch_flat)).reshape(b * t, *self.shape)
        x = F.silu(self.spnorm(x0 + x1))
        logits = self.net(x)
        if logits.shape[-2:] != self.spatial_grid:
            raise RuntimeError(
                f"Token decoder produced grid {tuple(logits.shape[-2:])}, "
                f"expected {self.spatial_grid}"
            )
        logits = logits.reshape(
            b, t, self.num_views, self.num_image_tokens_vocab, self.tokens_per_view,
        )
        return logits.permute(0, 1, 2, 4, 3).contiguous()  # [B,T,V,N,C]


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 1,
        units: int = 1024,
        act: str = "silu",
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        cur = int(in_dim)
        for _ in range(int(layers)):
            mods.extend([nn.Linear(cur, int(units)), RMSNorm(int(units)), _act(act)])
            cur = int(units)
        final = nn.Linear(cur, int(out_dim))
        if float(outscale) != 1.0:
            with torch.no_grad():
                final.weight.mul_(float(outscale))
                if final.bias is not None:
                    final.bias.mul_(float(outscale))
        mods.append(final)
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class SymexpTwoHotHead(nn.Module):
    """DreamerV3 ``symexp_twohot`` scalar head."""

    def __init__(
        self,
        in_dim: int,
        bins: int = 255,
        layers: int = 1,
        units: int = 1024,
        act: str = "silu",
        outscale: float = 0.0,
    ) -> None:
        super().__init__()
        self.bins_count = int(bins)
        self.net = MLPHead(
            in_dim,
            self.bins_count,
            layers=layers,
            units=units,
            act=act,
            outscale=outscale,
        )

    def _bins(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.bins_count % 2 == 1:
            half = torch.linspace(-20.0, 0.0, (self.bins_count - 1) // 2 + 1, device=device, dtype=dtype)
            half = symexp(half)
            return torch.cat([half, -half[:-1].flip(0)], dim=0)
        half = torch.linspace(-20.0, 0.0, self.bins_count // 2, device=device, dtype=dtype)
        half = symexp(half)
        return torch.cat([half, -half.flip(0)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(device=logits.device, dtype=logits.dtype).detach()
        bins = self._bins(logits.device, logits.dtype)
        below = (bins.view(*((1,) * target.ndim), -1) <= target.unsqueeze(-1)).sum(dim=-1) - 1
        above = self.bins_count - (
            bins.view(*((1,) * target.ndim), -1) > target.unsqueeze(-1)
        ).sum(dim=-1)
        below = below.clamp(0, self.bins_count - 1).long()
        above = above.clamp(0, self.bins_count - 1).long()
        equal = below == above
        below_bins = bins[below]
        above_bins = bins[above]
        dist_to_below = torch.where(equal, torch.ones_like(target), (below_bins - target).abs())
        dist_to_above = torch.where(equal, torch.ones_like(target), (above_bins - target).abs())
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total.clamp_min(1e-8)
        weight_above = dist_to_below / total.clamp_min(1e-8)
        target_dist = (
            F.one_hot(below, self.bins_count).to(dtype=logits.dtype) * weight_below.unsqueeze(-1)
            + F.one_hot(above, self.bins_count).to(dtype=logits.dtype) * weight_above.unsqueeze(-1)
        )
        return -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        bins = self._bins(logits.device, logits.dtype)
        n = logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = probs[..., :m]
            p2 = probs[..., m : m + 1]
            p3 = probs[..., m + 1 :]
            b1 = bins[..., :m]
            b2 = bins[..., m : m + 1]
            b3 = bins[..., m + 1 :]
            return (p2 * b2).sum(dim=-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(dim=-1)
        p1 = probs[..., : n // 2]
        p2 = probs[..., n // 2 :]
        b1 = bins[..., : n // 2]
        b2 = bins[..., n // 2 :]
        return ((p1 * b1).flip(-1) + (p2 * b2)).sum(dim=-1)


class BinaryRewardHead(nn.Module):
    """Bernoulli reward head for sparse 0/1 environments such as LIBERO."""

    def __init__(
        self,
        in_dim: int,
        layers: int = 1,
        units: int = 1024,
        act: str = "silu",
        init_logit: float = -5.0,
    ) -> None:
        super().__init__()
        self.net = MLPHead(in_dim, 1, layers=layers, units=units, act=act)
        final = self.net.net[-1]
        if isinstance(final, nn.Linear) and final.bias is not None:
            nn.init.constant_(final.bias, float(init_logit))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1).float()
        target = target.to(device=logits.device, dtype=torch.float32).detach().clamp(0.0, 1.0)
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits.squeeze(-1).float()).to(dtype=logits.dtype)


def _make_reward_head(
    feat_dim: int,
    reward_bins: int,
    hidden: int,
    act: str,
    reward_head_type: str = "twohot",
    reward_init_logit: float = -5.0,
) -> nn.Module:
    reward_head_type = str(reward_head_type).lower()
    if reward_head_type in {"binary", "bernoulli", "sigmoid"}:
        return BinaryRewardHead(
            feat_dim,
            layers=1,
            units=hidden,
            act=act,
            init_logit=reward_init_logit,
        )
    if reward_head_type not in {"twohot", "symexp_twohot", "regression", "mse"}:
        raise ValueError(f"Unsupported reward_head_type: {reward_head_type}")
    if int(reward_bins) <= 1:
        return MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)
    return SymexpTwoHotHead(feat_dim, bins=reward_bins, layers=1, units=hidden, act=act)


def _reward_loss(head: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=pred.device, dtype=pred.dtype)
    if hasattr(head, "loss"):
        return head.loss(pred, target).mean()
    return F.mse_loss(pred.squeeze(-1), target, reduction="none").mean()


def _reward_pred(head: nn.Module, pred: torch.Tensor) -> torch.Tensor:
    if hasattr(head, "pred"):
        return head.pred(pred)
    return pred.squeeze(-1)


@dataclass
class DreamerV3Loss:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class DreamerV3LatentState:
    """Single-step DreamerV3 RSSM state used by DreamerVLA actor cotrain."""

    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor | None = None

    def feature(self) -> torch.Tensor:
        return torch.cat([self.deter, self.stoch.reshape(*self.stoch.shape[:-2], -1)], dim=-1)


class DreamerV3ActorAdapterMixin:
    """Adds DreamerVLA's single-step actor interface to DreamerV3 WMs.

    The standalone DreamerV3 trainers call ``model(batch)`` and use the
    sequence loss.  DreamerVLA cotrain additionally needs:

      model({'mode': 'encode_latent', 'hidden': obs})
      model({'mode': 'predict_next', 'latent': state, 'actions': action})
      model({'mode': 'reward', ...})

    This mixin keeps both routes available without changing the public config
    surface of the standalone pixel/token baselines.
    """

    def _single_observation_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, DreamerV3TokenEncoder):
            if hidden.ndim in {2, 3}:
                return hidden[:, None]
            if hidden.ndim == 4 and hidden.shape[1] == 1:
                return hidden
            raise ValueError(f"Unsupported DreamerV3 token observation shape: {tuple(hidden.shape)}")

        # Pixel obs: [B,C,H,W] -> [B,1,C,H,W].
        if hidden.ndim == 4:
            return hidden[:, None]
        if hidden.ndim == 5 and hidden.shape[1] == 1:
            return hidden
        raise ValueError(f"Unsupported DreamerV3 single observation shape: {tuple(hidden.shape)}")

    def _feature_dim(self) -> int:
        return int(self.rssm.deter + self.rssm.stoch * self.rssm.classes)

    def _latent_from_feature(self, feature: torch.Tensor) -> DreamerV3LatentState:
        dtype = _module_dtype(self, feature.dtype)
        device = _module_device(self, feature.device)
        feature = feature.to(device=device, dtype=dtype)
        deter = feature[:, : self.rssm.deter]
        stoch_flat = feature[:, self.rssm.deter :]
        stoch = stoch_flat.reshape(feature.shape[0], self.rssm.stoch, self.rssm.classes)
        return DreamerV3LatentState(deter=deter, stoch=stoch)

    def encode_latent(self, hidden: torch.Tensor) -> DreamerV3LatentState:
        if hidden.ndim == 2 and hidden.shape[-1] == self._feature_dim() and torch.is_floating_point(hidden):
            return self._latent_from_feature(hidden)

        device = _module_device(self, hidden.device)
        obs = self._single_observation_sequence(hidden.to(device=device))
        enc = self.encoder(obs)
        batch_size = enc.shape[0]
        dtype = enc.dtype
        actions = torch.zeros(batch_size, 1, self.rssm.action_dim, device=device, dtype=dtype)
        is_first = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        seq = self.rssm.observe(enc, actions, is_first)
        return DreamerV3LatentState(
            deter=seq["deter"][:, 0],
            stoch=seq["stoch"][:, 0],
            logits=seq["post_logits"][:, 0],
        )

    def predict_next(self, latent: DreamerV3LatentState, actions: torch.Tensor) -> DreamerV3LatentState:
        device = _module_device(self, actions.device)
        dtype = latent.deter.dtype
        action = actions.to(device=device, dtype=dtype)
        if action.ndim == 3:
            action = action.mean(dim=1)
        deter = self.rssm._core(
            latent.deter.to(device=device, dtype=dtype),
            latent.stoch.to(device=device, dtype=dtype),
            action,
        )
        logits = self.rssm._prior(deter)
        stoch = self.rssm._sample(logits)
        return DreamerV3LatentState(deter=deter, stoch=stoch, logits=logits)

    def actor_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return latent.feature()

    def critic_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return latent.feature()

    def observe_sequence(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "images" in batch:
            obs = batch["images"]
        elif "tokens" in batch:
            obs = batch["tokens"]
        else:
            raise KeyError("DreamerV3 observe_sequence expects `images` or `tokens`.")
        device = _module_device(self, obs.device)
        enc = self.encoder(obs.to(device=device))
        actions = batch["actions"].to(device=device, dtype=enc.dtype)
        is_first = batch["is_first"].to(device=device)
        seq = self.rssm.observe(enc, actions, is_first)
        latent = DreamerV3LatentState(
            deter=seq["deter"],
            stoch=seq["stoch"],
            logits=seq["post_logits"],
        )
        return {"latent": latent, "feat": latent.feature()}

    def state_reward(self, latent: DreamerV3LatentState) -> torch.Tensor:
        pred = self.reward_head(latent.feature())
        return _reward_pred(self.reward_head, pred).squeeze(-1)

    def continue_prob(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(latent.feature()).squeeze(-1))

    def reward(
        self,
        latent: DreamerV3LatentState,
        actions: torch.Tensor,
        next_latent: DreamerV3LatentState,
    ) -> torch.Tensor:
        del latent, actions
        return self.state_reward(next_latent)

    def _forward_actor_adapter(self, batch: dict[str, Any]) -> Any:
        mode = batch.get("mode")
        if mode == "encode_latent":
            return self.encode_latent(batch["hidden"])
        if mode == "predict_next":
            return self.predict_next(batch["latent"], batch["actions"])
        if mode == "reward":
            if "next_latent" in batch:
                return self.reward(batch["latent"], batch.get("actions"), batch["next_latent"])
            return self.state_reward(batch["latent"])
        if mode == "continue":
            return self.continue_prob(batch["latent"])
        if mode == "actor_input":
            return self.actor_input(batch["latent"])
        if mode == "critic_input":
            return self.critic_input(batch["latent"])
        if mode == "observe_sequence":
            return self.observe_sequence(batch)
        raise ValueError(f"Unknown DreamerV3 actor-adapter mode: {mode!r}")

    @staticmethod
    def _compat_forward_dict(out: DreamerV3Loss) -> dict[str, torch.Tensor]:
        result = {"_loss": out.loss, **out.metrics}
        result["loss"] = out.loss
        zero = out.loss.new_zeros(())
        if "dyn_loss" in result:
            result.setdefault("dyn_kl", result["dyn_loss"])
        if "rep_loss" in result:
            result.setdefault("rep_kl", result["rep_loss"])
        if "rec_loss" in result:
            result.setdefault("image_decoder_loss", result["rec_loss"])
        if "image_mse" in result:
            result.setdefault("image_recon_mse_loss", result["image_mse"])
        if "token_ce" in result:
            result.setdefault("image_recon_ce_loss", result["token_ce"])
        if "token_acc" in result:
            result.setdefault("image_recon_accuracy", result["token_acc"])
        if "reward_pred_mean" in result:
            result.setdefault("predicted_reward_mean", result["reward_pred_mean"])
        result.setdefault("transition_loss", zero)
        result.setdefault("kl_loss", result.get("dyn_loss", zero) + result.get("rep_loss", zero))
        result.setdefault("delta_latent_loss", zero)
        result.setdefault("action_margin_loss", zero)
        return result


class DreamerV3PixelWorldModel(DreamerV3ActorAdapterMixin, nn.Module):
    """PyTorch DreamerV3 world model for pixel observations.

    This intentionally mirrors the public DreamerV3 architecture: CNN encoder,
    discrete RSSM with block-GRU deterministic state, decoder from (h, z),
    reward and continue heads, and split dyn/rep KL losses.
    """

    def __init__(
        self,
        action_dim: int = 7,
        image_channels: int = 6,
        image_size: int = 64,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        act: str = "silu",
        unimix: float = 0.01,
        free_nats: float = 1.0,
        reward_bins: int = 255,
        reward_head_type: str = "twohot",
        reward_init_logit: float = -5.0,
        contdisc: bool = True,
        horizon: int = 333,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = DreamerV3PixelEncoder(image_channels, image_size, depth, mults, kernel, act)
        self.rssm = DreamerV3RSSM(
            action_dim=action_dim,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=blocks,
            unimix=unimix,
            free_nats=free_nats,
            act=act,
        )
        self.rssm.build_posterior(self.encoder.out_dim, act=act)
        self.decoder = DreamerV3PixelDecoder(
            image_channels=image_channels,
            image_size=image_size,
            deter=deter,
            stoch=stoch,
            classes=classes,
            depth=depth,
            mults=mults,
            kernel=kernel,
            act=act,
        )
        feat_dim = int(deter) + int(stoch) * int(classes)
        self.reward_head = _make_reward_head(
            feat_dim,
            reward_bins,
            hidden,
            act,
            reward_head_type=reward_head_type,
            reward_init_logit=reward_init_logit,
        )
        self.continue_head = MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)
        self.dyn_scale = float(dyn_scale)
        self.rep_scale = float(rep_scale)
        self.rec_scale = float(rec_scale)
        self.rew_scale = float(rew_scale)
        self.con_scale = float(con_scale)
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)], dim=-1)

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        images = batch["images"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=images.device, dtype=torch.float32)
        dones = batch["dones"].to(device=images.device, dtype=torch.float32)
        is_first = batch["is_first"].to(device=images.device)

        tokens = self.encoder(images)
        seq = self.rssm.observe(tokens, actions, is_first)
        kls = self.rssm.kl_loss(seq["post_logits"], seq["prior_logits"])
        recon = self.decoder(seq["deter"], seq["stoch"])
        target = images.to(device=recon.device, dtype=recon.dtype) / 255.0
        rec_per = (recon - target).square().sum(dim=(-3, -2, -1))
        rec_loss = rec_per.mean()

        feat = self.feature(seq)
        reward_logits = self.reward_head(feat)
        cont_logits = self.continue_head(feat).squeeze(-1)
        reward_loss = _reward_loss(self.reward_head, reward_logits, rewards)
        cont_target = 1.0 - dones.to(device=cont_logits.device, dtype=cont_logits.dtype)
        if self.contdisc:
            cont_target = cont_target * (1.0 - 1.0 / float(self.horizon))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        loss = (
            self.rec_scale * rec_loss
            + self.dyn_scale * kls["dyn"]
            + self.rep_scale * kls["rep"]
            + self.rew_scale * reward_loss
            + self.con_scale * cont_loss
        )
        mse = (recon.detach() - target).square().mean()
        metrics = {
            "loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "dyn_loss": kls["dyn"].detach(),
            "rep_loss": kls["rep"].detach(),
            "reward_loss": reward_loss.detach(),
            "continue_loss": cont_loss.detach(),
            "reward_pred_mean": _reward_pred(self.reward_head, reward_logits.detach()).mean().detach(),
            "image_mse": mse.detach(),
            "image_psnr": (-10.0 * torch.log10(mse.clamp_min(1e-8))).detach(),
            "dyn_entropy": kls["dyn_entropy"].detach(),
            "rep_entropy": kls["rep_entropy"].detach(),
        }
        return DreamerV3Loss(loss=loss, metrics=metrics)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)


class DreamerV3TokenWorldModel(DreamerV3ActorAdapterMixin, nn.Module):
    """DreamerV3 RSSM with categorical image-token observations.

    This is the controlled token counterpart of ``DreamerV3PixelWorldModel``:
    same RSSM, same aggregate free-nats KL semantics, same reward and continue
    heads. The observation likelihood is the only intended change, replacing
    pixel MSE with categorical CE over image tokens.
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_image_tokens_vocab: int = 8192,
        n_image_tokens: int = 512,
        num_views: int = 2,
        spatial_grid: tuple[int, int] = (16, 16),
        token_embed_dim: int = 512,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3),
        kernel: int = 5,
        act: str = "silu",
        unimix: float = 0.01,
        free_nats: float = 1.0,
        reward_bins: int = 255,
        reward_head_type: str = "twohot",
        reward_init_logit: float = -5.0,
        contdisc: bool = True,
        horizon: int = 333,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        rec_reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.encoder = DreamerV3TokenEncoder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            n_image_tokens=n_image_tokens,
            num_views=num_views,
            spatial_grid=tuple(spatial_grid),
            token_embed_dim=token_embed_dim,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )
        self.rssm = DreamerV3RSSM(
            action_dim=action_dim,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=blocks,
            unimix=unimix,
            free_nats=free_nats,
            act=act,
        )
        self.rssm.build_posterior(self.encoder.out_dim, act=act)
        self.decoder = DreamerV3TokenDecoder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            n_image_tokens=n_image_tokens,
            num_views=num_views,
            spatial_grid=tuple(spatial_grid),
            deter=deter,
            stoch=stoch,
            classes=classes,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )
        feat_dim = int(deter) + int(stoch) * int(classes)
        self.reward_head = _make_reward_head(
            feat_dim,
            reward_bins,
            hidden,
            act,
            reward_head_type=reward_head_type,
            reward_init_logit=reward_init_logit,
        )
        self.continue_head = MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)
        self.dyn_scale = float(dyn_scale)
        self.rep_scale = float(rep_scale)
        self.rec_scale = float(rec_scale)
        self.rew_scale = float(rew_scale)
        self.con_scale = float(con_scale)
        self.rec_reduction = str(rec_reduction).lower()
        if self.rec_reduction not in {"sum", "mean"}:
            raise ValueError("rec_reduction must be 'sum' or 'mean'")
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)],
            dim=-1,
        )

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        tokens = batch["tokens"].long()
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=actions.device, dtype=actions.dtype)
        terminal = batch.get("is_terminal", batch["dones"])
        dones = terminal.to(device=actions.device, dtype=actions.dtype)
        is_first = batch["is_first"].to(device=actions.device)

        enc = self.encoder(tokens)
        seq = self.rssm.observe(enc, actions, is_first)
        kls = self.rssm.kl_loss(seq["post_logits"], seq["prior_logits"])
        logits = self.decoder(seq["deter"], seq["stoch"])  # [B,T,views,tokens,classes]

        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1).to(device=logits.device),
            reduction="none",
        ).reshape(tokens.shape)
        rec_per_step = ce.reshape(ce.shape[0], ce.shape[1], -1).sum(dim=-1)
        rec_loss = rec_per_step.mean() if self.rec_reduction == "sum" else ce.mean()

        feat = self.feature(seq)
        reward_logits = self.reward_head(feat)
        cont_logits = self.continue_head(feat).squeeze(-1)
        reward_loss = _reward_loss(self.reward_head, reward_logits, rewards)
        cont_target = 1.0 - dones.to(device=cont_logits.device, dtype=cont_logits.dtype)
        if self.contdisc:
            cont_target = cont_target * (1.0 - 1.0 / float(self.horizon))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target, reduction="none").mean()

        loss = (
            self.rec_scale * rec_loss
            + self.dyn_scale * kls["dyn"]
            + self.rep_scale * kls["rep"]
            + self.rew_scale * reward_loss
            + self.con_scale * cont_loss
        )

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            reward_pred = _reward_pred(self.reward_head, reward_logits.detach())
            token_acc = (pred == tokens.to(device=pred.device)).float().mean()
            token_ce = ce.detach().float().mean()
            log_probs = F.log_softmax(logits.detach().float(), dim=-1)
            probs = log_probs.exp()
            pred_entropy = -(probs * log_probs).sum(dim=-1).mean()
            flat_pred = pred.reshape(pred.shape[0] * pred.shape[1], -1)
            flat_gt = tokens.to(device=pred.device).reshape(tokens.shape[0] * tokens.shape[1], -1)
            pred_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_pred],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()
            gt_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_gt],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()
        metrics = {
            "loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "token_ce": token_ce,
            "token_acc": token_acc.detach(),
            "dyn_loss": kls["dyn"].detach(),
            "rep_loss": kls["rep"].detach(),
            "reward_loss": reward_loss.detach(),
            "reward_pred_mean": reward_pred.mean().detach(),
            "continue_loss": cont_loss.detach(),
            "pred_entropy": pred_entropy.detach(),
            "pred_unique_tokens": pred_unique.detach(),
            "gt_unique_tokens": gt_unique.detach(),
            "dyn_entropy": kls["dyn_entropy"].detach(),
            "rep_entropy": kls["rep_entropy"].detach(),
        }
        return DreamerV3Loss(loss=loss, metrics=metrics)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)


class DreamerV3TokenFromPixelWorldModel(DreamerV3ActorAdapterMixin, nn.Module):
    """Pixel-world-model copy with only the observation distribution changed.

    This is the controlled ablation requested for token observations: keep the
    same RSSM, reward head, continue head, KL scales, and loss aggregation as
    ``DreamerV3PixelWorldModel``. The only replacement is:

      pixel obs + MSE decoder loss -> spatial token obs + categorical CE loss.
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_image_tokens_vocab: int = 8192,
        n_image_tokens: int = 512,
        num_views: int = 2,
        spatial_grid: tuple[int, int] = (16, 16),
        token_embed_dim: int = 512,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3),
        kernel: int = 5,
        act: str = "silu",
        unimix: float = 0.01,
        free_nats: float = 1.0,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        rec_reduction: str = "sum",
        reward_bins: int = 255,
        reward_head_type: str = "twohot",
        reward_init_logit: float = -5.0,
        contdisc: bool = True,
        horizon: int = 333,
    ) -> None:
        super().__init__()
        self.encoder = DreamerV3TokenEncoder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            n_image_tokens=n_image_tokens,
            num_views=num_views,
            spatial_grid=tuple(spatial_grid),
            token_embed_dim=token_embed_dim,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )
        self.rssm = DreamerV3RSSM(
            action_dim=action_dim,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=blocks,
            unimix=unimix,
            free_nats=free_nats,
            act=act,
        )
        self.rssm.build_posterior(self.encoder.out_dim, act=act)
        self.decoder = DreamerV3TokenDecoder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            n_image_tokens=n_image_tokens,
            num_views=num_views,
            spatial_grid=tuple(spatial_grid),
            deter=deter,
            stoch=stoch,
            classes=classes,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )
        feat_dim = int(deter) + int(stoch) * int(classes)
        self.reward_head = _make_reward_head(
            feat_dim,
            reward_bins,
            hidden,
            act,
            reward_head_type=reward_head_type,
            reward_init_logit=reward_init_logit,
        )
        self.continue_head = MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)
        self.dyn_scale = float(dyn_scale)
        self.rep_scale = float(rep_scale)
        self.rec_scale = float(rec_scale)
        self.rew_scale = float(rew_scale)
        self.con_scale = float(con_scale)
        self.rec_reduction = str(rec_reduction).lower()
        if self.rec_reduction not in {"sum", "mean"}:
            raise ValueError("rec_reduction must be 'sum' or 'mean'")
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)],
            dim=-1,
        )

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        tokens = batch["tokens"].long()
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=actions.device, dtype=actions.dtype)
        dones = batch["dones"].to(device=actions.device, dtype=actions.dtype)
        is_first = batch["is_first"].to(device=actions.device)

        enc = self.encoder(tokens)
        seq = self.rssm.observe(enc, actions, is_first)
        kls = self.rssm.kl_loss(seq["post_logits"], seq["prior_logits"])
        logits = self.decoder(seq["deter"], seq["stoch"])  # [B,T,views,tokens,classes]

        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1).to(device=logits.device),
            reduction="none",
        ).reshape(tokens.shape)
        rec_per = ce.reshape(ce.shape[0], ce.shape[1], -1).sum(dim=-1)
        rec_loss = rec_per.mean() if self.rec_reduction == "sum" else ce.mean()

        feat = self.feature(seq)
        reward_logits = self.reward_head(feat)
        cont_logits = self.continue_head(feat).squeeze(-1)
        reward_loss = _reward_loss(self.reward_head, reward_logits, rewards)
        cont_target = 1.0 - dones.to(device=cont_logits.device, dtype=cont_logits.dtype)
        if self.contdisc:
            cont_target = cont_target * (1.0 - 1.0 / float(self.horizon))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        loss = (
            self.rec_scale * rec_loss
            + self.dyn_scale * kls["dyn"]
            + self.rep_scale * kls["rep"]
            + self.rew_scale * reward_loss
            + self.con_scale * cont_loss
        )

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            token_acc = (pred == tokens.to(device=pred.device)).float().mean()
            token_ce = ce.detach().float().mean()
            log_probs = F.log_softmax(logits.detach().float(), dim=-1)
            probs = log_probs.exp()
            pred_entropy = -(probs * log_probs).sum(dim=-1).mean()
            flat_pred = pred.reshape(pred.shape[0] * pred.shape[1], -1)
            flat_gt = tokens.to(device=pred.device).reshape(tokens.shape[0] * tokens.shape[1], -1)
            pred_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_pred],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()
            gt_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_gt],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()

        metrics = {
            "loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "token_ce": token_ce,
            "token_acc": token_acc.detach(),
            "dyn_loss": kls["dyn"].detach(),
            "rep_loss": kls["rep"].detach(),
            "reward_loss": reward_loss.detach(),
            "reward_pred_mean": _reward_pred(self.reward_head, reward_logits.detach()).mean().detach(),
            "continue_loss": cont_loss.detach(),
            "pred_entropy": pred_entropy.detach(),
            "pred_unique_tokens": pred_unique.detach(),
            "gt_unique_tokens": gt_unique.detach(),
            "dyn_entropy": kls["dyn_entropy"].detach(),
            "rep_entropy": kls["rep_entropy"].detach(),
        }
        return DreamerV3Loss(loss=loss, metrics=metrics)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)


class _RynnBackboneObsEncoder(nn.Module):
    """Identity/projection shim for frozen RynnVLA backbone outputs."""

    def __init__(
        self,
        obs_dim: int = 4096,
        embed_dim: int | None = None,
        hidden: int = 2048,
        layers: int = 2,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.out_dim = int(embed_dim if embed_dim is not None else obs_dim)
        self.register_buffer("_dtype_anchor", torch.empty(0), persistent=False)
        if self.out_dim == self.obs_dim:
            self.net = nn.Identity()
        else:
            modules: list[nn.Module] = [nn.LayerNorm(self.obs_dim)]
            dim = self.obs_dim
            for _ in range(max(int(layers) - 1, 0)):
                modules.extend([nn.Linear(dim, int(hidden)), _act(act)])
                dim = int(hidden)
            modules.append(nn.Linear(dim, self.out_dim))
            self.net = nn.Sequential(*modules)

    def forward(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        if obs_embedding.ndim == 2:
            obs_embedding = obs_embedding[:, None]
        if obs_embedding.ndim != 3:
            raise ValueError(
                f"Rynn backbone obs_embedding must be [B,T,D] or [B,D], got {tuple(obs_embedding.shape)}"
            )
        if obs_embedding.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Rynn backbone obs dim mismatch: got {obs_embedding.shape[-1]}, expected {self.obs_dim}"
            )
        dtype = _module_dtype(self, obs_embedding.dtype)
        return self.net(obs_embedding.to(dtype=dtype))


class DreamerV3PixelRynnBackboneWorldModel(DreamerV3ActorAdapterMixin, nn.Module):
    """Pixel DreamerV3 with the observation encoder replaced by frozen RynnVLA.

    The workspace supplies:

      images: raw pixel observations, used as the reconstruction target.
      obs_embedding: frozen RynnVLA-002 hidden vectors, used as RSSM observations.

    Thus only the ``encode`` slot changes.  RSSM, pixel decoder, reward head and
    continue head stay aligned with ``DreamerV3PixelWorldModel``.
    """

    def __init__(
        self,
        obs_dim: int = 4096,
        latent_dim: int | None = None,
        action_dim: int = 7,
        image_channels: int = 6,
        image_size: int = 64,
        embed_dim: int | None = None,
        encoder_hidden: int = 2048,
        encoder_layers: int = 2,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        act: str = "silu",
        unimix: float = 0.01,
        free_nats: float = 1.0,
        reward_bins: int = 255,
        reward_head_type: str = "twohot",
        reward_init_logit: float = -5.0,
        contdisc: bool = True,
        horizon: int = 333,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        hidden_rec_scale: float = 100.0,
        hidden_decoder_layers: int = 1,
        hidden_decoder_units: int = 2048,
    ) -> None:
        super().__init__()
        if latent_dim is not None:
            obs_dim = int(obs_dim if obs_dim is not None else latent_dim)
        self.obs_dim = int(obs_dim)
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.encoder = _RynnBackboneObsEncoder(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            hidden=encoder_hidden,
            layers=encoder_layers,
            act=act,
        )
        self.rssm = DreamerV3RSSM(
            action_dim=action_dim,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=blocks,
            unimix=unimix,
            free_nats=free_nats,
            act=act,
        )
        self.rssm.build_posterior(self.encoder.out_dim, act=act)
        self.decoder = DreamerV3PixelDecoder(
            image_channels=image_channels,
            image_size=image_size,
            deter=deter,
            stoch=stoch,
            classes=classes,
            depth=depth,
            mults=tuple(mults),
            kernel=kernel,
            act=act,
        )
        feat_dim = int(deter) + int(stoch) * int(classes)
        self.hidden_decoder = MLPHead(
            feat_dim,
            self.obs_dim,
            layers=int(hidden_decoder_layers),
            units=int(hidden_decoder_units),
            act=act,
        )
        self.reward_head = _make_reward_head(
            feat_dim,
            reward_bins,
            hidden,
            act,
            reward_head_type=reward_head_type,
            reward_init_logit=reward_init_logit,
        )
        self.continue_head = MLPHead(feat_dim, 1, layers=1, units=hidden, act=act)
        self.dyn_scale = float(dyn_scale)
        self.rep_scale = float(rep_scale)
        self.rec_scale = float(rec_scale)
        self.rew_scale = float(rew_scale)
        self.con_scale = float(con_scale)
        self.hidden_rec_scale = float(hidden_rec_scale)
        self.contdisc = bool(contdisc)
        self.horizon = int(horizon)

    def _feature_dim(self) -> int:
        return int(self.rssm.deter + self.rssm.stoch * self.rssm.classes)

    def feature(self, seq: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [seq["deter"], seq["stoch"].reshape(*seq["stoch"].shape[:2], -1)],
            dim=-1,
        )

    def _resize_target(self, images: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"images must be [B,T,C,H,W], got {tuple(images.shape)}")
        bsz, steps, channels, height, width = images.shape
        if channels != self.image_channels:
            raise ValueError(f"Expected {self.image_channels} image channels, got {channels}")
        target = images.to(device=device, dtype=dtype) / 255.0
        if (height, width) == (self.image_size, self.image_size):
            return target
        flat = target.reshape(bsz * steps, channels, height, width)
        flat = F.interpolate(flat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return flat.reshape(bsz, steps, channels, self.image_size, self.image_size)

    def encode_latent(self, hidden: torch.Tensor) -> DreamerV3LatentState:
        if hidden.ndim == 2 and hidden.shape[-1] == self._feature_dim() and torch.is_floating_point(hidden):
            return self._latent_from_feature(hidden)
        device = _module_device(self, hidden.device)
        obs = self.encoder(hidden.to(device=device))
        batch_size = obs.shape[0]
        dtype = obs.dtype
        actions = torch.zeros(batch_size, 1, self.rssm.action_dim, device=device, dtype=dtype)
        is_first = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        seq = self.rssm.observe(obs, actions, is_first)
        return DreamerV3LatentState(
            deter=seq["deter"][:, 0],
            stoch=seq["stoch"][:, 0],
            logits=seq["post_logits"][:, 0],
        )

    def observe_sequence(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        obs_embedding = batch["obs_embedding"]
        device = _module_device(self, obs_embedding.device)
        enc = self.encoder(obs_embedding.to(device=device))
        actions = batch["actions"].to(device=device, dtype=enc.dtype)
        is_first = batch["is_first"].to(device=device)
        seq = self.rssm.observe(enc, actions, is_first)
        latent = DreamerV3LatentState(
            deter=seq["deter"],
            stoch=seq["stoch"],
            logits=seq["post_logits"],
        )
        return {"latent": latent, "feat": latent.feature()}

    def actor_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return self.hidden_decoder(latent.feature())

    def critic_input(self, latent: DreamerV3LatentState) -> torch.Tensor:
        return latent.feature()

    def loss(self, batch: dict[str, torch.Tensor]) -> DreamerV3Loss:
        images = batch["images"]
        obs_embedding = batch["obs_embedding"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(device=actions.device, dtype=actions.dtype)
        terminal = batch.get("is_terminal", batch["dones"])
        dones = terminal.to(device=actions.device, dtype=actions.dtype)
        is_first = batch["is_first"].to(device=actions.device)

        enc = self.encoder(obs_embedding)
        seq = self.rssm.observe(enc, actions, is_first)
        kls = self.rssm.kl_loss(seq["post_logits"], seq["prior_logits"])
        recon = self.decoder(seq["deter"], seq["stoch"])
        target = self._resize_target(images, dtype=recon.dtype, device=recon.device)
        rec_per = (recon - target).square().sum(dim=(-3, -2, -1))
        rec_loss = rec_per.mean()

        feat = self.feature(seq)
        hidden_pred = self.hidden_decoder(feat)
        hidden_target = obs_embedding.to(device=hidden_pred.device, dtype=hidden_pred.dtype).detach()
        hidden_mse = (hidden_pred.float() - hidden_target.float()).square().mean()
        hidden_pred_norm = F.normalize(hidden_pred.float(), dim=-1)
        hidden_target_norm = F.normalize(hidden_target.float(), dim=-1)
        hidden_cosine = 1.0 - (hidden_pred_norm * hidden_target_norm).sum(dim=-1).mean()
        reward_logits = self.reward_head(feat)
        cont_logits = self.continue_head(feat).squeeze(-1)
        reward_loss = _reward_loss(self.reward_head, reward_logits, rewards)
        cont_target = 1.0 - dones.to(device=cont_logits.device, dtype=cont_logits.dtype)
        if self.contdisc:
            cont_target = cont_target * (1.0 - 1.0 / float(self.horizon))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        loss = (
            self.rec_scale * rec_loss
            + self.dyn_scale * kls["dyn"]
            + self.rep_scale * kls["rep"]
            + self.hidden_rec_scale * hidden_mse
            + self.rew_scale * reward_loss
            + self.con_scale * cont_loss
        )
        mse = (recon.detach() - target).square().mean()
        metrics = {
            "loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "dyn_loss": kls["dyn"].detach(),
            "rep_loss": kls["rep"].detach(),
            "reward_loss": reward_loss.detach(),
            "continue_loss": cont_loss.detach(),
            "hidden_rec_loss": hidden_mse.detach(),
            "hidden_rec_scaled_loss": (self.hidden_rec_scale * hidden_mse).detach(),
            "hidden_cosine_loss": hidden_cosine.detach(),
            "hidden_pred_norm": hidden_pred.detach().float().norm(dim=-1).mean().detach(),
            "hidden_target_norm": hidden_target.detach().float().norm(dim=-1).mean().detach(),
            "reward_pred_mean": _reward_pred(self.reward_head, reward_logits.detach()).mean().detach(),
            "image_mse": mse.detach(),
            "image_psnr": (-10.0 * torch.log10(mse.clamp_min(1e-8))).detach(),
            "dyn_entropy": kls["dyn_entropy"].detach(),
            "rep_entropy": kls["rep_entropy"].detach(),
        }
        return DreamerV3Loss(loss=loss, metrics=metrics)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict) and batch.get("mode") is not None:
            return self._forward_actor_adapter(batch)
        out = self.loss(batch)
        return self._compat_forward_dict(out)


__all__ = [
    "DreamerV3PixelWorldModel",
    "DreamerV3TokenWorldModel",
    "DreamerV3TokenFromPixelWorldModel",
    "DreamerV3PixelRynnBackboneWorldModel",
    "DreamerV3PixelEncoder",
    "DreamerV3TokenEncoder",
    "DreamerV3RSSM",
    "DreamerV3PixelDecoder",
    "DreamerV3TokenDecoder",
    "SymexpTwoHotHead",
]
