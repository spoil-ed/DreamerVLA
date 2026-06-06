from __future__ import annotations

# ruff: noqa: F822
# (names below are resolved lazily via module-level __getattr__)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer_vla.models.world_model.block_linear import BlockLinear
from dreamer_vla.models.world_model.base_world_model import (
    DreamerV3ActorAdapterMixin,
    DreamerV3LatentState,
    DreamerV3Loss,
)
from dreamer_vla.models.world_model.reward_heads import (
    BinaryRewardHead,
    SymexpTwoHotHead,
    _make_reward_head,
    _reward_loss,
    _reward_pred,
)


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
    if name == "elu":
        return nn.ELU()
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
            layers.append(
                nn.Conv2d(prev, out_ch, kernel_size=kernel, stride=stride, padding=pad)
            )
            if not self.strided:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(ChannelRMSNorm(out_ch))
            layers.append(_act(act))
            prev = out_ch
            h = math.ceil(h / 2) if self.strided else h // 2
            w = math.ceil(w / 2) if self.strided else w // 2
        if not (3 <= h <= 16 and 3 <= w <= 16):
            raise ValueError(
                f"DreamerV3 final image grid should be 3..16, got {(h, w)}"
            )
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
            layers.append(
                nn.Conv2d(prev, out_ch, kernel_size=kernel, stride=stride, padding=pad)
            )
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
                raise ValueError(
                    f"Expected {self.n_image_tokens} image tokens, got {n}"
                )
            tokens = tokens.reshape(b, t, self.num_views, self.tokens_per_view)
        elif tokens.ndim == 4:
            b, t, v, n = tokens.shape
            if v != self.num_views or n != self.tokens_per_view:
                raise ValueError(
                    f"Expected tokens [B,T,{self.num_views},{self.tokens_per_view}], "
                    f"got {tuple(tokens.shape)}"
                )
        else:
            raise ValueError(
                f"Expected token tensor [B,T,V,N], got {tuple(tokens.shape)}"
            )
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

        self.dynin0 = nn.Sequential(
            nn.Linear(self.deter, self.hidden), RMSNorm(self.hidden), activation
        )
        self.dynin1 = nn.Sequential(
            nn.Linear(self.flat_stoch, self.hidden), RMSNorm(self.hidden), _act(act)
        )
        self.dynin2 = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden), RMSNorm(self.hidden), _act(act)
        )
        core_in = self.deter + self.blocks * 3 * self.hidden
        dyn_layers: list[nn.Module] = []
        for _ in range(int(dynlayers)):
            dyn_layers.extend(
                [
                    BlockLinear(core_in, self.deter, self.blocks),
                    RMSNorm(self.deter),
                    _act(act),
                ]
            )
            core_in = self.deter
        self.dynhid = nn.Sequential(*dyn_layers)
        self.dyngru = BlockLinear(self.deter, 3 * self.deter, self.blocks)

        prior_layers: list[nn.Module] = []
        prior_in = self.deter
        for _ in range(int(imglayers)):
            prior_layers.extend(
                [nn.Linear(prior_in, self.hidden), RMSNorm(self.hidden), _act(act)]
            )
            prior_in = self.hidden
        prior_layers.append(nn.Linear(prior_in, self.flat_stoch))
        self.prior_net = nn.Sequential(*prior_layers)

        self.obslayers = int(obslayers)
        self._posterior: nn.Sequential | None = None

    def build_posterior(self, token_dim: int, act: str = "silu") -> None:
        layers: list[nn.Module] = []
        in_dim = self.deter + int(token_dim)
        for _ in range(self.obslayers):
            layers.extend(
                [nn.Linear(in_dim, self.hidden), RMSNorm(self.hidden), _act(act)]
            )
            in_dim = self.hidden
        layers.append(nn.Linear(in_dim, self.flat_stoch))
        self._posterior = nn.Sequential(*layers)

    def initial(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self.deter, device=device, dtype=dtype),
            "stoch": torch.zeros(
                batch_size, self.stoch, self.classes, device=device, dtype=dtype
            ),
        }

    def _core(
        self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        action = action / torch.maximum(torch.ones_like(action), action.abs()).detach()
        x0 = self.dynin0(deter)
        x1 = self.dynin1(stoch_flat)
        x2 = self.dynin2(action)
        x = torch.cat([x0, x1, x2], dim=-1)
        x = x[:, None, :].expand(-1, self.blocks, -1)
        deter_group = deter.reshape(
            deter.shape[0], self.blocks, self.deter // self.blocks
        )
        x = torch.cat([deter_group, x], dim=-1).reshape(deter.shape[0], -1)
        x = self.dynhid(x)
        gates = self.dyngru(x).reshape(
            deter.shape[0], self.blocks, 3 * (self.deter // self.blocks)
        )
        reset, cand, update = [
            g.reshape(deter.shape[0], self.deter) for g in gates.chunk(3, dim=-1)
        ]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter

    def _prior(self, deter: torch.Tensor) -> torch.Tensor:
        return self.prior_net(deter).reshape(deter.shape[0], self.stoch, self.classes)

    def _posterior_logits(
        self, deter: torch.Tensor, token: torch.Tensor
    ) -> torch.Tensor:
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
        sample_probs = sample_probs / sample_probs.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
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
            action = (
                actions[:, step].to(device=tokens.device, dtype=tokens.dtype) * keep
            )
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

    def observe_next(
        self,
        latent: DreamerV3LatentState,
        token: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor | bool | None = None,
    ) -> DreamerV3LatentState:
        if token.ndim != 2:
            raise ValueError(f"token must be [B,D], got {tuple(token.shape)}")
        batch = int(token.shape[0])
        if action.ndim == 3:
            action = action[:, 0]
        if action.ndim == 1:
            action = action[None]
        if action.shape[0] != batch:
            raise ValueError(
                f"action batch mismatch: got {action.shape[0]}, expected {batch}"
            )
        if is_first is None:
            reset = torch.zeros(batch, device=token.device, dtype=torch.bool)
        elif isinstance(is_first, bool):
            reset = torch.full(
                (batch,), bool(is_first), device=token.device, dtype=torch.bool
            )
        else:
            reset = is_first.to(device=token.device).reshape(batch).bool()
        keep = (~reset).to(dtype=token.dtype).view(batch, 1)
        deter = latent.deter.to(device=token.device, dtype=token.dtype) * keep
        stoch = latent.stoch.to(device=token.device, dtype=token.dtype) * keep.view(
            batch, 1, 1
        )
        action = action.to(device=token.device, dtype=token.dtype) * keep
        deter = self._core(deter, stoch, action)
        prior_logits = self._prior(deter)
        del prior_logits
        post_logits = self._posterior_logits(deter, token)
        post_stoch = self._sample(post_logits)
        return DreamerV3LatentState(deter=deter, stoch=post_stoch, logits=post_logits)

    def kl_loss(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
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
            raise ValueError(
                f"DreamerV3 decoder minres should be 3..16, got {self.minres}"
            )
        self.shape = (self.depths[-1], self.minres, self.minres)
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
            convs.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(prev, out_ch, kernel_size=kernel, padding=pad),
                    ChannelRMSNorm(out_ch),
                    _act(act),
                ]
            )
            prev = out_ch
        convs.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(prev, self.image_channels, kernel_size=kernel, padding=pad),
            ]
        )
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
            convs.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(prev, out_ch, kernel_size=kernel, padding=pad),
                    ChannelRMSNorm(out_ch),
                    _act(act),
                ]
            )
            prev = out_ch
        convs.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    prev,
                    self.num_views * self.num_image_tokens_vocab,
                    kernel_size=kernel,
                    padding=pad,
                ),
            ]
        )
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
            b,
            t,
            self.num_views,
            self.num_image_tokens_vocab,
            self.tokens_per_view,
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


class _ResBlock(nn.Module):
    def __init__(self, dim: int, act: str) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = _act(act)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        return x + self.fc2(h)


class ResMLPHead(nn.Module):
    """ResNet-style MLP head: input proj -> N residual blocks -> output proj.

    Each block: x + Linear(act(Linear(RMSNorm(x)))).
    Stable for deeper stacks than plain MLPHead because of skip connections.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 2,
        units: int = 8192,
        act: str = "silu",
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(in_dim), int(units))
        self.blocks = nn.ModuleList(
            [_ResBlock(int(units), act) for _ in range(int(layers))]
        )
        self.norm_out = RMSNorm(int(units))
        final = nn.Linear(int(units), int(out_dim))
        if float(outscale) != 1.0:
            with torch.no_grad():
                final.weight.mul_(float(outscale))
                if final.bias is not None:
                    final.bias.mul_(float(outscale))
        self.output_proj = final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(self.norm_out(h))


class Pi0StyleHiddenDecoder(nn.Module):
    """Pi0-action-head-style hidden decoder.

    Replicates the original RynnVLA action-head pattern: a TransformerEncoder
    over [memory_tokens, learned_query_tokens]. Memory comes from feat (deter +
    stoch*classes) split into ``mem_tokens`` tokens of width ``d_model``. We have
    ``num_queries = out_dim // d_model`` learned query embeddings; their output
    positions, concatenated, form the reconstructed action_hidden.

    Default ``d_model=1024, mem_tokens=8, nhead=8`` mirrors pi0's reduced_hidden
    width and gives an FF dim of 4096 -- compact compared to a 16384-unit MLP.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 4,
        d_model: int = 1024,
        nhead: int = 8,
        mem_tokens: int = 8,
        dim_feedforward_mult: int = 4,
        dropout: float = 0.0,
        act: str = "gelu",
        outscale: float = 1.0,
        token_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        mem_tokens = int(mem_tokens)
        out_dim = int(out_dim)
        token_dim = int(token_dim) if token_dim is not None else d_model
        if out_dim % token_dim != 0:
            raise ValueError(
                f"Pi0StyleHiddenDecoder: out_dim={out_dim} must be divisible by token_dim={token_dim}"
            )
        self.num_queries = out_dim // token_dim
        self.d_model = d_model
        self.token_dim = token_dim
        self.mem_tokens = mem_tokens
        self.out_dim = out_dim
        self.feat_proj = nn.Linear(int(in_dim), mem_tokens * d_model)
        self.queries = nn.Parameter(torch.randn(self.num_queries, d_model) * 0.02)
        self.out_proj = (
            nn.Linear(d_model, token_dim) if token_dim != d_model else nn.Identity()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(nhead),
            dim_feedforward=d_model * int(dim_feedforward_mult),
            dropout=float(dropout),
            activation=str(act).lower()
            if str(act).lower() in {"relu", "gelu"}
            else "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(layers),
            norm=nn.LayerNorm(d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        if float(outscale) != 1.0:
            with torch.no_grad():
                self.queries.mul_(float(outscale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        N = x_flat.shape[0]
        mem = self.feat_proj(x_flat).view(N, self.mem_tokens, self.d_model)
        queries = self.queries.unsqueeze(0).expand(N, -1, -1)
        seq = torch.cat([mem, queries], dim=1)
        out = self.transformer(seq)
        out_queries = self.output_norm(out[:, self.mem_tokens :, :])
        out_tokens = self.out_proj(out_queries)
        return out_tokens.reshape(*lead_shape, self.out_dim)


class Pi0TimeBroadcastDecoder(nn.Module):
    """Time-only transformer decoder with joint broadcast.

    Structural finding (docs/hidden_token_structure_report.md): the
    35 = 5*7 action_hidden tokens collapse to a 5-step time sequence
    with 7 statistically-identical joint copies per step (same-t residual
    cosine = 0.996). So only 5 learned query tokens are needed; the 7
    joints are produced by broadcasting.

    Architecture (default ``T_q=5, joint_broadcast=7, d_model=1024``):

        feat ``[B, ..., in_dim]``
            → Linear → ``mem [B, mem_tokens, d_model]``
        concat with learned ``queries [T_q, d_model]``
            → ``[B, mem_tokens + T_q, d_model]``
        TransformerEncoder × L
            → take last ``T_q`` positions → ``time_out [B, T_q, d_model]``
        broadcast along joint axis (repeat)
            → ``[B, T_q, J, d_model]`` → flatten → ``[B, T_q * J * d_model]``

    Compared to ``Pi0StyleHiddenDecoder``:
        - 35 queries → 5 queries  (-86 % query params)
        - attention seq 43 → 13   (-91 % attention ops)
        - feat_proj unchanged → total params ≈ same
        - removes the 30×30 degenerate joint-pair attention block
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 4,
        d_model: int = 1024,
        nhead: int = 8,
        mem_tokens: int = 8,
        n_time_queries: int = 5,
        joint_broadcast: int = 7,
        dim_feedforward_mult: int = 4,
        dropout: float = 0.0,
        act: str = "gelu",
        outscale: float = 1.0,
        token_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        mem_tokens = int(mem_tokens)
        out_dim = int(out_dim)
        n_time_queries = int(n_time_queries)
        joint_broadcast = int(joint_broadcast)
        token_dim = int(token_dim) if token_dim is not None else d_model
        expected = n_time_queries * joint_broadcast * token_dim
        if out_dim != expected:
            raise ValueError(
                f"Pi0TimeBroadcastDecoder: out_dim={out_dim} must equal "
                f"n_time_queries × joint_broadcast × token_dim "
                f"({n_time_queries}×{joint_broadcast}×{token_dim}={expected})"
            )
        self.d_model = d_model
        self.token_dim = token_dim
        self.mem_tokens = mem_tokens
        self.n_time_queries = n_time_queries
        self.joint_broadcast = joint_broadcast
        self.out_dim = out_dim
        self.feat_proj = nn.Linear(int(in_dim), mem_tokens * d_model)
        self.queries = nn.Parameter(torch.randn(n_time_queries, d_model) * 0.02)
        self.out_proj = (
            nn.Linear(d_model, token_dim) if token_dim != d_model else nn.Identity()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(nhead),
            dim_feedforward=d_model * int(dim_feedforward_mult),
            dropout=float(dropout),
            activation=str(act).lower()
            if str(act).lower() in {"relu", "gelu"}
            else "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(layers),
            norm=nn.LayerNorm(d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        if float(outscale) != 1.0:
            with torch.no_grad():
                self.queries.mul_(float(outscale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        N = x_flat.shape[0]
        mem = self.feat_proj(x_flat).view(N, self.mem_tokens, self.d_model)
        queries = self.queries.unsqueeze(0).expand(N, -1, -1)
        seq = torch.cat([mem, queries], dim=1)
        out = self.transformer(seq)
        time_out = self.output_norm(out[:, self.mem_tokens :, :])  # [N, T_q, d_model]
        time_tokens = self.out_proj(time_out)  # [N, T_q, token_dim]
        # broadcast across joint axis: [N, T_q, 1, token_dim] -> [N, T_q, J, token_dim]
        broadcast = time_tokens.unsqueeze(2).expand(-1, -1, self.joint_broadcast, -1)
        return broadcast.reshape(*lead_shape, self.out_dim)


class PerTokenMLPHead(nn.Module):
    """Per-token shared MLP decoder for structured hidden reconstruction.

    Treats the output ``[B, ..., out_dim]`` as ``n_tokens × token_dim`` and produces
    each of the ``n_tokens`` 1024-dim tokens via a SHARED MLP, conditioned on a
    small learned query embedding to distinguish token positions.

    Architecture:
        feat ``[B, T, in_dim]``
            → broadcast to ``[B, T, n_tokens, in_dim]``
            → concat learned ``query_emb [n_tokens, query_dim]`` → ``[B, T, n_tokens, in_dim+query_dim]``
            → shared ``MLPHead`` ``(in_dim+query_dim) → ... → token_dim`` → ``[B, T, n_tokens, token_dim]``
            → reshape → ``[B, T, n_tokens*token_dim]``

    Param budget (default ``layers=2, units=2048, query_dim=128``) for
    in_dim=10240, n_tokens=35, token_dim=1024:
        query_emb 35×128 = 4.5 K
        Linear(10368 → 2048) = 21.2 M
        Linear(2048 → 1024)  =  2.1 M
        ≈ 23 M total — matches the legacy RynnVLA action-head decoder's per-token capacity.
    """

    def __init__(
        self,
        in_dim: int,
        n_tokens: int,
        token_dim: int,
        query_dim: int = 128,
        layers: int = 2,
        units: int = 2048,
        act: str = "silu",
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.query_dim = int(query_dim)
        self.out_dim = self.n_tokens * self.token_dim
        self.query_emb = nn.Parameter(torch.randn(self.n_tokens, self.query_dim) * 0.02)
        self.mlp = MLPHead(
            int(in_dim) + self.query_dim,
            self.token_dim,
            layers=int(layers),
            units=int(units),
            act=act,
            outscale=outscale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[..., in_dim]`` → ``[..., n_tokens * token_dim]``"""
        lead_shape = x.shape[:-1]
        x_expanded = x.unsqueeze(-2).expand(*lead_shape, self.n_tokens, -1)
        q = self.query_emb.to(dtype=x.dtype).expand(*lead_shape, -1, -1)
        x_with_q = torch.cat([x_expanded, q], dim=-1)
        out = self.mlp(x_with_q)
        return out.reshape(*lead_shape, self.out_dim)


class FullHiddenSequenceDecoder(nn.Module):
    """Decode RSSM features into a fixed-length VLA token hidden sequence."""

    def __init__(
        self,
        in_dim: int,
        sequence_length: int,
        hidden_dim: int = 4096,
        query_dim: int = 1024,
        layers: int = 1,
        units: int = 2048,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.hidden_dim = int(hidden_dim)
        self.query_dim = int(query_dim)
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        self.feature_proj = MLPHead(
            in_dim, self.query_dim, layers=1, units=units, act=act
        )
        self.position = nn.Parameter(torch.zeros(self.sequence_length, self.query_dim))
        nn.init.normal_(self.position, std=0.02)
        mods: list[nn.Module] = []
        cur = self.query_dim
        for _ in range(int(layers)):
            mods.extend([nn.Linear(cur, int(units)), RMSNorm(int(units)), _act(act)])
            cur = int(units)
        mods.append(nn.Linear(cur, self.hidden_dim))
        self.token_head = nn.Sequential(*mods)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        context = self.feature_proj(feature).unsqueeze(-2)
        token_query = context + self.position.to(
            device=feature.device, dtype=context.dtype
        )
        return self.token_head(token_query)


class CompactTokenSequenceAutoencoder(nn.Module):
    """Compress VLA token hidden states into a small token latent and decode a tail context."""

    def __init__(
        self,
        in_dim: int = 4096,
        latent_tokens: int = 32,
        latent_dim: int = 1024,
        target_tokens: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.latent_tokens = int(latent_tokens)
        self.latent_dim = int(latent_dim)
        self.target_tokens = int(target_tokens)
        if self.latent_tokens <= 0 or self.target_tokens <= 0:
            raise ValueError("latent_tokens and target_tokens must be positive")
        self.input_norm = RMSNorm(self.in_dim)
        self.input_proj = nn.Linear(self.in_dim, self.latent_dim)
        self.latent_queries = nn.Parameter(
            torch.zeros(self.latent_tokens, self.latent_dim)
        )
        self.decoder_queries = nn.Parameter(
            torch.zeros(self.target_tokens, self.latent_dim)
        )
        nn.init.normal_(self.latent_queries, std=0.02)
        nn.init.normal_(self.decoder_queries, std=0.02)
        self.encoder_attn = nn.MultiheadAttention(
            self.latent_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.decoder_attn = nn.MultiheadAttention(
            self.latent_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.latent_norm = RMSNorm(self.latent_dim)
        self.decode_norm = RMSNorm(self.latent_dim)
        self.output_proj = nn.Linear(self.latent_dim, self.in_dim)

    @staticmethod
    def tail_tokens(
        hidden: torch.Tensor, mask: torch.Tensor | None, target_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim < 3:
            raise ValueError(f"hidden must end with [L,D], got {tuple(hidden.shape)}")
        length = int(hidden.shape[-2])
        target_tokens = int(target_tokens)
        if mask is None:
            mask = torch.ones(
                *hidden.shape[:-1], device=hidden.device, dtype=torch.bool
            )
        mask = mask.to(device=hidden.device).bool()
        valid = mask.long().sum(dim=-1).clamp_min(1)
        offsets = torch.arange(target_tokens, device=hidden.device)
        indices = valid.unsqueeze(-1) - target_tokens + offsets
        target_mask = indices >= 0
        indices = indices.clamp(0, length - 1)
        gather_index = indices.unsqueeze(-1).expand(
            *indices.shape, int(hidden.shape[-1])
        )
        tail = torch.gather(hidden, dim=-2, index=gather_index)
        return tail, target_mask

    def encode(
        self, hidden: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        original_shape = hidden.shape[:-2]
        length = int(hidden.shape[-2])
        hidden_flat = hidden.reshape(-1, length, int(hidden.shape[-1]))
        if hidden_flat.shape[-1] != self.in_dim:
            raise ValueError(
                f"hidden dim mismatch: got {hidden_flat.shape[-1]}, expected {self.in_dim}"
            )
        if mask is None:
            mask_flat = torch.ones(
                hidden_flat.shape[:2], device=hidden.device, dtype=torch.bool
            )
        else:
            mask_flat = mask.reshape(-1, length).to(device=hidden.device).bool()
        key = self.input_proj(self.input_norm(hidden_flat))
        queries = (
            self.latent_queries.to(device=hidden.device, dtype=key.dtype)
            .unsqueeze(0)
            .expand(key.shape[0], -1, -1)
        )
        latent, _ = self.encoder_attn(queries, key, key, key_padding_mask=~mask_flat)
        latent = self.latent_norm(latent + queries)
        return latent.reshape(*original_shape, self.latent_tokens, self.latent_dim)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        original_shape = latent.shape[:-2]
        latent_flat = latent.reshape(-1, self.latent_tokens, self.latent_dim)
        queries = (
            self.decoder_queries.to(device=latent.device, dtype=latent.dtype)
            .unsqueeze(0)
            .expand(
                latent_flat.shape[0],
                -1,
                -1,
            )
        )
        decoded, _ = self.decoder_attn(queries, latent_flat, latent_flat)
        decoded = self.decode_norm(decoded + queries)
        decoded = self.output_proj(decoded)
        return decoded.reshape(*original_shape, self.target_tokens, self.in_dim)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(hidden, mask)
        reconstruction = self.decode(latent)
        target, target_mask = self.tail_tokens(hidden, mask, self.target_tokens)
        return {
            "latent": latent,
            "reconstruction": reconstruction,
            "target": target,
            "target_mask": target_mask,
        }


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


_WORLD_MODEL_EXPORTS = {
    "DreamerV3PixelWorldModel": "dreamer_vla.models.world_model.dreamer_v3_pixel_world_model",
    "DreamerV3TokenWorldModel": "dreamer_vla.models.world_model.dreamer_v3_token_world_model",
    "DreamerV3TokenFromPixelWorldModel": "dreamer_vla.models.world_model.dreamer_v3_token_from_pixel_world_model",
    "DreamerV3PixelRynnBackboneWorldModel": "dreamer_vla.models.world_model.dreamer_v3_pixel_rynn_backbone_world_model",
    "RynnDinoWMWorldModel": "dreamer_vla.models.world_model.rynn_dino_wm",
    "OFTDinoWMWorldModel": "dreamer_vla.models.world_model.rynn_dino_wm",
}


def __getattr__(name: str):
    if name in _WORLD_MODEL_EXPORTS:
        from importlib import import_module

        module = import_module(_WORLD_MODEL_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DreamerV3ActorAdapterMixin",
    "DreamerV3LatentState",
    "DreamerV3Loss",
    "DreamerV3PixelWorldModel",
    "DreamerV3TokenWorldModel",
    "DreamerV3TokenFromPixelWorldModel",
    "DreamerV3PixelRynnBackboneWorldModel",
    "RynnDinoWMWorldModel",
    "OFTDinoWMWorldModel",
    "DreamerV3PixelEncoder",
    "DreamerV3TokenEncoder",
    "DreamerV3RSSM",
    "DreamerV3PixelDecoder",
    "DreamerV3TokenDecoder",
    "CompactTokenSequenceAutoencoder",
    "SymexpTwoHotHead",
    "BinaryRewardHead",
    "_make_reward_head",
    "_reward_loss",
    "_reward_pred",
    "Pi0StyleHiddenDecoder",
    "Pi0TimeBroadcastDecoder",
    "PerTokenMLPHead",
    "FullHiddenSequenceDecoder",
]
