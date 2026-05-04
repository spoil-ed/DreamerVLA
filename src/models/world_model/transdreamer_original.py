"""Standalone TransDreamer world model for DreamerVLA's LIBERO token interface.

This file deliberately does not inherit from the existing DreamerVLA world
models and does not reuse the DreamerVLA image-codec blocks.  The model body is
made from the architectural pieces in ``/home/user01/liops/workspace/TransDreamer``:

* Conv2D/ConvTranspose2D blocks with Xavier init and ELU,
* ImgEncoder-style four-layer strided CNN,
* custom causal Transformer with sinusoidal absolute position embedding,
* observation-only posterior q(z_t | o_t),
* action-conditioned prior p(z_{t+1} | z_t, a_{t+1}),
* concat_o deterministic state,
* DenseDecoder-style reward/pcont heads and balanced KL.

The only adapter is at the boundary: LIBERO observations arrive as Chameleon
image-token ids, so token ids are embedded into a small channel grid before the
TransDreamer CNN, and the TransDreamer deconv decoder emits token-vocab logits
instead of RGB pixels.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import OneHotCategorical


def _init_xavier(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(int(in_features), int(out_features), bias=bias)
        _init_xavier(self)


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        padding: int,
        *,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _init_xavier(
                nn.Conv2d(
                    int(in_ch),
                    int(out_ch),
                    kernel_size=int(kernel),
                    stride=int(stride),
                    padding=int(padding),
                    bias=True,
                )
            )
        ]
        if act:
            layers.append(nn.ELU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTranspose2DBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        padding: int,
        *,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _init_xavier(
                nn.ConvTranspose2d(
                    int(in_ch),
                    int(out_ch),
                    kernel_size=int(kernel),
                    stride=int(stride),
                    padding=int(padding),
                    bias=True,
                )
            )
        ]
        if act:
            layers.append(nn.ELU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, act: str = "elu") -> None:
        super().__init__()
        act_layer: nn.Module
        if act == "elu":
            act_layer = nn.ELU(inplace=True)
        elif act == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported MLP activation {act!r}")
        self.net = nn.Sequential(
            Linear(in_dim, hidden_dim),
            act_layer,
            Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.net(x.to(dtype=dtype)).float()


class ImgTokenEncoder(nn.Module):
    """TransDreamer ImgEncoder structure applied to a token-id embedding grid."""

    def __init__(
        self,
        in_channels: int,
        spatial: tuple[int, int],
        obs_dim: int = 1536,
        depth: int = 48,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.spatial = (int(spatial[0]), int(spatial[1]))
        self.obs_dim = int(obs_dim)
        self.enc = nn.Sequential(
            Conv2DBlock(self.in_channels, depth, 4, 2, padding, act=True),
            Conv2DBlock(depth, 2 * depth, 4, 2, padding, act=True),
            Conv2DBlock(2 * depth, 4 * depth, 4, 2, padding, act=True),
            Conv2DBlock(4 * depth, 8 * depth, 4, 2, padding, act=True),
        )
        h, w = self.spatial
        for _ in range(4):
            h = (h + 2 * padding - 4) // 2 + 1
            w = (w + 2 * padding - 4) // 2 + 1
            if h <= 0 or w <= 0:
                raise ValueError(
                    f"Invalid token grid after TransDreamer CNN: {(h, w)}; "
                    f"spatial={self.spatial}, padding={padding}"
                )
        self.final_spatial = (h, w)
        self.proj = Linear(8 * depth * h * w, self.obs_dim)

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        *leading, n_img, c_in = token_grid.shape
        if c_in != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {c_in}")
        h, w = self.spatial
        if n_img != h * w:
            raise ValueError(f"Expected {h * w} tokens for spatial={self.spatial}, got {n_img}")
        x = token_grid.reshape(-1, n_img, c_in).permute(0, 2, 1).contiguous()
        x = x.view(-1, c_in, h, w)
        x = self.enc(x).flatten(1)
        x = self.proj(x)
        return x.view(*leading, self.obs_dim)


class ImgTokenDecoder(nn.Module):
    """TransDreamer ImgDecoder-style deconv head, adapted to token logits."""

    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        out_spatial: tuple[int, int],
        depth: int = 48,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.vocab_size = int(vocab_size)
        self.out_spatial = (int(out_spatial[0]), int(out_spatial[1]))
        self.fc = Linear(self.input_size, 1536)
        self.dec = nn.Sequential(
            ConvTranspose2DBlock(1536, 4 * depth, 5, 2, 0, act=True),
            ConvTranspose2DBlock(4 * depth, 2 * depth, 5, 2, 0, act=True),
            ConvTranspose2DBlock(2 * depth, depth, 6, 2, 0, act=True),
            # Keep the original deconv geometry, but postpone the huge vocab
            # projection until after pooling to the LIBERO token grid.
            ConvTranspose2DBlock(depth, depth, 6, 2, 0, act=True),
        )
        self.to_vocab = _init_xavier(nn.Conv2d(depth, self.vocab_size, kernel_size=1, bias=True))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        *leading, _ = feature.shape
        x = self.fc(feature.to(dtype=dtype))
        x = x.reshape(-1, 1536, 1, 1)
        x = self.dec(x)
        x = F.adaptive_avg_pool2d(x, self.out_spatial)
        x = self.to_vocab(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        return x.view(*leading, self.out_spatial[0] * self.out_spatial[1], self.vocab_size).float()


class DenseDecoder(nn.Module):
    def __init__(self, input_size: int, layers: int, units: int, act: str = "elu") -> None:
        super().__init__()
        act_cls = nn.ELU if act == "elu" else nn.ReLU
        mods: list[nn.Module] = []
        dim = int(input_size)
        for _ in range(int(layers)):
            mods.extend([Linear(dim, units), act_cls(inplace=True) if act == "elu" else act_cls()])
            dim = int(units)
        mods.append(Linear(dim, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.net(x.to(dtype=dtype)).float()


class GRUGatingMechanism(nn.Module):
    def __init__(self, d_input: int, bg: float = 0.1) -> None:
        super().__init__()
        self.Wr = Linear(d_input, d_input, bias=False)
        self.Ur = Linear(d_input, d_input, bias=False)
        self.Wz = Linear(d_input, d_input, bias=False)
        self.Uz = Linear(d_input, d_input)
        self.Wg = Linear(d_input, d_input, bias=False)
        self.Ug = Linear(d_input, d_input, bias=False)
        self.bg = float(bg)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = torch.tanh(self.Wg(y) + r * self.Ug(x))
        return (1.0 - z) * x + z * h


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        return torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model: int, d_inner: int, dropout: float, pre_lnorm: bool) -> None:
        super().__init__()
        self.pre_lnorm = bool(pre_lnorm)
        self.core = nn.Sequential(
            Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.pre_lnorm:
            return self.core(self.layer_norm(inp))
        return self.layer_norm(self.core(inp))


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_inner: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
    ) -> None:
        super().__init__()
        self.d_inner = int(d_inner)
        self.n_head = int(n_head)
        self.q_net = Linear(d_model, d_inner * n_head, bias=False)
        self.k_net = Linear(d_model, d_inner * n_head, bias=False)
        self.v_net = Linear(d_model, d_inner * n_head, bias=False)
        self.out_net = Linear(d_inner * n_head, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1.0 / math.sqrt(d_inner)
        self.pre_lnorm = bool(pre_lnorm)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        T_q, bsz = q.shape[:2]
        T_k = k.shape[0]
        if self.pre_lnorm:
            q_in = k_in = v_in = self.layer_norm(q)
            if k is not q:
                k_in = self.layer_norm(k)
                v_in = self.layer_norm(v)
            w_head_q = self.q_net(q_in)
            w_head_k = self.k_net(k_in)
            w_head_v = self.v_net(v_in)
        else:
            w_head_q = self.q_net(q)
            w_head_k = self.k_net(k)
            w_head_v = self.v_net(v)
        w_head_q = w_head_q.view(T_q, bsz, self.n_head, self.d_inner)
        w_head_k = w_head_k.view(T_k, bsz, self.n_head, self.d_inner)
        w_head_v = w_head_v.view(T_k, bsz, self.n_head, self.d_inner)
        attn_score = torch.einsum("ibnd,jbnd->ijbn", w_head_q, w_head_k) * self.scale
        attn_score = attn_score.float().masked_fill(
            attn_mask[:, :, None, None].bool(), -float("inf")
        ).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        attn_vec = attn_vec.contiguous().view(T_q, bsz, self.n_head * self.d_inner)
        attn_out = self.drop(self.out_net(attn_vec))
        if self.pre_lnorm:
            return attn_out
        return self.layer_norm(attn_out)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_inner: int,
        d_ff_inner: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
        gating: bool,
    ) -> None:
        super().__init__()
        self.mah = MultiheadAttention(d_model, n_head, d_inner, dropout, dropatt, pre_lnorm)
        self.pos_ff = PositionwiseFF(d_model, d_ff_inner, dropout, pre_lnorm)
        self.gating = bool(gating)
        if self.gating:
            self.gate1 = GRUGatingMechanism(d_model)
            self.gate2 = GRUGatingMechanism(d_model)

    def forward(self, inpts: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        src2 = self.mah(inpts, inpts, inpts, attn_mask=attn_mask)
        src = self.gate1(inpts, src2) if self.gating else inpts + src2
        src2 = self.pos_ff(src)
        return self.gate2(src, src2) if self.gating else src + src2


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 600,
        n_layers: int = 6,
        num_heads: int = 8,
        d_inner: int = 64,
        d_ff_inner: int = 1024,
        dropout: float = 0.1,
        dropatt: float = 0.1,
        pre_lnorm: bool = True,
        gating: bool = False,
        last_ln: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.last_ln = bool(last_ln)
        self.pos_embs = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.d_model, num_heads, d_inner, d_ff_inner,
                dropout, dropatt, pre_lnorm, gating,
            )
            for _ in range(self.n_layers)
        ])
        if self.last_ln:
            self.ln = nn.LayerNorm(self.d_model)

    @staticmethod
    def _generate_square_subsequent_mask(T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        N = H * W
        mask = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-1e10")).masked_fill(mask == 1, 0.0)
        mask = torch.repeat_interleave(mask, N, dim=0)
        mask = torch.repeat_interleave(mask, N, dim=1)
        return mask

    def forward(self, z: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        del actions
        B, T, D, H, W = z.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")
        attn_mask = self._generate_square_subsequent_mask(T, H, W, z.device)
        pos_ips = torch.arange(T * H * W, dtype=torch.float, device=z.device)
        pos_embs = self.drop(self.pos_embs(pos_ips)).to(dtype=z.dtype)
        z = rearrange(z, "b t d h w -> (t h w) b d")
        output = z + pos_embs
        output_list: list[torch.Tensor] = []
        for layer in self.layers:
            output = layer(output, attn_mask=attn_mask)
            if self.last_ln:
                output = self.ln(output)
            output_list.append(output)
        output = torch.stack(output_list, dim=1)
        return rearrange(output, "(t h w) l b d -> b t l d h w", h=H, w=W)


class OriginalTransDreamerDiscrete(nn.Module):
    """TransDreamer copied into the current pretokenized WM training interface."""

    def __init__(
        self,
        action_dim: int = 7,
        io_mode: str = "token",
        token_embed_dim: int = 3,
        num_image_tokens_vocab: int | None = None,
        spatial_grid: tuple[int, int] = (32, 16),
        obs_dim: int = 1536,
        stoch_dims: int = 32,
        stoch_categories: int = 32,
        td_d_model: int = 600,
        td_n_layers: int = 6,
        td_num_heads: int = 8,
        td_d_inner: int = 64,
        td_d_ff_inner: int = 1024,
        td_dropout: float = 0.1,
        td_dropatt: float = 0.1,
        td_pre_lnorm: bool = True,
        td_gating: bool = False,
        td_last_ln: bool = False,
        td_deter_type: str = "concat_o",
        td_hidden_size: int = 600,
        td_obs_encoder_depth: int = 48,
        td_obs_encoder_padding: int = 1,
        td_decoder_depth: int = 48,
        kl_balance: float = 0.8,
        kl_loss_coef: float = 0.1,
        free_nats: float = 0.0,
        image_decoder_loss_coef: float = 1.0,
        reward_loss_coef: float = 1.0,
        pcont_loss_coef: float = 5.0,
        pcont_done_threshold: float = 0.5,
        **unused: Any,
    ) -> None:
        super().__init__()
        del unused
        if str(io_mode) != "token":
            raise ValueError("OriginalTransDreamerDiscrete only supports io_mode='token'")
        if num_image_tokens_vocab is None:
            raise ValueError("num_image_tokens_vocab must be provided by the workspace")
        self.io_mode = "token"
        self.spatial_codec = True
        self.image_decoder_enabled = True
        self.state_conditioning = False
        self.action_dim = int(action_dim)
        self.token_embed_dim = int(token_embed_dim)
        self.num_image_tokens_vocab = int(num_image_tokens_vocab)
        self.spatial_grid = (int(spatial_grid[0]), int(spatial_grid[1]))
        self.n_image_tokens = self.spatial_grid[0] * self.spatial_grid[1]
        self.stoch_dims = int(stoch_dims)
        self.stoch_categories = int(stoch_categories)
        self.latent_dim = self.stoch_dims * self.stoch_categories
        self.td_d_model = int(td_d_model)
        self.td_n_layers = int(td_n_layers)
        self.td_deter_type = str(td_deter_type)
        if self.td_deter_type not in {"concat_o", "last"}:
            raise ValueError("td_deter_type must be 'concat_o' or 'last'")
        self.deter_dim = (
            self.td_d_model * self.td_n_layers
            if self.td_deter_type == "concat_o"
            else self.td_d_model
        )
        self.d_model = self.deter_dim
        self.obs_dim = int(obs_dim)
        self.kl_balance = float(kl_balance)
        self.kl_loss_coef = float(kl_loss_coef)
        self.free_nats = float(free_nats)
        self.image_decoder_loss_coef = float(image_decoder_loss_coef)
        self.reward_loss_coef = float(reward_loss_coef)
        self.pcont_loss_coef = float(pcont_loss_coef)
        self.pcont_done_threshold = float(pcont_done_threshold)

        self.token_embedder = nn.Embedding(self.num_image_tokens_vocab, self.token_embed_dim)
        nn.init.uniform_(self.token_embedder.weight, -0.05, 0.05)
        self.img_enc = ImgTokenEncoder(
            in_channels=self.token_embed_dim,
            spatial=self.spatial_grid,
            obs_dim=self.obs_dim,
            depth=int(td_obs_encoder_depth),
            padding=int(td_obs_encoder_padding),
        )
        self.cell = Transformer(
            d_model=self.td_d_model,
            n_layers=self.td_n_layers,
            num_heads=int(td_num_heads),
            d_inner=int(td_d_inner),
            d_ff_inner=int(td_d_ff_inner),
            dropout=float(td_dropout),
            dropatt=float(td_dropatt),
            pre_lnorm=bool(td_pre_lnorm),
            gating=bool(td_gating),
            last_ln=bool(td_last_ln),
        )
        self.act_stoch_mlp = Linear(self.action_dim + self.latent_dim, self.td_d_model)
        self.post_stoch_mlp = MLP(self.obs_dim, int(td_hidden_size), self.latent_dim, act="elu")
        self.prior_stoch_mlp = MLP(self.deter_dim, int(td_hidden_size), self.latent_dim, act="elu")
        feature_dim = self.latent_dim + self.deter_dim
        self.img_dec = ImgTokenDecoder(
            input_size=feature_dim,
            vocab_size=self.num_image_tokens_vocab,
            out_spatial=self.spatial_grid,
            depth=int(td_decoder_depth),
        )
        self.reward = DenseDecoder(feature_dim, layers=4, units=400, act="elu")
        self.pcont = DenseDecoder(feature_dim, layers=4, units=400, act="elu")

        self.register_buffer("image_token_bpe_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_bpe_to_img_idx", torch.empty(0, dtype=torch.long), persistent=False)

    def attach_lm_head(
        self,
        lm_head: nn.Module | None,
        image_token_bpe_ids: torch.Tensor,
        full_vocab_size: int,
    ) -> None:
        del lm_head
        device = next(self.parameters()).device
        image_token_bpe_ids = image_token_bpe_ids.to(device=device, dtype=torch.long).clone()
        self.image_token_bpe_ids = image_token_bpe_ids
        rev = torch.full((int(full_vocab_size),), -1, dtype=torch.long, device=device)
        rev[image_token_bpe_ids] = torch.arange(image_token_bpe_ids.numel(), device=device)
        self._bpe_to_img_idx = rev

    def _tokens_to_obs_emb(self, bpe_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._bpe_to_img_idx.numel() == 0:
            raise RuntimeError("attach_lm_head() must be called before token-mode training")
        img_idx = self._bpe_to_img_idx[bpe_ids.long()]
        if (img_idx < 0).any():
            raise ValueError("Input contains non-image BPE ids")
        tok = self.token_embedder(img_idx)
        return self.img_enc(tok), img_idx

    def _stat_layer(self, logits_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits_flat.reshape(*logits_flat.shape[:-1], self.stoch_dims, self.stoch_categories)
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample()
        stoch = sample + dist.probs - dist.probs.detach()
        return logits, stoch.reshape(*logits_flat.shape[:-1], self.latent_dim)

    def _infer_post(self, obs_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._stat_layer(self.post_stoch_mlp(obs_emb))

    def _infer_prior(self, prev_stoch: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = prev_stoch.shape[:2]
        dtype = self.act_stoch_mlp.weight.dtype
        act_sto_emb = self.act_stoch_mlp(torch.cat([actions, prev_stoch], dim=-1).to(dtype=dtype))
        act_sto_emb = F.elu(act_sto_emb)
        o_t = self.cell(act_sto_emb.reshape(B, T, self.td_d_model, 1, 1), None)
        o_t = o_t.reshape(B, T, self.td_n_layers, self.td_d_model)
        if self.td_deter_type == "concat_o":
            deter = o_t.reshape(B, T, self.deter_dim)
        else:
            deter = o_t[:, :, -1]
        prior_logits, prior_stoch = self._stat_layer(self.prior_stoch_mlp(deter))
        return prior_logits, prior_stoch, deter

    @staticmethod
    def _cat_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        log_post = F.log_softmax(post_logits, dim=-1)
        post_p = log_post.exp()
        log_prior = F.log_softmax(prior_logits, dim=-1)
        return (post_p * (log_post - log_prior)).sum(dim=-1).sum(dim=-1)

    def _effective_rank(self, logits: torch.Tensor) -> torch.Tensor:
        x = logits.reshape(-1, self.latent_dim).float()
        if x.shape[0] < 2:
            return x.new_zeros(())
        x = x - x.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(x)
        p = s / s.sum().clamp_min(1e-8)
        return torch.exp(-(p * p.clamp_min(1e-8).log()).sum())

    @staticmethod
    def _pairwise_cos(logits: torch.Tensor) -> torch.Tensor:
        x = logits.reshape(-1, logits.shape[-2] * logits.shape[-1]).float()
        if x.shape[0] < 2:
            return x.new_zeros(())
        x = F.normalize(x, dim=-1)
        sim = x @ x.t()
        n = sim.shape[0]
        return (sim.sum() - sim.diag().sum()) / max(n * (n - 1), 1)

    def _token_ce_and_metrics(
        self,
        logits: torch.Tensor,
        target_idx: torch.Tensor,
        prev_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_idx.reshape(-1),
            reduction="none",
        ).view_as(target_idx)
        image_ce = ce.reshape(*ce.shape[:-1], -1).sum(dim=-1).mean()
        dynamic_mask = target_idx != prev_idx
        static_mask = ~dynamic_mask
        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1)
            acc = (pred_idx == target_idx).float().mean()
            static_acc = (
                (pred_idx[static_mask] == target_idx[static_mask]).float().mean()
                if static_mask.any() else acc.new_zeros(())
            )
            dynamic_acc = (
                (pred_idx[dynamic_mask] == target_idx[dynamic_mask]).float().mean()
                if dynamic_mask.any() else acc.new_zeros(())
            )
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            pred_entropy = -(probs * log_probs).sum(dim=-1).mean()
            flat_pred = pred_idx.reshape(-1, pred_idx.shape[-1])
            pred_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_pred],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()
            flat_gt = target_idx.reshape(-1, target_idx.shape[-1])
            gt_unique = torch.tensor(
                [int(torch.unique(row).numel()) for row in flat_gt],
                dtype=logits.dtype,
                device=logits.device,
            ).mean()
        return {
            "image_recon_ce_loss": image_ce,
            "image_static_ce_loss": ce[static_mask].mean() if static_mask.any() else image_ce.new_zeros(()),
            "image_dynamic_ce_loss": ce[dynamic_mask].mean() if dynamic_mask.any() else image_ce.new_zeros(()),
            "image_recon_accuracy": acc,
            "image_static_accuracy": static_acc,
            "image_dynamic_accuracy": dynamic_acc,
            "image_dynamic_fraction": dynamic_mask.float().mean(),
            "pred_entropy": pred_entropy,
            "pred_unique_tokens": pred_unique,
            "gt_unique_tokens": gt_unique,
        }

    def pretrain_loss(
        self,
        hidden_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor | None = None,
        done_seq: torch.Tensor | None = None,
        global_step: int | None = None,
        **unused: Any,
    ) -> dict[str, torch.Tensor]:
        del global_step, unused
        obs_emb, img_idx_seq = self._tokens_to_obs_emb(hidden_seq)
        post_logits_all, post_stoch_all = self._infer_post(obs_emb)
        prior_logits, _prior_stoch, deter = self._infer_prior(
            post_stoch_all[:, :-1],
            action_seq[:, 1:],
        )
        post_logits = post_logits_all[:, 1:]
        post_stoch = post_stoch_all[:, 1:]
        K = post_logits.shape[1]

        rep_kl_t = self._cat_kl(post_logits, prior_logits.detach())
        dyn_kl_t = self._cat_kl(post_logits.detach(), prior_logits)
        rep_kl_mean = rep_kl_t.sum(dim=-1).mean() if rep_kl_t.ndim == 3 else rep_kl_t.mean()
        dyn_kl_mean = dyn_kl_t.sum(dim=-1).mean() if dyn_kl_t.ndim == 3 else dyn_kl_t.mean()
        rep_kl = torch.maximum(rep_kl_mean, rep_kl_mean.new_tensor(self.free_nats))
        dyn_kl = torch.maximum(dyn_kl_mean, dyn_kl_mean.new_tensor(self.free_nats))
        kl_unscaled = (1.0 - self.kl_balance) * rep_kl + self.kl_balance * dyn_kl
        kl_loss = self.kl_loss_coef * kl_unscaled

        feature = torch.cat([post_stoch, deter], dim=-1)
        logits = self.img_dec(feature)
        ce_metrics = self._token_ce_and_metrics(
            logits=logits,
            target_idx=img_idx_seq[:, 1:],
            prev_idx=img_idx_seq[:, :-1],
        )
        image_decoder_loss = self.image_decoder_loss_coef * ce_metrics["image_recon_ce_loss"]

        reward_pred = self.reward(feature).squeeze(-1)
        if reward_seq is None:
            reward_target = torch.zeros_like(reward_pred)
        else:
            reward_target = reward_seq[:, 1 : 1 + K].reshape_as(reward_pred)
        reward_loss = F.mse_loss(reward_pred, reward_target)

        pcont_logits = self.pcont(feature).squeeze(-1)
        if done_seq is not None:
            done_target = done_seq[:, 1 : 1 + K].reshape_as(pcont_logits).float()
        elif reward_seq is not None:
            done_target = (reward_seq[:, 1 : 1 + K].reshape_as(pcont_logits) > self.pcont_done_threshold).float()
        else:
            done_target = torch.zeros_like(pcont_logits)
        pcont_target = 1.0 - done_target
        pcont_loss_raw = F.binary_cross_entropy_with_logits(pcont_logits, pcont_target)
        pcont_loss = self.pcont_loss_coef * pcont_loss_raw

        loss = image_decoder_loss + kl_loss + self.reward_loss_coef * reward_loss + pcont_loss

        with torch.no_grad():
            post_probs = F.softmax(post_logits, dim=-1)
            prior_probs = F.softmax(prior_logits, dim=-1)
            post_entropy = -(post_probs * post_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            prior_entropy = -(prior_probs * prior_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

        zero = loss.new_zeros(())
        return {
            "loss": loss,
            "kl_loss": kl_loss,
            "vae_warmup": zero,
            "eff_kl_loss_coef": loss.new_tensor(self.kl_loss_coef),
            "dyn_kl": dyn_kl.detach(),
            "rep_kl": rep_kl.detach(),
            "kl_post_prior": rep_kl_t.mean().detach(),
            "dyn_loss": (self.kl_balance * dyn_kl).detach(),
            "rep_loss": ((1.0 - self.kl_balance) * rep_kl).detach(),
            "transition_loss": zero,
            "reward_loss": reward_loss,
            "delta_latent_loss": zero,
            "action_margin_loss": zero,
            "action_margin_active": zero,
            "action_ranking_loss": zero,
            "pcont_loss": pcont_loss,
            "pcont_acc": ((pcont_logits > 0).float() == pcont_target).float().mean().detach(),
            "image_decoder_loss": image_decoder_loss,
            "image_recon_mse_loss": zero,
            **ce_metrics,
            "post_eff_rank": self._effective_rank(post_logits.detach()),
            "prior_eff_rank": self._effective_rank(prior_logits.detach()),
            "post_pairwise_cos": self._pairwise_cos(post_logits.detach()),
            "prior_pairwise_cos": self._pairwise_cos(prior_logits.detach()),
            "post_entropy": post_entropy.detach(),
            "prior_entropy": prior_entropy.detach(),
            "post_max_prob": post_probs.max(dim=-1).values.mean().detach(),
            "prior_max_prob": prior_probs.max(dim=-1).values.mean().detach(),
            "post_std_mean": (1.0 - post_probs.max(dim=-1).values).mean().detach(),
            "prior_std_mean": (1.0 - prior_probs.max(dim=-1).values).mean().detach(),
            "post_logits_std_across_batch": post_logits.reshape(-1, self.latent_dim).std(dim=0).mean().detach(),
            "prior_logits_std_across_batch": prior_logits.reshape(-1, self.latent_dim).std(dim=0).mean().detach(),
            "post_logits_mean_abs": post_logits.abs().mean().detach(),
            "prior_logits_mean_abs": prior_logits.abs().mean().detach(),
            "sequence_loss_steps": zero,
            "sequence_context_steps": zero,
            "sequence_loss_scale": zero,
            "imagine_ce_loss": zero,
            "imagine_static_ce_loss": zero,
            "imagine_dynamic_ce_loss": zero,
            "imagine_loss": zero,
            "imagine_recon_accuracy": zero,
            "imagine_static_accuracy": zero,
            "imagine_dynamic_accuracy": zero,
            "imagine_dynamic_fraction": zero,
        }

    def compute_loss_dict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        first = next(self.parameters())
        device, dtype = first.device, first.dtype
        if "obs_embedding_seq" in batch and "action_seq" in batch:
            hidden_seq = batch["obs_embedding_seq"].to(device=device, dtype=torch.long)
            action_seq = batch["action_seq"].to(device=device, dtype=dtype)
            reward_seq = batch.get("reward_seq")
            done_seq = batch.get("done_seq")
            if reward_seq is not None:
                reward_seq = reward_seq.to(device=device, dtype=dtype)
            if done_seq is not None:
                done_seq = done_seq.to(device=device, dtype=dtype)
            return self.pretrain_loss(hidden_seq, action_seq, reward_seq=reward_seq, done_seq=done_seq)

        if "obs_embedding" not in batch or "next_obs_embedding" not in batch:
            raise ValueError("OriginalTransDreamerDiscrete expects sequence or single-step token inputs")
        obs = batch["obs_embedding"].to(device=device, dtype=torch.long)
        nxt = batch["next_obs_embedding"].to(device=device, dtype=torch.long)
        action = batch["action"].to(device=device, dtype=dtype)
        if action.ndim == 3:
            action = action.mean(dim=1)
        zero_action = torch.zeros_like(action)
        hidden_seq = torch.stack([obs, nxt], dim=1)
        action_seq = torch.stack([zero_action, action], dim=1)
        reward = batch.get("reward")
        reward_seq = None
        if reward is not None:
            reward = reward.to(device=device, dtype=dtype).reshape(action.shape[0])
            reward_seq = torch.stack([torch.zeros_like(reward), reward], dim=1)
        return self.pretrain_loss(hidden_seq, action_seq, reward_seq=reward_seq)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.compute_loss_dict(batch)

    @torch.no_grad()
    def predict_next_image_token_ids(
        self,
        current_bpe: torch.Tensor,
        action: torch.Tensor,
        state_token_ids: torch.Tensor | None = None,
        state_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del state_token_ids, state_token_mask
        first = next(self.parameters())
        device, dtype = first.device, first.dtype
        current_bpe = current_bpe.to(device=device, dtype=torch.long)
        action = action.to(device=device, dtype=dtype)
        if action.ndim == 3:
            action = action.mean(dim=1)
        obs_emb, _img_idx = self._tokens_to_obs_emb(current_bpe)
        _post_logits, post_stoch = self._infer_post(obs_emb.unsqueeze(1))
        prior_logits, prior_stoch, deter = self._infer_prior(post_stoch, action.unsqueeze(1))
        del prior_logits, prior_stoch
        feature = torch.cat([post_stoch, deter], dim=-1)
        logits = self.img_dec(feature).squeeze(1)
        pred_idx = logits.argmax(dim=-1)
        return self.image_token_bpe_ids[pred_idx]


__all__ = ["OriginalTransDreamerDiscrete"]
