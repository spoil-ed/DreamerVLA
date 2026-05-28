"""TSSM (Transformer State Space Model) port for DreamerVLA.

Faithful port of TransDreamer's TransformerDynamic:
    /mnt/data/spoil/workspace/worldmodel/TransDreamer/model/modules_transformer.py
    /mnt/data/spoil/workspace/worldmodel/TransDreamer/model/transformer.py

The Transformer cell, MultiheadAttention, PositionwiseFF, GRUGating, and sinusoidal
positional embedding are inlined here (TransDreamer's transformer.py classes) with
image-spatial H/W dims removed (we only have a T-axis sequence). All other behavior
(deter_type=concat_o, layer-wise output stacking, manual straight-through sample,
no unimix, no final LN) matches the original 1:1.

Public class:
    ``TSSMRynnBackboneWorldModel`` — drop-in replacement for
    ``DreamerV3PixelRynnBackboneWorldModel``.
"""

from __future__ import annotations

# ruff: noqa: F822
# (names below are resolved lazily via module-level __getattr__)

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Independent

from dreamer_vla.models.world_model.dreamerv3_torch import (
    MLPHead,
    Pi0StyleHiddenDecoder,
    ResMLPHead,
)


# ============================================================
# Faithful TransDreamer Transformer port (1D sequence, no H/W)
# ============================================================


class _SinusoidalPosEmb(nn.Module):
    """1:1 with TransDreamer.transformer.PositionalEmbedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [T] (float). returns [T, 1, dim].
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class _MultiheadAttention(nn.Module):
    """1:1 with TransDreamer.transformer.MultiheadAttention (separate q/k/v Linears + dropatt)."""

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
        self.n_head = int(n_head)
        self.d_inner = int(d_inner)
        self.q_net = nn.Linear(d_model, d_inner * n_head, bias=False)
        self.k_net = nn.Linear(d_model, d_inner * n_head, bias=False)
        self.v_net = nn.Linear(d_model, d_inner * n_head, bias=False)
        self.out_net = nn.Linear(d_inner * n_head, d_model, bias=False)
        self.drop = nn.Dropout(float(dropout))
        self.dropatt = nn.Dropout(float(dropatt))
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1.0 / (d_inner**0.5)
        self.pre_lnorm = bool(pre_lnorm)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q,k,v: [T, B, D]
        T_q, bsz = q.shape[:2]
        T_k = k.shape[0]
        if self.pre_lnorm:
            w_q = self.q_net(self.layer_norm(q))
            w_k = self.k_net(self.layer_norm(k))
            w_v = self.v_net(self.layer_norm(v))
        else:
            w_q = self.q_net(q)
            w_k = self.k_net(k)
            w_v = self.v_net(v)
        w_q = w_q.view(T_q, bsz, self.n_head, self.d_inner)
        w_k = w_k.view(T_k, bsz, self.n_head, self.d_inner)
        w_v = w_v.view(T_k, bsz, self.n_head, self.d_inner)
        attn_score = (
            torch.einsum("ibnd,jbnd->ijbn", w_q, w_k) * self.scale
        )  # [Tq, Tk, B, H]
        if attn_mask is not None:
            attn_score = (
                attn_score.float()
                .masked_fill(
                    attn_mask[:, :, None, None].bool(),
                    float("-1e10"),
                )
                .type_as(attn_score)
            )
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, w_v)
        attn_vec = attn_vec.contiguous().view(T_q, bsz, self.n_head * self.d_inner)
        attn_out = self.drop(self.out_net(attn_vec))
        if self.pre_lnorm:
            return attn_out
        return self.layer_norm(attn_out)


class _PositionwiseFF(nn.Module):
    """1:1 with TransDreamer.transformer.PositionwiseFF (Linear→ReLU→Linear→Dropout, LN per pre/post)."""

    def __init__(
        self, d_model: int, d_ff_inner: int, dropout: float, pre_lnorm: bool
    ) -> None:
        super().__init__()
        self.pre_lnorm = bool(pre_lnorm)
        self.core_net = nn.Sequential(
            nn.Linear(d_model, int(d_ff_inner)),
            nn.ReLU(inplace=True),
            nn.Linear(int(d_ff_inner), d_model),
            nn.Dropout(float(dropout)),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.pre_lnorm:
            return self.core_net(self.layer_norm(inp))
        return self.layer_norm(self.core_net(inp))


class _GRUGating(nn.Module):
    """1:1 with TransDreamer.transformer.GRUGatingMechanism."""

    def __init__(self, d_input: int, bg: float = 0.1) -> None:
        super().__init__()
        self.Wr = nn.Linear(d_input, d_input, bias=False)
        self.Ur = nn.Linear(d_input, d_input, bias=False)
        self.Wz = nn.Linear(d_input, d_input, bias=False)
        self.Uz = nn.Linear(d_input, d_input)
        self.Wg = nn.Linear(d_input, d_input, bias=False)
        self.Ug = nn.Linear(d_input, d_input, bias=False)
        self.bg = float(bg)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = torch.tanh(self.Wg(y) + r * self.Ug(x))
        return (1 - z) * x + z * h


class _TransformerLayer(nn.Module):
    """1:1 with TransDreamer.transformer.TransformerEncoderLayer (MHA + FFN, optional GRU gating)."""

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
        self.mha = _MultiheadAttention(
            d_model, n_head, d_inner, dropout, dropatt, pre_lnorm
        )
        self.ff = _PositionwiseFF(d_model, d_ff_inner, dropout, pre_lnorm)
        self.gating = bool(gating)
        if self.gating:
            self.gate1 = _GRUGating(d_model)
            self.gate2 = _GRUGating(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        # x: [T, B, D]
        src2 = self.mha(x, x, x, attn_mask=attn_mask)
        src = self.gate1(x, src2) if self.gating else x + src2
        src2 = self.ff(src)
        src = self.gate2(src, src2) if self.gating else src + src2
        return src


class _Transformer(nn.Module):
    """1:1 with TransDreamer.transformer.Transformer minus image H/W handling.

    Returns ALL layer outputs stacked: [B, T, L, D] (used for deter_type='concat_o').
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_head: int,
        d_inner: int,
        d_ff_inner: int,
        dropout: float,
        dropatt: float,
        pre_lnorm: bool,
        gating: bool,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.pos_embs = _SinusoidalPosEmb(d_model)
        self.drop = nn.Dropout(float(dropout))
        self.layers = nn.ModuleList(
            [
                _TransformerLayer(
                    d_model,
                    n_head,
                    d_inner,
                    d_ff_inner,
                    dropout,
                    dropatt,
                    pre_lnorm,
                    gating,
                )
                for _ in range(n_layers)
            ]
        )
        # last_ln declared in TransDreamer but NEVER applied in forward — we omit it.

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # 1:1 with TransDreamer: mask[i,j] = True if j > i (upper triangular, masked)
        m = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
        # m True = allowed, False = masked. We need True = masked for masked_fill.
        # Match TransDreamer: masked_fill(mask==0, -1e10) — so mask==0 (i.e., j>i) gets masked.
        # Our boolean mask: True where (j > i)
        return ~m.bool()  # True where masked

    def forward(
        self, z: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """z: [B, S, D]; output: [B, S, L, D]. attn_mask: optional [S, S] boolean (True = masked)."""
        B, S, D = z.shape
        device = z.device
        if attn_mask is None:
            attn_mask = self._causal_mask(S, device)
        pos_ips = torch.arange(S, dtype=torch.float, device=device)
        pos_embs = self.drop(self.pos_embs(pos_ips)).to(dtype=z.dtype)  # [S, 1, D]
        # to (S, B, D)
        x = z.transpose(0, 1).contiguous() + pos_embs
        output = x
        outputs: list[torch.Tensor] = []
        for layer in self.layers:
            output = layer(output, attn_mask=attn_mask)
            outputs.append(output)
        stacked = torch.stack(outputs, dim=1)  # [S, L, B, D]
        return stacked.permute(2, 0, 1, 3).contiguous()  # [B, S, L, D]

    @staticmethod
    def spatio_temporal_mask(T: int, N: int, device: torch.device) -> torch.Tensor:
        """Build a (T*N) x (T*N) causal-at-time mask.
        Token (t, i) attends to (t', j) iff t' <= t (any j). 1:1 with TransDreamer's
        ``_generate_square_subsequent_mask(T, H, W)`` semantics.
        """
        m = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(
            0, 1
        )  # True if t' <= t (allowed)
        m = m.repeat_interleave(N, dim=0).repeat_interleave(N, dim=1)
        return ~m.bool()  # True where masked


# ============================================================
# Latent state, dynamics, world model
# ============================================================


@dataclass
class TSSMLatentState:
    stoch: torch.Tensor  # [B, stoch, classes]
    deter: torch.Tensor  # [B, deter_dim]
    logits: torch.Tensor | None = None
    history_stoch: torch.Tensor | None = None
    history_action: torch.Tensor | None = None

    def feature(self) -> torch.Tensor:
        stoch_flat = self.stoch.reshape(*self.stoch.shape[:-2], -1)
        return torch.cat([stoch_flat, self.deter], dim=-1)


def _onehot_st_sample(logits: torch.Tensor) -> torch.Tensor:
    """Manual straight-through sample: stoch + p - p.detach()  (1:1 with TransDreamer 'discrete')."""
    dist = Independent(OneHotCategorical(logits=logits), 1)
    sample = dist.sample()
    probs = F.softmax(logits, dim=-1)
    return sample + probs - probs.detach()


class TSSMDynamic(nn.Module):
    """TSSM dynamics — 1:1 with TransDreamer.modules_transformer.TransformerDynamic (q_trans=False, discrete)."""

    def __init__(
        self,
        obs_emb_dim: int,
        action_dim: int,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 32,
        # Transformer hyperparams (defaults follow TransDreamer Atari config)
        n_layers: int = 4,
        n_head: int = 4,
        d_model: int = 384,
        d_inner: int = 96,  # per-head d_inner (so d_model = n_head * d_inner)
        d_ff_inner: int = 1536,  # FFN inner
        dropout: float = 0.1,
        dropatt: float = 0.0,
        pre_lnorm: bool = True,
        gating: bool = False,
        deter_type: str = "concat_o",  # 'concat_o' or 'last'
        tssm_window: int = 64,
        free_nats: float = 1.0,
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.stoch = int(stoch)
        self.classes = int(classes)
        self.flat_stoch = self.stoch * self.classes
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.deter_type = str(deter_type).lower()
        self.tssm_window = int(tssm_window)
        self.free_nats = float(free_nats)

        # Input to transformer: Linear(action+flat_stoch → d_model), with ELU (1:1 TransDreamer)
        self.act_stoch_mlp = nn.Linear(self.action_dim + self.flat_stoch, self.d_model)

        self.cell = _Transformer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_head=int(n_head),
            d_inner=int(d_inner),
            d_ff_inner=int(d_ff_inner),
            dropout=dropout,
            dropatt=dropatt,
            pre_lnorm=pre_lnorm,
            gating=gating,
        )

        # deter dim: concat_o → L*d_model, else d_model
        self.deter = (
            self.n_layers * self.d_model
            if self.deter_type == "concat_o"
            else self.d_model
        )

        # prior MLP: deter → hidden → flat_stoch (3-layer MLP matches TransDreamer's `MLP([d_model, hidden, latent_dim_out])`)
        self.prior_stoch_mlp = nn.Sequential(
            nn.Linear(self.deter, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.flat_stoch),
        )
        # posterior MLP: obs_emb → hidden → flat_stoch (q_trans=False path)
        self.obs_emb_dim = int(obs_emb_dim)
        self.post_stoch_mlp = nn.Sequential(
            nn.Linear(self.obs_emb_dim, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.flat_stoch),
        )

    # ---- helpers ----

    def _logit_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.stoch, self.classes)

    def _build_tx_input(
        self, stoch: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """stoch: [B,T,stoch,classes] or [B,T,flat]; action: [B,T,action_dim] → [B,T,d_model]."""
        if stoch.ndim == 4:
            stoch_flat = stoch.reshape(*stoch.shape[:2], self.flat_stoch)
        else:
            stoch_flat = stoch
        # TransDreamer: act_stoch_mlp(cat([action, prev_stoch])) then F.elu
        x = self.act_stoch_mlp(torch.cat([action, stoch_flat], dim=-1))
        return F.elu(x)

    def _deter_from_layers(self, o_t: torch.Tensor) -> torch.Tensor:
        """o_t: [B, T, L, D] → deter [B, T, L*D] (concat_o) or [B, T, D] (last)."""
        if self.deter_type == "concat_o":
            return o_t.reshape(*o_t.shape[:2], -1)
        return o_t[:, :, -1]

    # ---- KL ----

    def _dist(self, logits: torch.Tensor) -> torch.distributions.Distribution:
        return Independent(OneHotCategorical(logits=logits), 1)

    def kl_loss(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # post / prior logits: [B, T, stoch, classes]
        post = self._dist(post_logits)
        prior = self._dist(prior_logits)
        dyn = torch.distributions.kl_divergence(
            self._dist(post_logits.detach()), prior
        ).mean()
        rep = torch.distributions.kl_divergence(
            post, self._dist(prior_logits.detach())
        ).mean()
        if self.free_nats > 0:
            dyn = torch.clamp(dyn, min=self.free_nats)
            rep = torch.clamp(rep, min=self.free_nats)
        return {"dyn": dyn, "rep": rep}

    # ---- sequence-mode observe (training) ----

    def observe(
        self, enc: torch.Tensor, actions: torch.Tensor, is_first: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """1:1 with TransDreamer's TransformerDynamic.forward:
        - posterior from obs only
        - prior on (s_t[:-1], action[1:]) predicts s_{1..T-1}
        - is_first masking is applied to prev_stoch/prev_action just like RSSM convention.
        """
        B, T = enc.shape[:2]
        if T < 2:
            raise ValueError(
                "TSSMDynamic.observe requires at least 2 timesteps for TransDreamer alignment"
            )

        # 1) Posterior from obs only (logits computed in float32 for numerical stability,
        #    but stoch sample cast back to enc.dtype so downstream Linear layers match weights)
        post_logits_flat = self.post_stoch_mlp(enc).float()
        post_logits = self._logit_view(
            post_logits_flat
        )  # [B, T, stoch, classes] float32
        post_stoch = _onehot_st_sample(post_logits).to(
            dtype=enc.dtype
        )  # cast to compute dtype

        # 2) Prior from (s_t, a_t) for targets t=1..T-1, matching TransDreamer:
        #    s_t = post_stoch[:, :-1], action = actions[:, 1:].
        prev_stoch = post_stoch[:, :-1]
        prev_action = actions[:, 1:]
        target_is_first = is_first[:, 1:]
        mask_s = (
            (~target_is_first.bool())
            .to(dtype=prev_stoch.dtype)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        mask_a = (~target_is_first.bool()).to(dtype=prev_action.dtype).unsqueeze(-1)
        prev_stoch = prev_stoch * mask_s
        prev_action = prev_action * mask_a

        # 3) Transformer over the input sequence
        tx_in = self._build_tx_input(prev_stoch, prev_action)  # [B, T-1, d_model]
        steps = T - 1
        if steps > self.tssm_window:
            tx_in = tx_in[:, -self.tssm_window :]
            o_t = self.cell(tx_in)
            pad = steps - self.tssm_window
            pad_zeros = o_t.new_zeros(B, pad, self.n_layers, self.d_model)
            o_t = torch.cat([pad_zeros, o_t], dim=1)
        else:
            o_t = self.cell(tx_in)  # [B, T-1, L, D]
        deter = self._deter_from_layers(o_t)  # [B, T-1, deter_dim]

        # 4) Prior logits
        prior_logits_flat = self.prior_stoch_mlp(deter).float()
        prior_logits = self._logit_view(prior_logits_flat)

        return {
            "deter": deter,
            "stoch": post_stoch[:, 1:],
            "post_logits": post_logits[:, 1:],
            "prior_logits": prior_logits,
        }

    # ---- single-step inference (rollout) ----

    def observe_next(
        self,
        latent: TSSMLatentState,
        enc: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | bool | None = None,
    ) -> TSSMLatentState:
        device = enc.device
        B = enc.shape[0]
        action = actions if actions.ndim == 2 else actions[:, 0]

        # 1) Posterior from new obs (1:1 TransDreamer post_stoch_mlp)
        post_logits = self._logit_view(self.post_stoch_mlp(enc).float())
        new_stoch = _onehot_st_sample(post_logits).to(dtype=enc.dtype)

        # 2) Update history with the *previous* (stoch, action)
        prev_stoch_step = latent.stoch.unsqueeze(1)
        action_step = action.unsqueeze(1)
        if latent.history_stoch is None:
            new_h_stoch = prev_stoch_step
            new_h_action = action_step
        else:
            new_h_stoch = torch.cat([latent.history_stoch, prev_stoch_step], dim=1)
            new_h_action = torch.cat([latent.history_action, action_step], dim=1)
        # is_first reset
        if is_first is not None:
            if not isinstance(is_first, torch.Tensor):
                is_first = torch.tensor(bool(is_first), device=device).expand(B)
            mask = (~is_first.bool()).to(dtype=new_h_stoch.dtype).reshape(B, 1, 1, 1)
            new_h_stoch = new_h_stoch * mask
            new_h_action = new_h_action * mask.squeeze(-1)
        # Window trunc
        if new_h_stoch.shape[1] > self.tssm_window:
            new_h_stoch = new_h_stoch[:, -self.tssm_window :]
            new_h_action = new_h_action[:, -self.tssm_window :]

        # 3) Transformer → deter at last position
        tx_in = self._build_tx_input(new_h_stoch, new_h_action)
        o_t = self.cell(tx_in)  # [B, T, L, D]
        new_deter = self._deter_from_layers(o_t)[:, -1]  # [B, deter_dim]

        return TSSMLatentState(
            stoch=new_stoch,
            deter=new_deter,
            logits=post_logits,
            history_stoch=new_h_stoch,
            history_action=new_h_action,
        )


# ============================================================
# Token-based TSSM (each of the 35 pi0 action-query tokens is its own latent token)
# ============================================================


@dataclass
class TSSMTokenLatentState:
    """Latent state for token-based TSSM. Carries per-token stoch/deter (N=35 tokens)."""

    stoch: torch.Tensor  # [B, N, stoch, classes]
    deter: torch.Tensor  # [B, N, deter_dim_per_token]
    logits: torch.Tensor | None = None
    history_stoch: torch.Tensor | None = None  # [B, T, N, stoch, classes]
    history_action: torch.Tensor | None = None  # [B, T, action_dim]

    def feature(self) -> torch.Tensor:
        """Per-token feat then flatten across tokens. Shape: [B, N * (deter+flat_stoch)]."""
        stoch_flat = self.stoch.reshape(
            *self.stoch.shape[:-2], -1
        )  # [..., N, stoch*classes]
        feat = torch.cat([stoch_flat, self.deter], dim=-1)  # [..., N, feat_per_tok]
        return feat.reshape(*feat.shape[:-2], -1)  # [..., N*feat_per_tok]


class TSSMTokenDynamic(nn.Module):
    """Token-aware TSSM: pi0's 35 action-query tokens are kept as a sequence (not flattened).

    Per timestep t we have N=35 tokens of dim D_tok=1024 (= pi0 action_hidden 5*7*1024 reshaped).
    The causal Transformer attends over the (T * N) sequence:
        - causal at the time axis (future t masked)
        - bidirectional within a time step (all 35 tokens see each other)
    This matches TransDreamer's spatio-temporal handling exactly (it originally used H*W image tokens).

    Per-token posterior and prior:
        q(z_{t,i} | x_{t,i})          posterior MLP on each token's embedding
        p(z_{t,i} | z_{<t,*}, a_{<t}) prior MLP on each token's transformer output
    Stoch shape: [B, T, N, stoch, classes]. Deter shape: [B, T, N, deter_per_tok].
    """

    def __init__(
        self,
        n_tokens: int = 35,
        token_dim: int = 1024,
        action_dim: int = 7,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 32,
        n_layers: int = 4,
        n_head: int = 4,
        d_model: int = 384,
        d_inner: int = 96,
        d_ff_inner: int = 1536,
        dropout: float = 0.1,
        dropatt: float = 0.0,
        pre_lnorm: bool = True,
        gating: bool = False,
        deter_type: str = "concat_o",
        tssm_window: int = 8,  # in TIMESTEPS (each step has n_tokens tokens; seq len = window*n_tokens)
        free_nats: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.action_dim = int(action_dim)
        self.stoch = int(stoch)
        self.classes = int(classes)
        self.flat_stoch = self.stoch * self.classes
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.deter_type = str(deter_type).lower()
        self.tssm_window = int(tssm_window)
        self.free_nats = float(free_nats)

        # Per-token embedding: token_dim → d_model (shared across tokens)
        self.token_embed = (
            nn.Identity()
            if self.token_dim == self.d_model
            else nn.Linear(self.token_dim, self.d_model)
        )
        # Embed prev_stoch (one-hot flat) → d_model (used for dynamics input)
        self.stoch_embed = nn.Linear(self.flat_stoch, self.d_model)
        # Embed action → d_model (broadcast over N tokens within the same timestep)
        self.action_embed = nn.Linear(self.action_dim, self.d_model)

        # Transformer
        self.cell = _Transformer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_head=int(n_head),
            d_inner=int(d_inner),
            d_ff_inner=int(d_ff_inner),
            dropout=dropout,
            dropatt=dropatt,
            pre_lnorm=pre_lnorm,
            gating=gating,
        )

        # Per-token deter dim: concat_o → L*d_model, else d_model
        self.deter_per_token = (
            self.n_layers * self.d_model
            if self.deter_type == "concat_o"
            else self.d_model
        )
        # Aggregate deter (concat across N tokens at one timestep) for downstream heads expecting [B, T, deter_dim]
        self.deter = self.n_tokens * self.deter_per_token

        # Per-token posterior MLP (from token embedding)
        self.post_stoch_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.flat_stoch),
        )
        # Per-token prior MLP (from deter_per_token)
        self.prior_stoch_mlp = nn.Sequential(
            nn.Linear(self.deter_per_token, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.flat_stoch),
        )

    # ---- distribution helpers ----

    def _logit_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.stoch, self.classes)

    def _dist(self, logits: torch.Tensor) -> torch.distributions.Distribution:
        return Independent(OneHotCategorical(logits=logits), 1)

    def kl_loss(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        post = self._dist(post_logits)
        prior = self._dist(prior_logits)
        dyn = torch.distributions.kl_divergence(
            self._dist(post_logits.detach()), prior
        ).mean()
        rep = torch.distributions.kl_divergence(
            post, self._dist(prior_logits.detach())
        ).mean()
        if self.free_nats > 0:
            dyn = torch.clamp(dyn, min=self.free_nats)
            rep = torch.clamp(rep, min=self.free_nats)
        return {"dyn": dyn, "rep": rep}

    # ---- core ops ----

    def _per_token_aggregated(self, o_t: torch.Tensor, T: int, N: int) -> torch.Tensor:
        """o_t: [B, T*N, L, D] → [B, T, N, L*D] (concat_o) or [B, T, N, D] (last)."""
        B = o_t.shape[0]
        L, D = o_t.shape[-2], o_t.shape[-1]
        if self.deter_type == "concat_o":
            return o_t.reshape(B, T, N, L * D)
        return o_t[..., -1, :].reshape(B, T, N, D)

    def _build_tx_input(
        self,
        prev_stoch_flat: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        prev_stoch_flat: [B, T, N, flat_stoch]
        prev_action:     [B, T, action_dim]
        Returns: [B, T*N, d_model] — stoch_embed(per-token) + action_embed(broadcast over N).
        """
        B, T, N, _ = prev_stoch_flat.shape
        stoch_e = self.stoch_embed(prev_stoch_flat)  # [B, T, N, d_model]
        action_e = self.action_embed(prev_action)  # [B, T, d_model]
        action_e = action_e.unsqueeze(2).expand(B, T, N, -1)  # [B, T, N, d_model]
        tx = F.elu(stoch_e + action_e)  # element-wise add then ELU (TransDreamer style)
        return tx.reshape(B, T * N, self.d_model)

    # ---- sequence-mode observe (training) ----

    def observe(
        self,
        obs_tokens: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        obs_tokens: [B, T, N=35, D_tok=1024]
        actions:    [B, T, action_dim]
        is_first:   [B, T] bool
        Returns: {deter[B,T-1,N,deter_per_tok], stoch[B,T-1,N,stoch,classes],
                  post_logits[B,T-1,N,stoch,classes], prior_logits[B,T-1,N,stoch,classes]}
        """
        B, T, N, _ = obs_tokens.shape
        if T < 2:
            raise ValueError(
                "TSSMTokenDynamic.observe requires at least 2 timesteps for TransDreamer alignment"
            )
        assert N == self.n_tokens, f"expected n_tokens={self.n_tokens}, got {N}"
        device = obs_tokens.device

        # 1) Posterior per token (from token embedding)
        tok_emb = self.token_embed(obs_tokens)  # [B, T, N, d_model]
        post_logits_flat = self.post_stoch_mlp(tok_emb).float()  # [B, T, N, flat_stoch]
        post_logits = self._logit_view(post_logits_flat)  # [B, T, N, stoch, classes]
        post_stoch = _onehot_st_sample(post_logits).to(dtype=obs_tokens.dtype)

        # 2) Prior input, matching TransDreamer: post_stoch[:, :-1] with actions[:, 1:].
        prev_stoch = post_stoch[:, :-1]  # [B, T-1, N, stoch, classes]
        prev_action = actions[:, 1:]  # [B, T-1, action_dim]
        # is_first reset
        steps = T - 1
        target_is_first = is_first[:, 1:]
        m_s = (
            (~target_is_first.bool()).to(dtype=prev_stoch.dtype).view(B, steps, 1, 1, 1)
        )
        m_a = (~target_is_first.bool()).to(dtype=prev_action.dtype).view(B, steps, 1)
        prev_stoch = prev_stoch * m_s
        prev_action = prev_action * m_a
        prev_stoch_flat = prev_stoch.reshape(B, steps, N, self.flat_stoch)

        # 3) Transformer over (T*N) tokens with spatio-temporal mask
        tx_in = self._build_tx_input(
            prev_stoch_flat, prev_action
        )  # [B, (T-1)*N, d_model]
        if steps > self.tssm_window:
            # truncate to last `tssm_window` timesteps (window*N tokens), pad deter for earlier steps
            tx_in_trunc = tx_in.reshape(B, steps, N, self.d_model)[
                :, -self.tssm_window :
            ].reshape(B, self.tssm_window * N, self.d_model)
            mask = _Transformer.spatio_temporal_mask(self.tssm_window, N, device)
            o_t = self.cell(tx_in_trunc, attn_mask=mask)  # [B, win*N, L, d_model]
            o_full = o_t.new_zeros(B, steps * N, o_t.shape[-2], o_t.shape[-1])
            o_full[:, -self.tssm_window * N :] = o_t
            o_t = o_full
        else:
            mask = _Transformer.spatio_temporal_mask(steps, N, device)
            o_t = self.cell(tx_in, attn_mask=mask)  # [B, T*N, L, d_model]
        deter = self._per_token_aggregated(o_t, steps, N)  # [B, T-1, N, deter_per_tok]

        # 4) Prior per token
        prior_logits_flat = self.prior_stoch_mlp(deter).float()
        prior_logits = self._logit_view(prior_logits_flat)

        return {
            "deter": deter,
            "stoch": post_stoch[:, 1:],
            "post_logits": post_logits[:, 1:],
            "prior_logits": prior_logits,
        }

    def observe_next(
        self,
        latent: TSSMTokenLatentState,
        obs_tokens: torch.Tensor,
        actions: torch.Tensor,
        is_first: torch.Tensor | bool | None = None,
    ) -> TSSMTokenLatentState:
        """Single-step inference: obs_tokens [B, N, D_tok]; actions [B, action_dim] or [B,1,action_dim]."""
        device = obs_tokens.device
        B, N, _ = obs_tokens.shape
        action = actions if actions.ndim == 2 else actions[:, 0]

        # 1) Posterior from new obs
        tok_emb = self.token_embed(obs_tokens)
        post_logits = self._logit_view(self.post_stoch_mlp(tok_emb).float())
        new_stoch = _onehot_st_sample(post_logits).to(
            dtype=obs_tokens.dtype
        )  # [B, N, stoch, classes]

        # 2) Append (prev_stoch, action) to history
        prev_stoch_step = latent.stoch.unsqueeze(1)  # [B, 1, N, stoch, classes]
        action_step = action.unsqueeze(1)  # [B, 1, action_dim]
        if latent.history_stoch is None:
            new_h_stoch = prev_stoch_step
            new_h_action = action_step
        else:
            new_h_stoch = torch.cat([latent.history_stoch, prev_stoch_step], dim=1)
            new_h_action = torch.cat([latent.history_action, action_step], dim=1)
        if is_first is not None:
            if not isinstance(is_first, torch.Tensor):
                is_first = torch.tensor(bool(is_first), device=device).expand(B)
            m = (~is_first.bool()).to(dtype=new_h_stoch.dtype).view(B, 1, 1, 1, 1)
            new_h_stoch = new_h_stoch * m
            new_h_action = new_h_action * m.view(B, 1, 1)
        if new_h_stoch.shape[1] > self.tssm_window:
            new_h_stoch = new_h_stoch[:, -self.tssm_window :]
            new_h_action = new_h_action[:, -self.tssm_window :]

        # 3) Transformer over (T*N)
        T = new_h_stoch.shape[1]
        prev_stoch_flat = new_h_stoch.reshape(B, T, N, self.flat_stoch)
        tx_in = self._build_tx_input(prev_stoch_flat, new_h_action)
        mask = _Transformer.spatio_temporal_mask(T, N, device)
        o_t = self.cell(tx_in, attn_mask=mask)
        deter_seq = self._per_token_aggregated(o_t, T, N)
        new_deter = deter_seq[:, -1]  # [B, N, deter_per_tok]

        return TSSMTokenLatentState(
            stoch=new_stoch,
            deter=new_deter,
            logits=post_logits,
            history_stoch=new_h_stoch,
            history_action=new_h_action,
        )


# ============================================================
# World model wrapper
# ============================================================


# ============================================================
# helpers
# ============================================================


def _build_hidden_decoder(
    kind,
    in_dim,
    out_dim,
    *,
    layers,
    units,
    d_model,
    nhead,
    mem_tokens,
    dropout,
    act,
    n_time_queries: int = 5,
    joint_broadcast: int = 7,
):
    kind = str(kind).lower()
    if kind == "mlp":
        return MLPHead(in_dim, out_dim, layers=layers, units=units, act=act)
    if kind == "resnet":
        return ResMLPHead(in_dim, out_dim, layers=layers, units=units, act=act)
    if kind in {"pi0_transformer", "transformer", "pi0"}:
        return Pi0StyleHiddenDecoder(
            in_dim,
            out_dim,
            layers=layers,
            d_model=d_model,
            nhead=nhead,
            mem_tokens=mem_tokens,
            dropout=dropout,
            act=act,
        )
    if kind in {"pi0_time_broadcast", "time_broadcast", "pi0_time"}:
        # Lazy import to avoid circular: lives in dreamerv3_torch.
        from dreamer_vla.models.world_model.dreamerv3_torch import Pi0TimeBroadcastDecoder

        return Pi0TimeBroadcastDecoder(
            in_dim,
            out_dim,
            layers=layers,
            d_model=d_model,
            nhead=nhead,
            mem_tokens=mem_tokens,
            n_time_queries=int(n_time_queries),
            joint_broadcast=int(joint_broadcast),
            dropout=dropout,
            act=act,
        )
    raise ValueError(f"Unknown hidden_decoder_kind: {kind}")


class _PerTokenMLPDecoder(nn.Module):
    """Per-token shared MLP decoder. Each of N tokens maps (deter+stoch) → token_dim.
    Output is then flattened to obs_dim = N * token_dim.
    """

    def __init__(
        self,
        in_dim_per_tok: int,
        out_dim_per_tok: int,
        n_tokens: int,
        layers: int = 2,
        units: int = 4096,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.out_dim_per_tok = int(out_dim_per_tok)
        self.mlp = MLPHead(
            int(in_dim_per_tok),
            int(out_dim_per_tok),
            layers=int(layers),
            units=int(units),
            act=act,
        )

    def forward(self, feat_per_tok: torch.Tensor) -> torch.Tensor:
        # feat_per_tok: [..., N, in_dim_per_tok] → [..., N*out_dim_per_tok]
        out = self.mlp(feat_per_tok)  # MLP broadcasts over leading dims
        return out.reshape(*out.shape[:-2], self.n_tokens * self.out_dim_per_tok)


_WORLD_MODEL_EXPORTS = {
    "TSSMRynnBackboneWorldModel": "dreamer_vla.models.world_model.tssm_rynn_backbone_world_model",
    "TSSMTokenRynnBackboneWorldModel": "dreamer_vla.models.world_model.tssm_token_rynn_backbone_world_model",
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
    "TSSMRynnBackboneWorldModel",
    "TSSMDynamic",
    "TSSMLatentState",
    "TSSMTokenRynnBackboneWorldModel",
    "TSSMTokenDynamic",
    "TSSMTokenLatentState",
]
