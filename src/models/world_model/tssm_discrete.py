"""Discrete-stoch variant of TSSMWorldModelTransDreamer.

Replaces the diagonal-Gaussian latent z with `stoch_dims` independent
`stoch_categories`-way OneHotCategoricals (DreamerV2 / DreamerV3 / TransDreamer
default for image domains).

Why discrete:
  * Posterior std cannot collapse to 0 (entropy bounded by log K), so the
    "prior_std too small / posterior collapse" failure mode of the Gaussian
    variant disappears.
  * KL is between two categorical distributions, well-conditioned even when
    one side gets very confident.
  * Empirically more stable on pixel-space tasks (Hafner et al., DreamerV2 §4.1).

Implementation:
  * Inherits from `TSSMWorldModelTransDreamer`. Forces
    `latent_dim = stoch_dims * stoch_categories` so the *flat* latent
    dimension matches everywhere downstream (act_stoch_emb input,
    transition_head input, reward_head input, image_decoder input,
    feature() = cat(stoch_flat, h)).
  * Rebuilds the three "stats heads" (obs_to_stoch / prior_head / posterior_head)
    to output `stoch_dims * stoch_categories` logits instead of
    `2 * latent_dim` Gaussian (mean, log_std) parameters.
  * Overrides `_stats_to_dist` to sample via straight-through Gumbel-softmax
    during training and argmax (deterministic one-hot) during eval.
  * Overrides `_gaussian_kl` to compute per-dim categorical KL.

DreamerV3 unimix prior:
  Optionally blends the categorical with a uniform distribution
  (probs = (1 - unimix) * softmax(logits) + unimix * uniform). Prevents the
  prior from putting zero mass on any class, which would make the categorical
  KL unbounded if posterior visits that class. Default `unimix=0.01`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_model.block_linear import BlockLinear

from .tssm import TSSMWorldModelTransDreamer


class TSSMWorldModelTransDreamerDiscrete(TSSMWorldModelTransDreamer):
    """Categorical-stoch TransDreamer WM (DreamerV2/V3 style)."""

    def __init__(
        self,
        *args,
        stoch_dims: int = 32,
        stoch_categories: int = 32,
        gumbel_temp: float = 1.0,
        unimix: float = 0.01,
        **kwargs,
    ) -> None:
        flat_dim = int(stoch_dims) * int(stoch_categories)

        # Force latent_dim so that downstream layers built by the parent
        # (act_stoch_emb, transition_head, reward_head, image_decoder, ...)
        # receive the correct flat-z dim.
        if "latent_dim" in kwargs and int(kwargs["latent_dim"]) != flat_dim:
            raise ValueError(
                "Discrete WM forces latent_dim = stoch_dims * stoch_categories "
                f"= {flat_dim}; got latent_dim={kwargs['latent_dim']}."
            )
        kwargs["latent_dim"] = flat_dim

        super().__init__(*args, **kwargs)

        self.stoch_dims = int(stoch_dims)
        self.stoch_categories = int(stoch_categories)
        self.gumbel_temp = float(gumbel_temp)
        self.unimix = float(unimix)

        # Probe the parent's mapper hidden dim by inspecting one of its heads.
        # All three stats heads share the same intermediate width.
        mapper_hidden_dim = int(self.posterior_head[1].out_features)

        # Replace stats heads: parent built them with output dim 2*latent_dim
        # (Gaussian (mean, log_std)); discrete needs latent_dim logits only.
        self.obs_to_stoch = self._build_logits_head(
            self.obs_dim, mapper_hidden_dim, flat_dim,
        )
        self.prior_head = self._build_logits_head(
            self.d_model, mapper_hidden_dim, flat_dim,
        )
        self.posterior_head = self._build_logits_head(
            self.d_model + self.obs_dim, mapper_hidden_dim, flat_dim,
        )

    # ── Layer construction ──────────────────────────────────────────────────

    @staticmethod
    def _build_logits_head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    # ── Distribution helpers ────────────────────────────────────────────────

    def _stats_to_dist(
        self, stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Discrete-stoch sampling.

        Returns the same 3-tuple shape as the Gaussian parent for compatibility
        with downstream code:
            return value 0  →  *mean slot*  carries flat logits         [..., S*K]
            return value 1  →  *std slot*   placeholder ones (unused)   [..., S*K]
            return value 2  →  *stoch slot* one-hot flattened sample    [..., S*K]

        Sampling:
            train mode: straight-through Gumbel-softmax with temperature
                        `self.gumbel_temp`
            eval  mode: deterministic argmax → one-hot
        """
        leading = stats.shape[:-1]
        logits = stats.reshape(*leading, self.stoch_dims, self.stoch_categories)

        # DreamerV3 unimix: keep KL bounded by giving every class a floor mass.
        if self.unimix > 0.0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.full_like(probs, 1.0 / self.stoch_categories)
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
            logits = probs.clamp_min(1e-8).log()

        if self.training:
            # Straight-through Gumbel: forward returns one-hot via argmax,
            # backward uses softmax(logits/tau) gradient. Standard trick from
            # DreamerV2/V3 categorical RSSM.
            sample = F.gumbel_softmax(
                logits, tau=self.gumbel_temp, hard=True, dim=-1,
            )
        else:
            idx = logits.argmax(dim=-1)                                  # [..., S]
            sample = F.one_hot(idx, num_classes=self.stoch_categories)
            sample = sample.to(dtype=logits.dtype)                       # [..., S, K]

        flat_dim = self.stoch_dims * self.stoch_categories
        sample_flat = sample.reshape(*leading, flat_dim)
        logits_flat = logits.reshape(*leading, flat_dim)
        std_placeholder = torch.ones_like(logits_flat)
        return logits_flat, std_placeholder, sample_flat

    def _gaussian_kl(  # name kept for parent-compat dispatch
        self,
        post_mean: torch.Tensor,
        post_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
    ) -> torch.Tensor:
        """Per-dim categorical KL.

        ``post_mean`` and ``prior_mean`` carry the flat logits emitted by
        ``_stats_to_dist`` (see docstring there); ``post_std`` / ``prior_std``
        are the placeholder-ones tensors and are unused.

        Returns per-latent-dim KL with shape ``[..., S]`` (no sum/mean over
        the latent or batch axes).  The caller (``pretrain_loss``) applies
        free_nats clamp PER LATENT DIM before summing — this is essential to
        prevent the optimizer from finding the degenerate "concentrate KL in
        one dim, collapse the other 31" solution.  Cf. DreamerV3's free_bits.
        """
        del post_std, prior_std  # not used in discrete KL

        S = self.stoch_dims
        K = self.stoch_categories
        post_logits = post_mean.reshape(*post_mean.shape[:-1], S, K)
        prior_logits = prior_mean.reshape(*prior_mean.shape[:-1], S, K)

        log_post = F.log_softmax(post_logits, dim=-1)
        post_p = log_post.exp()
        log_prior = F.log_softmax(prior_logits, dim=-1)
        # KL per cell: sum_k post_p[k] * (log_post[k] - log_prior[k])
        kl = (post_p * (log_post - log_prior)).sum(dim=-1)               # [..., S]
        return kl                                                         # [..., S] per-dim


class _RSSMRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        scale = x32.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * scale).to(dtype=dtype) * self.weight.to(dtype=dtype)


def _rssm_act(name: str) -> nn.Module:
    name = str(name).lower()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported RSSM activation: {name!r}")


class TSSMWorldModelRSSMDiscrete(TSSMWorldModelTransDreamerDiscrete):
    """DreamerV3-style block-GRU RSSM core with the existing DreamerVLA I/O.

    The outer interface intentionally matches ``TSSMWorldModelTransDreamerDiscrete``:
    token/state input, ConvEncoderStem, image-token decoder, metrics, save/resume,
    and workspace plumbing remain unchanged.  Only the deterministic dynamics
    kernel changes from causal Transformer/LLM history encoding to an RSSM
    recurrent core:

        h_t = GRUBlock(h_{t-1}, z_{t-1}, a_t)
        p(z_t | h_t) = prior_head(h_t)
        q(z_t | h_t, o_t) = posterior_head([h_t, o_t])
    """

    def __init__(
        self,
        *args,
        rssm_hidden: int | None = None,
        rssm_blocks: int = 8,
        rssm_dyn_layers: int = 1,
        rssm_act: str = "silu",
        **kwargs,
    ) -> None:
        # This variant does not use the pretrained Chameleon/LLM transition
        # backbone.  For config compatibility, force the lightweight path even
        # when inheriting from older configs that set use_pretrained_backbone.
        kwargs["use_pretrained_backbone"] = False
        # RSSM posterior must see the deterministic state, unlike the
        # TransDreamer-paper q(z|o) ablation used by the previous baseline.
        kwargs["posterior_uses_h"] = True

        super().__init__(*args, **kwargs)

        # Parent construction creates a lightweight CausalTransformerCell for
        # historical compatibility.  The RSSM variant never calls it, so drop
        # its parameters before the optimizer/FSDP wrapper is built.
        self.causal_transformer = nn.Identity()
        self.posterior_uses_h = True
        self.rssm_blocks = int(rssm_blocks)
        if self.rssm_blocks <= 0:
            raise ValueError(f"rssm_blocks must be positive, got {rssm_blocks}")
        if self.d_model % self.rssm_blocks != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"rssm_blocks={self.rssm_blocks}"
            )
        self.rssm_hidden = int(rssm_hidden) if rssm_hidden is not None else self.d_model
        self.rssm_dyn_layers = int(rssm_dyn_layers)
        if self.rssm_dyn_layers < 1:
            raise ValueError("rssm_dyn_layers must be >= 1")

        act = str(rssm_act)
        self.rssm_dynin_h = nn.Sequential(
            nn.Linear(self.d_model, self.rssm_hidden),
            _RSSMRMSNorm(self.rssm_hidden),
            _rssm_act(act),
        )
        self.rssm_dynin_z = nn.Sequential(
            nn.Linear(self.latent_dim, self.rssm_hidden),
            _RSSMRMSNorm(self.rssm_hidden),
            _rssm_act(act),
        )
        self.rssm_dynin_a = nn.Sequential(
            nn.Linear(self.action_dim, self.rssm_hidden),
            _RSSMRMSNorm(self.rssm_hidden),
            _rssm_act(act),
        )

        core_in = self.d_model + self.rssm_blocks * 3 * self.rssm_hidden
        layers: list[nn.Module] = []
        for _ in range(self.rssm_dyn_layers):
            layers.extend(
                [
                    BlockLinear(core_in, self.d_model, self.rssm_blocks),
                    _RSSMRMSNorm(self.d_model),
                    _rssm_act(act),
                ]
            )
            core_in = self.d_model
        self.rssm_dynhid = nn.Sequential(*layers)
        self.rssm_gru = BlockLinear(self.d_model, 3 * self.d_model, self.rssm_blocks)

    def _rssm_core(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        # Match DreamerV3's action normalization guard: large action magnitudes
        # should not dominate the recurrent core input scale.
        action = action / torch.maximum(torch.ones_like(action), action.abs()).detach()
        x_h = self.rssm_dynin_h(deter)
        x_z = self.rssm_dynin_z(stoch)
        x_a = self.rssm_dynin_a(action)
        x = torch.cat([x_h, x_z, x_a], dim=-1)
        x = x[:, None, :].expand(-1, self.rssm_blocks, -1)
        deter_group = deter.reshape(
            deter.shape[0],
            self.rssm_blocks,
            self.d_model // self.rssm_blocks,
        )
        x = torch.cat([deter_group, x], dim=-1).reshape(deter.shape[0], -1)
        x = self.rssm_dynhid(x)
        gates = self.rssm_gru(x).reshape(
            deter.shape[0],
            self.rssm_blocks,
            3 * (self.d_model // self.rssm_blocks),
        )
        reset, cand, update = [
            gate.reshape(deter.shape[0], self.d_model)
            for gate in gates.chunk(3, dim=-1)
        ]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter

    def _infer_prior_seq(
        self,
        stoch_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential RSSM prior p(z_t | h_t), t=1..K.

        ``stoch_seq[:, i]`` is z_i and ``action_seq[:, i]`` is the action that
        carries the system to the next observation.  The returned h sequence is
        the deterministic state used for prior/posterior at those next frames.
        """
        B, K, _ = stoch_seq.shape
        h = stoch_seq.new_zeros(B, self.d_model)
        prior_mean_list: list[torch.Tensor] = []
        prior_std_list: list[torch.Tensor] = []
        prior_stoch_list: list[torch.Tensor] = []
        h_list: list[torch.Tensor] = []
        for t in range(K):
            h = self._rssm_core(h, stoch_seq[:, t], action_seq[:, t])
            prior_stats = self.prior_head(h)
            prior_mean, prior_std, prior_stoch = self._stats_to_dist(prior_stats)
            prior_mean_list.append(prior_mean)
            prior_std_list.append(prior_std)
            prior_stoch_list.append(prior_stoch)
            h_list.append(h)
        return (
            torch.stack(prior_mean_list, dim=1),
            torch.stack(prior_std_list, dim=1),
            torch.stack(prior_stoch_list, dim=1),
            torch.stack(h_list, dim=1),
        )

    def _infer_dreamer_seq(
        self,
        hidden_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        B, T, _ = hidden_seq.shape
        if T < 2:
            raise ValueError("RSSM sequence loss requires T >= 2")

        h = hidden_seq.new_zeros(B, self.d_model)
        post_mean_list: list[torch.Tensor] = []
        post_std_list: list[torch.Tensor] = []
        post_stoch_list: list[torch.Tensor] = []
        prior_mean_list: list[torch.Tensor] = []
        prior_std_list: list[torch.Tensor] = []
        prior_stoch_list: list[torch.Tensor] = []
        h_list: list[torch.Tensor] = []

        mean_t, std_t, stoch_t = self._posterior_from_obs_h(hidden_seq[:, 0], h)
        post_mean_list.append(mean_t)
        post_std_list.append(std_t)
        post_stoch_list.append(stoch_t)

        for t in range(1, T):
            h = self._rssm_core(h, stoch_t, action_seq[:, t])
            prior_stats = self.prior_head(h)
            prior_mean_t, prior_std_t, prior_stoch_t = self._stats_to_dist(prior_stats)
            prior_mean_list.append(prior_mean_t)
            prior_std_list.append(prior_std_t)
            prior_stoch_list.append(prior_stoch_t)
            h_list.append(h)

            mean_t, std_t, stoch_t = self._posterior_from_obs_h(hidden_seq[:, t], h)
            post_mean_list.append(mean_t)
            post_std_list.append(std_t)
            post_stoch_list.append(stoch_t)

        return (
            torch.stack(post_mean_list, dim=1),
            torch.stack(post_std_list, dim=1),
            torch.stack(post_stoch_list, dim=1),
            torch.stack(prior_mean_list, dim=1),
            torch.stack(prior_std_list, dim=1),
            torch.stack(prior_stoch_list, dim=1),
            torch.stack(h_list, dim=1),
        )


__all__ = [
    "TSSMWorldModelTransDreamerDiscrete",
    "TSSMWorldModelRSSMDiscrete",
]
