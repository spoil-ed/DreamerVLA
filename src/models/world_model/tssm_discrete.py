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


__all__ = ["TSSMWorldModelTransDreamerDiscrete"]
