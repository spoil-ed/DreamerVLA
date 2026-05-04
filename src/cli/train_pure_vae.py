"""
Pure VAE-style autoencoder ablation.

Strips the WM down to:

    image_id (+ optional state tokens) → token_embedder → obs CNN stem → posterior_head
                                                              │
                                      (categorical gumbel-ST or Gaussian sample)
                                                              │
                                         decoder (h=0) → image-token logits
                                         state_decoder → optional state-token logits
                                                              │
                                     CE vs same-frame image/state target tokens

No prior, no dynamics (causal_transformer), no KL, no reward, no continue.

Goal:  verify that posterior_head + decoder can learn an obs ↔ z ↔ obs
identity mapping.  If post_logits_std stays near 0 here, the issue is in the
encoder / decoder / sampling link, not in the dynamics interference.

Run (single card):
    cd /home/user01/liops/workspace/DreamerVLA
    CUDA_VISIBLE_DEVICES=4 python -m src.cli.train_pure_vae \\
        --config-name pretokenize_wm_libero_10_discrete_minimal \\
        --max-steps 5000 --batch-size 32 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ───────────────────────────────────────────────────────────────────


def build_cfg(config_name: str, overrides: list[str]):
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if not isinstance(value, str):
        try:
            return [int(v) for v in list(value)]
        except TypeError:
            pass
    parts = str(value).replace(";", ",").split(",")
    out = [int(part.strip()) for part in parts if part.strip()]
    if not out:
        raise ValueError(f"expected at least one integer, got {value!r}")
    return out


def extract_image_bpe_ids(input_ids_list, which_blocks, n_img_tok_per_block, img_bpe_set):
    from src.utils.wm_image_viz import extract_image_blocks

    if isinstance(which_blocks, int):
        which_blocks = [which_blocks]
    which_blocks = [int(v) for v in which_blocks]
    rows = []
    for sample_idx, seq in enumerate(input_ids_list):
        blocks = extract_image_blocks(list(seq))
        if not blocks:
            raise ValueError(f"sample {sample_idx}: no image block found")
        row: list[int] = []
        for which_block in which_blocks:
            bidx = which_block if which_block >= 0 else len(blocks) + which_block
            if not (0 <= bidx < len(blocks)):
                raise ValueError(
                    f"sample {sample_idx}: which_block={which_block} out of range "
                    f"(have {len(blocks)} blocks)"
                )
            _, _, block_ids = blocks[bidx]
            tok_ids = [int(t) for t in block_ids if int(t) in img_bpe_set]
            if len(tok_ids) != n_img_tok_per_block:
                raise ValueError(
                    f"sample {sample_idx}: selected image block {which_block} has "
                    f"{len(tok_ids)} image tokens, expected {n_img_tok_per_block}"
                )
            row.extend(tok_ids)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def infer_state_token_bounds(encoder: Any) -> tuple[int, int]:
    config = getattr(getattr(encoder, "backbone", None), "config", None)
    vocab_map = getattr(config, "vocabulary_map", None)
    if not isinstance(vocab_map, dict):
        raise ValueError("encoder config does not expose vocabulary_map for state-token ids")
    try:
        start_id = int(vocab_map["<reserved15500>"])
        end_id = int(vocab_map["<reserved16000>"])
    except KeyError as exc:
        raise KeyError("state start/end reserved tokens missing from vocabulary_map") from exc
    if end_id < start_id:
        raise ValueError(f"invalid state token range: {start_id}..{end_id}")
    return start_id, end_id


def extract_state_bpe_ids(
    input_ids_list,
    state_start_id: int,
    state_end_id: int,
    max_state_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    inferred_max = 0
    for seq in input_ids_list:
        tokens = [int(tok) for tok in seq]
        state_tokens: list[int] = []
        idx = 0
        while idx < len(tokens):
            if tokens[idx] != state_start_id:
                idx += 1
                continue
            end = idx + 1
            while end < len(tokens) and tokens[end] != state_end_id:
                end += 1
            if end < len(tokens):
                state_tokens.extend(tokens[idx : end + 1])
                idx = end + 1
            else:
                state_tokens.extend(tokens[idx:])
                break
        rows.append(state_tokens)
        inferred_max = max(inferred_max, len(state_tokens))

    width = inferred_max if max_state_tokens is None else int(max_state_tokens)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for row_idx, row in enumerate(rows):
        if not row or width <= 0:
            continue
        clipped = row[:width]
        row_t = torch.tensor(clipped, dtype=torch.long)
        ids[row_idx, : row_t.numel()] = row_t
        mask[row_idx, : row_t.numel()] = True
    return ids, mask


def collate(batch):
    """Sequence or single-transition format → list of cur image bpe id streams."""
    if "wm_obs_input_ids_seq" in batch[0]:
        # Use the LAST step in each sequence as a single autoencoder example.
        return {"input_ids": [s["wm_obs_input_ids_seq"][-1] for s in batch]}
    return {"input_ids": [list(s["wm_obs_input_ids"]) for s in batch]}


def sample_current_input_ids(sample: dict[str, Any]) -> list[int]:
    if "wm_obs_input_ids_seq" in sample:
        return list(sample["wm_obs_input_ids_seq"][-1])
    return list(sample["wm_obs_input_ids"])


# ── Pure VAE model ────────────────────────────────────────────────────────────


class PureVAE(nn.Module):
    def __init__(
        self,
        num_image_tokens_vocab: int,
        n_image_tokens: int = 256,
        spatial: tuple[int, int] = (16, 16),
        token_embed_dim: int = 512,
        obs_dim: int = 1024,
        stoch_dims: int = 32,
        stoch_categories: int = 32,
        mapper_hidden_dim: int = 512,
        gumbel_temp: float = 1.0,
        unimix: float = 0.01,
        decoder_mid_channels: int = 192,
        decoder_bspace_groups: int = 8,
        decoder_minres: tuple[int, int] = (4, 4),
        decoder_stage_channels: tuple[int, ...] = (96, 48),
        decoder_stoch_hidden: int = 512,
        stem_post_norm: bool = True,
        include_state: bool = False,
        num_state_tokens_vocab: int | None = None,
        max_state_tokens: int = 0,
        state_conditioning_scale: float = 1.0,
        state_decoder_hidden_dim: int = 512,
        obs_encoder_type: str = "conv_stem",
        stem_init_proj_channels: int = 384,
        stem_stage_channels: tuple[int, ...] = (96, 192),
        stem_kernel: int = 4,
        stem_stride: int = 2,
        stem_padding: int = 1,
        dreamer_cnn_depth: int = 64,
        dreamer_cnn_mults: tuple[int, ...] = (2, 3, 4, 4),
        dreamer_cnn_kernel: int = 5,
        dreamer_cnn_layers: int = 1,
        dreamer_cnn_norm: bool = True,
        dreamer_cnn_act: str = "gelu",
        dreamer_cnn_strided: bool = False,
        dreamer_cnn_post_norm: bool = True,
        latent_type: str = "discrete",
        latent_dim: int | None = None,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        from src.models.world_model.image_codec import (
            BspaceConvDecoderHead,
            ConvEncoderStem,
            DreamerCNNEncoderStem,
        )
        from src.models.world_model.tssm import ImageTokenEmbedder

        self.num_image_tokens_vocab = num_image_tokens_vocab
        self.n_image_tokens = n_image_tokens
        self.latent_type = str(latent_type).lower()
        if self.latent_type not in {"discrete", "gaussian"}:
            raise ValueError(f"latent_type must be 'discrete' or 'gaussian', got {latent_type!r}")
        self.stoch_dims = int(stoch_dims)
        self.stoch_categories = int(stoch_categories)
        self.gumbel_temp = float(gumbel_temp)
        self.unimix = float(unimix)
        self.latent_dim = (
            int(latent_dim)
            if latent_dim is not None
            else self.stoch_dims * self.stoch_categories
        )
        self.min_std = float(min_std)
        self.flat_z_dim = (
            self.stoch_dims * self.stoch_categories
            if self.latent_type == "discrete"
            else self.latent_dim
        )
        self.include_state = bool(include_state)
        self.num_state_tokens_vocab = (
            int(num_state_tokens_vocab) if num_state_tokens_vocab is not None else 0
        )
        self.max_state_tokens = int(max_state_tokens)
        self.state_conditioning_scale = float(state_conditioning_scale)
        self.obs_encoder_type = str(obs_encoder_type).lower()
        if self.obs_encoder_type not in {"conv_stem", "dreamer_cnn"}:
            raise ValueError(
                "obs_encoder_type must be one of {'conv_stem', 'dreamer_cnn'}, "
                f"got {obs_encoder_type!r}"
            )

        # Encoder
        self.token_embedder = ImageTokenEmbedder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            d_embed=token_embed_dim,
            spatial=spatial,
        )
        if self.include_state:
            if self.num_state_tokens_vocab <= 0:
                raise ValueError("include_state=True requires num_state_tokens_vocab > 0")
            if self.max_state_tokens <= 0:
                raise ValueError("include_state=True requires max_state_tokens > 0")
            self.state_token_embedder = nn.Embedding(
                self.num_state_tokens_vocab,
                token_embed_dim,
            )
            self.state_context_proj = nn.Sequential(
                nn.LayerNorm(token_embed_dim),
                nn.Linear(token_embed_dim, token_embed_dim),
                nn.GELU(),
                nn.Linear(token_embed_dim, token_embed_dim),
            )
        else:
            self.state_token_embedder = None
            self.state_context_proj = None
        if self.obs_encoder_type == "dreamer_cnn":
            self.conv_stem = DreamerCNNEncoderStem(
                in_channels=token_embed_dim,
                spatial=spatial,
                obs_dim=obs_dim,
                depth=dreamer_cnn_depth,
                mults=tuple(dreamer_cnn_mults),
                kernel=dreamer_cnn_kernel,
                layers=dreamer_cnn_layers,
                norm=dreamer_cnn_norm,
                act=dreamer_cnn_act,
                strided=dreamer_cnn_strided,
                post_norm=dreamer_cnn_post_norm,
            )
        else:
            self.conv_stem = ConvEncoderStem(
                in_channels=token_embed_dim,
                spatial=spatial,
                obs_dim=obs_dim,
                init_proj_channels=stem_init_proj_channels,
                stage_channels=tuple(stem_stage_channels),
                kernel=stem_kernel,
                stride=stem_stride,
                padding=stem_padding,
                post_norm=stem_post_norm,
            )
        # Posterior head: obs -> categorical logits or Gaussian mean/std stats.
        posterior_out_dim = self.flat_z_dim if self.latent_type == "discrete" else 2 * self.flat_z_dim
        self.posterior_head = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(mapper_hidden_dim, posterior_out_dim),
        )
        # Decoder: pure z → token logits.  We pass zero h.
        d_deter = obs_dim   # arbitrary; we feed zeros
        self.d_deter = d_deter
        self.decoder = BspaceConvDecoderHead(
            deter_dim=d_deter,
            stoch_dim=self.flat_z_dim,
            minres=decoder_minres,
            mid_channels=decoder_mid_channels,
            bspace_groups=decoder_bspace_groups,
            stage_channels=decoder_stage_channels,
            out_channels=num_image_tokens_vocab,
            out_spatial=spatial,
            stoch_hidden=decoder_stoch_hidden,
        )
        if self.include_state:
            self.state_decoder = nn.Sequential(
                nn.LayerNorm(self.flat_z_dim),
                nn.Linear(self.flat_z_dim, state_decoder_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    state_decoder_hidden_dim,
                    self.max_state_tokens * self.num_state_tokens_vocab,
                ),
            )
        else:
            self.state_decoder = None

    def _sample_z(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (sampled_z, distribution_stats_for_metrics)."""
        if self.latent_type == "gaussian":
            mean, raw_std = stats.chunk(2, dim=-1)
            std = F.softplus(raw_std) + self.min_std
            if self.training:
                sample = mean + std * torch.randn_like(std)
            else:
                sample = mean
            return sample, torch.cat([mean, std], dim=-1)

        S, K = self.stoch_dims, self.stoch_categories
        x = stats.reshape(*stats.shape[:-1], S, K)
        if self.unimix > 0.0:
            probs = F.softmax(x, dim=-1)
            uniform = torch.full_like(probs, 1.0 / K)
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
            x = probs.clamp_min(1e-8).log()
        if self.training:
            # Straight-through Gumbel-softmax.
            soft = F.gumbel_softmax(x, tau=self.gumbel_temp, hard=False, dim=-1)
            idx = soft.argmax(dim=-1)
            hard = F.one_hot(idx, num_classes=K).to(soft.dtype)
            sample = hard - soft.detach() + soft   # ST trick
        else:
            idx = x.argmax(dim=-1)
            sample = F.one_hot(idx, num_classes=K).to(x.dtype)
        return sample.reshape(*sample.shape[:-2], S * K), x.reshape(*x.shape[:-2], S * K)

    def _state_context_from_indices(
        self,
        state_idx: torch.Tensor | None,
        state_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            not self.include_state
            or self.state_token_embedder is None
            or self.state_context_proj is None
        ):
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if state_idx is None:
            return torch.zeros(batch_size, self.token_embedder.d_embed, device=device, dtype=dtype)
        state_idx = state_idx.to(device=device, dtype=torch.long)
        if state_mask is None:
            state_mask = torch.ones_like(state_idx, dtype=torch.bool)
        else:
            state_mask = state_mask.to(device=device, dtype=torch.bool)
        if state_idx.shape != state_mask.shape:
            raise ValueError(f"state_idx and state_mask shapes differ: {state_idx.shape} vs {state_mask.shape}")

        valid_range = (state_idx >= 0) & (state_idx < self.num_state_tokens_vocab)
        invalid_visible = state_mask & ~valid_range
        if invalid_visible.any():
            bad = state_idx[invalid_visible][:8].detach().cpu().tolist()
            raise ValueError(f"state_idx contains ids outside configured range: {bad}")

        safe_idx = state_idx.clamp(min=0, max=self.num_state_tokens_vocab - 1)
        emb = self.state_token_embedder(safe_idx).to(dtype=dtype)
        valid = state_mask & valid_range
        weights = valid.to(dtype=emb.dtype).unsqueeze(-1)
        denom = weights.sum(dim=-2).clamp_min(1.0)
        context = (emb * weights).sum(dim=-2) / denom
        context = self.state_context_proj(context)
        has_state = valid.any(dim=-1, keepdim=True)
        context = torch.where(has_state, context, torch.zeros_like(context))
        return context * self.state_conditioning_scale

    def forward(
        self,
        image_idx: torch.Tensor,
        state_idx: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
    ):
        """image_idx: [B, N_img] image-vocab ids (0..num_image_tokens_vocab-1)"""
        embed = self.token_embedder(image_idx)        # [B, N_img, d_embed]
        if self.include_state:
            state_context = self._state_context_from_indices(
                state_idx,
                state_mask,
                batch_size=image_idx.shape[0],
                device=image_idx.device,
                dtype=embed.dtype,
            )
            embed = embed + state_context.unsqueeze(-2)
        obs = self.conv_stem(embed)                    # [B, obs_dim]
        post_stats = self.posterior_head(obs)          # [B, S*K] or [B, 2*D]
        z, post_metric_stats = self._sample_z(post_stats)
        h_zero = torch.zeros(z.shape[0], self.d_deter, device=z.device, dtype=z.dtype)
        decoded = self.decoder(h_zero, z)              # [B, N_img, num_image_tokens_vocab]
        state_logits = None
        if self.state_decoder is not None:
            state_logits = self.state_decoder(z).view(
                z.shape[0],
                self.max_state_tokens,
                self.num_state_tokens_vocab,
            )
        return decoded, state_logits, post_stats, post_metric_stats


def categorical_metrics(logits_flat: torch.Tensor, S: int, K: int) -> dict[str, float]:
    """Compute entropy / max_prob / std / pairwise_cos / eff_rank for monitoring."""
    out: dict[str, float] = {}
    flat = logits_flat.detach().float()
    # std across batch dim (mean over feature)
    out["logits_std"] = float(flat.std(dim=0, unbiased=False).mean())
    out["logits_mean_abs"] = float(flat.abs().mean())
    # categorical entropy
    x = flat.reshape(*flat.shape[:-1], S, K)
    log_p = F.log_softmax(x, dim=-1)
    p = log_p.exp()
    ent = -(p * log_p).sum(dim=-1).mean()
    out["entropy"] = float(ent)
    out["max_prob"] = float(p.max(dim=-1).values.mean())
    # pairwise cos
    N = flat.shape[0]
    if N >= 2:
        norms = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = flat / norms
        cos = unit @ unit.T
        triu = torch.triu_indices(N, N, offset=1, device=flat.device)
        out["pairwise_cos"] = float(cos[triu[0], triu[1]].mean())
    return out


def gaussian_metrics(mean_std_flat: torch.Tensor, latent_dim: int) -> dict[str, float]:
    """Compute Gaussian-posterior diagnostics with keys compatible with logs."""
    out: dict[str, float] = {}
    flat = mean_std_flat.detach().float()
    mean, std = flat[..., :latent_dim], flat[..., latent_dim:]
    std = std.clamp_min(1e-8)
    out["logits_std"] = float(mean.std(dim=0, unbiased=False).mean())
    out["logits_mean_abs"] = float(mean.abs().mean())
    out["std_mean"] = float(std.mean())
    out["std_min"] = float(std.min())
    out["std_max"] = float(std.max())
    out["mean_rms"] = float(mean.pow(2).mean().sqrt())
    entropy = 0.5 * torch.log(2.0 * math.pi * math.e * std.square())
    out["entropy"] = float(entropy.mean())
    # Keep this historical log key populated; for Gaussian it is max density.
    out["max_prob"] = float((1.0 / (math.sqrt(2.0 * math.pi) * std)).mean())
    N = mean.shape[0]
    if N >= 2:
        norms = mean.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = mean / norms
        cos = unit @ unit.T
        triu = torch.triu_indices(N, N, offset=1, device=mean.device)
        out["pairwise_cos"] = float(cos[triu[0], triu[1]].mean())
    return out


def latent_metrics(dist_stats: torch.Tensor, model: PureVAE) -> dict[str, float]:
    if getattr(model, "latent_type", "discrete") == "gaussian":
        return gaussian_metrics(dist_stats, model.latent_dim)
    return categorical_metrics(dist_stats, model.stoch_dims, model.stoch_categories)


# ── Train loop ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pretokenize_wm_libero_10_discrete_minimal")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Wrap the pure VAE in torch.nn.DataParallel across visible CUDA devices.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--which-blocks", default=None,
                        help="Comma-separated image block indices. Example: -2,-1 for third+wrist.")
    parser.add_argument("--tokens-per-image-block", type=int, default=None,
                        help="Image VQ tokens per selected block. Defaults to config world_model.n_image_tokens for one block, or 256 for multi-view.")
    parser.add_argument("--spatial-grid", default=None,
                        help="Override spatial grid as H,W. For third+wrist 512 tokens use 32,16.")
    parser.add_argument("--decoder-minres", default=None,
                        help="Override decoder bottleneck as H,W. For spatial 32,16 with two stride-2 stages use 8,4.")
    parser.add_argument("--include-state", action="store_true",
                        help="Condition the VAE encoder on state tokens and reconstruct the same state-token block.")
    parser.add_argument("--state-loss-coef", type=float, default=1.0)
    parser.add_argument("--max-state-tokens", type=int, default=0,
                        help="0 = infer from the dataset scan.")
    parser.add_argument("--state-scan-samples", type=int, default=512)
    parser.add_argument("--state-decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--viz-every", type=int, default=500,
                        help="Save reconstruction PNG grid every N steps (0 to disable)")
    parser.add_argument("--latent-type", choices=["discrete", "gaussian"], default=None,
                        help="Override latent family. Defaults to config target: Gaussian for non-discrete TSSM, else discrete.")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Gaussian latent width. Defaults to world_model.latent_dim.")
    parser.add_argument("--min-std", type=float, default=None,
                        help="Minimum Gaussian posterior std. Defaults to world_model.min_std.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    cfg = build_cfg(args.config_name, [])

    # Encoder just for vocab mapping.
    print("[vae] loading encoder for vocab mapping ...")
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    vocab_mapping = encoder.backbone.model.vocabulary_mapping
    img_bpe_set = set(vocab_mapping.bpe2img.keys())
    image_token_bpe_ids = torch.tensor(sorted(img_bpe_set), dtype=torch.long, device=device)
    n_image_tokens_vocab = len(img_bpe_set)
    full_vocab_size = int(getattr(encoder.backbone.lm_head, "out_features",
                                  encoder.backbone.lm_head.weight.shape[0]))
    bpe_to_imgidx = torch.full((full_vocab_size,), -1, dtype=torch.long, device=device)
    bpe_to_imgidx[image_token_bpe_ids] = torch.arange(n_image_tokens_vocab, device=device)
    state_start_id = state_end_id = None
    num_state_tokens_vocab = 0
    if args.include_state:
        state_start_id, state_end_id = infer_state_token_bounds(encoder)
        num_state_tokens_vocab = state_end_id - state_start_id + 1

    # Save vqgan handles before freeing encoder, for image viz.
    vq_model = None
    if args.viz_every > 0:
        from src.utils.vq_image_decoder import load_vq_model, build_bpe2vq_tensor
        vq_model = load_vq_model(
            cfg_path=cfg.encoder.chameleon_vqgan_config,
            ckpt_path=cfg.encoder.chameleon_vqgan_ckpt,
            device=device,
        )
        bpe2vq_table = build_bpe2vq_tensor(vocab_mapping).to(device)
        # imgidx_to_bpe table: maps post-decode 0..n_image_tokens_vocab → original BPE id
        imgidx_to_bpe = image_token_bpe_ids.clone()
    else:
        bpe2vq_table = None
        imgidx_to_bpe = None

    # Free encoder GPU memory.
    encoder = encoder.cpu()
    torch.cuda.empty_cache()
    del encoder

    # Dataset
    print("[vae] building dataset ...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    print(f"[vae]   dataset size = {len(dataset)}")

    which_blocks = (
        parse_int_list(args.which_blocks)
        if args.which_blocks is not None
        else parse_int_list(
            OmegaConf.select(
                cfg,
                "viz.which_blocks",
                default=[int(OmegaConf.select(cfg, "viz.which_block", default=-2))],
            )
        )
    )
    n_views = len(which_blocks)
    cfg_n_image_tokens = int(OmegaConf.select(cfg, "world_model.n_image_tokens", default=256))
    tokens_per_block = (
        int(args.tokens_per_image_block)
        if args.tokens_per_image_block is not None
        else (cfg_n_image_tokens if n_views == 1 else 256)
    )
    n_image_tokens = tokens_per_block * n_views

    spatial = (
        tuple(parse_int_list(args.spatial_grid))
        if args.spatial_grid is not None
        else tuple(OmegaConf.select(cfg, "world_model.spatial_grid", default=[16, 16]))
    )
    if len(spatial) != 2:
        raise ValueError(f"--spatial-grid must be H,W, got {spatial}")
    if int(spatial[0]) * int(spatial[1]) != n_image_tokens:
        if args.spatial_grid is not None:
            raise ValueError(
                f"spatial grid {spatial} has product {spatial[0] * spatial[1]}, "
                f"but selected image tokens are {n_image_tokens}"
            )
        side = int(np.sqrt(tokens_per_block))
        if side * side != tokens_per_block:
            raise ValueError(
                "Cannot infer a rectangular multi-view spatial grid because "
                f"tokens_per_block={tokens_per_block} is not a square."
            )
        spatial = (side * n_views, side)

    decoder_stage_channels = tuple(OmegaConf.select(cfg, "world_model.decoder_stage_channels", default=[96, 48]))
    decoder_minres = (
        tuple(parse_int_list(args.decoder_minres))
        if args.decoder_minres is not None
        else tuple(OmegaConf.select(cfg, "world_model.decoder_minres", default=[4, 4]))
    )
    if len(decoder_minres) != 2:
        raise ValueError(f"--decoder-minres must be H,W, got {decoder_minres}")
    decoder_stride = int(OmegaConf.select(cfg, "world_model.decoder_stride", default=2))
    upsample_factor = decoder_stride ** len(decoder_stage_channels)
    expected_spatial = (
        int(decoder_minres[0]) * upsample_factor,
        int(decoder_minres[1]) * upsample_factor,
    )
    if expected_spatial != tuple(spatial):
        if args.decoder_minres is not None:
            raise ValueError(
                f"decoder_minres={decoder_minres} with {len(decoder_stage_channels)} "
                f"stride-{decoder_stride} stages reaches {expected_spatial}, "
                f"but spatial={spatial}"
            )
        if int(spatial[0]) % upsample_factor != 0 or int(spatial[1]) % upsample_factor != 0:
            raise ValueError(
                f"spatial={spatial} is not divisible by decoder upsample factor {upsample_factor}"
            )
        decoder_minres = (int(spatial[0]) // upsample_factor, int(spatial[1]) // upsample_factor)

    max_state_tokens = 0
    if args.include_state:
        assert state_start_id is not None and state_end_id is not None
        max_state_tokens = int(args.max_state_tokens)
        if max_state_tokens <= 0:
            scan_n = min(int(args.state_scan_samples), len(dataset))
            for scan_idx in range(scan_n):
                ids = sample_current_input_ids(dataset[scan_idx])
                _, state_mask = extract_state_bpe_ids(
                    [ids],
                    state_start_id,
                    state_end_id,
                    max_state_tokens=None,
                )
                max_state_tokens = max(max_state_tokens, int(state_mask.shape[-1]))
        if max_state_tokens <= 0:
            raise ValueError("--include-state was set, but no state-token block was found")
        print(
            f"[vae]   state tokens = {state_start_id}..{state_end_id} "
            f"(vocab={num_state_tokens_vocab}, width={max_state_tokens})"
        )

    print(
        f"[vae]   image blocks = {which_blocks}; "
        f"tokens_per_block={tokens_per_block}; total_tokens={n_image_tokens}; "
        f"spatial={tuple(spatial)}; decoder_minres={tuple(decoder_minres)}"
    )
    obs_encoder_type = str(OmegaConf.select(cfg, "world_model.obs_encoder_type", default="conv_stem"))
    print(f"[vae]   obs encoder = {obs_encoder_type}")
    wm_target = str(OmegaConf.select(cfg, "world_model._target_", default=""))
    latent_type = args.latent_type
    if latent_type is None:
        latent_type = str(OmegaConf.select(cfg, "world_model.latent_type", default="")).lower()
    if not latent_type:
        latent_type = "gaussian" if wm_target.endswith("TSSMWorldModelTransDreamer") else "discrete"
    latent_dim = (
        int(args.latent_dim)
        if args.latent_dim is not None
        else int(OmegaConf.select(cfg, "world_model.latent_dim", default=1024))
    )
    min_std = (
        float(args.min_std)
        if args.min_std is not None
        else float(OmegaConf.select(cfg, "world_model.min_std", default=0.1))
    )
    print(f"[vae]   latent = {latent_type} (dim={latent_dim}, min_std={min_std})")

    # Build pure VAE
    print("[vae] building model ...")
    model_core = PureVAE(
        num_image_tokens_vocab=n_image_tokens_vocab,
        n_image_tokens=n_image_tokens,
        spatial=tuple(spatial),
        token_embed_dim=int(OmegaConf.select(cfg, "world_model.token_embed_dim", default=512)),
        obs_dim=int(OmegaConf.select(cfg, "world_model.obs_dim", default=1024)),
        stoch_dims=int(OmegaConf.select(cfg, "world_model.stoch_dims", default=32)),
        stoch_categories=int(OmegaConf.select(cfg, "world_model.stoch_categories", default=32)),
        mapper_hidden_dim=int(OmegaConf.select(cfg, "world_model.mapper_hidden_dim", default=512)),
        gumbel_temp=float(OmegaConf.select(cfg, "world_model.gumbel_temp", default=1.0)),
        unimix=float(OmegaConf.select(cfg, "world_model.unimix", default=0.01)),
        decoder_mid_channels=int(OmegaConf.select(cfg, "world_model.decoder_mid_channels", default=192)),
        decoder_bspace_groups=int(OmegaConf.select(cfg, "world_model.decoder_bspace_groups", default=8)),
        decoder_minres=tuple(decoder_minres),
        decoder_stage_channels=decoder_stage_channels,
        decoder_stoch_hidden=int(OmegaConf.select(cfg, "world_model.decoder_stoch_hidden", default=512)),
        stem_post_norm=bool(OmegaConf.select(cfg, "world_model.stem_post_norm", default=False)),
        obs_encoder_type=obs_encoder_type,
        stem_init_proj_channels=int(OmegaConf.select(cfg, "world_model.stem_init_proj_channels", default=384)),
        stem_stage_channels=tuple(OmegaConf.select(cfg, "world_model.stem_stage_channels", default=[96, 192])),
        stem_kernel=int(OmegaConf.select(cfg, "world_model.stem_kernel", default=4)),
        stem_stride=int(OmegaConf.select(cfg, "world_model.stem_stride", default=2)),
        stem_padding=int(OmegaConf.select(cfg, "world_model.stem_padding", default=1)),
        dreamer_cnn_depth=int(OmegaConf.select(cfg, "world_model.dreamer_cnn_depth", default=64)),
        dreamer_cnn_mults=tuple(OmegaConf.select(cfg, "world_model.dreamer_cnn_mults", default=[2, 3, 4, 4])),
        dreamer_cnn_kernel=int(OmegaConf.select(cfg, "world_model.dreamer_cnn_kernel", default=5)),
        dreamer_cnn_layers=int(OmegaConf.select(cfg, "world_model.dreamer_cnn_layers", default=1)),
        dreamer_cnn_norm=bool(OmegaConf.select(cfg, "world_model.dreamer_cnn_norm", default=True)),
        dreamer_cnn_act=str(OmegaConf.select(cfg, "world_model.dreamer_cnn_act", default="gelu")),
        dreamer_cnn_strided=bool(OmegaConf.select(cfg, "world_model.dreamer_cnn_strided", default=False)),
        dreamer_cnn_post_norm=bool(OmegaConf.select(cfg, "world_model.dreamer_cnn_post_norm", default=True)),
        include_state=bool(args.include_state),
        num_state_tokens_vocab=num_state_tokens_vocab if args.include_state else None,
        max_state_tokens=max_state_tokens,
        state_conditioning_scale=float(OmegaConf.select(cfg, "world_model.state_conditioning_scale", default=1.0)),
        state_decoder_hidden_dim=int(args.state_decoder_hidden_dim),
        latent_type=latent_type,
        latent_dim=latent_dim,
        min_std=min_std,
    ).to(device)
    model: nn.Module = model_core
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data-parallel requires a CUDA device")
        num_cuda = torch.cuda.device_count()
        if num_cuda < 2:
            print("[vae]   --data-parallel requested, but only one visible CUDA device; using single GPU.")
        else:
            model = nn.DataParallel(model_core)
            print(f"[vae]   data parallel = {num_cuda} visible CUDA devices")
    n_params = sum(p.numel() for p in model_core.parameters() if p.requires_grad)
    print(f"[vae]   trainable params = {n_params:,}")

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-6)

    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=2, collate_fn=collate, drop_last=True,
    )

    # Output dir
    from datetime import datetime
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("/home/user01/liops/workspace/DreamerVLA/data/outputs/worldmodel/vae/pure_vae") / datetime.now().strftime("pure_vae_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vae_logs.json.txt"
    print(f"[vae] out_dir = {out_dir}")

    @torch.no_grad()
    def save_viz_grid(step: int, gt_img_idx: torch.Tensor, pred_img_idx: torch.Tensor,
                      n_show: int = 4):
        """Save side-by-side [GT | recon] decoded to pixels via VQGAN."""
        if vq_model is None:
            return
        from src.utils.vq_image_decoder import vq_tokens_to_pixels, tensor_to_pil
        from PIL import Image, ImageDraw
        gt = gt_img_idx[:n_show]                       # [n, 256] image-vocab
        pr = pred_img_idx[:n_show]
        side = int(np.sqrt(tokens_per_block))
        if side * side != tokens_per_block:
            print(f"[viz] step {step}: skipped (tokens_per_block={tokens_per_block} is not square)")
            return

        gt_pix_by_view = []
        pr_pix_by_view = []
        for view_idx in range(n_views):
            start = view_idx * tokens_per_block
            end = start + tokens_per_block
            gt_bpe = imgidx_to_bpe[gt[:, start:end]]    # → BPE ids
            pr_bpe = imgidx_to_bpe[pr[:, start:end]]
            gt_vq = bpe2vq_table[gt_bpe]                # → VQ codebook ids
            pr_vq = bpe2vq_table[pr_bpe]
            if (gt_vq < 0).any() or (pr_vq < 0).any():
                print(f"[viz] step {step}: skipped (non-image bpe in tokens)")
                return
            gt_pix_by_view.append(vq_tokens_to_pixels(gt_vq, vq_model, h_latent=side, w_latent=side))
            pr_pix_by_view.append(vq_tokens_to_pixels(pr_vq, vq_model, h_latent=side, w_latent=side))

        # Build an n_show × (GT/recon per view) grid.
        cell = 256
        canvas = Image.new("RGB", (cell * 2 * n_views, cell * n_show + 24), (32, 32, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4),  f"step {step}  |  per view: GT / recon", fill=(230, 230, 230))
        for i in range(n_show):
            for view_idx in range(n_views):
                x = view_idx * cell * 2
                canvas.paste(
                    tensor_to_pil(gt_pix_by_view[view_idx][i]).convert("RGB").resize((cell, cell)),
                    (x, 24 + i * cell),
                )
                canvas.paste(
                    tensor_to_pil(pr_pix_by_view[view_idx][i]).convert("RGB").resize((cell, cell)),
                    (x + cell, 24 + i * cell),
                )
        path = out_dir / "viz" / f"step_{step:06d}.png"
        canvas.save(path)
        print(f"[viz] step {step}: saved {path.name}")

    # Train
    print(f"[vae] training for {args.max_steps} steps ...")
    model.train()
    step = 0
    log_handle = open(log_path, "w")
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break
            input_ids_list = batch["input_ids"]
            if args.lr_warmup_steps > 0:
                lr_scale = min(1.0, float(step + 1) / float(args.lr_warmup_steps))
                for group in opt.param_groups:
                    group["lr"] = args.lr * lr_scale
            current_lr = float(opt.param_groups[0]["lr"])

            obs_emb_bpe = extract_image_bpe_ids(
                input_ids_list, which_blocks, tokens_per_block, img_bpe_set,
            ).to(device)                                 # [B, N_img]
            img_idx = bpe_to_imgidx[obs_emb_bpe]
            if (img_idx < 0).any():
                raise ValueError("non-image bpe id in input")

            state_idx = state_mask = None
            if args.include_state:
                assert state_start_id is not None and state_end_id is not None
                state_bpe, state_mask = extract_state_bpe_ids(
                    input_ids_list,
                    state_start_id,
                    state_end_id,
                    max_state_tokens=max_state_tokens,
                )
                state_bpe = state_bpe.to(device)
                state_mask = state_mask.to(device)
                state_idx = state_bpe - int(state_start_id)
                state_idx = state_idx.clamp(min=0, max=num_state_tokens_vocab - 1)

            decoded_logits, state_logits, _post_stats, post_metric_stats = model(
                img_idx,
                state_idx=state_idx,
                state_mask=state_mask,
            )
            target = img_idx                              # autoencoder: target = input
            image_loss = F.cross_entropy(
                decoded_logits.reshape(-1, decoded_logits.shape[-1]),
                target.reshape(-1),
            )
            state_loss = decoded_logits.new_zeros(())
            state_acc = decoded_logits.new_zeros(())
            if args.include_state:
                if state_logits is None or state_idx is None or state_mask is None:
                    raise RuntimeError("include_state=True but state logits/targets were not produced")
                state_ce = F.cross_entropy(
                    state_logits.reshape(-1, state_logits.shape[-1]),
                    state_idx.reshape(-1),
                    reduction="none",
                ).view_as(state_idx)
                weights = state_mask.to(dtype=state_ce.dtype)
                denom = weights.sum().clamp_min(1.0)
                state_loss = (state_ce * weights).sum() / denom
                with torch.no_grad():
                    state_pred = state_logits.argmax(dim=-1)
                    state_acc = ((state_pred == state_idx) & state_mask).to(torch.float32).sum() / denom

            loss = image_loss + float(args.state_loss_coef) * state_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            opt.step()

            need_log = args.log_every > 0 and step % args.log_every == 0
            need_viz = args.viz_every > 0 and step % args.viz_every == 0
            if need_log or need_viz:
                with torch.no_grad():
                    pred = decoded_logits.argmax(dim=-1)
                    acc = (pred == target).float().mean()
                    pred_unique = pred.unique().numel()
                    latent_m = latent_metrics(post_metric_stats, model_core)

            # Image viz at viz_every cadence
            if need_viz:
                save_viz_grid(step, target, pred, n_show=4)
            if need_log:
                row = {
                    "step": step,
                    "loss": float(loss),
                    "image_loss": float(image_loss),
                    "state_loss": float(state_loss),
                    "state_acc": float(state_acc),
                    "recon_acc": float(acc),
                    "pred_unique_tokens": int(pred_unique),
                    "grad_norm": float(grad_norm),
                    "lr": current_lr,
                    **{f"post_{k}": v for k, v in latent_m.items()},
                }
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
                if model_core.latent_type == "gaussian":
                    print(
                        f"step {step:5d}  loss={row['loss']:.3f}  "
                        f"img={row['image_loss']:.3f}  state={row['state_loss']:.3f}  "
                        f"acc={row['recon_acc']:.4f}  s_acc={row['state_acc']:.4f}  "
                        f"uniq={row['pred_unique_tokens']:3d}  "
                        f"post_mean_std={row['post_logits_std']:.5f}  "
                        f"post_std={row['post_std_mean']:.3f}  "
                        f"post_ent={row['post_entropy']:.3f}  "
                        f"post_cos={row.get('post_pairwise_cos', float('nan')):.4f}  "
                        f"lr={row['lr']:.2e}  |g|={row['grad_norm']:.2f}"
                    )
                else:
                    print(
                        f"step {step:5d}  loss={row['loss']:.3f}  "
                        f"img={row['image_loss']:.3f}  state={row['state_loss']:.3f}  "
                        f"acc={row['recon_acc']:.4f}  s_acc={row['state_acc']:.4f}  "
                        f"uniq={row['pred_unique_tokens']:3d}  "
                        f"post_std={row['post_logits_std']:.5f}  "
                        f"post_ent={row['post_entropy']:.3f}  "
                        f"post_max_p={row['post_max_prob']:.3f}  "
                        f"post_cos={row.get('post_pairwise_cos', float('nan')):.4f}  "
                        f"lr={row['lr']:.2e}  |g|={row['grad_norm']:.2f}"
                    )
            step += 1

    log_handle.close()
    # Final checkpoint
    ckpt_path = out_dir / "final.ckpt"
    torch.save(
        {
            "model": model_core.state_dict(),
            "step": step,
            "meta": {
                "which_blocks": which_blocks,
                "tokens_per_block": tokens_per_block,
                "n_image_tokens": n_image_tokens,
                "spatial": list(spatial),
                "decoder_minres": list(decoder_minres),
                "include_state": bool(args.include_state),
                "state_start_id": state_start_id,
                "state_end_id": state_end_id,
                "num_state_tokens_vocab": num_state_tokens_vocab,
                "max_state_tokens": max_state_tokens,
                "state_loss_coef": float(args.state_loss_coef),
                "lr_warmup_steps": int(args.lr_warmup_steps),
                "latent_type": model_core.latent_type,
                "latent_dim": int(model_core.latent_dim),
                "flat_z_dim": int(model_core.flat_z_dim),
                "min_std": float(model_core.min_std),
                "stoch_dims": int(model_core.stoch_dims),
                "stoch_categories": int(model_core.stoch_categories),
                "obs_encoder_type": obs_encoder_type,
                "dreamer_cnn_depth": int(OmegaConf.select(cfg, "world_model.dreamer_cnn_depth", default=64)),
                "dreamer_cnn_mults": list(OmegaConf.select(cfg, "world_model.dreamer_cnn_mults", default=[2, 3, 4, 4])),
                "dreamer_cnn_kernel": int(OmegaConf.select(cfg, "world_model.dreamer_cnn_kernel", default=5)),
                "dreamer_cnn_layers": int(OmegaConf.select(cfg, "world_model.dreamer_cnn_layers", default=1)),
                "dreamer_cnn_norm": bool(OmegaConf.select(cfg, "world_model.dreamer_cnn_norm", default=True)),
                "dreamer_cnn_act": str(OmegaConf.select(cfg, "world_model.dreamer_cnn_act", default="gelu")),
                "dreamer_cnn_strided": bool(OmegaConf.select(cfg, "world_model.dreamer_cnn_strided", default=False)),
                "dreamer_cnn_post_norm": bool(OmegaConf.select(cfg, "world_model.dreamer_cnn_post_norm", default=True)),
            },
        },
        ckpt_path,
    )
    # Final viz
    if args.viz_every > 0:
        with torch.no_grad():
            decoded_logits, _state_logits, _post_stats, _post_metric_stats = model(
                img_idx,
                state_idx=state_idx,
                state_mask=state_mask,
            )
            pred = decoded_logits.argmax(dim=-1)
            save_viz_grid(step, img_idx, pred, n_show=8)
    print(f"\n[vae] done.")
    print(f"  log    → {log_path}")
    print(f"  ckpt   → {ckpt_path}")
    print(f"  viz    → {out_dir / 'viz'}")


if __name__ == "__main__":
    main()
