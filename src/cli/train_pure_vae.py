"""
Pure VAE-style autoencoder ablation.

Strips the WM down to:

    image_id → token_embedder → conv_stem → posterior_head
                                                   │
                                            (categorical sample, gumbel-ST)
                                                   │
                                       decoder (h=0) → token logits
                                                   │
                                     CE vs same-frame target tokens

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


def extract_image_bpe_ids(input_ids_list, which_block, n_img_tok, img_bpe_set):
    from src.utils.wm_image_viz import extract_image_blocks
    rows = []
    for seq in input_ids_list:
        blocks = extract_image_blocks(list(seq))
        bidx = which_block if which_block >= 0 else len(blocks) + which_block
        _, _, block_ids = blocks[bidx]
        tok_ids = [int(t) for t in block_ids if int(t) in img_bpe_set]
        if len(tok_ids) != n_img_tok:
            raise ValueError(f"got {len(tok_ids)} image tokens, expected {n_img_tok}")
        rows.append(tok_ids)
    return torch.tensor(rows, dtype=torch.long)


def collate(batch):
    """Sequence or single-transition format → list of cur image bpe id streams."""
    if "wm_obs_input_ids_seq" in batch[0]:
        # Use the LAST step in each sequence as a single autoencoder example.
        return {"input_ids": [s["wm_obs_input_ids_seq"][-1] for s in batch]}
    return {"input_ids": [list(s["wm_obs_input_ids"]) for s in batch]}


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
    ) -> None:
        super().__init__()
        from src.models.world_model.image_codec import ConvEncoderStem, BspaceConvDecoderHead
        from src.models.world_model.tssm import ImageTokenEmbedder

        self.num_image_tokens_vocab = num_image_tokens_vocab
        self.n_image_tokens = n_image_tokens
        self.stoch_dims = stoch_dims
        self.stoch_categories = stoch_categories
        self.gumbel_temp = gumbel_temp
        self.unimix = unimix
        self.flat_z_dim = stoch_dims * stoch_categories

        # Encoder
        self.token_embedder = ImageTokenEmbedder(
            num_image_tokens_vocab=num_image_tokens_vocab,
            d_embed=token_embed_dim,
            spatial=spatial,
        )
        self.conv_stem = ConvEncoderStem(
            in_channels=token_embed_dim,
            spatial=spatial,
            obs_dim=obs_dim,
            post_norm=stem_post_norm,
        )
        # Posterior head: obs → categorical logits over (stoch_dims, stoch_categories)
        self.posterior_head = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(mapper_hidden_dim, self.flat_z_dim),
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

    def _sample_z(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_one_hot_flat, logits_with_unimix)."""
        S, K = self.stoch_dims, self.stoch_categories
        x = logits.reshape(*logits.shape[:-1], S, K)
        if self.unimix > 0.0:
            probs = F.softmax(x, dim=-1)
            uniform = torch.full_like(probs, 1.0 / K)
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
            x = probs.clamp_min(1e-8).log()
        if self.training:
            # Straight-through Gumbel-softmax
            soft = F.gumbel_softmax(x, tau=self.gumbel_temp, hard=False, dim=-1)
            idx = soft.argmax(dim=-1)
            hard = F.one_hot(idx, num_classes=K).to(soft.dtype)
            sample = hard - soft.detach() + soft   # ST trick
        else:
            idx = x.argmax(dim=-1)
            sample = F.one_hot(idx, num_classes=K).to(x.dtype)
        return sample.reshape(*sample.shape[:-2], S * K), x.reshape(*x.shape[:-2], S * K)

    def forward(self, image_idx: torch.Tensor):
        """image_idx: [B, N_img] image-vocab ids (0..num_image_tokens_vocab-1)"""
        embed = self.token_embedder(image_idx)        # [B, N_img, d_embed]
        obs = self.conv_stem(embed)                    # [B, obs_dim]
        post_logits = self.posterior_head(obs)         # [B, S*K]
        z, post_logits_unimix = self._sample_z(post_logits)
        h_zero = torch.zeros(z.shape[0], self.d_deter, device=z.device, dtype=z.dtype)
        decoded = self.decoder(h_zero, z)              # [B, N_img, num_image_tokens_vocab]
        return decoded, post_logits, post_logits_unimix


def categorical_metrics(logits_flat: torch.Tensor, S: int, K: int) -> dict[str, float]:
    """Compute entropy / max_prob / std / pairwise_cos / eff_rank for monitoring."""
    out: dict[str, float] = {}
    flat = logits_flat.detach().float()
    # std across batch dim (mean over feature)
    out["logits_std"] = float(flat.std(dim=0).mean())
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


# ── Train loop ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pretokenize_wm_libero_10_discrete_minimal")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--viz-every", type=int, default=500,
                        help="Save reconstruction PNG grid every N steps (0 to disable)")
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

    # Build pure VAE
    print("[vae] building model ...")
    n_image_tokens = int(OmegaConf.select(cfg, "world_model.n_image_tokens", default=256))
    model = PureVAE(
        num_image_tokens_vocab=n_image_tokens_vocab,
        n_image_tokens=n_image_tokens,
        spatial=tuple(OmegaConf.select(cfg, "world_model.spatial_grid", default=[16, 16])),
        token_embed_dim=int(OmegaConf.select(cfg, "world_model.token_embed_dim", default=512)),
        obs_dim=int(OmegaConf.select(cfg, "world_model.obs_dim", default=1024)),
        stoch_dims=int(OmegaConf.select(cfg, "world_model.stoch_dims", default=32)),
        stoch_categories=int(OmegaConf.select(cfg, "world_model.stoch_categories", default=32)),
        mapper_hidden_dim=int(OmegaConf.select(cfg, "world_model.mapper_hidden_dim", default=512)),
        gumbel_temp=float(OmegaConf.select(cfg, "world_model.gumbel_temp", default=1.0)),
        unimix=float(OmegaConf.select(cfg, "world_model.unimix", default=0.01)),
        decoder_mid_channels=int(OmegaConf.select(cfg, "world_model.decoder_mid_channels", default=192)),
        decoder_bspace_groups=int(OmegaConf.select(cfg, "world_model.decoder_bspace_groups", default=8)),
        decoder_minres=tuple(OmegaConf.select(cfg, "world_model.decoder_minres", default=[4, 4])),
        decoder_stage_channels=tuple(OmegaConf.select(cfg, "world_model.decoder_stage_channels", default=[96, 48])),
        decoder_stoch_hidden=int(OmegaConf.select(cfg, "world_model.decoder_stoch_hidden", default=512)),
        stem_post_norm=bool(OmegaConf.select(cfg, "world_model.stem_post_norm", default=False)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[vae]   trainable params = {n_params:,}")

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-6)

    # Dataset
    print("[vae] building dataset ...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    print(f"[vae]   dataset size = {len(dataset)}")
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=2, collate_fn=collate, drop_last=True,
    )

    # Output dir
    from datetime import datetime
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("/home/user01/liops/workspace/DreamerVLA/data/outputs/pure_vae") / datetime.now().strftime("pure_vae_%Y%m%d_%H%M%S")
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
        gt_bpe = imgidx_to_bpe[gt]                     # → BPE ids
        pr_bpe = imgidx_to_bpe[pr]
        gt_vq = bpe2vq_table[gt_bpe]                   # → VQ codebook ids
        pr_vq = bpe2vq_table[pr_bpe]
        if (gt_vq < 0).any() or (pr_vq < 0).any():
            print(f"[viz] step {step}: skipped (non-image bpe in tokens)")
            return
        gt_pix = vq_tokens_to_pixels(gt_vq, vq_model, h_latent=16, w_latent=16)
        pr_pix = vq_tokens_to_pixels(pr_vq, vq_model, h_latent=16, w_latent=16)
        # Build an n_show × 2 grid (col0 = GT, col1 = recon)
        cell = 256
        canvas = Image.new("RGB", (cell * 2, cell * n_show + 24), (32, 32, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4),  f"step {step}  |  col0=GT  col1=recon", fill=(230, 230, 230))
        for i in range(n_show):
            canvas.paste(tensor_to_pil(gt_pix[i]).convert("RGB").resize((cell, cell)),
                         (0, 24 + i * cell))
            canvas.paste(tensor_to_pil(pr_pix[i]).convert("RGB").resize((cell, cell)),
                         (cell, 24 + i * cell))
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
            obs_emb_bpe = extract_image_bpe_ids(
                input_ids_list, -2, n_image_tokens, img_bpe_set,
            ).to(device)                                 # [B, N_img]
            img_idx = bpe_to_imgidx[obs_emb_bpe]
            if (img_idx < 0).any():
                raise ValueError("non-image bpe id in input")

            decoded_logits, post_logits, _ = model(img_idx)
            target = img_idx                              # autoencoder: target = input
            loss = F.cross_entropy(
                decoded_logits.reshape(-1, decoded_logits.shape[-1]),
                target.reshape(-1),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            opt.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    pred = decoded_logits.argmax(dim=-1)
                    acc = (pred == target).float().mean()
                    pred_unique = pred.unique().numel()
                    cat_m = categorical_metrics(post_logits, model.stoch_dims, model.stoch_categories)

            # Image viz at viz_every cadence
            if args.viz_every > 0 and step % args.viz_every == 0:
                with torch.no_grad():
                    pred = decoded_logits.argmax(dim=-1)
                    save_viz_grid(step, target, pred, n_show=4)
                row = {
                    "step": step,
                    "loss": float(loss),
                    "recon_acc": float(acc),
                    "pred_unique_tokens": int(pred_unique),
                    "grad_norm": float(grad_norm),
                    **{f"post_{k}": v for k, v in cat_m.items()},
                }
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
                print(
                    f"step {step:5d}  loss={row['loss']:.3f}  acc={row['recon_acc']:.4f}  "
                    f"uniq={row['pred_unique_tokens']:3d}  "
                    f"post_std={row['post_logits_std']:.5f}  "
                    f"post_ent={row['post_entropy']:.3f}  "
                    f"post_max_p={row['post_max_prob']:.3f}  "
                    f"post_cos={row.get('post_pairwise_cos', float('nan')):.4f}  "
                    f"|g|={row['grad_norm']:.2f}"
                )
            step += 1

    log_handle.close()
    # Final checkpoint
    ckpt_path = out_dir / "final.ckpt"
    torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
    # Final viz
    if args.viz_every > 0:
        with torch.no_grad():
            decoded_logits, post_logits, _ = model(img_idx)
            pred = decoded_logits.argmax(dim=-1)
            save_viz_grid(step, img_idx, pred, n_show=8)
    print(f"\n[vae] done.")
    print(f"  log    → {log_path}")
    print(f"  ckpt   → {ckpt_path}")
    print(f"  viz    → {out_dir / 'viz'}")


if __name__ == "__main__":
    main()
