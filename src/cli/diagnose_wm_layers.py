"""
Layer-wise collapse localization.

Run a small batch of *very different* samples through the WM, capture every
intermediate tensor, and report how diverse those tensors still are at each
layer.  Three metrics per layer (all computed over the [B, T, ..., D] tensor
flattened to a population of vectors of dim D):

  • mean_pairwise_cos     — average cosine similarity between distinct vectors.
                            1.0 = collapsed to one direction.
  • centered_rel_norm     — ||x - mean|| / ||mean||, averaged over vectors.
                            0.0 = all identical (single point).
  • pc1_explained_var     — fraction of variance captured by the top PCA
                            component.  ~1.0 = collapsed onto a 1-D line.

Reads the v2/transdreamer-discrete checkpoint and the seq dataset.

Run:
    python -m src.cli.diagnose_wm_layers \
        --config-name pretokenize_wm_libero_10_discrete_longt_zfocus_v2 \
        --ckpt <path>.ckpt \
        --num-samples 8 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_cfg(config_name: str, overrides: list[str]):
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _strip_fsdp_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        cleaned = key
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        out[cleaned] = value
    return out


def load_wm_state_dict(ckpt_path: Path):
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return payload["state_dicts"]["world_model"]


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


def collate_seq(batch):
    return {
        "wm_obs_input_ids_seq": [s["wm_obs_input_ids_seq"] for s in batch],
        "action_seq":   torch.stack([s["action_seq"] for s in batch], dim=0),
        "reward_seq":   torch.stack([s["reward_seq"] for s in batch], dim=0),
        "done_seq":     torch.stack([s["done_seq"]   for s in batch], dim=0),
        "task_name":    [s["task_name"] for s in batch],
    }


def collate_single(batch):
    """Single-transition format: build T=2 pseudo-seq per sample."""
    return {
        "wm_obs_input_ids":      [list(s["wm_obs_input_ids"])      for s in batch],
        "wm_next_obs_input_ids": [list(s["wm_next_obs_input_ids"]) for s in batch],
        "wm_action": torch.stack([torch.as_tensor(s["wm_action"], dtype=torch.float32) for s in batch], dim=0),
        "task_name": [s["task_name"] for s in batch],
    }


def diversity_stats(x: torch.Tensor, name: str):
    """
    x: [B, T, ..., D] or [B, ..., D]; will be flattened to [N, D].
    Returns dict of three diversity metrics + shape info.
    """
    x = x.detach().float().cpu()
    if x.ndim < 2:
        x = x.unsqueeze(0)
    # Flatten everything but last dim
    flat = x.reshape(-1, x.shape[-1])           # [N, D]
    N, D = flat.shape

    # Subsample for cost if N is large
    if N > 256:
        idx = torch.randperm(N)[:256]
        flat = flat[idx]
        N = flat.shape[0]

    # ---- pairwise cosine
    norms = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = flat / norms
    cos = unit @ unit.T   # [N, N]
    triu = cos[torch.triu_indices(N, N, offset=1).unbind(0)]
    mean_pair_cos = float(triu.mean())

    # ---- centered relative norm
    mean_vec = flat.mean(dim=0, keepdim=True)
    centered = flat - mean_vec
    mean_norm = float(mean_vec.norm())
    centered_norm = float(centered.norm(dim=-1).mean())
    rel_norm = centered_norm / max(mean_norm, 1e-12)

    # ---- PC1 explained variance
    try:
        # SVD on centered
        u, s, vh = torch.linalg.svd(centered, full_matrices=False)
        s2 = (s ** 2)
        pc1 = float(s2[0] / s2.sum().clamp_min(1e-12))
        # also entropy-based effective rank
        p = (s2 / s2.sum().clamp_min(1e-12)).clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p * p.log()).sum()))
    except Exception:
        pc1 = float("nan")
        eff_rank = float("nan")

    return {
        "name": name,
        "shape": tuple(x.shape),
        "N_vectors": N,
        "D": D,
        "mean_pairwise_cos": mean_pair_cos,
        "centered_rel_norm": rel_norm,
        "pc1_explained_var": pc1,
        "effective_rank": eff_rank,
    }


def fmt_row(s):
    return (f"  {s['name']:35s} shape={str(s['shape']):28s} "
            f"cos={s['mean_pairwise_cos']:+.4f}  "
            f"rel_spread={s['centered_rel_norm']:.4f}  "
            f"PC1={s['pc1_explained_var']:.3f}  "
            f"eff_rank={s['effective_rank']:6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    cfg = build_cfg(args.config_name, args.overrides)

    # --- Encoder for vocab mapping
    print("[layers] building encoder ...")
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    vocab_mapping = encoder.backbone.model.vocabulary_mapping
    img_bpe_set = set(vocab_mapping.bpe2img.keys())
    n_image_tokens_vocab = len(img_bpe_set)
    image_token_bpe_ids = torch.tensor(sorted(img_bpe_set), dtype=torch.long, device=device)
    full_vocab_size = int(getattr(encoder.backbone.lm_head, "out_features",
                                  encoder.backbone.lm_head.weight.shape[0]))

    # --- WM
    print("[layers] building world model ...")
    hidden_dim = int(OmegaConf.select(cfg, "world_model.hidden_dim", default=4096))
    wm_kwargs: dict[str, Any] = {"hidden_dim": hidden_dim}
    io_mode = str(OmegaConf.select(cfg, "world_model.io_mode", default="hidden"))
    if io_mode == "token" and OmegaConf.select(cfg, "world_model.num_image_tokens_vocab") is None:
        wm_kwargs["num_image_tokens_vocab"] = n_image_tokens_vocab
    world_model = hydra.utils.instantiate(cfg.world_model, **wm_kwargs)
    world_model = world_model.to(dtype=torch.bfloat16).to(device)
    sd = _strip_fsdp_prefix(load_wm_state_dict(Path(args.ckpt)))
    world_model.load_state_dict(sd, strict=False)
    world_model.eval()
    n_image_tokens = int(getattr(world_model, "n_image_tokens", 256))
    world_model.attach_lm_head(
        lm_head=None if io_mode == "token" else encoder.backbone.lm_head,
        image_token_bpe_ids=image_token_bpe_ids,
        full_vocab_size=full_vocab_size,
    )

    # --- Dataset: pull samples from K *different tasks* if possible
    print("[layers] building dataset ...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    n_total = len(dataset)
    rng = np.random.default_rng(args.seed)

    # Pick samples spread across the dataset, prefer different tasks
    samples = []
    seen_tasks = set()
    indices = rng.permutation(n_total).tolist()
    for idx in indices:
        s = dataset[idx]
        t = s.get("task_name", "?")
        if t in seen_tasks and len(seen_tasks) < args.num_samples:
            continue
        samples.append(s)
        seen_tasks.add(t)
        if len(samples) >= args.num_samples:
            break
    print(f"[layers] picked {len(samples)} samples from {len(seen_tasks)} tasks: {sorted(seen_tasks)[:8]}")

    # Detect dataset format and build [B, T, N_img] BPE id tensor + [B, T, action_dim]
    sample0 = samples[0]
    if "wm_obs_input_ids_seq" in sample0:
        batch = collate_seq(samples)
        seq_ids = batch["wm_obs_input_ids_seq"]
        B = len(seq_ids); T = len(seq_ids[0])
        flat = [step for sample in seq_ids for step in sample]
        obs_emb_bpe = extract_image_bpe_ids(flat, -2, n_image_tokens, img_bpe_set)\
                        .view(B, T, -1).to(device)
        action_seq = batch["action_seq"].to(device, dtype=torch.bfloat16)
    else:
        # single-transition format → build T=2 seq
        batch = collate_single(samples)
        cur_ids  = batch["wm_obs_input_ids"]
        next_ids = batch["wm_next_obs_input_ids"]
        B = len(cur_ids); T = 2
        cur_bpe  = extract_image_bpe_ids(cur_ids,  -2, n_image_tokens, img_bpe_set).to(device)
        next_bpe = extract_image_bpe_ids(next_ids, -2, n_image_tokens, img_bpe_set).to(device)
        obs_emb_bpe = torch.stack([cur_bpe, next_bpe], dim=1)              # [B, 2, N_img]
        # action: [B, H, A] → mean to single step, then stack zero+action for T=2
        wm_action = batch["wm_action"].to(device, dtype=torch.bfloat16)    # [B, H, A]
        action_step = wm_action.mean(dim=1)
        action_seq = torch.stack([torch.zeros_like(action_step), action_step], dim=1)  # [B, 2, A]

    print(f"\n[layers] running forward; B={B} T={T} N_img={n_image_tokens}\n")

    # ──────────────────── Manual layer-by-layer forward ───────────────────────
    layer_stats: list[dict] = []
    with torch.no_grad():
        # Sanity: input BPE ids diversity (raw, integer)
        # Convert to one-hot frequency vector per (sample,time) so diversity is meaningful
        # — easier: just record stat
        layer_stats.append({
            "name": "input_bpe_ids (raw)",
            "shape": tuple(obs_emb_bpe.shape),
            "N_vectors": int(B*T),
            "D": int(obs_emb_bpe.shape[-1]),
            "mean_pairwise_cos": float("nan"),
            "centered_rel_norm": float("nan"),
            "pc1_explained_var": float("nan"),
            "effective_rank": float("nan"),
            "note": f"unique tokens across batch = {int(obs_emb_bpe.unique().numel())}",
        })

        # 1) bpe → image-vocab idx → token_embedder
        img_idx_seq = world_model._bpe_to_img_idx[obs_emb_bpe]                # [B, T, N_img]
        per_token = world_model.token_embedder(img_idx_seq)                    # [B, T, N_img, d_embed]
        layer_stats.append(diversity_stats(per_token.float(), "1. token_embedder out"))

        # 2) conv_stem
        hidden_seq = world_model.conv_stem(per_token)                          # [B, T, obs_dim]
        layer_stats.append(diversity_stats(hidden_seq.float(), "2. conv_stem out (obs)"))

        # 3) Dreamer sequence inference
        (post_mean, post_std, post_stoch,
         prior_mean, prior_std, prior_stoch,
         h_seq) = world_model._infer_dreamer_seq(hidden_seq, action_seq)
        # post_*: [B, T, latent_dim],  prior_*: [B, T-1, latent_dim],  h: [B, T-1, d_model]

        layer_stats.append(diversity_stats(post_mean.float(),  "3a. posterior mean (z logits)"))
        layer_stats.append(diversity_stats(post_stoch.float(), "3b. posterior z (sampled, t=0..T-1)"))
        layer_stats.append(diversity_stats(prior_mean.float(),  "3c. prior mean (z logits, t=1..T-1)"))
        layer_stats.append(diversity_stats(prior_stoch.float(), "3d. prior z (sampled)"))
        layer_stats.append(diversity_stats(h_seq.float(),      "3e. h (causal transformer)"))

        # 4) transition_head: predicts next obs
        post_state = torch.cat([post_stoch[:, :-1], h_seq], dim=-1)            # [B, T-1, latent+d_model]
        # transition_head expects state.feature() = cat(stoch, deter)
        from src.models.world_model.tssm import TSSMState  # for typing only
        # Use the model helper transition_head directly; it's a simple MLP
        pred_next_obs = world_model.transition_head(post_state)                # [B, T-1, obs_dim]
        layer_stats.append(diversity_stats(pred_next_obs.float(), "4. transition_head out (pred next obs)"))

        # 5) image_decoder logits (if enabled)
        if getattr(world_model, "image_decoder", None) is not None:
            # The image_decoder typically takes (deter h, stoch z); signature:
            try:
                logits = world_model.image_decoder(h_seq, post_stoch[:, 1:])   # [B, T-1, N_img, V]
                layer_stats.append(diversity_stats(logits.float(),
                                                   "5. image_decoder logits (post-conditioned)"))
                # Predicted token argmax diversity
                pred_top1 = logits.argmax(dim=-1)                              # [B, T-1, N_img]
                # For an integer-valued tensor, fall back to a unique-count metric
                pred_uniques = int(pred_top1.unique().numel())
                layer_stats.append({
                    "name": "5b. pred top-1 token ids",
                    "shape": tuple(pred_top1.shape),
                    "N_vectors": int(B*(T-1)),
                    "D": int(pred_top1.shape[-1]),
                    "mean_pairwise_cos": float("nan"),
                    "centered_rel_norm": float("nan"),
                    "pc1_explained_var": float("nan"),
                    "effective_rank": float("nan"),
                    "note": f"unique top-1 ids across batch = {pred_uniques}",
                })
            except Exception as exc:
                print(f"[layers] image_decoder probe failed: {exc}")

    # Print
    print("─" * 130)
    print(" Layer-wise diversity (lower cos / lower rel_spread / higher PC1 = more collapsed)")
    print("─" * 130)
    for s in layer_stats:
        line = fmt_row(s)
        if s.get("note"):
            line += f"  ◀ {s['note']}"
        print(line)
    print("─" * 130)

    # Save
    out_path = args.out_json or str(Path(args.ckpt).parent.parent / f"layers_collapse_{Path(args.ckpt).stem}.json")
    Path(out_path).write_text(json.dumps({"layers": layer_stats,
                                          "ckpt": args.ckpt,
                                          "config_name": args.config_name,
                                          "tasks_in_batch": sorted(seen_tasks)}, indent=2))
    print(f"\n[layers] wrote {out_path}")


if __name__ == "__main__":
    main()
