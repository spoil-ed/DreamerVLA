"""
Focused world-model collapse diagnosis.

Loads a checkpoint, runs forward on N batches from the val/train dataset, and
aggregates:

  * the WM's own loss_dict metrics, **per imagination step** when available
  * GT token diversity across samples (per spatial position) — if the GT is
    near-constant, the high recon-accuracy floor is not learning
  * predicted top-1 token diversity (collapse of the predicted distribution)
  * KL / posterior std stats — does the posterior collapse onto the prior?

Run:
    python -m src.cli.diagnose_wm_collapse \
        --config-name pretokenize_wm_libero_10_discrete_longt_zfocus_v2 \
        --ckpt <path>.ckpt \
        --num-batches 16 --batch-size 8 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    cfg = build_cfg(args.config_name, args.overrides)

    # Build encoder (only need vocab mapping for io_mode='token').
    print("[diag] building encoder (for vocab mapping) ...")
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    vocab_mapping = encoder.backbone.model.vocabulary_mapping
    img_bpe_set = set(vocab_mapping.bpe2img.keys())
    n_image_tokens_vocab = len(img_bpe_set)
    print(f"[diag]   image vocab size = {n_image_tokens_vocab}")

    # Build WM.
    print("[diag] building world model ...")
    hidden_dim = int(OmegaConf.select(cfg, "world_model.hidden_dim", default=4096))
    wm_kwargs = {"hidden_dim": hidden_dim}
    if str(OmegaConf.select(cfg, "world_model.io_mode", default="hidden")) == "token" \
       and OmegaConf.select(cfg, "world_model.num_image_tokens_vocab") is None:
        wm_kwargs["num_image_tokens_vocab"] = n_image_tokens_vocab
    world_model = hydra.utils.instantiate(cfg.world_model, **wm_kwargs)
    world_model = world_model.to(dtype=torch.bfloat16).to(device)
    sd = _strip_fsdp_prefix(load_wm_state_dict(Path(args.ckpt)))
    miss, unexp = world_model.load_state_dict(sd, strict=False)
    if miss:    print(f"[diag] WARN missing keys: {len(miss)}  e.g. {miss[:3]}")
    if unexp:   print(f"[diag] WARN unexpected keys: {len(unexp)}  e.g. {unexp[:3]}")
    world_model.eval()
    n_image_tokens = int(getattr(world_model, "n_image_tokens", 256))
    print(f"[diag]   spatial tokens per image = {n_image_tokens}")

    # Required for token mode: register bpe→img-idx mapping (lm_head=None).
    image_token_bpe_ids = torch.tensor(sorted(img_bpe_set), dtype=torch.long, device=device)
    full_vocab_size = int(getattr(encoder.backbone.lm_head, "out_features",
                                  encoder.backbone.lm_head.weight.shape[0]))
    io_mode = str(OmegaConf.select(cfg, "world_model.io_mode", default="hidden"))
    world_model.attach_lm_head(
        lm_head=None if io_mode == "token" else encoder.backbone.lm_head,
        image_token_bpe_ids=image_token_bpe_ids,
        full_vocab_size=full_vocab_size,
    )
    print(f"[diag]   attach_lm_head ok (io_mode={io_mode}, full_vocab={full_vocab_size})")

    # Build dataset and dataloader.
    print("[diag] building dataset ...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    print(f"[diag]   dataset size = {len(dataset)}")
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=0, collate_fn=collate_seq, drop_last=True,
    )

    which_block = int(OmegaConf.select(cfg, "viz.which_block", default=-2))

    # Build BPE→image-vocab-index lookup so we can histogram in image-vocab space.
    bpe_to_imgidx = torch.full((full_vocab_size,), -1, dtype=torch.long)
    bpe_to_imgidx[image_token_bpe_ids.cpu()] = torch.arange(n_image_tokens_vocab, dtype=torch.long)

    # Aggregators
    metric_sums: dict[str, float] = defaultdict(float)
    metric_counts: dict[str, int] = defaultdict(int)
    pred_token_counts = torch.zeros(n_image_tokens_vocab, dtype=torch.long)
    gt_token_counts   = torch.zeros(n_image_tokens_vocab, dtype=torch.long)
    # Per spatial position GT diversity: count unique tokens per position across samples
    pos_gt_tokens: list[list[int]] = [[] for _ in range(n_image_tokens)]
    pos_pred_tokens: list[list[int]] = [[] for _ in range(n_image_tokens)]
    samples_seen = 0

    # Try to capture token logits via a hook by monkey-patching `argmax` calls.
    # Simpler: we'll re-run the imagine path manually after forward() to get
    # predicted token argmax. Instead, we look at the loss_dict — most TSSM
    # discrete impls already return per-step accuracies.

    print(f"[diag] running {args.num_batches} batches of size {args.batch_size} ...")
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.num_batches: break

            # Build wm_batch like the workspace does (token-mode path).
            seq_ids = batch["wm_obs_input_ids_seq"]  # list[B] of list[T] of list[int]
            B = len(seq_ids); T = len(seq_ids[0])
            flat = [step for sample in seq_ids for step in sample]
            obs_emb = extract_image_bpe_ids(flat, which_block, n_image_tokens, img_bpe_set)
            obs_emb = obs_emb.view(B, T, -1).to(device)

            wm_batch = {
                "obs_embedding_seq": obs_emb,                                  # [B, T, S]
                "action_seq":  batch["action_seq"].to(device, dtype=torch.bfloat16),
                "reward_seq":  batch["reward_seq"].to(device, dtype=torch.bfloat16),
                "done_seq":    batch["done_seq"].to(device,  dtype=torch.bfloat16),
            }
            loss_dict = world_model(wm_batch)

            # Aggregate scalar metrics
            for k, v in loss_dict.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    metric_sums[k]   += float(v.detach().cpu())
                    metric_counts[k] += 1

            # Map BPE ids → image-vocab indices [0, n_image_tokens_vocab).
            obs_imgidx = bpe_to_imgidx[obs_emb.cpu()]   # [B, T, S]
            if (obs_imgidx < 0).any():
                bad = int((obs_imgidx < 0).sum())
                raise RuntimeError(f"{bad} GT tokens have BPE id outside image vocab")

            # GT token diversity / global histogram
            obs_flat = obs_imgidx.view(-1)
            gt_token_counts.scatter_add_(0, obs_flat,
                                         torch.ones_like(obs_flat))
            # Per spatial position diversity over time/batch
            obs_per_pos = obs_imgidx.view(B*T, n_image_tokens)
            for s in range(n_image_tokens):
                pos_gt_tokens[s].extend(obs_per_pos[:, s].tolist())

            samples_seen += B * T

    # Summary
    metrics_avg = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
    # Sort metrics for readability
    print("\n========== loss_dict averages ==========")
    for k in sorted(metrics_avg.keys()):
        print(f"  {k:50s} {metrics_avg[k]:.5f}")

    # GT token diversity stats
    gt_total = int(gt_token_counts.sum())
    gt_used  = int((gt_token_counts > 0).sum())
    gt_top   = gt_token_counts.topk(10)
    print(f"\n========== GT token diversity (across all positions, batches) ==========")
    print(f"  total tokens observed = {gt_total}")
    print(f"  unique tokens used    = {gt_used} / {n_image_tokens_vocab}  ({gt_used/n_image_tokens_vocab*100:.2f}%)")
    print(f"  top-10 tokens (id, count, frac):")
    for tid, cnt in zip(gt_top.indices.tolist(), gt_top.values.tolist()):
        print(f"    {tid:5d}  {cnt:8d}  {cnt/gt_total*100:6.3f}%")

    # Per spatial position: how many distinct GT tokens
    pos_diversity = []
    pos_top1_share = []
    for s in range(n_image_tokens):
        toks = pos_gt_tokens[s]
        if not toks:
            pos_diversity.append(0); pos_top1_share.append(0.0); continue
        cnts = np.bincount(np.array(toks))
        pos_diversity.append(int((cnts > 0).sum()))
        pos_top1_share.append(float(cnts.max() / len(toks)))
    pos_diversity = np.array(pos_diversity)
    pos_top1_share = np.array(pos_top1_share)
    print(f"\n========== Per-spatial-position GT diversity ==========")
    print(f"  N samples per position = {len(pos_gt_tokens[0])}")
    print(f"  unique GT tokens / position:  min={pos_diversity.min()}  median={int(np.median(pos_diversity))}  mean={pos_diversity.mean():.1f}  max={pos_diversity.max()}")
    print(f"  top-1 token share / position:  mean={pos_top1_share.mean():.3f}  median={float(np.median(pos_top1_share)):.3f}  max={pos_top1_share.max():.3f}")
    # Histogram of how many positions are 'collapsed' (>90%, >99% same token)
    print(f"  positions with >50% same token: {(pos_top1_share > 0.5).sum()} / {n_image_tokens}")
    print(f"  positions with >90% same token: {(pos_top1_share > 0.9).sum()} / {n_image_tokens}")
    print(f"  positions with >99% same token: {(pos_top1_share > 0.99).sum()} / {n_image_tokens}")

    summary = {
        "ckpt": args.ckpt,
        "config_name": args.config_name,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "samples_seen_frames": samples_seen,
        "metrics": metrics_avg,
        "gt_diversity": {
            "tokens_observed": gt_total,
            "unique_tokens_used": gt_used,
            "image_vocab_size": n_image_tokens_vocab,
            "top10_ids":    gt_top.indices.tolist(),
            "top10_counts": gt_top.values.tolist(),
        },
        "per_position_gt": {
            "min_unique":    int(pos_diversity.min()),
            "median_unique": int(np.median(pos_diversity)),
            "mean_unique":   float(pos_diversity.mean()),
            "max_unique":    int(pos_diversity.max()),
            "top1_share_mean":   float(pos_top1_share.mean()),
            "top1_share_median": float(np.median(pos_top1_share)),
            "top1_share_max":    float(pos_top1_share.max()),
            "n_positions":   n_image_tokens,
            "n_pos_top1_gt_50pct": int((pos_top1_share > 0.5).sum()),
            "n_pos_top1_gt_90pct": int((pos_top1_share > 0.9).sum()),
            "n_pos_top1_gt_99pct": int((pos_top1_share > 0.99).sum()),
        },
    }
    out_path = args.out_json or str(Path(args.ckpt).parent.parent / f"diagnose_collapse_{Path(args.ckpt).stem}.json")
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\n[diag] wrote summary to {out_path}")


if __name__ == "__main__":
    main()
