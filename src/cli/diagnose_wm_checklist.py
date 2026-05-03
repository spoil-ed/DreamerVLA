"""WM diagnostic checklist (Levels 0–5).

Loads a trained WM ckpt and computes:
  Level 0  representation alignment   (post vs prior feature norms / cos / diff)
  Level 1  KL distribution health     (kl, μ-diff, std-of-logits)
  Level 2  transition collapse        (prior diversity across batch / time)
  Level 3  action conditioning        (real vs zero vs shuffled action margins)
  Level 4  multi-step rollout         (KL / feature-l2 vs imagine horizon)
  Level 5  token-level reconstruction (image_decoder argmax acc, static vs dynamic)

Output: <out-dir>/checklist.json with the full set of numbers + pass/fail flags.

Usage:
  python -m src.cli.diagnose_wm_checklist \\
    --config-name pretokenize_wm_libero_10_discrete_minimal_warmup \\
    --ckpt <path>.ckpt \\
    --num-samples 32
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.cli.eval_wm import (
    PROJECT_ROOT,
    extract_image_blocks,
    load_wm_state_dict,
    _strip_fsdp_prefix,
)

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def build_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(dim=-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def _f(t: torch.Tensor | float) -> float:
    return float(t.float().mean().item()) if isinstance(t, torch.Tensor) else float(t)


# ───────────────────────────────────────────────────────────────────────────────
# Per-level computation
# ───────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def level0_representation(
    h_seq: torch.Tensor,           # [B, T-1, d_model]
    post_stoch: torch.Tensor,      # [B, T-1, latent]   (sliced post[:, 1:])
    prior_stoch: torch.Tensor,     # [B, T-1, latent]
) -> dict[str, float]:
    """Posterior vs prior feature alignment.  In our WM h is shared (prior+
    posterior get the same h_t), so we report the joint-feature metrics that
    Dreamer downstream heads actually consume: feature = cat(h, z)."""
    feat_post  = torch.cat([h_seq, post_stoch],  dim=-1)         # [B, T-1, d+latent]
    feat_prior = torch.cat([h_seq, prior_stoch], dim=-1)
    fp_n  = _l2(feat_post)
    fpr_n = _l2(feat_prior)
    diff = feat_post - feat_prior
    return {
        "feature_post_norm":   _f(fp_n),
        "feature_prior_norm":  _f(fpr_n),
        "feature_diff_l2":     _f(_l2(diff)),
        "feature_cos_mean":    _f(_cos(feat_post, feat_prior)),
        "z_diff_l2":           _f(_l2(post_stoch - prior_stoch)),
        "h_diff_l2":           0.0,   # h is shared in our impl
    }


@torch.no_grad()
def level1_kl(
    post_logits: torch.Tensor,     # [B, T-1, S*K]
    prior_logits: torch.Tensor,    # [B, T-1, S*K]
    S: int, K: int,
) -> dict[str, float]:
    pl = post_logits.float().reshape(*post_logits.shape[:-1], S, K)
    qr = prior_logits.float().reshape(*prior_logits.shape[:-1], S, K)
    log_p = F.log_softmax(pl, dim=-1)
    p     = log_p.exp()
    log_q = F.log_softmax(qr, dim=-1)
    kl    = (p * (log_p - log_q)).sum(dim=-1)            # [B, T-1, S]   per-dim
    mu_diff = (pl - qr).norm(dim=-1).mean()              # mean over S
    return {
        "kl_per_dim_mean":    _f(kl.mean()),             # nats per latent dim
        "kl_total_mean":      _f(kl.sum(dim=-1).mean()), # summed over 32 dims
        "mu_diff_l2_mean":    _f(mu_diff),
        # discrete logits "std" within a sample (sharpness proxy)
        "std_post_logits":    _f(pl.std(dim=-1).mean()),
        "std_prior_logits":   _f(qr.std(dim=-1).mean()),
        # peak prob (closer to 1 = sharper, closer to 1/K = flatter)
        "max_prob_post":      _f(p.max(dim=-1).values.mean()),
        "max_prob_prior":     _f(log_q.exp().max(dim=-1).values.mean()),
    }


@torch.no_grad()
def level2_transition_collapse(
    prior_stoch: torch.Tensor,     # [B, T-1, latent]
) -> dict[str, float]:
    p = prior_stoch.float()                              # [B, T-1, L]
    B, Tm1, L = p.shape
    flat = p.reshape(B * Tm1, L)                         # [B*T-1, L]
    norms = _l2(flat)
    centered = flat - flat.mean(dim=0, keepdim=True)
    cent_norms = _l2(centered)
    # Pairwise L2 across batch (subset for speed if large)
    n = min(64, flat.shape[0])
    pick = flat[:n]
    diffs = pick.unsqueeze(0) - pick.unsqueeze(1)         # [n, n, L]
    pairwise_l2 = _l2(diffs).flatten()
    triu = torch.triu_indices(n, n, offset=1)
    pairwise_l2 = _l2(diffs)[triu[0], triu[1]]
    # Adjacent-time cosine (along T)
    if Tm1 >= 2:
        adj_cos = _cos(p[:, :-1], p[:, 1:])              # [B, T-2]
        adj_cos_mean = _f(adj_cos)
    else:
        adj_cos_mean = float("nan")
    rel_centered = (cent_norms / norms.clamp_min(1e-8)).mean()
    return {
        "prior_norm_mean":              _f(norms),
        "prior_centered_norm_mean":     _f(cent_norms),
        "prior_relative_centered_norm": _f(rel_centered),
        "prior_pairwise_l2_mean":       _f(pairwise_l2),
        "prior_adjacent_cos_mean":      adj_cos_mean,
    }


@torch.no_grad()
def level3_action_conditioning(
    world_model,
    post_stoch_prefix: torch.Tensor,    # [B, T-1, latent]  z_{0:T-2}
    action_real: torch.Tensor,          # [B, T-1, A]      a_{1:T-1}
    target_z: torch.Tensor,             # [B, T-1, latent] z_t target = post_stoch[:, 1:]
) -> dict[str, float]:
    """Run prior dynamics with real / zeroed / batch-shuffled actions and
    measure how much the predicted z deviates from the target z (= posterior
    z at time t).  In a healthy WM, real_to_target should be smallest."""
    B = action_real.shape[0]
    # Match dtype to model params (bf16) so _infer_prior_seq doesn't error.
    target_dtype = post_stoch_prefix.dtype
    action_real_b = action_real.to(dtype=target_dtype)
    target_z_b    = target_z.to(dtype=target_dtype)

    def run(action: torch.Tensor) -> torch.Tensor:
        prior_mean, _, prior_stoch_seq, _ = world_model._infer_prior_seq(
            stoch_seq=post_stoch_prefix,
            action_seq=action.to(dtype=target_dtype),
        )
        return prior_stoch_seq                                 # [B, T-1, latent]

    pred_real = run(action_real_b)
    pred_zero = run(torch.zeros_like(action_real_b))
    perm = torch.randperm(B, device=action_real_b.device)
    pred_shuffle = run(action_real_b[perm])

    real_l2    = _l2(pred_real    - target_z_b).mean()
    zero_l2    = _l2(pred_zero    - target_z_b).mean()
    shuffle_l2 = _l2(pred_shuffle - target_z_b).mean()
    return {
        "real_to_target_l2":      _f(real_l2),
        "zero_to_target_l2":      _f(zero_l2),
        "shuffle_to_target_l2":   _f(shuffle_l2),
        "margin_zero":            _f(zero_l2 - real_l2),
        "margin_shuffle":         _f(shuffle_l2 - real_l2),
    }


@torch.no_grad()
def level4_multistep_rollout(
    world_model,
    post_stoch: torch.Tensor,       # [B, T, latent]
    post_logits: torch.Tensor,      # [B, T, S*K]
    h_seq_post: torch.Tensor,       # [B, T-1, d_model]
    action: torch.Tensor,           # [B, T, A]
    horizons: list[int],
    S: int, K: int,
) -> dict[str, float]:
    """Imagine k steps forward from z_0 conditioned on real actions, compare
    to posterior z_k.  We need T >= max(horizons)+1 in the data."""
    B, T, _ = post_stoch.shape
    out: dict[str, float] = {}
    for k in horizons:
        if k + 1 > T:
            out[f"kl@{k}"] = float("nan")
            out[f"feat_l2@{k}"] = float("nan")
            continue
        # Use post[:, :1] as the seed; iterate k steps with real action
        cur_z = post_stoch[:, :1]                         # [B, 1, latent]
        for step in range(k):
            act_step = action[:, step:step+1]              # [B, 1, A]
            _, _, prior_stoch_seq, h_step = world_model._infer_prior_seq(
                stoch_seq=cur_z, action_seq=act_step,
            )
            cur_z = torch.cat([cur_z, prior_stoch_seq[:, -1:]], dim=1)  # grow prefix
        imagined_z = cur_z[:, -1]                          # z_k
        target_z   = post_stoch[:, k]                      # post z_k
        feat_l2 = _l2(imagined_z - target_z).mean()
        # KL from posterior @ k vs imagined-prior @ k
        # Use post_logits (computed earlier for posterior) and imagined logits
        # (we have stochs not logits for imagined; approximate KL via L2 here.)
        out[f"kl_proxy@{k}"] = _f(feat_l2)
        out[f"feat_l2@{k}"]  = _f(feat_l2)
    return out


@torch.no_grad()
def level5_token_recon(
    world_model,
    h_seq: torch.Tensor,           # [B, T-1, d_model]
    post_stoch: torch.Tensor,      # [B, T-1, latent]   for posterior recon
    prior_stoch: torch.Tensor,     # [B, T-1, latent]
    next_img_idx: torch.Tensor,    # [B, T-1, N_img]    target image-vocab ids
    cur_img_idx: torch.Tensor,     # [B, T-1, N_img]    previous-frame ids (for static/dynamic split)
) -> dict[str, float]:
    if not getattr(world_model, "image_decoder", None):
        return {}
    # Posterior-decoded
    logits_post  = world_model.image_decoder(h_seq, post_stoch)   # [B, T-1, N_img, V]
    logits_prior = world_model.image_decoder(h_seq, prior_stoch)
    pred_post  = logits_post.argmax(dim=-1)
    pred_prior = logits_prior.argmax(dim=-1)
    correct_post  = (pred_post  == next_img_idx).float()
    correct_prior = (pred_prior == next_img_idx).float()
    static_mask  = (next_img_idx == cur_img_idx).float()
    dynamic_mask = 1.0 - static_mask
    s_sum = static_mask.sum().clamp_min(1.0)
    d_sum = dynamic_mask.sum().clamp_min(1.0)
    return {
        "post_recon_acc":         _f(correct_post.mean()),
        "prior_recon_acc":        _f(correct_prior.mean()),
        "post_static_acc":        _f((correct_post  * static_mask ).sum() / s_sum),
        "post_dynamic_acc":       _f((correct_post  * dynamic_mask).sum() / d_sum),
        "prior_static_acc":       _f((correct_prior * static_mask ).sum() / s_sum),
        "prior_dynamic_acc":      _f((correct_prior * dynamic_mask).sum() / d_sum),
        "dynamic_fraction":       _f(dynamic_mask.mean()),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Pass/fail evaluation against the user's checklist
# ───────────────────────────────────────────────────────────────────────────────

def evaluate_pass_fail(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, bool]]:
    L0 = metrics.get("level0", {})
    L1 = metrics.get("level1", {})
    L2 = metrics.get("level2", {})
    L3 = metrics.get("level3", {})
    L5 = metrics.get("level5", {})
    fp_n = L0.get("feature_post_norm", 1.0)
    return {
        "level0": {
            "feature_norms_match":      abs(L0.get("feature_post_norm", 0) - L0.get("feature_prior_norm", 0)) < 0.2 * fp_n,
            "feature_diff_small":       L0.get("feature_diff_l2", 1e9) < 0.5 * fp_n,
            "feature_cos_high":         L0.get("feature_cos_mean", 0.0) >= 0.9,
        },
        "level1": {
            "kl_in_range":              0.05 <= L1.get("kl_per_dim_mean", 0.0) <= 4.0,
            "post_logits_alive":        L1.get("std_post_logits", 0.0) > 0.05,
            "prior_logits_alive":       L1.get("std_prior_logits", 0.0) > 0.05,
        },
        "level2": {
            "prior_not_collapsed":      L2.get("prior_relative_centered_norm", 0.0) > 0.1,
            "prior_pairwise_nonzero":   L2.get("prior_pairwise_l2_mean", 0.0) > 0.1,
            "prior_adjacent_diverse":   L2.get("prior_adjacent_cos_mean", 1.0) < 0.999,
        },
        "level3": {
            "real_beats_zero":          L3.get("margin_zero", 0.0) > 0.0,
            "real_beats_shuffle":       L3.get("margin_shuffle", 0.0) > 0.0,
        },
        "level5": {
            "dynamic_acc_above_random": L5.get("post_dynamic_acc", 0.0) > 0.01,
        },
    }


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="WM diagnostic checklist")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="*", default=[1, 5])
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = (Path(args.out_dir) if args.out_dir else
               PROJECT_ROOT / "data" / "outputs" / "diagnose_wm" /
               datetime.now().strftime("checklist_%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] config={args.config_name}  ckpt={args.ckpt}  out={out_dir}")

    cfg = build_cfg(args.config_name, args.overrides)

    # Build encoder (frozen) and WM
    print("[diag] building encoder ...")
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("[diag] building world model ...")
    hidden_dim = int(OmegaConf.select(cfg, "world_model.hidden_dim", default=4096))
    wm_kwargs: dict[str, Any] = {"hidden_dim": hidden_dim}
    if (str(OmegaConf.select(cfg, "world_model.io_mode", default="hidden")) == "token"
            and OmegaConf.select(cfg, "world_model.num_image_tokens_vocab") is None):
        wm_kwargs["num_image_tokens_vocab"] = len(
            encoder.backbone.model.vocabulary_mapping.bpe2img
        )
    world_model = hydra.utils.instantiate(cfg.world_model, **wm_kwargs)
    world_model = world_model.to(dtype=torch.bfloat16).to(device)
    state_dict = _strip_fsdp_prefix(load_wm_state_dict(Path(args.ckpt)))
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    print(f"[diag] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    world_model.eval()

    # Attach lm_head (token mode needs it)
    vocab_mapping = encoder.backbone.model.vocabulary_mapping
    image_token_bpe_ids = torch.tensor(
        sorted(vocab_mapping.bpe2img.keys()), dtype=torch.long, device=device,
    )
    if getattr(world_model, "spatial_codec", False):
        wm_io_mode = str(getattr(world_model, "io_mode", "hidden"))
        full_vocab_size = int(encoder.backbone.lm_head.weight.shape[0])
        world_model.attach_lm_head(
            encoder.backbone.lm_head if wm_io_mode == "hidden" else None,
            image_token_bpe_ids,
            full_vocab_size=full_vocab_size,
        )

    print("[diag] building dataset ...")
    dataset = hydra.utils.instantiate(cfg.dataset_val_ind)
    n_total = len(dataset)
    print(f"[diag] dataset size = {n_total}")

    n_image_tokens = int(getattr(world_model, "n_image_tokens", 256))
    image_bpe_set = set(image_token_bpe_ids.tolist())

    def extract_block(seq, block_idx=-2):
        blocks = extract_image_blocks(list(seq))
        if not blocks:
            return None
        bidx = block_idx if block_idx >= 0 else len(blocks) + block_idx
        ids = [t for t in blocks[bidx][2] if t in image_bpe_set]
        if len(ids) != n_image_tokens:
            return None
        return ids

    # Build a batch of [obs, next_obs] pairs.
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_total, size=min(args.num_samples * 4, n_total), replace=False).tolist()

    obs_list, next_obs_list, action_list = [], [], []
    for idx in indices:
        if len(obs_list) >= args.num_samples:
            break
        sample = dataset[idx]
        wm_obs_ids      = sample.get("wm_obs_input_ids")
        wm_next_obs_ids = sample.get("wm_next_obs_input_ids")
        wm_action       = sample.get("wm_action")
        if wm_obs_ids is None or wm_next_obs_ids is None or wm_action is None:
            continue
        cur_block  = extract_block(wm_obs_ids,      block_idx=-2)
        next_block = extract_block(wm_next_obs_ids, block_idx=-2)
        if cur_block is None or next_block is None:
            continue
        if not isinstance(wm_action, torch.Tensor) or wm_action.numel() == 0:
            continue
        obs_list.append(cur_block)
        next_obs_list.append(next_block)
        action_list.append(wm_action.float())

    n = len(obs_list)
    if n == 0:
        raise RuntimeError("no usable samples")
    print(f"[diag] using {n} samples")

    obs       = torch.tensor(obs_list,      dtype=torch.long, device=device)         # [B, N_img]
    next_obs  = torch.tensor(next_obs_list, dtype=torch.long, device=device)
    # Average action across the chunk (matches workspace _build_world_model_batch).
    action_chunks = [a.to(device) for a in action_list]
    A = action_chunks[0].shape[-1]
    action_mean = torch.stack([a.mean(dim=0) if a.ndim == 2 else a for a in action_chunks], dim=0)  # [B, A]

    # Stack into T=2 sequence (obs, next_obs)
    hidden_seq = torch.stack([obs, next_obs], dim=1).to(dtype=torch.long)              # [B, 2, N_img]
    # Build raw_bpe_ids_seq matching the WM's expected long shape
    # (token mode: feed hidden_seq directly through token_embedder + conv_stem).
    # Action sequence: [zero, action]  (matches single-transition format).
    action_seq = torch.stack([torch.zeros_like(action_mean), action_mean], dim=1)      # [B, 2, A]

    # Run posterior + dynamics
    print("[diag] running posterior + dynamics ...")
    with torch.no_grad():
        # Re-create the same path the WM uses internally in token mode.
        # token_embedder expects image-vocab indices (0..vocab-1), so we have
        # to remap raw BPE ids through _bpe_to_img_idx first.
        bpe = hidden_seq.long()                                                              # [B, 2, N_img] BPE ids
        img_idx_seq = world_model._bpe_to_img_idx[bpe]                                       # [B, 2, N_img]
        if (img_idx_seq < 0).any():
            raise RuntimeError("non-image BPE ids leaked through extract_block")
        per_tok = world_model.token_embedder(img_idx_seq)                                    # [B, 2, N_img, d_embed]
        B, T = per_tok.shape[:2]
        per_tok_flat = per_tok.reshape(B * T, *per_tok.shape[2:])
        obs_seq_flat = world_model.conv_stem(per_tok_flat)                                   # [B*T, obs_dim]
        obs_seq = obs_seq_flat.reshape(B, T, -1)                                             # [B, 2, obs_dim]

        # Posterior + prior + h
        (post_mean, post_std, post_stoch,
         prior_mean, prior_std, prior_stoch,
         h_seq) = world_model._infer_dreamer_seq(obs_seq.to(dtype=torch.bfloat16),
                                                  action_seq.to(dtype=torch.bfloat16))
        # post_*: [B, T, latent_dim], prior_*/h_seq: [B, T-1, *]

    # Slice for loss-time alignment (post_t for t=1..T-1)
    post_mean_loss   = post_mean[:, 1:]
    post_stoch_loss  = post_stoch[:, 1:]
    prior_mean_loss  = prior_mean
    prior_stoch_loss = prior_stoch

    S = int(getattr(world_model, "stoch_dims", 32))
    K = int(getattr(world_model, "stoch_categories", 32))

    metrics: dict[str, dict[str, float]] = {}

    print("[diag] level 0: representation alignment ...")
    metrics["level0"] = level0_representation(h_seq, post_stoch_loss, prior_stoch_loss)

    print("[diag] level 1: KL distribution health ...")
    metrics["level1"] = level1_kl(post_mean_loss, prior_mean_loss, S, K)

    print("[diag] level 2: transition collapse ...")
    metrics["level2"] = level2_transition_collapse(prior_stoch_loss)

    print("[diag] level 3: action conditioning ...")
    metrics["level3"] = level3_action_conditioning(
        world_model,
        post_stoch_prefix=post_stoch[:, :-1],         # z_{0:T-2}
        action_real=action_seq[:, 1:],                 # a_{1:T-1}
        target_z=post_stoch_loss,
    )

    if max(args.horizons) + 1 <= T:
        print("[diag] level 4: multi-step rollout ...")
        metrics["level4"] = level4_multistep_rollout(
            world_model, post_stoch, post_mean, h_seq,
            action=action_seq, horizons=args.horizons, S=S, K=K,
        )
    else:
        metrics["level4"] = {
            "_skipped": (
                f"requires sequence length T >= {max(args.horizons)+1}; "
                f"current single-transition data only provides T={T}"
            )
        }

    # Level 5: token-level recon
    print("[diag] level 5: token-level recon ...")
    cur_img_idx = world_model._bpe_to_img_idx[obs]                                     # [B, N_img]
    next_img_idx = world_model._bpe_to_img_idx[next_obs]
    metrics["level5"] = level5_token_recon(
        world_model,
        h_seq=h_seq,
        post_stoch=post_stoch_loss,
        prior_stoch=prior_stoch_loss,
        next_img_idx=next_img_idx.unsqueeze(1),
        cur_img_idx=cur_img_idx.unsqueeze(1),
    )

    pass_fail = evaluate_pass_fail(metrics)

    # Minimal-required quick-look subset
    L0, L1, L2, L3 = metrics["level0"], metrics["level1"], metrics["level2"], metrics["level3"]
    quick_look = {
        "feature_post_norm":            L0["feature_post_norm"],
        "feature_prior_norm":           L0["feature_prior_norm"],
        "feature_diff_l2":              L0["feature_diff_l2"],
        "feature_cos":                  L0["feature_cos_mean"],
        "std_post_mean":                L1["std_post_logits"],
        "std_prior_mean":               L1["std_prior_logits"],
        "prior_relative_centered_norm": L2["prior_relative_centered_norm"],
        "real_to_target_l2":            L3["real_to_target_l2"],
        "zero_to_target_l2":            L3["zero_to_target_l2"],
        "shuffle_to_target_l2":         L3["shuffle_to_target_l2"],
    }

    payload = {
        "ckpt":            str(args.ckpt),
        "config":          args.config_name,
        "num_samples":     n,
        "T":               int(T),
        "S":               S,
        "K":               K,
        "quick_look":      quick_look,
        "metrics":         metrics,
        "pass_fail":       pass_fail,
    }

    out_path = out_dir / "checklist.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print("=" * 72)
    print("WM CHECKLIST RESULT")
    print("=" * 72)
    print(json.dumps(quick_look, indent=2))
    print()
    print("PASS/FAIL by level:")
    print(json.dumps(pass_fail, indent=2))
    print()
    print(f"[diag] full report: {out_path}")


if __name__ == "__main__":
    main()
