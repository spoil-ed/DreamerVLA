"""Embedding-distribution eval for the discrete WM (longt sequence mode).

Computes embedding/latent distribution diagnostics specific to the discrete
TSSMWorldModelTransDreamerDiscrete model trained with the
PretokenizeDataset sequence mode (T = batch_length + replay_context).

Reports, in one JSON:

  ── Categorical health (S × K = stoch_dims × stoch_categories) ──────────
    posterior / prior, separately:
        per-dim entropy mean   (vs log K)
        per-dim entropy of the marginal (averaged across batch & T)
        argmax-class usage histogram → fraction of dead dims
        mean active categories per dim
        usage entropy of marginal categorical mass per dim
  ── Distribution alignment ───────────────────────────────────────────────
        KL(post || prior) per timestep (mean / median / max)
        KL split per stoch_dim (mean / max)
        symmetric KL
  ── Feature-space spread ─────────────────────────────────────────────────
        feature [h, z] norm / centered-norm / pairwise L2 / cosine
        post vs prior feature L2 + cosine
  ── Action conditioning (optional) ───────────────────────────────────────
        prior with real / zero / shuffled action; L2 to posterior target
  ── Token reconstruction ─────────────────────────────────────────────────
        token CE / accuracy via compute_loss_dict
        predicted vs GT unique tokens

Usage:
  CKPT=... CONFIG_NAME=pretokenize_wm_libero_10_discrete_longt_zfocus \
      python -m src.cli.eval_embedding_distribution
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _strip_fsdp_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        cleaned = key
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        out[cleaned] = value
    return out


def _extract_image_bpe(seqs, bpe_set, n_img_tok, which_block, device):
    from src.utils.wm_image_viz import extract_image_blocks
    rows = []
    for seq in seqs:
        blocks = extract_image_blocks(list(seq))
        bidx = which_block if which_block >= 0 else len(blocks) + which_block
        _s, _e, block_ids = blocks[bidx]
        tok_ids = [int(t) for t in block_ids if int(t) in bpe_set]
        if len(tok_ids) != n_img_tok:
            raise ValueError(f"got {len(tok_ids)} image tokens, expected {n_img_tok}")
        rows.append(tok_ids)
    return torch.tensor(rows, dtype=torch.long, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset-key", default="dataset_val_ind")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    run_dir = ckpt_path.parent.parent
    out_path = Path(args.out).resolve() if args.out else (
        run_dir / f"embed_dist_{ckpt_path.stem}_s{args.num_samples}.json"
    )

    print(f"[eval-embed] config={args.config_name}  ckpt={ckpt_path}")
    print(f"[eval-embed] out={out_path}")

    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg: DictConfig = compose(config_name=args.config_name)
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True

    print("[eval-embed] building encoder ...")
    encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("[eval-embed] building world model ...")
    wm_cfg = cfg.world_model
    instantiate_kwargs: dict[str, Any] = {}
    hidden_dim = OmegaConf.select(wm_cfg, "hidden_dim", default=None)
    if hidden_dim is None:
        hidden_dim = int(encoder.backbone.config.hidden_size)
    instantiate_kwargs["hidden_dim"] = int(hidden_dim)
    if (
        str(OmegaConf.select(wm_cfg, "io_mode", default="hidden")) == "token"
        and OmegaConf.select(wm_cfg, "num_image_tokens_vocab") is None
    ):
        vocab_mapping = encoder.backbone.model.vocabulary_mapping
        instantiate_kwargs["num_image_tokens_vocab"] = len(vocab_mapping.bpe2img)
    world_model = hydra.utils.instantiate(wm_cfg, **instantiate_kwargs).to(device)

    fsdp_precision = str(OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16"))
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    wm_dtype = dtype_map.get(fsdp_precision, torch.bfloat16)
    world_model = world_model.to(dtype=wm_dtype)

    if getattr(world_model, "spatial_codec", False):
        lm_head = encoder.backbone.lm_head
        vocab_mapping = encoder.backbone.model.vocabulary_mapping
        image_token_bpe_ids = torch.tensor(sorted(vocab_mapping.bpe2img.keys()), dtype=torch.long)
        full_vocab_size = int(lm_head.weight.shape[0])
        wm_io_mode = str(getattr(world_model, "io_mode", "hidden"))
        world_model.attach_lm_head(
            lm_head if wm_io_mode == "hidden" else None,
            image_token_bpe_ids,
            full_vocab_size=full_vocab_size,
        )

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = _strip_fsdp_prefix(payload["state_dicts"]["world_model"])
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[eval-embed] WARN missing {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[eval-embed] WARN unexpected {len(unexpected)} (first 5: {unexpected[:5]})")
    world_model.eval()

    ds_cfg = OmegaConf.select(cfg, args.dataset_key)
    dataset = hydra.utils.instantiate(ds_cfg)
    print(f"[eval-embed] dataset={args.dataset_key}  size={len(dataset)}")

    dl_kwargs = dict(cfg.dataloader)
    dl_kwargs["shuffle"] = False
    dl_kwargs["drop_last"] = False
    dl_kwargs.setdefault("batch_size", 4)
    collate_fn = getattr(dataset, "collate_fn", None)
    if callable(collate_fn):
        dl_kwargs["collate_fn"] = collate_fn
    dl_kwargs.pop("persistent_workers", None)
    dl_kwargs.pop("pin_memory", None)
    loader = DataLoader(dataset, **dl_kwargs)

    io_mode = str(getattr(world_model, "io_mode", "hidden"))
    is_discrete = hasattr(world_model, "stoch_dims") and hasattr(world_model, "stoch_categories")
    S = int(getattr(world_model, "stoch_dims", 0))
    K = int(getattr(world_model, "stoch_categories", 0))
    n_img_tok = int(getattr(world_model, "n_image_tokens", 256))
    which_block = int(OmegaConf.select(cfg, "viz.which_block", default=-2))
    img_bpe_set = set(encoder.backbone.model.vocabulary_mapping.bpe2img.keys())
    n_image_tokens_vocab = int(getattr(world_model, "num_image_tokens_vocab", 0))

    # Accumulators (keep on GPU to save host memory).
    post_logits_chunks = []   # [N*T, S, K]
    prior_logits_chunks = []  # [N*(T-1), S, K]
    post_argmax_count = torch.zeros(S, K, device=device)   # only timesteps t=1..T-1 to align with prior
    prior_argmax_count = torch.zeros(S, K, device=device)
    post_marginal_probs_sum = torch.zeros(S, K, device=device)
    prior_marginal_probs_sum = torch.zeros(S, K, device=device)
    post_marginal_n = 0
    prior_marginal_n = 0

    kl_per_step_chunks = []   # [N, T-1]
    kl_per_dim_chunks = []    # [N*(T-1), S]

    feat_post_norm_chunks = []
    feat_prior_norm_chunks = []
    feat_post_chunks = []     # for global-scale collapse stats (subset)
    feat_prior_chunks = []
    feat_diff_l2_chunks = []
    feat_cos_chunks = []

    # Action sensitivity (use last transition).
    real_to_target_chunks = []
    zero_to_target_chunks = []
    shuffle_to_target_chunks = []

    # Token recon
    token_ce_all = []
    token_acc_all = []
    static_ce_all = []
    dyn_ce_all = []
    dyn_acc_all = []
    dyn_frac_all = []
    pred_uniq_all = []
    gt_uniq_all = []
    transition_loss_all = []

    consumed = 0
    print("[eval-embed] running ...")
    with torch.no_grad():
        for batch in loader:
            if consumed >= args.num_samples:
                break
            seq_ids = batch.get("wm_obs_input_ids_seq")
            action_seq_t = batch.get("action_seq")
            if not isinstance(seq_ids, list) or not isinstance(action_seq_t, torch.Tensor):
                continue
            B = len(seq_ids)
            T = len(seq_ids[0])
            take = min(B, args.num_samples - consumed)
            if take <= 0:
                break

            flat = [list(step_ids) for sample in seq_ids[:take] for step_ids in sample]
            obs_seq_flat = _extract_image_bpe(flat, img_bpe_set, n_img_tok, which_block, device)
            obs_seq = obs_seq_flat.view(take, T, -1).long()  # [B, T, N_img]

            action_seq = action_seq_t[:take].to(device=device, dtype=wm_dtype)  # [B, T, A]

            # Build hidden_seq via token_embedder + conv_stem (matches WM forward path).
            img_idx_seq = world_model._bpe_to_img_idx[obs_seq]
            per_token = world_model.token_embedder(img_idx_seq)
            hidden_seq = world_model.conv_stem(per_token)  # [B, T, obs_dim] (or compatible)

            (
                post_mean, post_std, post_stoch,
                prior_mean, prior_std, prior_stoch,
                h_seq,
            ) = world_model._infer_dreamer_seq(hidden_seq, action_seq)
            # post_*: [B, T, latent], prior_*: [B, T-1, latent], h_seq: [B, T-1, d_model]

            # Discrete logits live in the "mean" slot. Reshape to [.., S, K].
            post_logits = post_mean.float().reshape(take, T, S, K)            # [B, T, S, K]
            prior_logits = prior_mean.float().reshape(take, T - 1, S, K)       # [B, T-1, S, K]

            # Skip t=0 of post for KL alignment; KL is computed on t=1..T-1.
            post_logits_kl = post_logits[:, 1:]                                # [B, T-1, S, K]

            # KL per (sample, timestep, stoch_dim): sum over K (categorical KL).
            log_p_post = F.log_softmax(post_logits_kl, dim=-1)
            log_p_prior = F.log_softmax(prior_logits, dim=-1)
            p_post = log_p_post.exp()
            kl_per_dim = (p_post * (log_p_post - log_p_prior)).sum(dim=-1)     # [B, T-1, S]
            kl_per_step = kl_per_dim.sum(dim=-1)                                # [B, T-1]
            kl_per_step_chunks.append(kl_per_step.detach())
            kl_per_dim_chunks.append(kl_per_dim.reshape(-1, S).detach())

            # Argmax-class & marginal-prob accumulators (post over t=1..T-1 to mirror prior).
            post_argmax = post_logits_kl.argmax(dim=-1).reshape(-1, S)         # [N, S]
            prior_argmax = prior_logits.argmax(dim=-1).reshape(-1, S)
            for s in range(S):
                post_argmax_count[s].scatter_add_(
                    0, post_argmax[:, s], torch.ones(post_argmax.shape[0], device=device),
                )
                prior_argmax_count[s].scatter_add_(
                    0, prior_argmax[:, s], torch.ones(prior_argmax.shape[0], device=device),
                )
            post_marginal_probs_sum += p_post.reshape(-1, S, K).sum(dim=0)
            prior_marginal_probs_sum += log_p_prior.exp().reshape(-1, S, K).sum(dim=0)
            post_marginal_n += p_post.reshape(-1, S, K).shape[0]
            prior_marginal_n += log_p_prior.shape[0] * log_p_prior.shape[1]

            # Feature collapse stats — use h_seq + post/prior stoch at t=1..T-1.
            post_stoch_kl = post_stoch[:, 1:].float()                          # [B, T-1, latent]
            prior_stoch_f = prior_stoch.float()
            h_seq_f = h_seq.float()
            feat_post = torch.cat([h_seq_f, post_stoch_kl], dim=-1)            # [B, T-1, D]
            feat_prior = torch.cat([h_seq_f, prior_stoch_f], dim=-1)
            feat_post_flat = feat_post.reshape(-1, feat_post.shape[-1])
            feat_prior_flat = feat_prior.reshape(-1, feat_prior.shape[-1])

            feat_post_norm_chunks.append(feat_post_flat.norm(dim=-1).detach())
            feat_prior_norm_chunks.append(feat_prior_flat.norm(dim=-1).detach())
            feat_diff_l2_chunks.append((feat_post_flat - feat_prior_flat).norm(dim=-1).detach())
            feat_cos_chunks.append(F.cosine_similarity(feat_post_flat, feat_prior_flat, dim=-1).detach())
            # Subsample for collapse stats: just keep the mean-time-step feature per sample.
            feat_post_chunks.append(feat_post[:, 0].detach())
            feat_prior_chunks.append(feat_prior[:, 0].detach())

            # Action sensitivity — use the LAST transition only (t=T-1):
            #   z_prefix = post_stoch[:, :T-1], action under test = action_seq[:, T-1:T].
            # _infer_prior_seq returns prior states for the last input = z_{T-1}.
            z_prefix = post_stoch[:take, :T - 1]
            a_real = action_seq[:take, T - 1:T]                                # [B, 1, A]
            target_feat = feat_post[:, -1]                                     # [B, D] (last step)

            def _prior_feat_for_action(act_step):
                act_full = action_seq[:take, 1:T - 1]                          # [B, T-2, A]
                act_chain = torch.cat([act_full, act_step], dim=1)             # [B, T-1, A]
                pm, ps, pz, hs = world_model._infer_prior_seq(z_prefix, act_chain)
                return torch.cat([hs[:, -1].float(), pz[:, -1].float()], dim=-1)

            real_feat = _prior_feat_for_action(a_real)
            zero_feat = _prior_feat_for_action(torch.zeros_like(a_real))
            shuffled = a_real.roll(shifts=1, dims=0) if a_real.shape[0] > 1 else a_real
            shuf_feat = _prior_feat_for_action(shuffled)
            real_to_target_chunks.append((real_feat - target_feat).norm(dim=-1).detach())
            zero_to_target_chunks.append((zero_feat - target_feat).norm(dim=-1).detach())
            shuffle_to_target_chunks.append((shuf_feat - target_feat).norm(dim=-1).detach())

            # Token recon via compute_loss_dict (sequence mode).
            ld = world_model.compute_loss_dict({
                "obs_embedding_seq": obs_seq,
                "action_seq": action_seq,
            })
            token_ce_all.append(float(ld.get("image_recon_ce_loss", torch.tensor(0.)).detach().cpu()))
            token_acc_all.append(float(ld.get("image_recon_accuracy", torch.tensor(0.)).detach().cpu()))
            static_ce_all.append(float(ld.get("image_static_ce_loss", torch.tensor(0.)).detach().cpu()))
            dyn_ce_all.append(float(ld.get("image_dynamic_ce_loss", torch.tensor(0.)).detach().cpu()))
            dyn_acc_all.append(float(ld.get("image_dynamic_accuracy", torch.tensor(0.)).detach().cpu()))
            dyn_frac_all.append(float(ld.get("image_dynamic_fraction", torch.tensor(0.)).detach().cpu()))
            pred_uniq_all.append(float(ld.get("pred_unique_tokens", torch.tensor(0.)).detach().cpu()))
            gt_uniq_all.append(float(ld.get("gt_unique_tokens", torch.tensor(0.)).detach().cpu()))
            transition_loss_all.append(float(ld.get("transition_loss", torch.tensor(0.)).detach().cpu()))

            consumed += take
            print(f"[eval-embed] {consumed}/{args.num_samples}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    cat = torch.cat
    kl_per_step = cat(kl_per_step_chunks, dim=0).reshape(-1)                   # [N*(T-1)]
    kl_per_dim = cat(kl_per_dim_chunks, dim=0)                                 # [N*(T-1), S]
    feat_post_norm = cat(feat_post_norm_chunks)
    feat_prior_norm = cat(feat_prior_norm_chunks)
    feat_diff_l2 = cat(feat_diff_l2_chunks)
    feat_cos = cat(feat_cos_chunks)
    real_to_t = cat(real_to_target_chunks)
    zero_to_t = cat(zero_to_target_chunks)
    shuf_to_t = cat(shuffle_to_target_chunks)
    feat_post_sub = cat(feat_post_chunks, dim=0)
    feat_prior_sub = cat(feat_prior_chunks, dim=0)

    def _stats(x: torch.Tensor) -> dict[str, float]:
        x = x.float()
        return {
            "mean": float(x.mean().cpu()),
            "median": float(x.median().cpu()),
            "min": float(x.min().cpu()),
            "max": float(x.max().cpu()),
            "std": float(x.std().cpu()),
        }

    # Categorical health metrics
    log_K = math.log(K) if K > 0 else 1.0

    def _categorical_health(argmax_count: torch.Tensor, marginal_sum: torch.Tensor, marginal_n: int):
        # argmax_count: [S, K] of how often each class was chosen as argmax across samples.
        # marginal_sum: [S, K] of summed soft probs.
        # Per-dim: how many distinct categories ever appeared as argmax (active);
        # max-class fraction (peakedness), entropy of marginal.
        n_per_dim = argmax_count.sum(dim=-1).clamp_min(1.0)                    # [S]
        max_frac = argmax_count.max(dim=-1).values / n_per_dim                  # [S]
        active_cats = (argmax_count > 0).sum(dim=-1).float()                    # [S]
        dead_dim_count = int((max_frac > 0.99).sum().cpu())                     # >=99% in one class
        # Marginal entropy (per-dim soft mean distribution).
        marginal_p = marginal_sum / max(marginal_n, 1)
        marginal_p = marginal_p / marginal_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        marg_ent = -(marginal_p * marginal_p.clamp_min(1e-12).log()).sum(dim=-1)  # [S], in nats
        # Per-sample entropy mean: avg of -sum p log p averaged over batch
        # We approximate from marginal_p -- this is already the per-dim marginal H.
        return {
            "stoch_dims": S,
            "stoch_categories": K,
            "log_K_nats": log_K,
            "argmax_max_class_frac_mean": float(max_frac.mean().cpu()),
            "argmax_max_class_frac_max": float(max_frac.max().cpu()),
            "argmax_max_class_frac_min": float(max_frac.min().cpu()),
            "argmax_active_categories_mean": float(active_cats.mean().cpu()),
            "argmax_active_categories_min": float(active_cats.min().cpu()),
            "argmax_active_categories_max": float(active_cats.max().cpu()),
            "dead_dims_count_argmax_99": dead_dim_count,
            "marginal_entropy_mean_nats": float(marg_ent.mean().cpu()),
            "marginal_entropy_min_nats": float(marg_ent.min().cpu()),
            "marginal_entropy_max_nats": float(marg_ent.max().cpu()),
            "marginal_entropy_normalized": float((marg_ent.mean() / log_K).cpu()),
        }

    post_health = _categorical_health(post_argmax_count, post_marginal_probs_sum, post_marginal_n)
    prior_health = _categorical_health(prior_argmax_count, prior_marginal_probs_sum, prior_marginal_n)

    # Argmax agreement post vs prior — fraction of (sample, dim) where both pick the same class.
    # Easier: do it on-the-fly via a fresh pass — but cheaper to use marginal correlations.
    # Use joint stat: cosine sim between marginal distributions per dim.
    post_marg_p = post_marginal_probs_sum / max(post_marginal_n, 1)
    post_marg_p = post_marg_p / post_marg_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    prior_marg_p = prior_marginal_probs_sum / max(prior_marginal_n, 1)
    prior_marg_p = prior_marg_p / prior_marg_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    marginal_kl_post_vs_prior = (
        post_marg_p * (post_marg_p.clamp_min(1e-12).log() - prior_marg_p.clamp_min(1e-12).log())
    ).sum(dim=-1)                                                              # [S]
    marginal_l1 = (post_marg_p - prior_marg_p).abs().sum(dim=-1) / 2            # [S]

    # Feature collapse stats — between-sample spread (last-t feature).
    def _collapse(feat: torch.Tensor) -> dict[str, float]:
        feat = feat.float()
        norm = feat.norm(dim=-1)
        center = feat - feat.mean(dim=0, keepdim=True)
        cnorm = center.norm(dim=-1)
        # pairwise within batch (shuffled adjacent)
        if feat.shape[0] > 1:
            pdists = (feat[:-1] - feat[1:]).norm(dim=-1)
            cos_adj = F.cosine_similarity(feat[:-1], feat[1:], dim=-1)
        else:
            pdists = torch.zeros(1)
            cos_adj = torch.ones(1)
        return {
            "feature_norm_mean": float(norm.mean().cpu()),
            "centered_norm_mean": float(cnorm.mean().cpu()),
            "relative_centered_norm": float((cnorm.mean() / norm.mean().clamp_min(1e-8)).cpu()),
            "pairwise_l2_mean": float(pdists.mean().cpu()),
            "adjacent_cos_mean": float(cos_adj.mean().cpu()),
        }

    post_collapse = _collapse(feat_post_sub)
    prior_collapse = _collapse(feat_prior_sub)

    report = {
        "ckpt": str(ckpt_path),
        "config": args.config_name,
        "dataset_key": args.dataset_key,
        "num_samples_processed": consumed,
        "stoch_grid": {"S": S, "K": K, "log_K_nats": log_K},
        "categorical_health": {
            "posterior": post_health,
            "prior": prior_health,
            "marginal_kl_post_vs_prior_per_dim": {
                "mean": float(marginal_kl_post_vs_prior.mean().cpu()),
                "median": float(marginal_kl_post_vs_prior.median().cpu()),
                "max": float(marginal_kl_post_vs_prior.max().cpu()),
            },
            "marginal_l1_post_vs_prior_per_dim": {
                "mean": float(marginal_l1.mean().cpu()),
                "median": float(marginal_l1.median().cpu()),
                "max": float(marginal_l1.max().cpu()),
            },
        },
        "kl_post_prior": {
            "per_step_total_nats": _stats(kl_per_step),
            "per_dim_nats": _stats(kl_per_dim.reshape(-1)),
            "per_dim_mean_per_dim": _stats(kl_per_dim.mean(dim=0)),
        },
        "feature_alignment": {
            "feat_post_norm": _stats(feat_post_norm),
            "feat_prior_norm": _stats(feat_prior_norm),
            "feat_diff_l2": _stats(feat_diff_l2),
            "feat_cos": _stats(feat_cos),
            "relative_feat_diff_mean": float((feat_diff_l2.float().mean() / feat_post_norm.float().mean().clamp_min(1e-8)).cpu()),
        },
        "feature_collapse_t1": {
            "posterior": post_collapse,
            "prior": prior_collapse,
        },
        "action_conditioning": {
            "real_to_target_l2_mean": float(real_to_t.float().mean().cpu()),
            "zero_to_target_l2_mean": float(zero_to_t.float().mean().cpu()),
            "shuffle_to_target_l2_mean": float(shuf_to_t.float().mean().cpu()),
            "margin_zero": float((zero_to_t.float().mean() - real_to_t.float().mean()).cpu()),
            "margin_shuffle": float((shuf_to_t.float().mean() - real_to_t.float().mean()).cpu()),
        },
        "token_reconstruction": {
            "n_image_tokens_vocab": n_image_tokens_vocab,
            "random_acc_baseline": 1.0 / max(n_image_tokens_vocab, 1),
            "token_ce_mean": float(sum(token_ce_all) / len(token_ce_all)) if token_ce_all else None,
            "token_acc_mean": float(sum(token_acc_all) / len(token_acc_all)) if token_acc_all else None,
            "static_ce_mean": float(sum(static_ce_all) / len(static_ce_all)) if static_ce_all else None,
            "dynamic_ce_mean": float(sum(dyn_ce_all) / len(dyn_ce_all)) if dyn_ce_all else None,
            "dynamic_acc_mean": float(sum(dyn_acc_all) / len(dyn_acc_all)) if dyn_acc_all else None,
            "dynamic_fraction_mean": float(sum(dyn_frac_all) / len(dyn_frac_all)) if dyn_frac_all else None,
            "pred_unique_tokens_mean": float(sum(pred_uniq_all) / len(pred_uniq_all)) if pred_uniq_all else None,
            "gt_unique_tokens_mean": float(sum(gt_uniq_all) / len(gt_uniq_all)) if gt_uniq_all else None,
            "transition_loss_mean": float(sum(transition_loss_all) / len(transition_loss_all)) if transition_loss_all else None,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[eval-embed] wrote {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
