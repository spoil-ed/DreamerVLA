"""WM Checklist: prior/posterior alignment diagnostic.

Computes the WM Checklist (Level 0–5) metrics on a saved TSSMWorldModelTransDreamer
checkpoint and writes a JSON file matching the schema of
``wm_checklist_*.json``.

Levels:
  0  representation-space consistency  (post vs prior features)
  1  KL / distribution health
  2  prior collapse detection
  3  action-conditioning sensitivity
  4  multi-step rollout            (skipped for the T=2 pretokenized format)
  5  token-level reconstruction    (read from compute_loss_dict)

Usage:
  python -m src.cli.diagnose_wm \
      --config-name pretokenize_wm_libero_10 \
      --ckpt data/outputs/pretokenize_wm/<run>/checkpoints/<file>.ckpt \
      --num-samples 128 \
      --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _strip_fsdp_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned = key
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        out[cleaned] = value
    return out


def _load_wm_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dicts" not in payload or "world_model" not in payload["state_dicts"]:
        raise ValueError(f"ckpt {ckpt_path} has no state_dicts.world_model")
    return _strip_fsdp_prefix(payload["state_dicts"]["world_model"])


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1).float().norm(dim=-1)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.flatten(start_dim=1).float()
    bf = b.flatten(start_dim=1).float()
    return F.cosine_similarity(af, bf, dim=-1)


def _mean(x: torch.Tensor) -> float:
    return float(x.float().mean().detach().cpu())


def _median(x: torch.Tensor) -> float:
    return float(x.float().median().detach().cpu())


def _max(x: torch.Tensor) -> float:
    return float(x.float().max().detach().cpu())


def _min(x: torch.Tensor) -> float:
    return float(x.float().min().detach().cpu())


# ── Per-batch posterior/prior extraction ─────────────────────────────────────


def _to_long_or_dtype(x: torch.Tensor, io_mode: str, device, dtype) -> torch.Tensor:
    return x.to(device=device, dtype=torch.long) if io_mode == "token" else x.to(device=device, dtype=dtype)


def _reduce_action(action: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    """Match TSSMWorldModelTransDreamer.compute_loss_dict's action chunk reduction."""
    if action.ndim == 3:
        if action_mask is not None:
            mask = action_mask.to(device=action.device, dtype=action.dtype).unsqueeze(-1)
            action = action * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            return action.sum(dim=1) / denom
        return action.mean(dim=1)
    return action


def _is_discrete(world_model) -> bool:
    """Returns True for TSSMWorldModelTransDreamerDiscrete (has stoch_dims attr)."""
    return hasattr(world_model, "stoch_dims") and hasattr(world_model, "stoch_categories")


def _entropy_from_dist(world_model, mean_or_logits: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy. For discrete uses logits (stored in mean slot), for
    Gaussian uses (μ, σ). Returns [B] (or matching leading dims).
    """
    if _is_discrete(world_model):
        S, K = world_model.stoch_dims, world_model.stoch_categories
        logits = mean_or_logits.reshape(*mean_or_logits.shape[:-1], S, K)
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        # H = -sum_k p log p, sum over S independent dims
        return -(p * log_p).sum(dim=-1).sum(dim=-1)
    # Gaussian: H(N(μ, σ)) = 0.5 log(2πe σ²) per dim, summed.
    std = std.clamp_min(1e-6)
    return (0.5 * (1.0 + torch.log(2.0 * math.pi * std.pow(2)))).sum(dim=-1)


def _imagine_rollout(
    world_model,
    z0: torch.Tensor,                # [B, latent_dim] from posterior at t=0
    action_repeat: torch.Tensor,      # [B, action_dim] reused for all H steps
    horizons: list[int],
):
    """Sequential imagination rollout (no GT comparison).

    Starts from z_post[0] and repeats `action_repeat` for max(horizons) steps.
    At each requested horizon h returns:
        feature[h]      = cat(z_h, h_h)            [B, latent+d_model]
        feature_norm[h]
        prior_entropy[h]
        prior_std_min[h]   (Gaussian only; for discrete returns min class probability)
    """
    H_max = max(horizons)
    B = z0.shape[0]
    # Repeat action over horizon: [B, H_max, A]
    actions = action_repeat.unsqueeze(1).expand(B, H_max, -1).contiguous()

    z_list = [z0]
    feat_at_h: dict[int, torch.Tensor] = {}
    norm_at_h: dict[int, torch.Tensor] = {}
    entropy_at_h: dict[int, torch.Tensor] = {}
    std_min_at_h: dict[int, torch.Tensor] = {}

    for t in range(H_max):
        z_prefix = torch.stack(z_list, dim=1)             # [B, t+1, latent]
        a_prefix = actions[:, : t + 1]                    # [B, t+1, action]
        prior_mean, prior_std, prior_stoch, h_seq = world_model._infer_prior_seq(
            z_prefix, a_prefix,
        )
        z_next = prior_stoch[:, -1]                       # [B, latent]
        h_next = h_seq[:, -1]                             # [B, d_model]
        z_list.append(z_next)

        feat = torch.cat([z_next, h_next], dim=-1).float()
        feat_norm = feat.norm(dim=-1)                     # [B]

        ent = _entropy_from_dist(
            world_model, prior_mean[:, -1].float(), prior_std[:, -1].float(),
        )                                                  # [B]
        if _is_discrete(world_model):
            S, K = world_model.stoch_dims, world_model.stoch_categories
            logits = prior_mean[:, -1].reshape(B, S, K).float()
            probs = F.softmax(logits, dim=-1)
            std_min_proxy = probs.min(dim=-1).values.mean(dim=-1)   # mean-min class prob per stoch_dim
        else:
            std_min_proxy = prior_std[:, -1].float().min(dim=-1).values

        h_idx = t + 1                                     # 1-indexed horizon
        if h_idx in horizons:
            feat_at_h[h_idx] = feat
            norm_at_h[h_idx] = feat_norm
            entropy_at_h[h_idx] = ent
            std_min_at_h[h_idx] = std_min_proxy

    return {
        "feat": feat_at_h,
        "feat_norm": norm_at_h,
        "entropy": entropy_at_h,
        "std_min": std_min_at_h,
    }


def _build_seq_inputs(world_model, batch: dict[str, Any], device, dtype):
    """Replicate the (obs, next_obs, action) → (hidden_seq, action_seq) wiring in
    compute_loss_dict, but stop just before the RSSM call so we can probe it.

    Returns:
        hidden_seq: [B, 2, obs_dim]   (continuous, post conv_stem)
        action_seq: [B, 2, action_dim]
        raw_bpe_ids_seq: [B, 2, n_img_tok] long  (token mode only, else None)
    """
    io_mode = str(getattr(world_model, "io_mode", "hidden"))

    obs = _to_long_or_dtype(batch["obs_embedding"], io_mode, device, dtype)
    next_obs = _to_long_or_dtype(batch["next_obs_embedding"], io_mode, device, dtype)
    action = batch["action"].to(device=device, dtype=dtype)
    action_step = _reduce_action(action, batch.get("action_mask"))

    raw_bpe_ids_seq = None
    if io_mode == "token":
        raw_bpe_ids_seq = torch.stack([obs, next_obs], dim=1).long()         # [B, 2, N_img]
        bpe2img = world_model._bpe_to_img_idx
        img_idx_seq = bpe2img[raw_bpe_ids_seq]
        if (img_idx_seq < 0).any():
            raise ValueError("token mode: input contains non-image BPE ids")
        per_token = world_model.token_embedder(img_idx_seq)                  # [B, 2, N_img, d_embed]
        hidden_seq = world_model.conv_stem(per_token)                        # [B, 2, obs_dim]
    elif getattr(world_model, "spatial_codec", False):
        hidden_seq_raw = torch.stack([obs, next_obs], dim=1)                 # [B, 2, N_img, C_in]
        hidden_seq = world_model.conv_stem(hidden_seq_raw)
    else:
        hidden_seq = torch.stack([obs, next_obs], dim=1)                     # [B, 2, obs_dim]

    action_seq = torch.stack(
        [torch.zeros_like(action_step), action_step], dim=1
    )                                                                         # [B, 2, action_dim]
    return hidden_seq, action_seq, raw_bpe_ids_seq


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WM prior/posterior alignment diagnostic")
    parser.add_argument("--config-name", default="pretokenize_wm_libero_10")
    parser.add_argument("--ckpt", required=True, help="Path to a workspace .ckpt with state_dicts.world_model")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-key", default="dataset_val_ind",
                        help="Which dataset section to use (dataset / dataset_val_ind / dataset_val_ood)")
    parser.add_argument("--out", default=None, help="Output JSON path. Default: <ckpt-dir>/../wm_checklist_<tag>.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    run_dir = ckpt_path.parent.parent
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else run_dir / f"wm_checklist_{ckpt_path.stem}_s{args.num_samples}.json"
    )

    print(f"[diagnose_wm] config={args.config_name}  ckpt={ckpt_path}")
    print(f"[diagnose_wm] out={out_path}")

    # ── compose cfg & build modules ──────────────────────────────────────────
    configs_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg: DictConfig = compose(config_name=args.config_name)
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.encoder.freeze_backbone = True

    print("[diagnose_wm] building encoder ...")
    encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("[diagnose_wm] building world model ...")
    wm_cfg = cfg.world_model
    instantiate_kwargs: dict[str, Any] = {}
    hidden_dim = OmegaConf.select(wm_cfg, "hidden_dim", default=None)
    if hidden_dim is None:
        # Pull from encoder hidden size if cfg hides it
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

    # Attach image-token mapping (token-mode WM requires this for forward).
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

    # Load WM weights.
    state_dict = _load_wm_state_dict(ckpt_path)
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[diagnose_wm] WARN missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[diagnose_wm] WARN unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    world_model.eval()

    # ── Build dataset / dataloader ───────────────────────────────────────────
    ds_cfg = OmegaConf.select(cfg, args.dataset_key, default=None)
    if ds_cfg is None:
        ds_cfg = OmegaConf.select(cfg, "dataset")
    print(f"[diagnose_wm] dataset = {args.dataset_key}")
    dataset = hydra.utils.instantiate(ds_cfg)
    print(f"[diagnose_wm] dataset size = {len(dataset)}")

    dl_kwargs = dict(cfg.dataloader)
    dl_kwargs["shuffle"] = False
    dl_kwargs["drop_last"] = False
    dl_kwargs.setdefault("batch_size", 4)
    collate_fn = getattr(dataset, "collate_fn", None)
    if callable(collate_fn):
        dl_kwargs["collate_fn"] = collate_fn
    dl_kwargs.pop("persistent_workers", None)
    dl_kwargs.pop("pin_memory", None)
    dataloader = DataLoader(dataset, **dl_kwargs)

    # ── Accumulators ─────────────────────────────────────────────────────────
    feat_post_norm_all: list[torch.Tensor] = []
    feat_prior_norm_all: list[torch.Tensor] = []
    feat_diff_l2_all:   list[torch.Tensor] = []
    feat_cos_all:       list[torch.Tensor] = []
    z_diff_l2_all:      list[torch.Tensor] = []
    z_cos_all:          list[torch.Tensor] = []
    h_diff_l2_all:      list[torch.Tensor] = []

    kl_all:             list[torch.Tensor] = []
    mu_post_norm_all:   list[torch.Tensor] = []
    mu_prior_norm_all:  list[torch.Tensor] = []
    mu_diff_l2_all:     list[torch.Tensor] = []
    std_post_all:       list[torch.Tensor] = []
    std_prior_all:      list[torch.Tensor] = []

    prior_feat_all:     list[torch.Tensor] = []
    post_feat_all:      list[torch.Tensor] = []

    real_to_target_all:    list[torch.Tensor] = []
    zero_to_target_all:    list[torch.Tensor] = []
    shuffle_to_target_all: list[torch.Tensor] = []
    real_vs_zero_all:      list[torch.Tensor] = []
    real_vs_shuffle_all:   list[torch.Tensor] = []

    token_ce_all = []
    token_acc_all = []
    static_ce_all = []
    dynamic_ce_all = []
    static_acc_all = []
    dynamic_acc_all = []
    dynamic_frac_all = []

    # Auxiliary main-task metrics (transition / reward) and posterior/prior entropy.
    transition_loss_all: list[float] = []
    reward_loss_all: list[float] = []
    post_entropy_all: list[torch.Tensor] = []
    prior_entropy_all: list[torch.Tensor] = []

    # Latent dead-unit / saturation tracking.
    if _is_discrete(world_model):
        # For each stoch_dim, count category usage across all samples we see.
        S, K = world_model.stoch_dims, world_model.stoch_categories
        post_cat_count = torch.zeros(S, K, device=device)
        prior_cat_count = torch.zeros(S, K, device=device)
    else:
        # For Gaussian: track per-dim std distribution to spot floor saturation.
        std_dim_collect: list[torch.Tensor] = []

    # Level 4: imagination rollout horizons.
    H_LIST = [1, 5, 10, 20]
    rollout_norm_at_h: dict[int, list[torch.Tensor]] = {h: [] for h in H_LIST}
    rollout_entropy_at_h: dict[int, list[torch.Tensor]] = {h: [] for h in H_LIST}
    rollout_std_min_at_h: dict[int, list[torch.Tensor]] = {h: [] for h in H_LIST}
    rollout_drift_from_h1: dict[int, list[torch.Tensor]] = {h: [] for h in H_LIST}

    consumed = 0
    n_image_tokens_vocab = int(getattr(world_model, "num_image_tokens_vocab", 0))

    print("[diagnose_wm] running ...")
    with torch.no_grad():
        for batch in dataloader:
            if consumed >= args.num_samples:
                break

            # ── Build (obs, next_obs, action) batch via workspace pipeline ───
            obs_ids = batch.get("wm_obs_input_ids")
            nxt_ids = batch.get("wm_next_obs_input_ids")
            if not isinstance(obs_ids, list) or not isinstance(nxt_ids, list):
                continue

            wm_io_mode = str(getattr(world_model, "io_mode", "hidden"))
            if wm_io_mode == "token":
                # Encode-side image-BPE extraction: same as PretokenizeWMWorkspace.
                from src.utils.wm_image_viz import extract_image_blocks
                n_img_tok = int(getattr(world_model, "n_image_tokens", 256))
                img_bpe = set(encoder.backbone.model.vocabulary_mapping.bpe2img.keys())

                def _extract(seqs):
                    rows = []
                    for seq in seqs:
                        blocks = extract_image_blocks(list(seq))
                        which = -2  # third-view of the cur frame, mirrors workspace default
                        bidx = which if which >= 0 else len(blocks) + which
                        _s, _e, block_ids = blocks[bidx]
                        tok_ids = [int(t) for t in block_ids if int(t) in img_bpe]
                        if len(tok_ids) != n_img_tok:
                            raise ValueError(
                                f"image block has {len(tok_ids)} image tokens, expected {n_img_tok}"
                            )
                        rows.append(tok_ids)
                    return torch.tensor(rows, dtype=torch.long, device=device)

                obs_emb = _extract(obs_ids)
                nxt_emb = _extract(nxt_ids)
            else:
                # Hidden-mode: encode pooled hidden states from the encoder.
                def _encode_pooled(seqs):
                    labels = [[-100] * len(s) for s in seqs]
                    lengths = [len(s) for s in seqs]
                    _, _, _, h, _, _, _ = encoder.backbone(
                        input_ids=seqs, labels=labels, training=True,
                        output_hidden_states=True, att_mask=False,
                    )
                    mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
                    for i, L in enumerate(lengths):
                        mask[i, :L] = True
                    w = mask.to(h.dtype).unsqueeze(-1)
                    pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
                    return pooled.float()
                obs_emb = _encode_pooled(obs_ids)
                nxt_emb = _encode_pooled(nxt_ids)

            action = (batch.get("conditioning_action") or batch.get("action"))
            assert isinstance(action, torch.Tensor)

            wm_batch: dict[str, Any] = {
                "obs_embedding": obs_emb,
                "next_obs_embedding": nxt_emb,
                "action": action,
                "action_mask": batch.get("action_mask"),
            }
            reward = batch.get("reward")
            if isinstance(reward, torch.Tensor):
                wm_batch["reward"] = reward

            B = obs_emb.shape[0]
            take = min(B, args.num_samples - consumed)
            if take <= 0:
                break

            # Build the same hidden_seq + action_seq the WM expects.
            hidden_seq, action_seq, raw_bpe_ids_seq = _build_seq_inputs(
                world_model, wm_batch, device=device, dtype=wm_dtype,
            )
            hidden_seq = hidden_seq[:take]
            action_seq = action_seq[:take]
            if raw_bpe_ids_seq is not None:
                raw_bpe_ids_seq = raw_bpe_ids_seq[:take]

            # ── _infer_dreamer_seq: post and prior (under same h_t) ───────────
            (
                post_mean, post_std, post_stoch,
                prior_mean, prior_std, prior_stoch,
                h_seq,
            ) = world_model._infer_dreamer_seq(hidden_seq, action_seq)
            # Keep only the t=1 transition (the only one available for T=2).
            post_mean_t = post_mean[:, 1].float()           # [B, latent]
            post_std_t = post_std[:, 1].float()
            post_stoch_t = post_stoch[:, 1].float()
            prior_mean_t = prior_mean[:, 0].float()
            prior_std_t = prior_std[:, 0].float()
            prior_stoch_t = prior_stoch[:, 0].float()
            h_t = h_seq[:, 0].float()                       # [B, d_model]
            # In TransDreamer's inference order the same h is shared for post and prior.
            h_post = h_t
            h_prior = h_t

            feat_post = torch.cat([h_post, post_stoch_t], dim=-1)     # [B, d_model+latent]
            feat_prior = torch.cat([h_prior, prior_stoch_t], dim=-1)

            # Level 0
            feat_post_norm_all.append(_l2_norm(feat_post))
            feat_prior_norm_all.append(_l2_norm(feat_prior))
            feat_diff_l2_all.append(_l2_norm(feat_post - feat_prior))
            feat_cos_all.append(_cos_sim(feat_post, feat_prior))
            z_diff_l2_all.append(_l2_norm(post_stoch_t - prior_stoch_t))
            z_cos_all.append(_cos_sim(post_stoch_t, prior_stoch_t))
            h_diff_l2_all.append(_l2_norm(h_post - h_prior))

            # Level 1
            kl = world_model._gaussian_kl(
                post_mean=post_mean_t, post_std=post_std_t,
                prior_mean=prior_mean_t, prior_std=prior_std_t,
            )
            kl_all.append(kl.detach().reshape(-1))
            mu_post_norm_all.append(_l2_norm(post_mean_t))
            mu_prior_norm_all.append(_l2_norm(prior_mean_t))
            mu_diff_l2_all.append(_l2_norm(post_mean_t - prior_mean_t))
            std_post_all.append(post_std_t.flatten())
            std_prior_all.append(prior_std_t.flatten())

            # Level 2 — accumulate raw features for batch-stat computations later.
            prior_feat_all.append(feat_prior.detach())
            post_feat_all.append(feat_post.detach())

            # Level 3 — re-run prior with zero / shuffled actions.
            #   _infer_prior_seq inputs: (stoch_seq z_{0:K-1}, action_seq a_{1:K}).
            #   For T=2 the prefix is z_0 = post_stoch[:, 0] and the action under test is action_seq[:, 1].
            z0 = post_stoch[:, :1]                       # [B, 1, latent]
            a1 = action_seq[:, 1:2]                      # [B, 1, action]
            target_feat = feat_post                      # posterior at t=1 is the target

            def _prior_feat_with_action(act):
                pm, ps, pz, hs = world_model._infer_prior_seq(z0, act)
                feat = torch.cat([hs[:, 0], pz[:, 0]], dim=-1).float()
                return feat

            real_feat = _prior_feat_with_action(a1)
            zero_feat = _prior_feat_with_action(torch.zeros_like(a1))
            if a1.shape[0] > 1:
                shuffled = a1.roll(shifts=1, dims=0)
            else:
                shuffled = a1                             # degenerate; metric will be ~0
            shuffle_feat = _prior_feat_with_action(shuffled)

            real_to_target_all.append(_l2_norm(real_feat - target_feat))
            zero_to_target_all.append(_l2_norm(zero_feat - target_feat))
            shuffle_to_target_all.append(_l2_norm(shuffle_feat - target_feat))
            real_vs_zero_all.append(_l2_norm(real_feat - zero_feat))
            real_vs_shuffle_all.append(_l2_norm(real_feat - shuffle_feat))

            # ── Posterior / prior entropy at t=1 ──────────────────────────────
            post_entropy_all.append(_entropy_from_dist(world_model, post_mean_t, post_std_t).detach())
            prior_entropy_all.append(_entropy_from_dist(world_model, prior_mean_t, prior_std_t).detach())

            # ── Latent dead-unit / saturation tracking ────────────────────────
            if _is_discrete(world_model):
                S, K = world_model.stoch_dims, world_model.stoch_categories
                # Use argmax per stoch_dim as the "active" category for this sample.
                post_idx = post_mean_t.reshape(take, S, K).argmax(dim=-1)    # [B, S]
                prior_idx = prior_mean_t.reshape(take, S, K).argmax(dim=-1)
                post_cat_count.scatter_add_(
                    1, post_idx.permute(1, 0).reshape(S, take).clone().long(),  # [S, B]
                    torch.ones(S, take, device=device),
                ) if False else None  # avoid scatter idiom; fall through to simple loop below
                for s in range(S):
                    post_cat_count[s].scatter_add_(0, post_idx[:, s], torch.ones(take, device=device))
                    prior_cat_count[s].scatter_add_(0, prior_idx[:, s], torch.ones(take, device=device))
            else:
                std_dim_collect.append(post_std_t.detach().cpu())

            # ── Level 4 — imagination rollout ────────────────────────────────
            # Repeat the same action_step over horizon, start from post_0.
            # Keep inputs in the WM's compute dtype (bf16) — LayerNorm in
            # act_stoch_emb refuses mixed dtype.
            roll = _imagine_rollout(
                world_model,
                z0=post_stoch[:take, 0],
                action_repeat=action_seq[:take, 1],
                horizons=H_LIST,
            )
            feat_h1_for_drift = roll["feat"].get(1)
            for h in H_LIST:
                if h not in roll["feat_norm"]:
                    continue
                rollout_norm_at_h[h].append(roll["feat_norm"][h].detach())
                rollout_entropy_at_h[h].append(roll["entropy"][h].detach())
                rollout_std_min_at_h[h].append(roll["std_min"][h].detach())
                if feat_h1_for_drift is not None:
                    drift = (roll["feat"][h] - feat_h1_for_drift).norm(dim=-1)
                    rollout_drift_from_h1[h].append(drift.detach())

            # Level 5 — Token CE / accuracy + transition / reward losses via compute_loss_dict.
            if wm_io_mode == "token":
                ld = world_model.compute_loss_dict({
                    "obs_embedding": obs_emb[:take],
                    "next_obs_embedding": nxt_emb[:take],
                    "action": action[:take],
                    "action_mask": batch.get("action_mask"),
                })
                token_ce_all.append(float(ld.get("image_recon_ce_loss", torch.tensor(0.)).detach().cpu()))
                token_acc_all.append(float(ld.get("image_recon_accuracy", torch.tensor(0.)).detach().cpu()))
                static_ce_all.append(float(ld.get("image_static_ce_loss", torch.tensor(0.)).detach().cpu()))
                dynamic_ce_all.append(float(ld.get("image_dynamic_ce_loss", torch.tensor(0.)).detach().cpu()))
                static_acc_all.append(float(ld.get("image_static_accuracy", torch.tensor(0.)).detach().cpu()))
                dynamic_acc_all.append(float(ld.get("image_dynamic_accuracy", torch.tensor(0.)).detach().cpu()))
                dynamic_frac_all.append(float(ld.get("image_dynamic_fraction", torch.tensor(0.)).detach().cpu()))
                transition_loss_all.append(float(ld.get("transition_loss", torch.tensor(0.)).detach().cpu()))
                reward_loss_all.append(float(ld.get("reward_loss", torch.tensor(0.)).detach().cpu()))

            consumed += take
            print(f"[diagnose_wm] {consumed}/{args.num_samples} samples processed")

    # ── Aggregate ────────────────────────────────────────────────────────────
    cat = torch.cat
    feat_post_norm = cat(feat_post_norm_all)
    feat_prior_norm = cat(feat_prior_norm_all)
    feat_diff_l2 = cat(feat_diff_l2_all)
    feat_cos = cat(feat_cos_all)
    z_diff_l2 = cat(z_diff_l2_all)
    z_cos = cat(z_cos_all)
    h_diff_l2 = cat(h_diff_l2_all)

    kl = cat(kl_all)
    mu_post_norm = cat(mu_post_norm_all)
    mu_prior_norm = cat(mu_prior_norm_all)
    mu_diff_l2 = cat(mu_diff_l2_all)
    std_post = cat(std_post_all)
    std_prior = cat(std_prior_all)

    prior_feat = cat(prior_feat_all, dim=0)            # [N, D]
    post_feat = cat(post_feat_all, dim=0)

    real_to_target = cat(real_to_target_all)
    zero_to_target = cat(zero_to_target_all)
    shuffle_to_target = cat(shuffle_to_target_all)

    # Level 2 batch statistics.
    def _collapse_stats(feat: torch.Tensor) -> dict[str, float]:
        feat = feat.float()
        norm = feat.norm(dim=-1)
        center = feat - feat.mean(dim=0, keepdim=True)
        centered_norm = center.norm(dim=-1)
        n = feat.shape[0]
        if n > 1:
            # Pairwise distances between consecutive (random-order) samples are a cheap proxy.
            pdists = (feat[:-1] - feat[1:]).norm(dim=-1)
            cos_adj = F.cosine_similarity(feat[:-1], feat[1:], dim=-1)
        else:
            pdists = torch.zeros(1)
            cos_adj = torch.ones(1)
        return {
            "feature_norm_mean": _mean(norm),
            "centered_norm_mean": _mean(centered_norm),
            "relative_centered_norm": _mean(centered_norm) / max(_mean(norm), 1e-8),
            "pairwise_l2_mean": _mean(pdists),
            "pairwise_l2_median": _median(pdists),
            "adjacent_cos_mean": _mean(cos_adj),
        }

    prior_collapse = _collapse_stats(prior_feat)
    post_collapse = _collapse_stats(post_feat)

    # ── Build report ─────────────────────────────────────────────────────────
    config_meta = {
        "image_decoder_stoch_source": str(OmegaConf.select(wm_cfg, "image_decoder_stoch_source", default="post")),
        "image_decoder_detach_mode": str(OmegaConf.select(wm_cfg, "image_decoder_detach_mode", default="none")),
        "free_nats": float(OmegaConf.select(wm_cfg, "free_nats", default=0.0)),
        "kl_balance": float(OmegaConf.select(wm_cfg, "kl_balance", default=0.0)),
    }

    level0 = {
        "feature_post_norm_mean": _mean(feat_post_norm),
        "feature_prior_norm_mean": _mean(feat_prior_norm),
        "feature_diff_l2_mean": _mean(feat_diff_l2),
        "feature_diff_l2_median": _median(feat_diff_l2),
        "feature_cos_mean": _mean(feat_cos),
        "z_diff_l2_mean": _mean(z_diff_l2),
        "z_cos_mean": _mean(z_cos),
        "h_diff_l2_mean": _mean(h_diff_l2),
        "relative_feature_diff": _mean(feat_diff_l2) / max(_mean(feat_post_norm), 1e-8),
    }

    level1 = {
        "kl_mean": _mean(kl),
        "kl_median": _median(kl),
        "kl_max": _max(kl),
        "mu_post_norm_mean": _mean(mu_post_norm),
        "mu_prior_norm_mean": _mean(mu_prior_norm),
        "mu_diff_l2_mean": _mean(mu_diff_l2),
        "std_post_mean": _mean(std_post),
        "std_prior_mean": _mean(std_prior),
        "std_post_min": _min(std_post),
        "std_post_max": _max(std_post),
        "std_prior_min": _min(std_prior),
        "std_prior_max": _max(std_prior),
        "logstd_post_mean": _mean(std_post.log()),
        "logstd_prior_mean": _mean(std_prior.log()),
        "post_entropy_mean": _mean(cat(post_entropy_all)) if post_entropy_all else None,
        "prior_entropy_mean": _mean(cat(prior_entropy_all)) if prior_entropy_all else None,
    }

    level2 = {
        "prior_feature_norm_mean": prior_collapse["feature_norm_mean"],
        "prior_centered_norm_mean": prior_collapse["centered_norm_mean"],
        "prior_relative_centered_norm": prior_collapse["relative_centered_norm"],
        "prior_pairwise_l2_mean": prior_collapse["pairwise_l2_mean"],
        "prior_pairwise_l2_median": prior_collapse["pairwise_l2_median"],
        "prior_adjacent_cos_mean": prior_collapse["adjacent_cos_mean"],
        "post_feature_norm_mean": post_collapse["feature_norm_mean"],
        "post_centered_norm_mean": post_collapse["centered_norm_mean"],
        "post_relative_centered_norm": post_collapse["relative_centered_norm"],
        "post_pairwise_l2_mean": post_collapse["pairwise_l2_mean"],
        "post_pairwise_l2_median": post_collapse["pairwise_l2_median"],
        "post_adjacent_cos_mean": post_collapse["adjacent_cos_mean"],
    }

    level3 = {
        "real_to_target_l2": _mean(real_to_target),
        "zero_to_target_l2": _mean(zero_to_target),
        "shuffle_to_target_l2": _mean(shuffle_to_target),
        "real_vs_zero_l2": _mean(cat(real_vs_zero_all)),
        "real_vs_shuffle_l2": _mean(cat(real_vs_shuffle_all)),
        "margin_zero": _mean(zero_to_target) - _mean(real_to_target),
        "margin_shuffle": _mean(shuffle_to_target) - _mean(real_to_target),
    }

    # ── Level 4: imagination rollout drift / stability ──────────────────────
    feat_norm_per_h = {
        h: _mean(cat(rollout_norm_at_h[h])) if rollout_norm_at_h[h] else None
        for h in H_LIST
    }
    entropy_per_h = {
        h: _mean(cat(rollout_entropy_at_h[h])) if rollout_entropy_at_h[h] else None
        for h in H_LIST
    }
    drift_per_h = {
        h: _mean(cat(rollout_drift_from_h1[h])) if rollout_drift_from_h1[h] else None
        for h in H_LIST if h != 1
    }
    std_min_per_h = {
        h: _mean(cat(rollout_std_min_at_h[h])) if rollout_std_min_at_h[h] else None
        for h in H_LIST
    }
    norms_list = [feat_norm_per_h[h] for h in H_LIST if feat_norm_per_h[h] is not None]
    if norms_list:
        n0 = norms_list[0]
        norm_explosion_ratio = max(norms_list) / max(n0, 1e-8)
        norm_collapse_ratio = min(norms_list) / max(n0, 1e-8)
    else:
        norm_explosion_ratio = None
        norm_collapse_ratio = None

    level4 = {
        "status": "computed_imagination_only",
        "note": "Imagination rollout under repeated action; no GT future obs so KL@h not measured.",
        "horizons": H_LIST,
        "feature_norm_at_h": feat_norm_per_h,
        "feature_drift_from_h1": drift_per_h,
        "prior_entropy_at_h": entropy_per_h,
        "prior_std_min_at_h": std_min_per_h,
        "norm_explosion_ratio": norm_explosion_ratio,
        "norm_collapse_ratio": norm_collapse_ratio,
    }

    if token_ce_all:
        random_baseline = 1.0 / max(n_image_tokens_vocab, 1)
        level5 = {
            "token_ce_mean": float(statistics.fmean(token_ce_all)),
            "token_acc_mean": float(statistics.fmean(token_acc_all)),
            "dynamic_fraction_mean": float(statistics.fmean(dynamic_frac_all)),
            "static_ce_mean": float(statistics.fmean(static_ce_all)),
            "dynamic_ce_mean": float(statistics.fmean(dynamic_ce_all)),
            "static_acc_mean": float(statistics.fmean(static_acc_all)),
            "dynamic_acc_mean": float(statistics.fmean(dynamic_acc_all)),
            "random_acc_baseline": random_baseline,
        }
    else:
        level5 = {"status": "not_computed", "reason": "image_loss disabled or io_mode != token"}

    # ── Main-task losses + latent saturation ─────────────────────────────────
    main_task = {
        "transition_loss_mean": (
            float(statistics.fmean(transition_loss_all)) if transition_loss_all else None
        ),
        "reward_loss_mean": (
            float(statistics.fmean(reward_loss_all)) if reward_loss_all else None
        ),
    }

    if _is_discrete(world_model):
        S, K = world_model.stoch_dims, world_model.stoch_categories
        post_max_usage = post_cat_count.max(dim=-1).values / post_cat_count.sum(dim=-1).clamp_min(1)
        post_active_cats = (post_cat_count > 0).sum(dim=-1).float()
        prior_max_usage = prior_cat_count.max(dim=-1).values / prior_cat_count.sum(dim=-1).clamp_min(1)
        prior_active_cats = (prior_cat_count > 0).sum(dim=-1).float()
        latent_health = {
            "type": "discrete",
            "stoch_dims": S,
            "stoch_categories": K,
            "post_max_class_usage_mean": float(post_max_usage.mean().cpu()),
            "post_max_class_usage_max": float(post_max_usage.max().cpu()),
            "post_active_categories_mean": float(post_active_cats.mean().cpu()),
            "post_dead_dims_count": int(((post_max_usage > 0.99)).sum().cpu()),  # dim where >99% goes to one class
            "prior_max_class_usage_mean": float(prior_max_usage.mean().cpu()),
            "prior_active_categories_mean": float(prior_active_cats.mean().cpu()),
            "prior_dead_dims_count": int(((prior_max_usage > 0.99)).sum().cpu()),
        }
    else:
        std_dims = torch.cat(std_dim_collect, dim=0).float() if std_dim_collect else torch.empty(0)
        if std_dims.numel() > 0:
            min_std_floor = float(getattr(world_model, "min_std", 0.0))
            saturated_count = (std_dims < min_std_floor + 0.01).all(dim=0).sum().item()
            latent_health = {
                "type": "gaussian",
                "latent_dim": int(std_dims.shape[-1]),
                "per_dim_std_mean": float(std_dims.mean(dim=0).mean().cpu()),
                "per_dim_std_min": float(std_dims.min(dim=0).values.min().cpu()),
                "per_dim_std_max": float(std_dims.max(dim=0).values.max().cpu()),
                "saturated_dims_at_floor": int(saturated_count),
                "min_std_floor": min_std_floor,
            }
        else:
            latent_health = {"type": "gaussian", "note": "no std samples collected"}

    report = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "device": str(device),
        "num_samples": consumed,
        "config": config_meta,
        "level0_space": level0,
        "level1_kl_distribution": level1,
        "level2_transition_collapse": level2,
        "level3_action_conditioning": level3,
        "level4_multistep_rollout": level4,
        "level5_token": level5,
        "main_task_losses": main_task,
        "latent_health": latent_health,
        "minimum_required": {
            "feature_post_norm": level0["feature_post_norm_mean"],
            "feature_prior_norm": level0["feature_prior_norm_mean"],
            "feature_diff_l2": level0["feature_diff_l2_mean"],
            "feature_cos": level0["feature_cos_mean"],
            "std_post_mean": level1["std_post_mean"],
            "std_prior_mean": level1["std_prior_mean"],
            "prior_relative_centered_norm": level2["prior_relative_centered_norm"],
            "real_to_target_l2": level3["real_to_target_l2"],
            "zero_to_target_l2": level3["zero_to_target_l2"],
            "shuffle_to_target_l2": level3["shuffle_to_target_l2"],
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[diagnose_wm] wrote {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
