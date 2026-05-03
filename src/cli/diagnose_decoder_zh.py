"""
Decoder z-vs-h sensitivity diagnostic.

Measures whether the image_decoder actually uses the stochastic latent z
(posterior sample) versus relying solely on the deterministic state h
(causal-transformer output that is driven by action history).

For B distinct samples, computes:

  Δ_logits_z = mean_abs( decoder(h_i, z_j) - decoder(h_i, z_i) )
  Δ_logits_h = mean_abs( decoder(h_j, z_i) - decoder(h_i, z_i) )

If Δ_logits_z ≪ Δ_logits_h, the decoder is bypassing z — explains
posterior collapse to uniform.

Run:
    python -m src.cli.diagnose_decoder_zh \\
        --config-name pretokenize_wm_libero_10_discrete_minimal_kl0 \\
        --ckpt <path>.ckpt \\
        --num-samples 8 --device cuda:0
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

    print("[zh] building encoder ...")
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

    print("[zh] building world model ...")
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

    print("[zh] building dataset ...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    rng = np.random.default_rng(args.seed)
    samples = []
    seen_tasks = set()
    for idx in rng.permutation(len(dataset)).tolist():
        s = dataset[idx]
        t = s.get("task_name", "?")
        if t in seen_tasks and len(seen_tasks) < args.num_samples:
            continue
        samples.append(s)
        seen_tasks.add(t)
        if len(samples) >= args.num_samples:
            break
    print(f"[zh] picked {len(samples)} samples from {len(seen_tasks)} tasks")

    # Build batch
    sample0 = samples[0]
    if "wm_obs_input_ids_seq" in sample0:
        seq_ids = [s["wm_obs_input_ids_seq"] for s in samples]
        action_seq = torch.stack([s["action_seq"] for s in samples], dim=0)
        B, T = len(seq_ids), len(seq_ids[0])
        flat = [step for sample in seq_ids for step in sample]
        obs_emb_bpe = extract_image_bpe_ids(flat, -2, n_image_tokens, img_bpe_set)\
                        .view(B, T, -1).to(device)
        action_seq = action_seq.to(device, dtype=torch.bfloat16)
    else:
        cur_ids  = [list(s["wm_obs_input_ids"])      for s in samples]
        next_ids = [list(s["wm_next_obs_input_ids"]) for s in samples]
        B, T = len(cur_ids), 2
        cur_bpe  = extract_image_bpe_ids(cur_ids,  -2, n_image_tokens, img_bpe_set).to(device)
        next_bpe = extract_image_bpe_ids(next_ids, -2, n_image_tokens, img_bpe_set).to(device)
        obs_emb_bpe = torch.stack([cur_bpe, next_bpe], dim=1)
        wm_action = torch.stack([torch.as_tensor(s["wm_action"], dtype=torch.float32) for s in samples], dim=0)
        wm_action = wm_action.to(device, dtype=torch.bfloat16)
        action_step = wm_action.mean(dim=1)
        action_seq = torch.stack([torch.zeros_like(action_step), action_step], dim=1)

    print(f"\n[zh] B={B} T={T} N_img={n_image_tokens}")

    with torch.no_grad():
        # Forward to get post stoch and h
        img_idx_seq = world_model._bpe_to_img_idx[obs_emb_bpe]
        per_token = world_model.token_embedder(img_idx_seq)
        hidden_seq = world_model.conv_stem(per_token)
        (post_mean, post_std, post_stoch,
         prior_mean, prior_std, prior_stoch,
         h_seq) = world_model._infer_dreamer_seq(hidden_seq, action_seq)
        # post_stoch: [B, T, latent], h_seq: [B, T-1, d_model]
        # decoder uses h_seq + post_stoch[:, 1:] (last T-1 steps)
        z = post_stoch[:, 1:]                           # [B, T-1, latent_dim]
        h = h_seq                                        # [B, T-1, d_model]

        # We work on the LAST timestep only for clarity (most important slot for prediction).
        z_last = z[:, -1]                                # [B, latent_dim]
        h_last = h[:, -1]                                # [B, d_model]

        def decode(h_in, z_in):
            # decoder expects [B, T-1, ...] like in main path; expand to T-1=1
            return world_model.image_decoder(h_in.unsqueeze(1), z_in.unsqueeze(1))

        # ─── Baseline + swaps ─────────────────────────────────────────────
        # For each pair (i, j) compute swaps.  Average over all i!=j pairs.
        results = {"delta_logits_z": [], "delta_logits_h": [],
                   "delta_logits_zrand": [], "delta_logits_hrand": [],
                   "logits_baseline_norm": []}

        for i in range(B):
            base_logits = decode(h_last[i:i+1], z_last[i:i+1])      # [1,1,N_img,V]
            results["logits_baseline_norm"].append(float(base_logits.float().abs().mean()))

            for j in range(B):
                if i == j: continue

                swap_z = decode(h_last[i:i+1], z_last[j:j+1])
                swap_h = decode(h_last[j:j+1], z_last[i:i+1])

                d_z = (swap_z - base_logits).float().abs().mean()
                d_h = (swap_h - base_logits).float().abs().mean()

                results["delta_logits_z"].append(float(d_z))
                results["delta_logits_h"].append(float(d_h))

            # Also: random z (one-hot drawn uniformly from latent space)
            #   z is [latent_dim] = stoch_dims*stoch_categories one-hot flattened.
            S = world_model.stoch_dims
            K = world_model.stoch_categories
            rand_idx = torch.randint(K, (S,), device=device)
            rand_z = torch.zeros(S * K, device=device, dtype=z_last.dtype)
            rand_z[torch.arange(S, device=device) * K + rand_idx] = 1.0
            rand_z = rand_z.view(1, -1)
            swap_zrand = decode(h_last[i:i+1], rand_z)
            d_zrand = (swap_zrand - base_logits).float().abs().mean()
            results["delta_logits_zrand"].append(float(d_zrand))

            rand_h = torch.randn_like(h_last[i:i+1]) * h_last.float().std()
            rand_h = rand_h.to(h_last.dtype)
            swap_hrand = decode(rand_h, z_last[i:i+1])
            d_hrand = (swap_hrand - base_logits).float().abs().mean()
            results["delta_logits_hrand"].append(float(d_hrand))

    # ─── Aggregate + report ───────────────────────────────────────────────
    summary: dict[str, float] = {}
    for k, vs in results.items():
        if vs:
            summary[f"{k}_mean"] = float(np.mean(vs))
            summary[f"{k}_std"]  = float(np.std(vs))

    print("\n" + "─" * 90)
    print(f" Decoder z-vs-h sensitivity  (averaged over B*(B-1) = {B*(B-1)} cross-sample pairs)")
    print("─" * 90)

    name_w = 36
    fmt = lambda k: f"{summary.get(k+'_mean', 0):.5f}  ±{summary.get(k+'_std', 0):.5f}"
    print(f"  {'baseline |logits|.mean()':<{name_w}}  {fmt('logits_baseline_norm')}")
    print(f"  {'Δ_logits_z  (same h, swap z)':<{name_w}}  {fmt('delta_logits_z')}")
    print(f"  {'Δ_logits_h  (same z, swap h)':<{name_w}}  {fmt('delta_logits_h')}")
    print(f"  {'Δ_logits_z_random (random one-hot z)':<{name_w}}  {fmt('delta_logits_zrand')}")
    print(f"  {'Δ_logits_h_random (random h)':<{name_w}}  {fmt('delta_logits_hrand')}")
    print("─" * 90)

    z_mean = summary.get('delta_logits_z_mean', 0)
    h_mean = summary.get('delta_logits_h_mean', 0)
    if h_mean > 1e-9:
        ratio = z_mean / h_mean
        print(f"  Δ_z / Δ_h ratio (cross-sample) = {ratio:.4f}")
        if ratio < 0.1:
            print("  ⚠ Decoder mostly bypasses z (Δ_z < 10% of Δ_h)")
        elif ratio < 0.5:
            print("  ⚠ Decoder under-uses z (Δ_z < 50% of Δ_h)")
        else:
            print("  ✓ Decoder appears to use z and h in comparable amounts")

    out_path = args.out_json or str(Path(args.ckpt).parent.parent / f"decoder_zh_{Path(args.ckpt).stem}.json")
    Path(out_path).write_text(json.dumps({
        "ckpt": args.ckpt,
        "config_name": args.config_name,
        "num_samples": B,
        "summary": summary,
        "raw": {k: list(v) for k, v in results.items()},
    }, indent=2))
    print(f"\n[zh] wrote {out_path}")


if __name__ == "__main__":
    main()
