"""Ablate classifier depth/width/token_count for WMPO alignment.

Reports parameter count for every grid point and, when CUDA is available,
RL-shaped forward peak memory/time.

Usage:
  conda run -n dreamervla python -m dreamervla.diagnostics.classifier_depth_ablation \
    --batch 16 --window 8 --token-count 512 --token-dim 4096
"""

from __future__ import annotations

import argparse
import time

import torch

from dreamervla.models.reward.latent_success_classifier import LatentSuccessClassifier

PROPRIO_DIM = 8
PROPRIO_EMB = 10
LANG_DIM = 4096
LANG_EMB = 32


def build(
    num_layers: int,
    hidden_dim: int,
    token_count: int,
    token_dim: int,
    window: int,
) -> LatentSuccessClassifier:
    """Build a spatial classifier using production OFT side-channel dims."""
    latent_dim = token_dim + PROPRIO_EMB + LANG_EMB
    return LatentSuccessClassifier(
        latent_dim=latent_dim,
        token_dim=token_dim,
        token_count=token_count,
        token_pool="mean",
        head_type="spatial_tf",
        window=window,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        granularity="chunk",
        chunk_size=window,
        chunk_pool="last",
        proprio_dim=PROPRIO_DIM,
        proprio_emb_dim=PROPRIO_EMB,
        num_proprio_repeat=1,
        lang_dim=LANG_DIM,
        lang_emb_dim=LANG_EMB,
        num_lang_repeat=1,
    )


def profile_one(
    num_layers: int,
    hidden_dim: int,
    token_count: int,
    token_dim: int,
    window: int,
    batch: int,
    device: torch.device,
) -> tuple[int, float, float]:
    """Profile one classifier size.

    CPU runs intentionally report only params. CUDA runs include peak allocated
    memory and average forward latency over a short no-grad sweep.
    """
    model = build(num_layers, hidden_dim, token_count, token_dim, window).to(device).eval()
    nparam = sum(p.numel() for p in model.parameters())
    mem_mb = float("nan")
    fwd_ms = float("nan")
    if device.type == "cuda":
        vis = torch.randn(batch, window, token_count, token_dim, device=device)
        prop = torch.randn(batch, window, PROPRIO_DIM, device=device)
        lang = torch.randn(batch, LANG_DIM, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(3):
                model(vis, proprio=prop, lang_emb=lang)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(5):
                model(vis, proprio=prop, lang_emb=lang)
            torch.cuda.synchronize(device)
            fwd_ms = (time.perf_counter() - t0) / 5 * 1000.0
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        del vis, prop, lang
        torch.cuda.empty_cache()
    del model
    return nparam, mem_mb, fwd_ms


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--token-count", type=int, default=512)
    parser.add_argument("--token-dim", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = _device_from_arg(str(args.device))
    grid = [
        (4, 1024, int(args.token_count)),
        (6, 1024, int(args.token_count)),
        (8, 1024, int(args.token_count)),
        (12, 768, int(args.token_count)),
        (8, 1024, 196),
        (12, 768, 196),
    ]
    print(
        f"device={device} batch={int(args.batch)} window={int(args.window)} "
        f"token_dim={int(args.token_dim)}"
    )
    print(
        f"{'layers':>6} {'hidden':>6} {'tokens':>6} "
        f"{'params(M)':>10} {'fwd_mem(MB)':>12} {'fwd(ms)':>8}"
    )
    for num_layers, hidden_dim, token_count in grid:
        try:
            nparam, mem_mb, fwd_ms = profile_one(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                token_count=token_count,
                token_dim=int(args.token_dim),
                window=int(args.window),
                batch=int(args.batch),
                device=device,
            )
            print(
                f"{num_layers:>6} {hidden_dim:>6} {token_count:>6} "
                f"{nparam / 1e6:>10.1f} {mem_mb:>12.1f} {fwd_ms:>8.1f}"
            )
        except RuntimeError as exc:
            print(
                f"{num_layers:>6} {hidden_dim:>6} {token_count:>6} "
                f"  ERROR: {str(exc)[:60]}"
            )


if __name__ == "__main__":
    main()
