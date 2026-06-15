# ruff: noqa: E402
"""Section 5.1 follow-up: residual cosine across token pairs.

For each frame n and token pair (i, j), compute cosine of the residuals
r_n,i = H[n,i,:] - mu_i  vs  r_n,j = H[n,j,:] - mu_j.

If cos(r) is high  → tokens are genuinely redundant per-sample.
If cos(r) is low   → mean-vector alignment is post-LayerNorm bias only;
                     each token carries independent per-sample info.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_data"
    / "libero_goal/no_noops_t_256_legacy_action_hidden_vla_policy_h2"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "hidden_token_structure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=35)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    stats_path = args.stats.expanduser().resolve() if args.stats else out_dir / "stats.npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tokens = int(args.tokens)
    horizon = int(args.horizon)
    action_dim = int(args.action_dim)
    hidden_dim = int(args.hidden_dim)
    n_sample = int(args.num_samples)

    stats = np.load(stats_path)
    mean_td = stats["mean_td"].astype(np.float32)

    rng = np.random.default_rng(int(args.seed))
    files = sorted(glob.glob(str(data_dir / "*.hdf5")))

    counts = []
    metas = []
    for filename in files:
        with h5py.File(filename, "r") as handle:
            for key in handle["data"].keys():
                length = handle["data"][key]["action_hidden_states"].shape[0]
                metas.append((filename, key, length))
                counts.append(length)
    total = sum(counts)
    if total < n_sample:
        raise ValueError(f"requested {n_sample} samples but only {total} frames exist")
    flat_idx = np.sort(rng.choice(total, size=n_sample, replace=False))
    cum = np.cumsum([0] + counts)

    samples = np.empty((n_sample, n_tokens, hidden_dim), dtype=np.float32)
    wp = 0
    for demo_idx, (filename, key, _length) in enumerate(metas):
        lo, hi = cum[demo_idx], cum[demo_idx + 1]
        mask = (flat_idx >= lo) & (flat_idx < hi)
        if not mask.any():
            continue
        local = flat_idx[mask] - lo
        with h5py.File(filename, "r") as handle:
            arr = handle["data"][key]["action_hidden_states"][local].astype(np.float32)
        samples[wp : wp + arr.shape[0]] = arr
        wp += arr.shape[0]
    assert wp == n_sample

    print(f"loaded {n_sample} samples; shape={samples.shape}")

    residuals = samples - mean_td[None]
    norms = np.linalg.norm(residuals, axis=-1, keepdims=True) + 1e-12
    unit = residuals / norms
    cos_per_frame = np.einsum("nid,njd->nij", unit, unit).astype(np.float32)

    mask_off = ~np.eye(n_tokens, dtype=bool)
    off = cos_per_frame[:, mask_off]

    print("\n=== Residual cosine (per-frame) ===")
    print(f"min     = {off.min():.4f}")
    print(f"p1      = {np.percentile(off, 1):.4f}")
    print(f"p25     = {np.percentile(off, 25):.4f}")
    print(f"median  = {np.median(off):.4f}")
    print(f"p75     = {np.percentile(off, 75):.4f}")
    print(f"p99     = {np.percentile(off, 99):.4f}")
    print(f"max     = {off.max():.4f}")
    print(f"mean    = {off.mean():.4f}")

    same_t_mask = np.zeros((n_tokens, n_tokens), dtype=bool)
    same_j_mask = np.zeros((n_tokens, n_tokens), dtype=bool)
    cross_mask = np.zeros((n_tokens, n_tokens), dtype=bool)
    for idx_i in range(n_tokens):
        ti, ji = idx_i // action_dim, idx_i % action_dim
        for idx_j in range(n_tokens):
            if idx_i == idx_j:
                continue
            tj, jj = idx_j // action_dim, idx_j % action_dim
            if ti == tj:
                same_t_mask[idx_i, idx_j] = True
            elif ji == jj:
                same_j_mask[idx_i, idx_j] = True
            else:
                cross_mask[idx_i, idx_j] = True

    same_t = cos_per_frame[:, same_t_mask]
    same_j = cos_per_frame[:, same_j_mask]
    cross = cos_per_frame[:, cross_mask]

    print("\n=== Residual cosine by block ===")
    print(f"{'group':12s}  {'mean':>8s}  {'median':>8s}  {'p25':>8s}  {'p75':>8s}")
    print(
        f"{'same t (jj)':12s}  {same_t.mean():8.4f}  {np.median(same_t):8.4f}  "
        f"{np.percentile(same_t, 25):8.4f}  {np.percentile(same_t, 75):8.4f}"
    )
    print(
        f"{'same j (tt)':12s}  {same_j.mean():8.4f}  {np.median(same_j):8.4f}  "
        f"{np.percentile(same_j, 25):8.4f}  {np.percentile(same_j, 75):8.4f}"
    )
    print(
        f"{'cross':12s}  {cross.mean():8.4f}  {np.median(cross):8.4f}  "
        f"{np.percentile(cross, 25):8.4f}  {np.percentile(cross, 75):8.4f}"
    )

    mean_cos_pair = cos_per_frame.mean(axis=0)

    summary = {
        "n_samples": int(n_sample),
        "residual_cosine_offdiag": {
            "min": float(off.min()),
            "p1": float(np.percentile(off, 1)),
            "p25": float(np.percentile(off, 25)),
            "median": float(np.median(off)),
            "p75": float(np.percentile(off, 75)),
            "p99": float(np.percentile(off, 99)),
            "max": float(off.max()),
            "mean": float(off.mean()),
        },
        "block_means": {
            "same_t_mean": float(same_t.mean()),
            "same_t_median": float(np.median(same_t)),
            "same_j_mean": float(same_j.mean()),
            "same_j_median": float(np.median(same_j)),
            "cross_mean": float(cross.mean()),
            "cross_median": float(np.median(cross)),
        },
    }
    with open(out_dir / "residual_cosine_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = axes[0].imshow(mean_cos_pair, vmin=-0.2, vmax=1.0, cmap="RdBu_r")
    axes[0].set_title("mean residual cosine per (i,j)\nrow/col index = t*7 + j")
    plt.colorbar(im, ax=axes[0])
    for t_idx in range(1, horizon):
        axes[0].axhline(t_idx * action_dim - 0.5, color="white", lw=0.6, alpha=0.6)
        axes[0].axvline(t_idx * action_dim - 0.5, color="white", lw=0.6, alpha=0.6)

    axes[1].hist(
        off.flatten()[::100],
        bins=80,
        color="steelblue",
        alpha=0.75,
        label="all off-diag",
    )
    axes[1].hist(same_t.flatten()[::20], bins=80, color="green", alpha=0.5, label="same t")
    axes[1].hist(
        same_j.flatten()[::50],
        bins=80,
        color="orange",
        alpha=0.5,
        label="same j",
    )
    axes[1].set_xlabel("cosine of residuals")
    axes[1].set_ylabel("count (subsampled)")
    axes[1].set_title("residual cosine distribution")
    axes[1].legend()
    axes[1].axvline(0, color="black", lw=0.5, ls="--")

    plt.tight_layout()
    plt.savefig(out_dir / "fig6_residual_cosine.png", dpi=120)
    plt.close()

    print(f"\nfigure -> {out_dir / 'fig6_residual_cosine.png'}")
    print(f"summary -> {out_dir / 'residual_cosine_summary.json'}")


if __name__ == "__main__":
    main()
