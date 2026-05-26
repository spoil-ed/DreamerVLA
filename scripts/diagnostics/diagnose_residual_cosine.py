"""Section 5.1 follow-up: residual cosine across token pairs.

For each frame n and token pair (i, j), compute cosine of the residuals
r_n,i = H[n,i,:] - mu_i  vs  r_n,j = H[n,j,:] - mu_j.

If cos(r) is high  → tokens are genuinely redundant per-sample.
If cos(r) is low   → mean-vector alignment is post-LayerNorm bias only;
                     each token carries independent per-sample info.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(
    "/mnt/data/spoil/workspace/DreamerVLA/data/processed_data/"
    "libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2"
)
OUT_DIR = Path("/mnt/data/spoil/workspace/DreamerVLA/data/diagnostics/hidden_token_structure")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TOKENS = 35
T = 5
J = 7
D = 1024

# Reuse mean_td from Phase 1
stats = np.load(OUT_DIR / "stats.npz")
mean_td = stats["mean_td"].astype(np.float32)  # [35, 1024]

# Random subsample 4000 frames (4000 * 35 * 35 = 4.9M pair cosines)
rng = np.random.default_rng(42)
files = sorted(glob.glob(str(DATA_DIR / "*.hdf5")))

counts = []
metas = []
for f in files:
    with h5py.File(f, "r") as h:
        for k in h["data"].keys():
            t_i = h["data"][k]["action_hidden_states"].shape[0]
            metas.append((f, k, t_i))
            counts.append(t_i)
total = sum(counts)
n_sample = 4000
flat_idx = np.sort(rng.choice(total, size=n_sample, replace=False))
cum = np.cumsum([0] + counts)

# Load samples
samples = np.empty((n_sample, N_TOKENS, D), dtype=np.float32)
wp = 0
for di, (f, k, t_i) in enumerate(metas):
    lo, hi = cum[di], cum[di + 1]
    mask = (flat_idx >= lo) & (flat_idx < hi)
    if not mask.any():
        continue
    local = flat_idx[mask] - lo
    with h5py.File(f, "r") as h:
        arr = h["data"][k]["action_hidden_states"][local].astype(np.float32)
    samples[wp : wp + arr.shape[0]] = arr
    wp += arr.shape[0]
assert wp == n_sample

print(f"loaded {n_sample} samples; shape={samples.shape}")

# Compute residuals
residuals = samples - mean_td[None]  # [N, 35, 1024]
# Normalize each token-vector
norms = np.linalg.norm(residuals, axis=-1, keepdims=True) + 1e-12
unit = residuals / norms  # [N, 35, 1024]

# Pairwise cosine of residuals per frame: [N, 35, 35]
# This is 4000 * 35 * 35 = 4.9M values, fine
cos_per_frame = np.einsum("nid,njd->nij", unit, unit).astype(np.float32)

# Overall stats on off-diagonal
mask_off = ~np.eye(N_TOKENS, dtype=bool)
off = cos_per_frame[:, mask_off]  # [N, 35*34]

print("\n=== Residual cosine (per-frame) ===")
print(f"min     = {off.min():.4f}")
print(f"p1      = {np.percentile(off, 1):.4f}")
print(f"p25     = {np.percentile(off, 25):.4f}")
print(f"median  = {np.median(off):.4f}")
print(f"p75     = {np.percentile(off, 75):.4f}")
print(f"p99     = {np.percentile(off, 99):.4f}")
print(f"max     = {off.max():.4f}")
print(f"mean    = {off.mean():.4f}")

# Block-averaged by (same_t / same_j / cross)
same_t_mask = np.zeros((N_TOKENS, N_TOKENS), dtype=bool)
same_j_mask = np.zeros((N_TOKENS, N_TOKENS), dtype=bool)
cross_mask = np.zeros((N_TOKENS, N_TOKENS), dtype=bool)
for i in range(N_TOKENS):
    ti, ji = i // J, i % J
    for j in range(N_TOKENS):
        if i == j: continue
        tj, jj = j // J, j % J
        if ti == tj: same_t_mask[i, j] = True
        elif ji == jj: same_j_mask[i, j] = True
        else: cross_mask[i, j] = True

same_t = cos_per_frame[:, same_t_mask]
same_j = cos_per_frame[:, same_j_mask]
cross = cos_per_frame[:, cross_mask]

print("\n=== Residual cosine by block ===")
print(f"{'group':12s}  {'mean':>8s}  {'median':>8s}  {'p25':>8s}  {'p75':>8s}")
print(f"{'same t (jj)':12s}  {same_t.mean():8.4f}  {np.median(same_t):8.4f}  {np.percentile(same_t,25):8.4f}  {np.percentile(same_t,75):8.4f}")
print(f"{'same j (tt)':12s}  {same_j.mean():8.4f}  {np.median(same_j):8.4f}  {np.percentile(same_j,25):8.4f}  {np.percentile(same_j,75):8.4f}")
print(f"{'cross':12s}  {cross.mean():8.4f}  {np.median(cross):8.4f}  {np.percentile(cross,25):8.4f}  {np.percentile(cross,75):8.4f}")

# Token-pair mean residual cosine matrix
mean_cos_pair = cos_per_frame.mean(axis=0)  # [35, 35]

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
with open(OUT_DIR / "residual_cosine_summary.json", "w") as fp:
    json.dump(summary, fp, indent=2)

# Save figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
im = axes[0].imshow(mean_cos_pair, vmin=-0.2, vmax=1.0, cmap="RdBu_r")
axes[0].set_title("mean residual cosine per (i,j)\nrow/col index = t*7 + j")
plt.colorbar(im, ax=axes[0])
for t in range(1, T):
    axes[0].axhline(t * J - 0.5, color="white", lw=0.6, alpha=0.6)
    axes[0].axvline(t * J - 0.5, color="white", lw=0.6, alpha=0.6)

axes[1].hist(off.flatten()[::100], bins=80, color="steelblue", alpha=0.75, label="all off-diag")
axes[1].hist(same_t.flatten()[::20], bins=80, color="green", alpha=0.5, label="same t")
axes[1].hist(same_j.flatten()[::50], bins=80, color="orange", alpha=0.5, label="same j")
axes[1].set_xlabel("cosine of residuals")
axes[1].set_ylabel("count (subsampled)")
axes[1].set_title("residual cosine distribution")
axes[1].legend()
axes[1].axvline(0, color="black", lw=0.5, ls="--")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_residual_cosine.png", dpi=120)
plt.close()

print(f"\nfigure -> {OUT_DIR / 'fig6_residual_cosine.png'}")
print(f"summary -> {OUT_DIR / 'residual_cosine_summary.json'}")
