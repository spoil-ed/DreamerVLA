"""Phase 1 statistics for the 35x1024 action_hidden_states tensor.

Tests whether the 35 = 5*7 tokens are statistically homogeneous along the
time axis (t=0..4) and the joint axis (j=0..6).

Outputs: JSON summary + PNG figures under data/diagnostics/hidden_token_structure/.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----- paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    PROJECT_ROOT
    / "data"
    / "processed_data"
    / "libero_goal/no_noops_t_256_legacy_action_hidden_vla_policy_h2"
)
OUT_DIR = (
    PROJECT_ROOT / "data" / "diagnostics" / "hidden_token_structure"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TOKENS = 35
T = 5
J = 7
D = 1024
DTYPE = np.float64  # accumulators in fp64 to keep numerics clean

SUBSAMPLE_FOR_PCA = 12000
RNG = np.random.default_rng(0)


def iter_demo_arrays(files: list[str]):
    """Yield (filename, demo_key, np.ndarray[T_i, 35, 1024] fp32)."""
    for f in files:
        with h5py.File(f, "r") as h:
            for k in h["data"].keys():
                arr = np.asarray(
                    h["data"][k]["action_hidden_states"][...], dtype=np.float32
                )
                yield f, k, arr


def streaming_stats(files: list[str]) -> dict:
    """Single pass: per-(token, dim) sum, sumsq, count."""
    sum_td = np.zeros((N_TOKENS, D), dtype=DTYPE)
    sumsq_td = np.zeros((N_TOKENS, D), dtype=DTYPE)
    n_total = 0
    n_demos = 0
    t0 = time.time()
    last_f = None
    for _f, _k, arr in iter_demo_arrays(files):
        # arr: [T_i, 35, 1024]
        n_i = arr.shape[0]
        a = arr.astype(DTYPE, copy=False)
        sum_td += a.sum(axis=0)
        sumsq_td += (a * a).sum(axis=0)
        n_total += n_i
        n_demos += 1
        if _f != last_f:
            print(
                f"  streaming: {Path(_f).name}  cum_frames={n_total}  cum_demos={n_demos}",
                flush=True,
            )
            last_f = _f
    elapsed = time.time() - t0
    mean_td = sum_td / max(n_total, 1)
    var_td = sumsq_td / max(n_total, 1) - mean_td * mean_td
    var_td = np.maximum(var_td, 0.0)
    std_td = np.sqrt(var_td)
    return {
        "n_total": int(n_total),
        "n_demos": int(n_demos),
        "elapsed_streaming_sec": float(elapsed),
        "mean_td": mean_td,  # [35, 1024]
        "std_td": std_td,  # [35, 1024]
    }


def subsample_frames(files: list[str], n: int) -> np.ndarray:
    """Return [n, 35, 1024] fp32 by reservoir-ish sampling across demos."""
    counts = []
    metas = []
    for f in files:
        with h5py.File(f, "r") as h:
            for k in h["data"].keys():
                t_i = h["data"][k]["action_hidden_states"].shape[0]
                metas.append((f, k, t_i))
                counts.append(t_i)
    total = sum(counts)
    n = min(n, total)
    flat_idx = RNG.choice(total, size=n, replace=False)
    flat_idx.sort()

    # Map flat indices to (demo_idx, local_idx)
    cum = np.cumsum([0] + counts)
    out = np.empty((n, N_TOKENS, D), dtype=np.float32)
    write_pos = 0
    for di, (f, k, _t_i) in enumerate(metas):
        lo, hi = cum[di], cum[di + 1]
        mask = (flat_idx >= lo) & (flat_idx < hi)
        if not mask.any():
            continue
        local = flat_idx[mask] - lo
        with h5py.File(f, "r") as h:
            arr = h["data"][k]["action_hidden_states"][local].astype(np.float32)
        out[write_pos : write_pos + arr.shape[0]] = arr
        write_pos += arr.shape[0]
    assert write_pos == n
    return out


def pca_via_gram(X: np.ndarray, k_top: int = 200) -> dict:
    """PCA on X: [N, D] with N << D. Use X @ X.T (N x N) gram matrix.

    Returns:
        eigvals: all N nonzero singular values^2 / (N-1), sorted desc
        explained_variance_ratio
        participation_ratio: (sum lambda)^2 / sum lambda^2
        effective_rank_99: smallest k s.t. cumvar >= 99%
    """
    N = X.shape[0]
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # Gram in sample space: N x N
    gram = Xc @ Xc.T  # [N, N]
    # Symmetric eig
    eigvals, _ = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals[::-1], 0.0)  # desc, drop tiny negatives
    # These are singular values squared of Xc. Variance per component = eigval / (N - 1)
    var = eigvals / max(N - 1, 1)
    total = var.sum()
    ratio = var / max(total, 1e-12)
    cum = np.cumsum(ratio)

    # participation ratio
    pr = float((var.sum() ** 2) / max((var * var).sum(), 1e-12))

    # effective rank at 90/95/99%
    def first_above(c, thr):
        idx = np.searchsorted(c, thr) + 1
        return int(min(idx, len(c)))

    return {
        "n_components": int(len(var)),
        "top_k_var": var[: min(k_top, len(var))].tolist(),
        "top_k_ratio": ratio[: min(k_top, len(ratio))].tolist(),
        "cum_ratio_at_5": float(cum[min(4, len(cum) - 1)]),
        "cum_ratio_at_50": float(cum[min(49, len(cum) - 1)]),
        "cum_ratio_at_200": float(cum[min(199, len(cum) - 1)]),
        "effective_rank_90": first_above(cum, 0.90),
        "effective_rank_95": first_above(cum, 0.95),
        "effective_rank_99": first_above(cum, 0.99),
        "participation_ratio": pr,
        "total_variance": float(total),
    }


def per_token_pca(X: np.ndarray) -> dict:
    """For each of the 35 tokens, run PCA on [N, 1024] separately.

    Uses the D x D covariance path (D=1024 < N=12000), so each token's
    eigendecomp is ~0.1s.

    Returns per-token participation ratio and 99% effective rank.
    """
    N = X.shape[0]
    out_pr = []
    out_r99 = []
    out_r95 = []
    out_total_var = []
    for ti in range(N_TOKENS):
        Xi = X[:, ti, :]  # [N, 1024]
        mean = Xi.mean(axis=0, keepdims=True)
        Xc = Xi - mean
        # Covariance in feature space: D x D (1024 x 1024)
        cov = (Xc.T @ Xc) / max(N - 1, 1)
        eigvals = np.linalg.eigvalsh(cov)  # ascending
        var = np.maximum(eigvals[::-1], 0.0)
        total = var.sum()
        if total <= 0:
            out_pr.append(0.0)
            out_r99.append(0)
            out_r95.append(0)
            out_total_var.append(0.0)
            continue
        pr = float((var.sum() ** 2) / max((var * var).sum(), 1e-12))
        cum = np.cumsum(var / total)
        r99 = int(min(np.searchsorted(cum, 0.99) + 1, len(cum)))
        r95 = int(min(np.searchsorted(cum, 0.95) + 1, len(cum)))
        out_pr.append(pr)
        out_r99.append(r99)
        out_r95.append(r95)
        out_total_var.append(float(total))
        print(
            f"    token {ti:2d} (t={ti // J}, j={ti % J}): pr={pr:.1f}  r99={r99}  total_var={total:.2f}",
            flush=True,
        )
    return {
        "per_token_participation_ratio": out_pr,
        "per_token_effective_rank_95": out_r95,
        "per_token_effective_rank_99": out_r99,
        "per_token_total_variance": out_total_var,
    }


def gaussian_frechet_diag(mu_i, var_i, mu_j, var_j) -> float:
    """Fréchet distance assuming diagonal covariances.

    FID = ||mu1 - mu2||^2 + sum (sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    d2 = float(((mu_i - mu_j) ** 2).sum())
    tr = float((var_i + var_j - 2.0 * np.sqrt(np.maximum(var_i * var_j, 0.0))).sum())
    return d2 + tr


def main():
    files = sorted(glob.glob(str(DATA_DIR / "*.hdf5")))
    print(f"[load] {len(files)} files in {DATA_DIR.name}")

    # ---- streaming stats on full dataset ----
    print("[phase] streaming per-(token,dim) sufficient statistics...")
    s = streaming_stats(files)
    mean_td = s["mean_td"]  # [35, 1024] fp64
    std_td = s["std_td"]
    print(
        f"  N={s['n_total']} frames, {s['n_demos']} demos, {s['elapsed_streaming_sec']:.1f}s"
    )

    # Per-token scalar stats
    per_token_mean_scalar = mean_td.mean(axis=1)  # [35]
    per_token_std_scalar = std_td.mean(axis=1)  # [35]
    per_token_norm = np.linalg.norm(mean_td, axis=1)  # [35]

    # Pairwise cosine on per-token mean vectors
    nrm = np.linalg.norm(mean_td, axis=1, keepdims=True) + 1e-12
    cos_means = (mean_td @ mean_td.T) / (nrm @ nrm.T)  # [35, 35]

    # Pairwise cosine on per-token std vectors (feature dispersion shape)
    snrm = np.linalg.norm(std_td, axis=1, keepdims=True) + 1e-12
    cos_stds = (std_td @ std_td.T) / (snrm @ snrm.T)

    # Pairwise Fréchet (diag) between (mu_i, var_i) and (mu_j, var_j)
    var_td = std_td**2
    frechet = np.zeros((N_TOKENS, N_TOKENS), dtype=np.float64)
    for i in range(N_TOKENS):
        for j in range(i + 1, N_TOKENS):
            d = gaussian_frechet_diag(mean_td[i], var_td[i], mean_td[j], var_td[j])
            frechet[i, j] = d
            frechet[j, i] = d

    # ---- subsample for PCA ----
    print(f"[phase] subsampling {SUBSAMPLE_FOR_PCA} frames for PCA...")
    t0 = time.time()
    X = subsample_frames(files, SUBSAMPLE_FOR_PCA)  # [n, 35, 1024] fp32
    print(f"  subsample done in {time.time() - t0:.1f}s, shape={X.shape}")

    # Flat PCA on [n, D]
    print("[phase] flat PCA via gram matrix...")
    t0 = time.time()
    Xflat = X.reshape(X.shape[0], N_TOKENS * D)
    pca_flat = pca_via_gram(Xflat, k_top=200)
    print(f"  flat PCA done in {time.time() - t0:.1f}s")
    print(
        f"  flat eff-rank 90/95/99 = {pca_flat['effective_rank_90']}/{pca_flat['effective_rank_95']}/{pca_flat['effective_rank_99']}"
    )
    print(f"  flat participation ratio = {pca_flat['participation_ratio']:.1f}")

    # Per-token PCA
    print("[phase] per-token PCA...")
    t0 = time.time()
    per_tok = per_token_pca(X)
    print(f"  per-token PCA done in {time.time() - t0:.1f}s")

    # ---- save figures ----
    print("[plot] writing figures...")

    # Fig 1: per-token mean/std heatmaps reshaped as (T=5, J=7)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    m = per_token_mean_scalar.reshape(T, J)
    s_ = per_token_std_scalar.reshape(T, J)
    im0 = axes[0].imshow(m, aspect="auto", cmap="viridis")
    axes[0].set_title(
        "per-token feature-mean (avg over 1024 dims)\nrows=t (0..4), cols=joint (0..6)"
    )
    axes[0].set_xlabel("joint")
    axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(s_, aspect="auto", cmap="magma")
    axes[1].set_title("per-token feature-std (avg over 1024 dims)")
    axes[1].set_xlabel("joint")
    axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_per_token_mean_std.png", dpi=120)
    plt.close()

    # Fig 2: cosine similarity matrices [35, 35] (token order is t*7+j)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(cos_means, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title("cosine(mean_i, mean_j)\n(grouped: row=t*7+j)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(cos_stds, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("cosine(std_i, std_j)")
    plt.colorbar(im1, ax=axes[1])
    for ax in axes:
        for t in range(1, T):
            ax.axhline(t * J - 0.5, color="white", lw=0.6, alpha=0.6)
            ax.axvline(t * J - 0.5, color="white", lw=0.6, alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_cosine_matrices.png", dpi=120)
    plt.close()

    # Fig 3: Fréchet (diag) distance matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frechet, cmap="magma")
    ax.set_title(
        "Gaussian Fréchet distance (diagonal Σ)\nbetween token marginal distributions"
    )
    plt.colorbar(im, ax=ax)
    for t in range(1, T):
        ax.axhline(t * J - 0.5, color="white", lw=0.6, alpha=0.6)
        ax.axvline(t * J - 0.5, color="white", lw=0.6, alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_frechet_diag.png", dpi=120)
    plt.close()

    # Fig 4: flat PCA cumulative variance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cum = np.cumsum(pca_flat["top_k_ratio"])
    axes[0].plot(np.arange(1, len(pca_flat["top_k_var"]) + 1), pca_flat["top_k_var"])
    axes[0].set_yscale("log")
    axes[0].set_title("flat PCA: top-200 eigenvalue spectrum (log)")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("variance")
    axes[1].plot(np.arange(1, len(cum) + 1), cum)
    axes[1].axhline(0.9, color="gray", ls="--", lw=0.7)
    axes[1].axhline(0.95, color="gray", ls="--", lw=0.7)
    axes[1].axhline(0.99, color="gray", ls="--", lw=0.7)
    axes[1].set_title(
        f"flat PCA: cumulative explained variance\n"
        f"eff-rank 90/95/99 = {pca_flat['effective_rank_90']}/{pca_flat['effective_rank_95']}/{pca_flat['effective_rank_99']}"
    )
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("cum ratio")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_flat_pca.png", dpi=120)
    plt.close()

    # Fig 5: per-token PCA effective rank
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pr = np.array(per_tok["per_token_participation_ratio"]).reshape(T, J)
    r99 = np.array(per_tok["per_token_effective_rank_99"]).reshape(T, J)
    im0 = axes[0].imshow(pr, aspect="auto", cmap="viridis")
    axes[0].set_title(
        "per-token participation ratio\n(higher = uses more of 1024 dims)"
    )
    axes[0].set_xlabel("joint")
    axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(r99, aspect="auto", cmap="viridis")
    axes[1].set_title("per-token PCA 99% eff-rank\n(out of min(N, 1024))")
    axes[1].set_xlabel("joint")
    axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_per_token_pca.png", dpi=120)
    plt.close()

    # ---- summary numbers ----

    # Block-average cosine: within-same-t, within-same-j, cross
    def block_avg(M):
        same_t = []
        same_j = []
        cross = []
        for i in range(N_TOKENS):
            ti, ji = i // J, i % J
            for j in range(N_TOKENS):
                if i == j:
                    continue
                tj, jj = j // J, j % J
                v = float(M[i, j])
                if ti == tj:
                    same_t.append(v)
                elif ji == jj:
                    same_j.append(v)
                else:
                    cross.append(v)
        return {
            "same_t_mean": float(np.mean(same_t)),
            "same_j_mean": float(np.mean(same_j)),
            "cross_mean": float(np.mean(cross)),
        }

    cos_means_blocks = block_avg(cos_means)
    cos_stds_blocks = block_avg(cos_stds)
    frechet_blocks = block_avg(frechet)

    # off-diagonal cosine summary
    off_diag = cos_means[np.triu_indices(N_TOKENS, k=1)]
    summary = {
        "n_frames": s["n_total"],
        "n_demos": s["n_demos"],
        "per_token_mean_scalar": per_token_mean_scalar.tolist(),
        "per_token_std_scalar": per_token_std_scalar.tolist(),
        "per_token_mean_norm": per_token_norm.tolist(),
        "cos_means_offdiag": {
            "min": float(off_diag.min()),
            "p25": float(np.percentile(off_diag, 25)),
            "median": float(np.median(off_diag)),
            "p75": float(np.percentile(off_diag, 75)),
            "max": float(off_diag.max()),
            "mean": float(off_diag.mean()),
        },
        "cos_means_block_avg": cos_means_blocks,
        "cos_stds_block_avg": cos_stds_blocks,
        "frechet_block_avg": frechet_blocks,
        "flat_pca": {
            k: v for k, v in pca_flat.items() if k not in ("top_k_var", "top_k_ratio")
        },
        "per_token_pca_summary": {
            "participation_ratio_mean": float(
                np.mean(per_tok["per_token_participation_ratio"])
            ),
            "participation_ratio_min": float(
                np.min(per_tok["per_token_participation_ratio"])
            ),
            "participation_ratio_max": float(
                np.max(per_tok["per_token_participation_ratio"])
            ),
            "eff_rank_99_mean": float(np.mean(per_tok["per_token_effective_rank_99"])),
            "eff_rank_99_min": int(np.min(per_tok["per_token_effective_rank_99"])),
            "eff_rank_99_max": int(np.max(per_tok["per_token_effective_rank_99"])),
        },
        "subsample_n_for_pca": int(X.shape[0]),
    }

    with open(OUT_DIR / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    # save the heavy arrays as npz for reuse
    np.savez_compressed(
        OUT_DIR / "stats.npz",
        mean_td=mean_td.astype(np.float32),
        std_td=std_td.astype(np.float32),
        cos_means=cos_means.astype(np.float32),
        cos_stds=cos_stds.astype(np.float32),
        frechet=frechet.astype(np.float32),
        per_token_pr=np.array(
            per_tok["per_token_participation_ratio"], dtype=np.float32
        ),
        per_token_r99=np.array(per_tok["per_token_effective_rank_99"], dtype=np.int32),
        flat_top_k_var=np.array(pca_flat["top_k_var"], dtype=np.float32),
    )

    print("\n========== Phase 1 summary ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nfigures + summary -> {OUT_DIR}")


if __name__ == "__main__":
    main()
