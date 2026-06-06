"""Estimate the *theoretical* ceiling on LatentSuccessClassifier F1.

Motivation
----------
sklearn LR(C=0.01) on real pi0 hidden W=8 windows hits F1 ≈ 0.87. This is the
*lower bound* on what's achievable (any signal LR captures is real). It is NOT
the upper bound — a nonlinear model with enough capacity could find structure
LR misses, up to the Bayes-optimal F1 implied by H(y|x).

This script runs three complementary estimates on the same data split:

1. **sklearn LR baseline** (sanity peg vs CLAUDE.md's recorded 0.87).
2. **sklearn LR + frame-position features** (the label generation rule is
   ``pos = end_idx >= finish_step``, so frame position is a causal parent of y —
   injecting it tests how much LR was leaving on the table by ignoring it).
3. **kNN classifiers (k=1, 5, 25)** with cosine metric — kNN with k → ∞ and
   n → ∞ is an asymptotically tight estimate of the Bayes error rate, so the
   best of (k=5, 25) is a *non-parametric upper bound* on achievable F1.
4. **Small MLP** (per-frame shared linear → small MLP head, ~1.2M params)
   trained with strong dropout + weight decay — tests whether a properly
   sized nonlinear model exceeds LR. 1.2M params on n≈800 is well below the
   PAC-Bayes overfitting regime that breaks the 137M Transformer.

Run on BOTH real hidden (the latent's intrinsic separability) and WM-imagined
hidden (the v2/v3 training distribution) to isolate (i) what the latent
*could* do at best, vs (ii) what imagine drift takes away.

Data protocol (matches the "1000-sample, 1:1.3 ratio" baseline in
``progress.md``):
    for each demo:
        emit 1 end window (label = int(complete))
        emit 1 random earlier window (label = 0)
    → ~1000 windows, 1:1.3 pos:neg
    stratified 80/20 train/val split, fixed seed.

Usage
-----
    python -u \\
        scripts/estimate_classifier_ceiling.py \\
        --config configs/wmpo_classifier_libero_goal_v4_real_hidden.yaml \\
        --out data/outputs/dreamervla/outcome_classifier/libero_goal/ceiling_real_hidden \\
        --feature-source real_hidden
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from dreamer_vla.dataset.wm_replay_classifier_dataset import _find_demo_pairs  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_demo(raw_p: Path, hid_p: Path, demo_key: str):
    """Return (obs[T, L] float32, finish_step, complete, episode_id)."""
    with h5py.File(str(raw_p), "r") as fr:
        grp = fr[demo_key]
        dones = np.asarray(grp["dones"][...]) if "dones" in grp else None
        rewards = np.asarray(grp["rewards"][...]) if "rewards" in grp else None
    with h5py.File(str(hid_p), "r") as fh:
        obs = np.asarray(fh[f"{demo_key}/obs_embedding"][...]).astype(np.float32)
    T = obs.shape[0]
    obs = obs.reshape(T, -1)
    if dones is not None and bool(dones[:T].any()):
        finish_step = int(np.argmax(dones[:T])) + 1
    else:
        finish_step = T
    complete = bool(rewards[:T].sum() > 0) if rewards is not None else True
    eid = f"{raw_p.stem}/{demo_key}"
    return obs, finish_step, complete, eid


def build_windows(
    pairs: list[tuple[Path, Path, str]],
    W: int,
    rng: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Emit 1 end + 1 random earlier window per demo.

    Returns:
        X: [N, W*L] flat features
        y: [N] labels
        pos: [N] normalized end-frame index (end_idx / finish_step)
        eid_idx: [N] integer episode id
    """
    X, y, pos, eids = [], [], [], []
    eid_map: dict[str, int] = {}
    for i, (raw_p, hid_p, demo_key) in enumerate(pairs):
        obs, fs, complete, eid = _load_demo(raw_p, hid_p, demo_key)
        T = int(min(fs, obs.shape[0]))
        if T < W + 1:
            continue
        if eid not in eid_map:
            eid_map[eid] = len(eid_map)
        eid_i = eid_map[eid]
        # end window
        X.append(obs[T - W : T].reshape(-1))
        y.append(int(complete))
        pos.append((T - 1) / max(fs, 1))
        eids.append(eid_i)
        # one random earlier window (end ∈ [W, T - W - 1], strict)
        if T - W - 1 > W:
            end = int(rng.integers(W, T - W))
            X.append(obs[end - W : end].reshape(-1))
            y.append(0)
            pos.append((end - 1) / max(fs, 1))
            eids.append(eid_i)
        if (i + 1) % 100 == 0:
            print(
                f"  [{label}] {i + 1}/{len(pairs)} demos, windows={len(X)}", flush=True
            )
    if not X:
        raise RuntimeError(f"no windows produced for {label}")
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.float32)
    eids = np.asarray(eids, dtype=np.int64)
    return X, y, pos, eids


# ---------------------------------------------------------------------------
# Small MLP — Option B in the design doc (per-frame shared proj + small head)
# ---------------------------------------------------------------------------


class SmallLatentMLP(nn.Module):
    """Per-frame shared Linear(L → d) → flatten → 2-layer MLP head.

    Total params ≈ L*d + (W*d)*h + h*2.
    PAC-Bayes-friendly for n≈800.
    """

    def __init__(
        self,
        latent_dim: int,
        window: int,
        frame_dim: int = 32,
        hidden: int = 64,
        dropout: float = 0.5,
        pos_dim: int = 0,
    ):
        super().__init__()
        self.W = window
        self.L = latent_dim
        self.pos_dim = pos_dim
        self.frame_proj = nn.Linear(latent_dim, frame_dim)
        self.norm = nn.LayerNorm(window * frame_dim + pos_dim)
        self.head = nn.Sequential(
            nn.Linear(window * frame_dim + pos_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(
        self, flat: torch.Tensor, pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        # flat: [B, W*L]
        B = flat.shape[0]
        x = flat.view(B, self.W, self.L)
        x = self.frame_proj(x).reshape(B, -1)
        if self.pos_dim:
            x = torch.cat([x, pos], dim=-1)
        x = self.norm(x)
        return self.head(x)


def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    p_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    p_va: np.ndarray,
    latent_dim: int,
    window: int,
    use_pos: bool,
    device: str = "cuda",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    wd: float = 1e-2,
    seed: int = 0,
):
    torch.manual_seed(seed)
    pos_dim = 2 if use_pos else 0
    model = SmallLatentMLP(latent_dim, window, pos_dim=pos_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # balanced class weights — same as sklearn balanced
    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    cw = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    Xtr_t = torch.from_numpy(X_tr).to(device)
    ytr_t = torch.from_numpy(y_tr).to(device)
    ptr_t = torch.from_numpy(np.stack([p_tr, np.ones_like(p_tr) - p_tr], axis=-1)).to(
        device
    )
    Xva_t = torch.from_numpy(X_va).to(device)
    pva_t = torch.from_numpy(np.stack([p_va, np.ones_like(p_va) - p_va], axis=-1)).to(
        device
    )

    best_val_f1 = -1.0
    best_thresh = 0.5
    best_epoch = -1
    n = Xtr_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x = Xtr_t[idx]
            p = ptr_t[idx] if use_pos else None
            yhat = model(x, p)
            loss = loss_fn(yhat, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            logits = model(Xva_t, pva_t if use_pos else None)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        f1, thresh = sweep_f1(probs, y_va)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh
            best_epoch = ep
    return {
        "f1": float(best_val_f1),
        "thresh": float(best_thresh),
        "epoch": int(best_epoch),
        "n_params": int(n_params),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def sweep_f1(probs: np.ndarray, y: np.ndarray, steps: int = 30) -> tuple[float, float]:
    best_f1, best_th = -1.0, 0.5
    for th in np.linspace(0.1, 0.95, steps):
        preds = (probs >= th).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return best_f1, best_th


def full_metrics(probs: np.ndarray, y: np.ndarray, thresh: float) -> dict:
    preds = (probs >= thresh).astype(int)
    return {
        "f1": float(f1_score(y, preds, zero_division=0)),
        "acc": float(accuracy_score(y, preds)),
        "prec": float(precision_score(y, preds, zero_division=0)),
        "rec": float(recall_score(y, preds, zero_division=0)),
        "thresh": float(thresh),
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--feature-source",
        choices=["real_hidden"],
        default="real_hidden",
        help="Currently only real_hidden — imagined needs WM rollout; can extend later.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional /dev/shm cache for the X,y arrays",
    )
    parser.add_argument(
        "--skip-knn", action="store_true", help="Skip kNN (slowest step on CPU)"
    )
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    W = int(cfg.classifier.window)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ceiling_log.jsonl"
    log_f = open(log_path, "w")

    def log(d: dict) -> None:
        d["ts"] = time.strftime("%H:%M:%S")
        print(json.dumps(d), flush=True)
        log_f.write(json.dumps(d) + "\n")
        log_f.flush()

    log(
        {
            "event": "config",
            "W": W,
            "feature_source": args.feature_source,
            "seed": args.seed,
            "val_frac": args.val_frac,
        }
    )

    cache_p = Path(args.cache) if args.cache else None
    if cache_p and cache_p.exists():
        log({"event": "loading_cache", "path": str(cache_p)})
        z = np.load(str(cache_p))
        X, y, pos, eids = z["X"], z["y"], z["pos"], z["eids"]
    else:
        rng = np.random.default_rng(args.seed)
        log({"event": "discover_pairs"})
        succ = _find_demo_pairs(cfg.wm_replay.raw_dir, cfg.wm_replay.hidden_dir)
        fail = _find_demo_pairs(
            cfg.wm_replay.failure_raw_dir, cfg.wm_replay.failure_hidden_dir
        )
        log({"event": "pair_counts", "succ": len(succ), "fail": len(fail)})
        log({"event": "build_windows_start"})
        X_s, y_s, p_s, e_s = build_windows(succ, W, rng, "succ")
        X_f, y_f, p_f, e_f = build_windows(fail, W, rng, "fail")
        # offset failure episode ids so they don't collide
        e_f = e_f + (int(e_s.max()) + 1)
        X = np.concatenate([X_s, X_f], axis=0)
        y = np.concatenate([y_s, y_f], axis=0)
        pos = np.concatenate([p_s, p_f], axis=0)
        eids = np.concatenate([e_s, e_f], axis=0)
        if cache_p:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(cache_p), X=X, y=y, pos=pos, eids=eids)
            log({"event": "cache_saved", "path": str(cache_p)})

    log(
        {
            "event": "data_summary",
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "pos": int((y == 1).sum()),
            "neg": int((y == 0).sum()),
        }
    )

    # ------------ stratified split ----------------------------------------
    idx_tr, idx_va = train_test_split(
        np.arange(len(y)), test_size=args.val_frac, stratify=y, random_state=args.seed
    )
    X_tr, X_va = X[idx_tr], X[idx_va]
    y_tr, y_va = y[idx_tr], y[idx_va]
    p_tr, p_va = pos[idx_tr], pos[idx_va]
    log(
        {
            "event": "split",
            "n_tr": len(y_tr),
            "n_va": len(y_va),
            "pos_tr": int((y_tr == 1).sum()),
            "neg_tr": int((y_tr == 0).sum()),
            "pos_va": int((y_va == 1).sum()),
            "neg_va": int((y_va == 0).sum()),
        }
    )

    results: dict[str, dict] = {}

    # ------------ 1. sklearn LR baseline ----------------------------------
    log({"event": "fit_lr"})
    t0 = time.time()
    lr = LogisticRegression(
        C=0.01, class_weight="balanced", max_iter=200, solver="lbfgs", n_jobs=-1
    )
    lr.fit(X_tr, y_tr)
    probs_va = lr.predict_proba(X_va)[:, 1]
    f1, th = sweep_f1(probs_va, y_va)
    m = full_metrics(probs_va, y_va, th)
    m["wall_s"] = time.time() - t0
    results["LR"] = m
    log({"event": "result", "method": "LR", **m})

    # ------------ 2. sklearn LR + position feature ------------------------
    log({"event": "fit_lr_pos"})
    t0 = time.time()
    pos_feat_tr = np.stack([p_tr, 1.0 - p_tr], axis=-1).astype(np.float32)
    pos_feat_va = np.stack([p_va, 1.0 - p_va], axis=-1).astype(np.float32)
    Xp_tr = np.concatenate([X_tr, pos_feat_tr], axis=-1)
    Xp_va = np.concatenate([X_va, pos_feat_va], axis=-1)
    lr_pos = LogisticRegression(
        C=0.01, class_weight="balanced", max_iter=200, solver="lbfgs", n_jobs=-1
    )
    lr_pos.fit(Xp_tr, y_tr)
    probs_va = lr_pos.predict_proba(Xp_va)[:, 1]
    f1, th = sweep_f1(probs_va, y_va)
    m = full_metrics(probs_va, y_va, th)
    m["wall_s"] = time.time() - t0
    results["LR+pos"] = m
    log({"event": "result", "method": "LR+pos", **m})

    # ------------ 3. kNN with cosine — Bayes proxy ------------------------
    if not args.skip_knn:
        # L2 normalize features so euclidean distance ≈ cosine
        Xn_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
        Xn_va = X_va / (np.linalg.norm(X_va, axis=1, keepdims=True) + 1e-8)
        for k in [1, 5, 25, 50]:
            if k > len(y_tr):
                continue
            log({"event": "fit_knn", "k": k})
            t0 = time.time()
            knn = KNeighborsClassifier(
                n_neighbors=k, algorithm="brute", metric="cosine", n_jobs=-1
            )
            knn.fit(Xn_tr, y_tr)
            probs_va = knn.predict_proba(Xn_va)[:, 1]
            f1, th = sweep_f1(probs_va, y_va)
            m = full_metrics(probs_va, y_va, th)
            m["wall_s"] = time.time() - t0
            results[f"kNN(k={k})"] = m
            log({"event": "result", "method": f"kNN(k={k})", **m})

    # ------------ 4. small MLP (proper regularization) --------------------
    if not args.skip_mlp:
        latent_dim = X_tr.shape[1] // W
        for use_pos in [False, True]:
            tag = "MLP+pos" if use_pos else "MLP"
            log({"event": "train_mlp", "use_pos": use_pos})
            t0 = time.time()
            r = train_mlp(
                X_tr,
                y_tr,
                p_tr,
                X_va,
                y_va,
                p_va,
                latent_dim=latent_dim,
                window=W,
                use_pos=use_pos,
                device=args.device,
                seed=args.seed,
            )
            r["wall_s"] = time.time() - t0
            results[tag] = r
            log({"event": "result", "method": tag, **r})

    # ------------ summary -------------------------------------------------
    summary = sorted(results.items(), key=lambda kv: -kv[1].get("f1", -1))
    log({"event": "summary"})
    for name, m in summary:
        log(
            {
                "event": "rank",
                "method": name,
                "f1": m.get("f1"),
                "thresh": m.get("thresh"),
                "n_params": m.get("n_params"),
                "wall_s": m.get("wall_s"),
            }
        )

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"results": results, "ranked": [s[0] for s in summary]}, fh, indent=2)
    log_f.close()
    print("\n[done] →", out_dir / "summary.json")


if __name__ == "__main__":
    main()
