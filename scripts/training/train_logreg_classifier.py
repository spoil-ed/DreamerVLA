# ruff: noqa: E402
"""Sklearn LogisticRegression baseline — the "LR ceiling" for LatentSuccessClassifier.

CLAUDE.md records that a sklearn LogisticRegression(C=0.01) on real pi0 hidden
W=8 windows reaches F1 ≈ 0.87 — far above the 130 M Transformer's F1 0.13–0.33.
This script re-establishes that ceiling under the current data layout and packs
the result as a LatentSuccessClassifier(head_type="linear") ckpt so the same
episode-eval harness can consume it.

Pipeline:
    1. Walk every success demo (real obs_embedding) and failure demo from
       the same paths as v4_real_hidden.yaml.
    2. For each demo extract W=8 windows under WMPO anchor:
         - 1 end window (label = int(complete))
         - all earlier stride-S windows (label = 0)
    3. Fit sklearn LogisticRegression(class_weight="balanced", C=cfg.C, max_iter=1000)
       on flat (L*W = 286720)-dim feature vectors. Validation split mirrors v3
       (last `val_demos_tail` / `val_failure_tail` demos).
    4. Convert (coef_, intercept_) into a LatentSuccessClassifier(head_type=linear)
       state_dict: weight = [zeros, coef], bias = [0, intercept]. Softmax of those
       logits exactly reproduces sklearn's `predict_proba(...)[:, 1]`.
    5. Save best.ckpt under `<out>/`, plus train/val per-window F1 sweep + a
       train_log.jsonl matching the v3 schema so downstream tools work unchanged.

Usage:
    PYTHONUNBUFFERED=1 /home/user01/miniconda3/envs/dreamervla/bin/python -u \\
        scripts/train_logreg_classifier.py \\
        --config configs/wmpo_classifier_libero_goal_v4_real_hidden.yaml \\
        --out data/outputs/dreamervla/outcome_classifier/libero_goal/lr_ceiling \\
        --C 0.01 --stride-train 8 --stride-val 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dreamer_vla.dataset.wm_replay_classifier_dataset import _find_demo_pairs
from dreamer_vla.models.reward import LatentSuccessClassifier, LatentSuccessClassifierConfig


def _load_demo_real_hidden(
    raw_p: Path, hid_p: Path, demo_key: str
) -> tuple[np.ndarray, int, bool]:
    """Read real obs_embedding (flattened to [T, L]), finish_step, complete."""
    with h5py.File(str(raw_p), "r") as fr:
        grp = fr[demo_key]
        dones = np.asarray(grp["dones"][...]) if "dones" in grp else None
        rewards = np.asarray(grp["rewards"][...]) if "rewards" in grp else None
    with h5py.File(str(hid_p), "r") as fh:
        obs = np.asarray(fh[f"{demo_key}/obs_embedding"][...], dtype=np.float32)
    T = int(obs.shape[0])
    obs = obs.reshape(T, -1)  # [T, L]
    if dones is not None:
        dones = dones[:T]
        finish_step = int(np.argmax(dones)) + 1 if bool(dones.any()) else T
    else:
        finish_step = T
    if rewards is not None:
        complete = bool(rewards[:T].sum() > 0)
    else:
        complete = True  # legacy demos without rewards array → assume success
    return obs, finish_step, complete


def _emit_windows_for_demo(
    obs: np.ndarray,
    finish_step: int,
    complete: bool,
    W: int,
    stride: int,
    earlier_mode: str = "all",
) -> list[tuple[np.ndarray, int]]:
    """WMPO-style windowing.

    earlier_mode:
      'all' — emit every stride-S earlier window (used for val)
      'one_random' — emit one random earlier window (used for train; rng inside caller)

    Here we just emit all earlier windows when earlier_mode='all' or skip earlier
    windows altogether when earlier_mode='end_only'. For 'one_random' use train
    sampling at fit time; sklearn LR doesn't need stochastic sampling so we just
    use 'all' for train and val both — sklearn will balance via class_weight.
    """
    T = int(min(finish_step, obs.shape[0]))
    if T < W:
        return []
    out: list[tuple[np.ndarray, int]] = []
    # end window
    out.append((obs[T - W : T].reshape(-1), int(complete)))
    if earlier_mode == "end_only":
        return out
    # all earlier windows, label=0
    for end in range(T - stride, W - 1, -stride):
        out.append((obs[end - W : end].reshape(-1), 0))
    return out


def _build_dataset(
    pairs: list[tuple[Path, Path, str]],
    W: int,
    stride: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[int] = []
    for i, (raw_p, hid_p, demo_key) in enumerate(pairs):
        obs, fs, complete = _load_demo_real_hidden(raw_p, hid_p, demo_key)
        for win, lab in _emit_windows_for_demo(obs, fs, complete, W, stride):
            X.append(win)
            y.append(lab)
        if (i + 1) % 50 == 0:
            print(
                f"  [{label}] loaded {i + 1}/{len(pairs)} demos, windows={len(X)}",
                flush=True,
            )
    if not X:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def _f1_sweep(
    probs: np.ndarray, ys: np.ndarray, thresholds: np.ndarray
) -> tuple[dict, dict]:
    metrics: dict = {}
    best = {"f1": -1.0, "thresh": float(thresholds[0])}
    for th in thresholds:
        preds = (probs >= th).astype(int)
        f1 = float(f1_score(ys, preds, zero_division=0))
        metrics[f"th_{th:.2f}"] = {
            "f1": f1,
            "acc": float(accuracy_score(ys, preds)),
            "prec": float(precision_score(ys, preds, zero_division=0)),
            "rec": float(recall_score(ys, preds, zero_division=0)),
        }
        if f1 > best["f1"]:
            best = {"f1": f1, "thresh": float(th)}
    return metrics, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="A wmpo_classifier_libero_goal_*.yaml — we read paths "
        "and classifier.window/latent_dim from it.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output dir; receives best.ckpt + train_log.jsonl + config.yaml",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=0.01,
        help="sklearn LogisticRegression inverse-of-regularization. "
        "0.01 matches CLAUDE.md's recorded F1 ≈ 0.87 setting.",
    )
    parser.add_argument("--stride-train", type=int, default=8)
    parser.add_argument("--stride-val", type=int, default=1)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="sklearn LR max_iter (default 200 is enough for L=286720).",
    )
    parser.add_argument("--thresh-steps", type=int, default=30)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    W = int(cfg.classifier.window)
    L = int(cfg.classifier.latent_dim)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")

    def log(line: dict) -> None:
        print(json.dumps(line))
        log_f.write(json.dumps(line) + "\n")
        log_f.flush()

    # ----- pair discovery (mirror v3 train script split) -----------------
    success_pairs = _find_demo_pairs(cfg.wm_replay.raw_dir, cfg.wm_replay.hidden_dir)
    failure_pairs = _find_demo_pairs(
        cfg.wm_replay.failure_raw_dir, cfg.wm_replay.failure_hidden_dir
    )
    val_succ_tail = int(cfg.wm_replay.val_demos_tail)
    val_fail_tail = int(cfg.wm_replay.val_failure_tail)
    tr_succ = success_pairs[:-val_succ_tail]
    va_succ = success_pairs[-val_succ_tail:]
    tr_fail = (
        failure_pairs[:-val_fail_tail]
        if len(failure_pairs) > val_fail_tail
        else failure_pairs[:]
    )
    va_fail = (
        failure_pairs[-val_fail_tail:] if len(failure_pairs) > val_fail_tail else []
    )
    log(
        {
            "event": "split",
            "tr_succ": len(tr_succ),
            "va_succ": len(va_succ),
            "tr_fail": len(tr_fail),
            "va_fail": len(va_fail),
        }
    )

    # ----- build flat (X, y) over real hidden ----------------------------
    print(f"[1/4] building train set (W={W}, stride={args.stride_train})", flush=True)
    X_succ_tr, y_succ_tr = _build_dataset(tr_succ, W, args.stride_train, "succ_tr")
    X_fail_tr, y_fail_tr = _build_dataset(tr_fail, W, args.stride_train, "fail_tr")
    X_tr = (
        np.concatenate([X_succ_tr, X_fail_tr], axis=0) if len(X_fail_tr) else X_succ_tr
    )
    y_tr = (
        np.concatenate([y_succ_tr, y_fail_tr], axis=0) if len(y_fail_tr) else y_succ_tr
    )
    log(
        {
            "event": "train_built",
            "n_train": int(len(y_tr)),
            "pos_train": int((y_tr == 1).sum()),
            "neg_train": int((y_tr == 0).sum()),
        }
    )

    print(f"[2/4] building val set (W={W}, stride={args.stride_val})", flush=True)
    X_succ_va, y_succ_va = _build_dataset(va_succ, W, args.stride_val, "succ_va")
    X_fail_va, y_fail_va = _build_dataset(va_fail, W, args.stride_val, "fail_va")
    X_va = (
        np.concatenate([X_succ_va, X_fail_va], axis=0) if len(X_fail_va) else X_succ_va
    )
    y_va = (
        np.concatenate([y_succ_va, y_fail_va], axis=0) if len(y_fail_va) else y_succ_va
    )
    log(
        {
            "event": "val_built",
            "n_val": int(len(y_va)),
            "pos_val": int((y_va == 1).sum()),
            "neg_val": int((y_va == 0).sum()),
        }
    )

    # ----- fit sklearn LR -----------------------------------------------
    print(
        f"[3/4] fitting LogisticRegression(C={args.C}, class_weight=balanced, "
        f"max_iter={args.max_iter}, lbfgs) on X_tr shape={X_tr.shape}",
        flush=True,
    )
    lr = LogisticRegression(
        C=float(args.C),
        class_weight="balanced",
        max_iter=int(args.max_iter),
        solver="lbfgs",
        n_jobs=-1,
        verbose=1,
    )
    lr.fit(X_tr, y_tr)
    log(
        {
            "event": "lr_fit_done",
            "n_iter": int(lr.n_iter_[0]),
            "coef_norm": float(np.linalg.norm(lr.coef_)),
            "intercept": float(lr.intercept_[0]),
        }
    )

    # per-window threshold sweep on val
    probs_va = lr.predict_proba(X_va)[:, 1]
    thresholds = np.linspace(0.3, 1.0, int(args.thresh_steps))
    metrics_va, best_va = _f1_sweep(probs_va, y_va, thresholds)
    log({"event": "val_window_f1", "best": best_va, "n_val": int(len(y_va))})

    # per-window F1 on train for sanity
    probs_tr = lr.predict_proba(X_tr)[:, 1]
    metrics_tr, best_tr = _f1_sweep(probs_tr, y_tr, thresholds)
    log({"event": "train_window_f1", "best": best_tr, "n_train": int(len(y_tr))})

    # ----- pack as LatentSuccessClassifier(head_type=linear) ckpt --------
    print("[4/4] packing as LatentSuccessClassifier(head_type=linear) ckpt", flush=True)
    coef = lr.coef_[0]  # [L*W]
    intercept = float(lr.intercept_[0])
    # Make logits[1] = coef·x + b, logits[0] = 0 so softmax recovers sklearn proba.
    weight = np.stack([np.zeros_like(coef), coef], axis=0).astype(
        np.float32
    )  # [2, L*W]
    bias = np.array([0.0, intercept], dtype=np.float32)

    cfg_dict = OmegaConf.to_container(cfg.classifier, resolve=True)
    cfg_dict["head_type"] = "linear"
    classifier_cfg = LatentSuccessClassifierConfig(**cfg_dict)
    if classifier_cfg.latent_dim != L or classifier_cfg.window != W:
        raise RuntimeError("config drift between fit and pack")
    model = LatentSuccessClassifier(classifier_cfg)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(weight))
        model.head.bias.copy_(torch.from_numpy(bias))

    # Sanity: torch forward matches sklearn predict_proba up to FP slop.
    sample_n = min(64, X_va.shape[0])
    if sample_n > 0:
        x_t = torch.from_numpy(X_va[:sample_n].reshape(sample_n, W, L))
        with torch.no_grad():
            torch_logits = model(x_t)
            torch_probs = torch.softmax(torch_logits, dim=-1)[:, 1].numpy()
        sklearn_probs = lr.predict_proba(X_va[:sample_n])[:, 1]
        max_abs_err = float(np.max(np.abs(torch_probs - sklearn_probs)))
        log(
            {
                "event": "torch_pack_sanity",
                "n": int(sample_n),
                "max_abs_err": max_abs_err,
            }
        )
        if max_abs_err > 5e-4:
            print(f"WARNING: torch/sklearn disagreement {max_abs_err:.6g}", flush=True)

    payload = {
        "model": model.state_dict(),
        "threshold": float(best_va["thresh"]),
        "f1": float(best_va["f1"]),
        "step": 0,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "head_type": "linear",
        "source": "sklearn-logreg",
        "sklearn": {
            "C": float(args.C),
            "max_iter": int(args.max_iter),
            "class_weight": "balanced",
            "n_iter": int(lr.n_iter_[0]),
            "coef_norm": float(np.linalg.norm(lr.coef_)),
            "intercept": float(lr.intercept_[0]),
        },
    }
    ckpt_path = out_dir / "best.ckpt"
    torch.save(payload, ckpt_path)

    cfg_save = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_save.classifier.head_type = "linear"
    OmegaConf.save(cfg_save, out_dir / "config.yaml")

    log(
        {
            "event": "done",
            "ckpt": str(ckpt_path),
            "best_val": best_va,
            "best_train": best_tr,
        }
    )
    log_f.close()
    print(
        f"\n[done] best val F1={best_va['f1']:.4f} thresh={best_va['thresh']:.2f}"
        f"  →  {ckpt_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
