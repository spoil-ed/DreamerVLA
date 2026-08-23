"""Shared classifier threshold-sweep metrics (LUMOS sweep protocol).

Used by both the standalone classifier runner and the online cotrain
warmup calibration/validation gate, so neither depends on the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def sweep_threshold_metrics(
    probs: np.ndarray, ys: np.ndarray, thresholds: np.ndarray, tag: str
) -> dict[str, Any]:
    best_f1 = -1.0
    best_thresh = float(thresholds[0])
    rows: dict[str, dict[str, float]] = {}
    for th in thresholds:
        preds = (probs >= th).astype(np.int64)
        f1 = float(f1_score(ys, preds, zero_division=0))
        tp = int(((preds == 1) & (ys == 1)).sum())
        tn = int(((preds == 0) & (ys == 0)).sum())
        fp = int(((preds == 1) & (ys == 0)).sum())
        fn = int(((preds == 0) & (ys == 1)).sum())
        rows[f"th_{th:.2f}"] = {
            "f1": f1,
            "acc": float(accuracy_score(ys, preds)),
            "prec": float(precision_score(ys, preds, zero_division=0)),
            "rec": float(recall_score(ys, preds, zero_division=0)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "pred_pos": int((preds == 1).sum()),
            "pred_neg": int((preds == 0).sum()),
            "true_pos": int((ys == 1).sum()),
            "true_neg": int((ys == 0).sum()),
        }
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(th)
    return {
        "best_f1": best_f1,
        "best_thresh": best_thresh,
        "n": int(len(ys)),
        "n_pos": int((ys == 1).sum()),
        "n_neg": int((ys == 0).sum()),
        "tag": tag,
        # full sweep retained for offline analysis; small dict, ok to log
        "per_thresh": rows,
    }


__all__ = ["sweep_threshold_metrics"]
