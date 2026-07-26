"""Shared classifier threshold-sweep metrics (LUMOS sweep protocol).

Used by both the standalone classifier runner and the online cotrain
warmup calibration/validation gate, so neither depends on the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def sweep_threshold_metrics(
    probs: np.ndarray,
    ys: np.ndarray,
    thresholds: np.ndarray,
    tag: str,
    *,
    selection_metric: str = "f1",
) -> dict[str, Any]:
    selection_metric = str(selection_metric).lower()
    supported = {"f1", "macro_f1", "failure_f1", "balanced_acc"}
    if selection_metric not in supported:
        raise ValueError(
            f"selection_metric must be one of {sorted(supported)}, got {selection_metric!r}"
        )
    best_score = -1.0
    best_thresh = float(thresholds[0])
    best_row: dict[str, float] | None = None
    rows: dict[str, dict[str, float]] = {}
    for th in thresholds:
        preds = (probs >= th).astype(np.int64)
        f1 = float(f1_score(ys, preds, zero_division=0))
        tp = int(((preds == 1) & (ys == 1)).sum())
        tn = int(((preds == 0) & (ys == 0)).sum())
        fp = int(((preds == 1) & (ys == 0)).sum())
        fn = int(((preds == 0) & (ys == 1)).sum())
        row = {
            "f1": f1,
            "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
            "failure_f1": float(f1_score(ys, preds, pos_label=0, zero_division=0)),
            "acc": float(accuracy_score(ys, preds)),
            "balanced_acc": float(balanced_accuracy_score(ys, preds)),
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
        rows[f"th_{th:.2f}"] = row
        score = float(row[selection_metric])
        if score > best_score:
            best_score, best_thresh, best_row = score, float(th), row
    assert best_row is not None
    all_positive = np.ones_like(ys)
    return {
        "selection_metric": selection_metric,
        "best_score": best_score,
        "best_f1": float(best_row["f1"]),
        "best_macro_f1": float(best_row["macro_f1"]),
        "best_failure_f1": float(best_row["failure_f1"]),
        "best_balanced_acc": float(best_row["balanced_acc"]),
        "best_thresh": best_thresh,
        "all_positive_f1_baseline": float(f1_score(ys, all_positive, zero_division=0)),
        "n": int(len(ys)),
        "n_pos": int((ys == 1).sum()),
        "n_neg": int((ys == 0).sum()),
        "tag": tag,
        # full sweep retained for offline analysis; small dict, ok to log
        "per_thresh": rows,
    }


__all__ = ["sweep_threshold_metrics"]
