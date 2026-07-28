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
    # Evaluate thresholds in float64. In NumPy scalar promotion, comparing a
    # float32 score array with ``nextafter(1.0, +inf)`` can otherwise round the
    # scalar back to float32 1.0 and silently destroy the reject-all boundary.
    probs = np.asarray(probs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    selection_metric = str(selection_metric).lower()
    supported = {
        "f1",
        "macro_f1",
        "failure_f1",
        "balanced_acc",
        "recall_at_zero_fp",
    }
    if selection_metric not in supported:
        raise ValueError(
            f"selection_metric must be one of {sorted(supported)}, got {selection_metric!r}"
        )
    if selection_metric == "recall_at_zero_fp":
        # The optimal zero-FP operating point sits immediately above the
        # largest held-out failure score. Include that exact boundary so a
        # coarse fixed grid cannot discard a narrow but valid safe margin.
        negative_probs = probs[ys == 0]
        safe_boundary = (
            np.nextafter(float(negative_probs.max()), np.inf)
            if negative_probs.size
            else float(np.asarray(thresholds, dtype=np.float64).min())
        )
        # Also include an explicit reject-all operating point above every
        # observed/configured score. This works for probability and logit
        # threshold spaces.
        reject_all_base = max(
            float(probs.max()) if probs.size else float("-inf"),
            float(thresholds.max()) if thresholds.size else float("-inf"),
        )
        reject_all = np.nextafter(reject_all_base, np.inf)
        thresholds = np.unique(
            np.concatenate(
                [
                    np.asarray(thresholds, dtype=np.float64),
                    np.asarray([safe_boundary], dtype=np.float64),
                    np.asarray([reject_all], dtype=np.float64),
                ]
            )
        )
    best_score = float("-inf")
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
        row["fpr"] = float(fp / max(1, fp + tn))
        row["specificity"] = float(tn / max(1, fp + tn))
        rows[f"th_{th:.2f}"] = row
        if selection_metric == "recall_at_zero_fp":
            score = float(row["rec"]) if int(row["fp"]) == 0 else -1.0
        else:
            score = float(row[selection_metric])
        if best_row is None or score > best_score:
            best_score, best_thresh, best_row = score, float(th), row
    assert best_row is not None
    all_positive = np.ones_like(ys)
    positive_probs = probs[ys == 1]
    negative_probs = probs[ys == 0]
    return {
        "selection_metric": selection_metric,
        "best_score": best_score,
        "best_f1": float(best_row["f1"]),
        "best_macro_f1": float(best_row["macro_f1"]),
        "best_failure_f1": float(best_row["failure_f1"]),
        "best_balanced_acc": float(best_row["balanced_acc"]),
        "best_precision": float(best_row["prec"]),
        "best_recall": float(best_row["rec"]),
        "best_false_positives": int(best_row["fp"]),
        "best_false_positive_rate": float(best_row["fpr"]),
        "best_specificity": float(best_row["specificity"]),
        "zero_false_positive_constraint_satisfied": bool(int(best_row["fp"]) == 0),
        "best_thresh": best_thresh,
        "all_positive_f1_baseline": float(f1_score(ys, all_positive, zero_division=0)),
        "n": int(len(ys)),
        "n_pos": int((ys == 1).sum()),
        "n_neg": int((ys == 0).sum()),
        "max_positive_score": (
            float(positive_probs.max()) if positive_probs.size else None
        ),
        "min_positive_score": (
            float(positive_probs.min()) if positive_probs.size else None
        ),
        "max_negative_score": (
            float(negative_probs.max()) if negative_probs.size else None
        ),
        "min_negative_score": (
            float(negative_probs.min()) if negative_probs.size else None
        ),
        "tag": tag,
        # full sweep retained for offline analysis; small dict, ok to log
        "per_thresh": rows,
    }


__all__ = ["sweep_threshold_metrics"]
