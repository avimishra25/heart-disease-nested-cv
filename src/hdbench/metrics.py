"""Evaluation metrics, operating-point selection and calibration error."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)

METRIC_NAMES = ["roc_auc", "recall", "specificity", "precision", "accuracy", "f1", "mcc", "brier"]


def classification_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return {
        "roc_auc": auc,
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "brier": brier_score_loss(y, p),
    }


def threshold_for_sensitivity(y, p, target: float) -> float:
    """Highest threshold whose sensitivity on (y, p) is at least ``target``."""
    fpr, tpr, thr = roc_curve(y, p)
    idx = int(np.argmax(tpr >= target))
    t = thr[idx]
    return float(min(t, 1.0)) if np.isfinite(t) else 1.0


def operating_point_metrics(y, p, thr: float) -> dict:
    m = classification_metrics(y, p, thr)
    return {"op_threshold": thr, "op_recall": m["recall"], "op_specificity": m["specificity"],
            "op_precision": m["precision"]}


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)
