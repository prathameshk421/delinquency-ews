"""Evaluation helpers: ranking, calibration and decision metrics."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 20:
            rows.append({"pred": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    return rows


def summary(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "threshold": float(thr),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }


def curves(y, p, n=60):
    """Down-sampled ROC / PR curves for the frontend."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    i = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    j = np.linspace(0, len(rec) - 1, min(n, len(rec))).astype(int)
    return {
        "roc": [{"x": float(fpr[k]), "y": float(tpr[k])} for k in i],
        "pr": [{"x": float(rec[k]), "y": float(prec[k])} for k in j],
    }


def expected_cost(y, p, thr, c_fn, c_fp):
    yhat = p >= thr
    fn = np.sum((~yhat) & (y == 1))
    fp = np.sum(yhat & (y == 0))
    return float((c_fn * fn + c_fp * fp) / len(y))
