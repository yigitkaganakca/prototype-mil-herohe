"""Shared slide-level metrics for MIL training / OOF evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_ovr_auc(y: np.ndarray, P: np.ndarray, num_classes: int) -> float:
    """Macro-averaged one-vs-rest AUROC (Valieris primary metric for 3-class)."""
    y = np.asarray(y, dtype=np.int64)
    P = np.asarray(P, dtype=np.float64)
    aucs: list[float] = []
    for c in range(num_classes):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        if not np.isfinite(P[:, c]).all():
            continue
        try:
            aucs.append(float(roc_auc_score(y_bin, P[:, c])))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics_from_prob_matrix(
    y: np.ndarray,
    p: np.ndarray,
    P: np.ndarray,
    num_classes: int,
) -> dict:
    """Classification metrics from hard preds and soft prob matrix."""
    P = np.asarray(P, dtype=np.float64).copy()
    p = np.asarray(p, dtype=np.int64).copy()
    y = np.asarray(y, dtype=np.int64)
    if not np.isfinite(P).all():
        bad = ~np.isfinite(P).all(axis=-1)
        P[bad] = 1.0 / num_classes
        p = np.where(bad, P.argmax(axis=-1), p)
    macro_f1 = f1_score(
        y, p, average="macro", labels=list(range(num_classes)), zero_division=0
    )
    out: dict = {
        "macro_f1": float(macro_f1),
        "n": int(len(y)),
    }
    if num_classes == 2:
        auc = float("nan")
        if len(np.unique(y)) == 2 and np.isfinite(P[:, 1]).all():
            try:
                auc = float(roc_auc_score(y, P[:, 1]))
            except ValueError:
                pass
        out["auc_positive"] = auc
    else:
        out["macro_auroc"] = macro_ovr_auc(y, P, num_classes)
    return out
