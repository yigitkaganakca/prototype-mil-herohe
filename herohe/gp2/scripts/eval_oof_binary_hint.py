"""Binary AUC/F1 hint from 4-class OOF probabilities (no retraining).

Merges ``oof_predictions.csv`` (from ``eval_clam_oof.py`` / ``eval_pheno_oof.py``)
with ``gt_binary`` in ``folds_v1.csv`` (ISH-aligned negative=0 / positive=1).

Scores (higher = more likely positive):
  - ``p2_plus_p3``: P(IHC 2+) + P(IHC 3+) — common IHC-high proxy
  - ``p3``: P(IHC 3+) only — stricter "amplified-looking" proxy

These are **hints**: the model was trained for 4-class CE, not for ``gt_binary``.

Example::

    python herohe/gp2/scripts/eval_oof_binary_hint.py \\
        --oof_csv herohe/gp2/runs/phenoher2_5fold_oof/oof_predictions.csv \\
        --folds_csv herohe/gp2/data/folds_v1.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof_csv", type=Path, required=True)
    ap.add_argument("--folds_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None, help="Optional path to write metrics JSON")
    args = ap.parse_args()

    oof = pd.read_csv(args.oof_csv)
    oof["slide_id"] = oof["slide_id"].astype(str)
    fdf = pd.read_csv(args.folds_csv)
    fdf["slide_id"] = fdf["slide_id"].astype(str)
    if "gt_binary" not in fdf.columns:
        raise ValueError(f"{args.folds_csv} must contain column gt_binary")
    m = oof.merge(fdf[["slide_id", "gt_binary"]], on="slide_id", how="inner")
    if len(m) != len(oof):
        raise ValueError(
            f"merge lost rows: oof={len(oof)} merged={len(m)}; check slide_id alignment"
        )

    y = m["gt_binary"].astype(int).to_numpy()
    p0 = m["p0"].to_numpy(dtype=np.float64)
    p1 = m["p1"].to_numpy(dtype=np.float64)
    p2 = m["p2"].to_numpy(dtype=np.float64)
    p3 = m["p3"].to_numpy(dtype=np.float64)
    s23 = p2 + p3
    s3 = p3.copy()

    def safe_auc(score: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        if not np.isfinite(score).all():
            return float("nan")
        try:
            return float(roc_auc_score(y, score))
        except ValueError:
            return float("nan")

    def bin_metrics(score: np.ndarray, name: str) -> dict:
        auc = safe_auc(score)
        pred = (score >= 0.5).astype(int)
        return {
            "score": name,
            "n": int(len(y)),
            "n_neg": int((y == 0).sum()),
            "n_pos": int((y == 1).sum()),
            "auc": auc,
            "acc_at_0.5": float(accuracy_score(y, pred)),
            "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
            "precision_at_0.5": float(precision_score(y, pred, zero_division=0)),
            "recall_at_0.5": float(recall_score(y, pred, zero_division=0)),
        }

    out = {
        "oof_csv": str(args.oof_csv),
        "folds_csv": str(args.folds_csv),
        "note": "4-class model; binary scores are post-hoc. gt_binary from folds CSV.",
        "p2_plus_p3": bin_metrics(s23, "p2+p3"),
        "p3_only": bin_metrics(s3, "p3"),
    }
    print(json.dumps(out, indent=2))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
