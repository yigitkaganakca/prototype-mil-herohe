#!/usr/bin/env python3
"""Reproducible prototype-geometry statistics for the report.

Deterministic (pool-free) numbers computed here:
  * raw off-diagonal cosine of the L medoid prototype vectors, per fold + mean
    (report quotes ~0.33);
  * cross-fold Hungarian-matched cosine: for each fold f, optimally match its L
    medoids to fold-0's by cosine (scipy ``linear_sum_assignment`` on 1-cos) and
    average the matched-pair cosines; reported mean over folds f!=0
    (report quotes 0.71, "the prototype *set* is reproducible across folds").

Pool-dependent numbers (within-cluster cosine, silhouette) are intentionally
NOT computed here because they depend on the exact patch pool; see
``prototype_cluster_quality.py`` once the pool definition is fixed.

Usage:
    python herohe/gp2/scripts/prototype_geometry.py [--L 8] [--folds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "herohe/gp2/data"


def load_medoids(fold: int, L: int) -> torch.Tensor:
    blob = torch.load(DATA / f"prototypes_medoid_phiher2fold_fold{fold}_train_L{L}.pt",
                      map_location="cpu", weights_only=False)
    return blob["centers"].float()


def offdiag_cos(c: torch.Tensor) -> float:
    c = F.normalize(c.float(), dim=1)
    S = (c @ c.T).numpy()
    L = len(S)
    return float(S[~np.eye(L, dtype=bool)].mean())


def matched_cosine(a: torch.Tensor, b: torch.Tensor) -> tuple[float, list[int]]:
    """Hungarian match rows of a to rows of b by cosine; return mean matched cos."""
    an = F.normalize(a.float(), dim=1)
    bn = F.normalize(b.float(), dim=1)
    C = (an @ bn.T).numpy()           # cosine similarity, LxL
    row, col = linear_sum_assignment(-C)  # maximise total cosine
    return float(C[row, col].mean()), col.tolist()


def projected_offdiag(checkpoint: Path) -> float:
    """Off-diagonal cosine of the trained/loaded prototypes living in the model's
    projected hidden space (report quotes ~0.14)."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    proto = blob["model_state"]["prototypes"].float()
    return offdiag_cos(proto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "herohe/gp2/runs/khead_hard_partition_medoid_proto_control/fold_0/best.pt",
                    help="checkpoint for the projected-space off-diagonal cosine")
    args = ap.parse_args()

    meds = {f: load_medoids(f, args.L) for f in args.folds}

    print(f"=== raw off-diagonal cosine (L={args.L}) ===")
    offs = []
    for f in args.folds:
        o = offdiag_cos(meds[f])
        offs.append(o)
        print(f"  fold {f}: {o:.4f}")
    print(f"  mean over folds: {np.mean(offs):.4f}   (report ~0.33)")

    if args.checkpoint and args.checkpoint.exists():
        print(f"\n=== projected-space off-diagonal cosine ({args.checkpoint.parent.parent.name}/fold_0) ===")
        print(f"  {projected_offdiag(args.checkpoint):.4f}   (report ~0.14)")

    if 0 in args.folds:
        print(f"\n=== cross-fold Hungarian matched cosine vs fold-0 (L={args.L}) ===")
        ms = []
        for f in args.folds:
            if f == 0:
                continue
            mc, perm = matched_cosine(meds[f], meds[0])
            ms.append(mc)
            print(f"  fold {f} -> fold 0: matched cos={mc:.4f}  perm={perm}")
        print(f"  mean matched cosine: {np.mean(ms):.4f}   (report 0.71)")
        identity = all(matched_cosine(meds[f], meds[0])[1] == list(range(args.L))
                       for f in args.folds if f != 0)
        print(f"  all permutations identity (index-aligned)? {identity}  "
              f"(report: NO, indices permuted)")


if __name__ == "__main__":
    main()
