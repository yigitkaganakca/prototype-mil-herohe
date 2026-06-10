#!/usr/bin/env python3
"""Build real-patch *medoid* prototype files (PhiHER2-faithful), per fold, L=8.

The report prototypes used stage-2 ``MiniBatchKMeans`` whose ``cluster_centers_`` are
*averaged* vectors. Averaging shrinks every center toward the global mean, so the 8
centers are mutually similar (off-diag cosine ~0.52) and correspond to no real patch
(hence not pathologist-labelable). PhiHER2 instead keeps real exemplars
(``cluster_centers_indices_``).

This script reproduces the report's k-means partition (same seed) on the cached stage-2
pool of stage-1 AP exemplars, then replaces each averaged centroid with its **medoid**
-- the real pooled exemplar nearest that centroid. Result: exactly L=8 prototypes that
are genuine patch embeddings, far more distinct (~0.34 off-diag cosine), drop-in
compatible with train_phenobin_mil.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "herohe/gp2/data"
SEED = 0  # match the report's stage-2 k-means seed for a controlled centroid->medoid swap


def offdiag_cos(c: torch.Tensor) -> float:
    c = F.normalize(c.float(), dim=1)
    S = c @ c.T
    return float(S[~torch.eye(len(S), dtype=bool)].mean())


def build_fold(fold: int, L: int, from_stored: bool, overwrite: bool) -> None:
    """Build medoid prototypes for one fold at a given L.

    ``from_stored=True`` (recommended for L in {4,16}): reuse the centres already stored in
    ``prototypes_ap_..._L{L}.pt`` and snap each to its nearest real patch in the cached pool
    (no re-clustering). ``from_stored=False`` re-runs k-means(L, seed) on the pool first
    (the original L8 behaviour).
    """
    outp = DATA / f"prototypes_medoid_phiher2fold_fold{fold}_train_L{L}.pt"
    if outp.is_file() and not overwrite:
        print(f"[skip] fold {fold} L{L}: {outp.name} exists")
        return
    ap = torch.load(
        DATA / f"prototypes_ap_phiher2fold_fold{fold}_train_L{L}.pt",
        map_location="cpu",
        weights_only=False,
    )
    pool = np.load(DATA / f"ap_stage2_pool_fold{fold}_train.npy").astype(np.float32)

    if from_stored:
        cent = ap["centers"].float().numpy()
        src = "stored AP centres"
    else:
        km = MiniBatchKMeans(
            n_clusters=L, random_state=SEED, n_init=10, batch_size=4096, max_iter=200
        ).fit(pool)
        cent = km.cluster_centers_
        src = f"k-means(L={L}, seed={SEED})"

    # medoid = real pooled exemplar nearest each centre
    med_idx = [int(np.argmin(np.linalg.norm(pool - c[None, :], axis=1))) for c in cent]
    medoids = torch.tensor(pool[med_idx], dtype=torch.float32)

    out = dict(ap)
    out["centers"] = medoids
    out["method"] = "hierarchical_ap_medoid"
    out["stage2_method"] = "kmeans_medoid"
    out["note"] = (
        f"PhiHER2-faithful real-patch medoids (fold {fold}, L={L}): each centre from "
        f"{src} replaced by its nearest real pooled exemplar"
    )
    torch.save(out, outp)

    cm = offdiag_cos(torch.tensor(cent))
    mm = offdiag_cos(medoids)
    print(
        f"fold {fold} L{L}: wrote {outp.name}  centroid off-diag cos={cm:.3f} -> "
        f"medoid={mm:.3f}  (real exemplars: {len(set(med_idx))}/{L} unique; src={src})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[4, 16],
                    help="prototype counts to build (default 4 16; L8 already exists)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--recluster", action="store_true",
                    help="re-run k-means instead of reusing stored AP centres")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for L in args.L:
        for fold in args.folds:
            build_fold(fold, L, from_stored=not args.recluster, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
