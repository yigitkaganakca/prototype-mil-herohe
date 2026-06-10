#!/usr/bin/env python3
"""Build real-patch *medoid* prototype files (per fold), the prototypes used by the
reported model.

Stage-2 ``MiniBatchKMeans`` returns ``cluster_centers_`` that are *averaged* vectors.
Averaging shrinks every center toward the global mean, so the L centers are mutually
similar (high off-diagonal cosine) and correspond to no real patch (hence not
inspectable / pathologist-labelable). The reported model instead keeps real exemplars:
each averaged centroid is replaced by its **medoid** -- the real pooled stage-1 exemplar
nearest that centroid. The result is exactly L prototypes that are genuine patch
embeddings, more mutually distinct, and drop-in compatible with ``train_phenobin_mil.py``.

Inputs per fold (produced by ``init_prototypes_ap.py``):
  * ``prototypes_ap_phiher2fold_fold{F}_train_L{L}.pt`` -- stage-2 k-means centroids.
  * ``ap_stage2_pool_fold{F}_train.npy``                -- pooled stage-1 AP exemplars
    (saved via ``--cache_stage2_pool``).

Output per fold:
  * ``prototypes_medoid_phiher2fold_fold{F}_train_L{L}.pt``

Example (after step 2 of the README):
    python herohe/gp2/scripts/make_medoid_prototypes.py --L 8 4 16 --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[3]))
DATA = REPO / "herohe/gp2/data"
SEED = 0  # match the stage-2 k-means seed for a controlled centroid -> medoid swap


def offdiag_cos(c: torch.Tensor) -> float:
    c = F.normalize(c.float(), dim=1)
    S = c @ c.T
    return float(S[~torch.eye(len(S), dtype=bool)].mean())


def build_fold(fold: int, L: int, from_stored: bool, overwrite: bool) -> None:
    """Build medoid prototypes for one fold at a given L.

    ``from_stored=True`` (default): reuse the centres already stored in
    ``prototypes_ap_..._L{L}.pt`` and snap each to its nearest real patch in the cached
    pool (no re-clustering). ``from_stored=False`` re-runs k-means(L, seed) on the pool
    first.
    """
    outp = DATA / f"prototypes_medoid_phiher2fold_fold{fold}_train_L{L}.pt"
    if outp.is_file() and not overwrite:
        print(f"[skip] fold {fold} L{L}: {outp.name} exists")
        return
    ap_path = DATA / f"prototypes_ap_phiher2fold_fold{fold}_train_L{L}.pt"
    pool_path = DATA / f"ap_stage2_pool_fold{fold}_train.npy"
    if not ap_path.is_file():
        raise SystemExit(f"missing {ap_path} (run init_prototypes_ap.py first)")
    if not pool_path.is_file():
        raise SystemExit(
            f"missing {pool_path} (run init_prototypes_ap.py with --cache_stage2_pool)"
        )
    ap = torch.load(ap_path, map_location="cpu", weights_only=False)
    pool = np.load(pool_path).astype(np.float32)

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
        f"real-patch medoids (fold {fold}, L={L}): each centre from "
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
    ap.add_argument("--L", type=int, nargs="+", default=[8, 4, 16],
                    help="prototype counts to build (default: 8 4 16)")
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
