#!/usr/bin/env python3
"""Cluster-quality of the medoid prototypes (silhouette + within-cluster cosine).

Reproduces the patch pool used by the prototype-discovery figure
(``render_report_figures.render_prototype_discovery``): the first 40 fold-0
training slides, 800 patches each, capped at 12,000, seed 42, with every patch
assigned to its nearest medoid by cosine (the model's routing rule). On that
pool it reports:

  * within-cluster cosine: mean cos(patch, its assigned medoid);
  * silhouette coefficient (cosine metric) on the hard assignment
    -> reproduces the figure caption's 0.095.

``--pool stage2`` instead uses the cached AP-exemplar discovery pool
(``ap_stage2_pool_fold{f}_train.npy``, ~4.9k patches per fold).

Usage:
    python herohe/gp2/scripts/prototype_cluster_quality.py            # figure pool, fold 0
    python herohe/gp2/scripts/prototype_cluster_quality.py --pool stage2 --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import silhouette_score

np.seterr(all="ignore")  # sklearn's float matmul path is noisy; inputs are clean

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "herohe/gp2/data"
FEAT = REPO / "herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"


def medoids(fold: int, L: int):
    blob = torch.load(DATA / f"prototypes_medoid_phiher2fold_fold{fold}_train_L{L}.pt",
                      map_location="cpu", weights_only=False)
    med = blob["centers"].double().numpy()
    return med, blob


def _assign(Xn, Mn):
    cos = Xn @ Mn.T
    return cos.argmax(axis=1), cos


def figure_pool(fold: int, L: int, seed: int = 42, per_slide: int = 800, cap: int = 12000):
    med, blob = medoids(fold, L)
    Mn = med / np.linalg.norm(med, axis=1, keepdims=True)
    ids = [str(x) for x in blob.get("train_slide_ids", [])][:40]
    rng = np.random.default_rng(seed)
    feats = []
    for sid in ids:
        h5 = FEAT / f"{sid}.h5"
        if not h5.is_file():
            continue
        with h5py.File(h5, "r") as f:
            x = f["features"][:]
        if len(x) > per_slide:
            x = x[rng.choice(len(x), per_slide, replace=False)]
        feats.append(x.astype(np.float64))
    X = np.vstack(feats)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels, cos = _assign(Xn, Mn)
    return Xn, labels, cos, len(X), len(ids)


def stage2_pool(fold: int, L: int):
    med, _ = medoids(fold, L)
    Mn = med / np.linalg.norm(med, axis=1, keepdims=True)
    pool = np.load(DATA / f"ap_stage2_pool_fold{fold}_train.npy").astype(np.float64)
    Xn = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    labels, cos = _assign(Xn, Mn)
    return Xn, labels, cos, len(pool), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--pool", choices=["figure", "stage2"], default="figure")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    print(f"pool={args.pool}  L={args.L}")
    print(f"{'fold':>4} {'n_patch':>8} {'within_cos':>11} {'silh_cos':>9} {'clusters':>9}")
    wins, sils = [], []
    for f in args.folds:
        Xn, labels, cos, n, nslide = (figure_pool(f, args.L) if args.pool == "figure"
                                      else stage2_pool(f, args.L))
        within = float(cos[np.arange(len(cos)), labels].mean())
        used = len(np.unique(labels))
        sil = float(silhouette_score(Xn, labels, metric="cosine")) if used > 1 else float("nan")
        wins.append(within); sils.append(sil)
        print(f"{f:>4} {n:>8} {within:>11.4f} {sil:>9.4f} {used:>9}")
    if len(args.folds) > 1:
        print(f"{'mean':>4} {'':>8} {np.mean(wins):>11.4f} {np.nanmean(sils):>9.4f}")
    print(f"\nfold-{args.folds[0]} within-cos={wins[0]:.4f}  silhouette(cos)={sils[0]:.4f}")


if __name__ == "__main__":
    main()
