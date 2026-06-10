"""Initialise PhenoHER2 phenotype prototypes via MiniBatch K-Means on training-fold features.

This corresponds to WP3 Activity 3.1 in the report (Phenotype Discovery Module bootstrap):
the K-Means centroids become the *initial* values of the learnable prototype matrix P.

We deliberately:
    - subsample patches per slide (bounded RAM, much faster K-Means)
    - run on the *training* slides only (no leakage into val/test)
    - support raw bags from any fixed-D patch encoder (e.g. Virchow2 2560-D, TRIDENT ResNet50 1024-D)
    - save the centroids as a torch tensor that the model can load via
      `PhenoHER2.load_prototypes_from_kmeans(...)` / ``PhenoHER2Binary.load_prototypes_from_kmeans(...)``

Usage:

    python herohe/gp2/scripts/init_prototypes.py \\
        --features_dir herohe/gp2/results_smoke_trident_mac/20x_256px_0px_overlap/features_virchow2 \\
        --slide_ids_csv herohe/gp2/data/wsi_list_1_5.csv \\
        --K 16 \\
        --patches_per_slide 512 \\
        --output herohe/gp2/data/prototypes_K16.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.prototype_discovery import save_prototype_checkpoint


def _slide_ids_from_csv(csv_path: str) -> list[str]:
    """Accept either a 'slide_id' column or a 'wsi' column (TRIDENT custom_list_of_wsis)."""
    df = pd.read_csv(csv_path)
    if "slide_id" in df.columns:
        ids = df["slide_id"].astype(str).tolist()
    elif "wsi" in df.columns:
        # TRIDENT-style: filename including ext (e.g. "1.mrxs"); strip extension
        ids = [Path(s).stem for s in df["wsi"].astype(str)]
    else:
        raise ValueError(f"{csv_path} must have a 'slide_id' or 'wsi' column")
    return ids


def collect_patch_sample(
    features_dir: str,
    slide_ids: list[str],
    patches_per_slide: int,
    seed: int,
) -> np.ndarray:
    """Concatenate up to `patches_per_slide` randomly sampled patches from each slide."""
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    missing: list[str] = []
    for sid in slide_ids:
        fp = os.path.join(features_dir, f"{sid}.h5")
        if not os.path.isfile(fp):
            missing.append(sid)
            continue
        with h5py.File(fp, "r") as f:
            feats = f["features"]
            n = feats.shape[0]
            if n == 0:
                continue
            if n <= patches_per_slide:
                arr = feats[:]
            else:
                idx = rng.choice(n, size=patches_per_slide, replace=False)
                idx.sort()
                arr = feats[idx]
        chunks.append(np.asarray(arr, dtype=np.float32))
    if missing:
        print(f"[init_prototypes] WARNING: {len(missing)} slide_ids had no .h5; e.g. {missing[:5]}")
    if not chunks:
        raise RuntimeError("No patches collected; check features_dir and slide_ids.")
    return np.concatenate(chunks, axis=0)


def fit_minibatch_kmeans(
    X: np.ndarray, K: int, batch_size: int, max_iter: int, seed: int
) -> np.ndarray:
    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init="auto",
        init="k-means++",
        verbose=1,
    )
    km.fit(X)
    print(
        f"[init_prototypes] inertia={km.inertia_:.2f}  "
        f"non-empty_clusters={len(np.unique(km.labels_))}"
    )
    return km.cluster_centers_.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--features_dir",
        required=True,
        help="dir containing <slide_id>.h5 patch bags (e.g. TRIDENT features_virchow2 or features_resnet50)",
    )
    ap.add_argument(
        "--slide_ids_csv",
        required=True,
        help="CSV with a 'slide_id' or 'wsi' column listing TRAINING slides only",
    )
    ap.add_argument("--K", type=int, default=16, help="number of prototypes")
    ap.add_argument("--patches_per_slide", type=int, default=512)
    ap.add_argument("--kmeans_batch_size", type=int, default=4096)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help=".pt file to write the (K, D_in) centroids to")
    args = ap.parse_args()

    slide_ids = _slide_ids_from_csv(args.slide_ids_csv)
    print(f"[init_prototypes] {len(slide_ids)} training slides; K={args.K}")

    X = collect_patch_sample(
        features_dir=args.features_dir,
        slide_ids=slide_ids,
        patches_per_slide=args.patches_per_slide,
        seed=args.seed,
    )
    print(f"[init_prototypes] sampled patch matrix: shape={X.shape}, dtype={X.dtype}")

    centers = fit_minibatch_kmeans(
        X=X,
        K=args.K,
        batch_size=args.kmeans_batch_size,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    print(f"[init_prototypes] cluster centers shape: {centers.shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_prototype_checkpoint(
        out_path,
        torch.from_numpy(centers),
        method="kmeans",
        feature_dim=int(centers.shape[1]),
        metadata={
            "n_slides_used": len(slide_ids),
            "patches_per_slide": args.patches_per_slide,
            "seed": args.seed,
        },
    )
    print(f"[init_prototypes] wrote {out_path}  (K={args.K}, method=kmeans)")


if __name__ == "__main__":
    main()
