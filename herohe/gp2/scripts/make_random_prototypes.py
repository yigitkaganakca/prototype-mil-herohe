#!/usr/bin/env python3
"""Build random-prototype control files: per fold, replace the AP+k-means centers with 8
random patch-embedding vectors sampled ONLY from that fold's training slides (no leakage).

Drop-in compatible with the AP prototype files consumed by train_phenobin_mil.py.
Decisive test of whether the discovered prototypes carry predictive signal: if hard-partition
test AUC is unchanged with random centers, the morphological discovery adds no predictive value.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
FEAT = REPO / "herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
DATA = REPO / "herohe/gp2/data"
L = 8


def main():
    for fold in range(5):
        ap = torch.load(DATA / f"prototypes_ap_phiher2fold_fold{fold}_train_L8.pt",
                        map_location="cpu", weights_only=False)
        train_ids = [str(s) for s in ap["train_slide_ids"]]
        rng = np.random.default_rng(1000 + fold)
        # gather a candidate pool of patches from a random subset of train slides
        pick_slides = rng.choice(train_ids, size=min(16, len(train_ids)), replace=False)
        pool = []
        for sid in pick_slides:
            fp = FEAT / f"{sid}.h5"
            if not fp.is_file():
                continue
            with h5py.File(fp, "r") as f:
                x = f["features"][:]
            idx = rng.choice(len(x), size=min(64, len(x)), replace=False)
            pool.append(x[idx])
        pool = np.vstack(pool)
        sel = rng.choice(len(pool), size=L, replace=False)
        centers = torch.tensor(pool[sel], dtype=torch.float32)
        out = dict(ap)
        out["centers"] = centers
        out["method"] = "random_patches"
        out["stage2_method"] = "random"
        out["note"] = f"random control: {L} random train patches (fold {fold}), seed {1000+fold}"
        outp = DATA / f"prototypes_random_phiher2fold_fold{fold}_train_L8.pt"
        torch.save(out, outp)
        # report cosine spread vs the AP centers (sanity)
        ap_c = torch.nn.functional.normalize(ap["centers"].float(), dim=1)
        rc = torch.nn.functional.normalize(centers, dim=1)
        print(f"fold {fold}: wrote {outp.name}; random within-set off-diag cos mean="
              f"{(rc @ rc.T)[~torch.eye(L, dtype=bool)].mean():.3f}")


if __name__ == "__main__":
    main()
