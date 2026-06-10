#!/usr/bin/env python3
"""Prototype gallery built from patches *nearest to each medoid* (not top-attended).

For every real-patch medoid prototype (fold-0, L=8) we search the fold-0 training
cohort for the patches whose Virchow2 embedding is closest (cosine) to that medoid,
keeping at most a couple per slide for diversity, and crop them from their WSIs.

This represents the prototype's whole neighbourhood/morphology rather than a single
high-attention outlier, so the rows can be inspected for visual distinctness. No
morphological labels are assigned to any prototype.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from herohe.gp2.scripts.herohe_wsi_paths import resolve_wsi
from herohe.gp2.scripts.interp_viz_utils import read_wsi_patch

REPO = _REPO
DATA = REPO / "herohe/gp2/data"
FEAT = REPO / "herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
PROTO = DATA / "prototypes_medoid_phiher2fold_fold0_train_L8.pt"
OUT = REPO / "herohe/report/figures/interp_proto_gallery_medoid.png"

PROTO_COLORS = [
    "#d62728", "#ff7f0e", "#9467bd", "#1f77b4",
    "#2ca02c", "#8c564b", "#e377c2", "#17becf",
]


def wsi_available(sid: str) -> bool:
    """True if the slide's MIRAX sidecar pixel-data folder is present locally."""
    return os.path.isdir(str(sid)) or os.path.isdir(f"herohe/{sid}")


def collect_nearest(centers: np.ndarray, slide_ids: list[str], per_slide_keep: int):
    """Return per-prototype candidate lists of (score, slide_id, x, y), sorted desc."""
    L = centers.shape[0]
    c_n = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    cands: list[list[tuple[float, str, int, int]]] = [[] for _ in range(L)]
    for n, sid in enumerate(slide_ids):
        h5 = FEAT / f"{sid}.h5"
        if not h5.is_file():
            continue
        with h5py.File(h5, "r") as f:
            x = f["features"][:].astype(np.float32)
            coords = f["coords"][:]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = np.nan_to_num(x_n @ c_n.T, nan=-1.0)  # (N, L)
        for k in range(L):
            sk = sim[:, k]
            m = min(per_slide_keep, len(sk))
            top = np.argpartition(-sk, m - 1)[:m]
            for pi in top:
                cands[k].append((float(sk[pi]), str(sid), int(coords[pi, 0]), int(coords[pi, 1])))
        if (n + 1) % 50 == 0:
            print(f"  scanned {n + 1}/{len(slide_ids)} slides")
    for k in range(L):
        cands[k].sort(key=lambda t: -t[0])
    return cands


def crop_row(cand, attrs, want: int, max_per_slide: int, out_px: int):
    tiles, used = [], {}
    for score, sid, x, y in cand:
        if used.get(sid, 0) >= max_per_slide:
            continue
        try:
            wsi = resolve_wsi(sid, split="train")
            tile = read_wsi_patch(wsi, x, y, attrs, out_px=out_px)
        except Exception:
            continue
        if tile is None:
            continue
        tiles.append(tile)
        used[sid] = used.get(sid, 0) + 1
        if len(tiles) >= want:
            break
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_row", type=int, default=8)
    ap.add_argument("--max_per_slide", type=int, default=2)
    ap.add_argument("--per_slide_keep", type=int, default=3)
    ap.add_argument("--out_px", type=int, default=224)
    args = ap.parse_args()

    blob = torch.load(PROTO, map_location="cpu", weights_only=False)
    centers = F.normalize(blob["centers"].float(), dim=1).numpy()
    L = centers.shape[0]
    slide_ids = [str(s) for s in blob["train_slide_ids"] if wsi_available(str(s))]
    print(f"L={L}, scanning {len(slide_ids)} fold-0 training slides with local pixel data "
          "for nearest-medoid patches")

    # one attrs dict (consistent across slides)
    with h5py.File(FEAT / f"{slide_ids[0]}.h5", "r") as f:
        attrs = dict(f["coords"].attrs)

    cands = collect_nearest(centers, slide_ids, args.per_slide_keep)

    rows = []
    for k in range(L):
        tiles = crop_row(cands[k], attrs, args.per_row, args.max_per_slide, args.out_px)
        print(f"  P{k}: {len(tiles)} tiles (top cosine {cands[k][0][0]:.3f})")
        rows.append(tiles)

    px = args.out_px
    gap = 6
    grid_w = args.per_row * px + (args.per_row - 1) * gap
    label_w = 150
    row_h = px + gap
    canvas = np.full((L * row_h - gap, label_w + grid_w, 3), 255, dtype=np.uint8)
    for k, tiles in enumerate(rows):
        y0 = k * row_h
        for j, t in enumerate(tiles):
            x0 = label_w + j * (px + gap)
            canvas[y0:y0 + px, x0:x0 + px] = t

    fig_w = (label_w + grid_w) / 220
    fig_h = (L * row_h) / 220
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(canvas)
    ax.set_xlim(0, canvas.shape[1])
    ax.set_ylim(canvas.shape[0], 0)
    for k in range(L):
        y0 = k * row_h
        ax.add_patch(Rectangle((label_w, y0), grid_w, px, fill=False,
                               edgecolor=PROTO_COLORS[k % len(PROTO_COLORS)], lw=3))
        ax.text(label_w - 24, y0 + px / 2, f"P{k}", ha="right", va="center",
                fontsize=20, fontweight="bold", color=PROTO_COLORS[k % len(PROTO_COLORS)])
    ax.axis("off")
    ax.set_title(
        "Patches nearest each prototype medoid (fold-0 training cohort, $L=8$; "
        "$\\leq$2 per slide)",
        fontsize=15, fontweight="bold", pad=12,
    )
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
