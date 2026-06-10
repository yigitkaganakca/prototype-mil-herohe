#!/usr/bin/env python3
"""PANTHER-style Figure 3 panel for a single HEROHE slide (prototype assignment + π + montage).

Inspired by Song et al. CVPR 2024 Fig. 3(A): tissue-cropped WSI, prototypical assignment map,
ROI zoom, estimated prototype proportions, and per-prototype exemplar patches.

Unlike a full-slide TRIDENT thumbnail (~900 px, mostly background glass), this script:
  1. Computes a **tissue bounding box** from all segmented patch coordinates in the .h5
  2. Reads that region from the **OpenSlide WSI** at up to ``--max_side`` pixels (default 3200)
  3. Renders assignment maps in the same crop space (white background outside patches)
  4. Uses **native 256 px patch crops** (512 px level-0) for the exemplar montage

Uses **test** features/WSIs by default (herohe/wsi_test/) — train/test IDs overlap but are
different specimens.

Example (held-out test slide 257):

    python herohe/gp2/scripts/render_panther_fig3.py \\
        --slide_id 257 \\
        --split test \\
        --checkpoint herohe/gp2/runs/khead_reg_sweep/reg_d06_wd4e3_ent15_pd20_ls10/fold_0/best.pt \\
        --out herohe/report/figures/interp_panther_fig3_slide257.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig
from herohe.gp2.scripts.herohe_wsi_paths import features_dir, resolve_wsi

PROTO_COLORS = [
    "#d62728", "#ff7f0e", "#9467bd", "#1f77b4",
    "#2ca02c", "#8c564b", "#e377c2", "#17becf",
]


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(PhenoHER2BinaryConfig)}
    cfg = PhenoHER2BinaryConfig(**{k: blob["config"][k] for k in names if k in blob["config"]})
    model = PhenoHER2Binary(cfg)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval().to(device)
    seed = int(blob.get("seed", 0))
    return model, cfg, seed


def load_h5_full(fdir: Path, slide_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(fdir / f"{slide_id}.h5", "r") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        attrs = dict(f["coords"].attrs)
    return feats, coords, attrs


def subsample_bag(
    feats: np.ndarray,
    coords: np.ndarray,
    max_patches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(feats))
    if len(feats) > max_patches:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(feats), max_patches, replace=False))
    return feats[idx], coords[idx]


def tissue_bbox_level0(
    coords: np.ndarray,
    attrs: dict,
    pad_frac: float = 0.015,
) -> tuple[int, int, int, int]:
    """Tight bbox around all tissue patches (excludes slide background)."""
    ps = float(attrs["patch_size_level0"])
    w0 = float(attrs["level0_width"])
    h0 = float(attrs["level0_height"])
    x0 = int(coords[:, 0].min())
    y0 = int(coords[:, 1].min())
    x1 = int(coords[:, 0].max() + ps)
    y1 = int(coords[:, 1].max() + ps)
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(int(w0), x1 + px)
    y1 = min(int(h0), y1 + py)
    return x0, y0, x1, y1


def read_wsi_crop(
    wsi_path: Path,
    bbox: tuple[int, int, int, int],
    max_side: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Read tissue crop from WSI; return RGB image, level0→pixel scale, origin (x0, y0)."""
    from openslide import OpenSlide

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid tissue bbox {bbox}")

    target_scale = min(1.0, max_side / max(w, h))
    out_w = max(1, int(round(w * target_scale)))
    out_h = max(1, int(round(h * target_scale)))

    with OpenSlide(str(wsi_path)) as slide:
        if target_scale >= 0.999:
            region = slide.read_region((x0, y0), 0, (w, h)).convert("RGB")
            if region.size != (out_w, out_h):
                region = region.resize((out_w, out_h), Image.Resampling.LANCZOS)
        else:
            target_ds = 1.0 / target_scale
            level = 0
            for i, ds in enumerate(slide.level_downsamples):
                if ds <= target_ds:
                    level = i
                else:
                    break
            ds = float(slide.level_downsamples[level])
            rw = max(1, int(round(w / ds)))
            rh = max(1, int(round(h / ds)))
            region = slide.read_region((x0, y0), level, (rw, rh)).convert("RGB")
            region = region.resize((out_w, out_h), Image.Resampling.LANCZOS)

    scale = out_w / w
    return np.asarray(region), scale, (x0, y0)


def rasterize_assignment_crop(
    coords: np.ndarray,
    assign: np.ndarray,
    attrs: dict,
    origin: tuple[int, int],
    scale: float,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Hard-assignment map in crop pixel space; white outside patch footprints."""
    x0o, y0o = origin
    ps_px = max(1, int(round(float(attrs["patch_size_level0"]) * scale)))
    th, tw = out_shape
    rgb = np.ones((th, tw, 3), dtype=np.float32)
    for (cx, cy), k in zip(coords, assign):
        px = int(round((int(cx) - x0o) * scale))
        py = int(round((int(cy) - y0o) * scale))
        x1, y1 = min(tw, px + ps_px), min(th, py + ps_px)
        if x1 <= px or y1 <= py:
            continue
        rgb[py:y1, px:x1] = matplotlib.colors.to_rgb(PROTO_COLORS[int(k) % len(PROTO_COLORS)])
    return rgb


def proto_roi_bbox(
    coords: np.ndarray,
    assign: np.ndarray,
    attrs: dict,
    proto_k: int,
    tissue_bbox: tuple[int, int, int, int],
    pad_frac: float = 0.12,
) -> tuple[int, int, int, int]:
    ps = float(attrs["patch_size_level0"])
    sel = coords[assign == proto_k]
    if len(sel) == 0:
        return tissue_bbox
    tx0, ty0, tx1, ty1 = tissue_bbox
    x0, y0 = int(sel[:, 0].min()), int(sel[:, 1].min())
    x1, y1 = int(sel[:, 0].max() + ps), int(sel[:, 1].max() + ps)
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x0 = max(tx0, x0 - px)
    y0 = max(ty0, y0 - py)
    x1 = min(tx1, x1 + px)
    y1 = min(ty1, y1 + py)
    return x0, y0, x1, y1


def load_test_label(slide_id: str, labels_xlsx: Path) -> tuple[str, int | None]:
    if not labels_xlsx.is_file():
        return "unknown", None
    df = pd.read_excel(labels_xlsx)
    id_col = next((c for c in df.columns if "slide" in c.lower() or c.lower() == "id"), df.columns[0])
    df[id_col] = df[id_col].astype(str)
    row = df[df[id_col] == str(slide_id)]
    if row.empty:
        return "unknown", None
    for col in ("gt_binary", "GT_binary", "binary", "HER2", "label"):
        if col in row.columns:
            val = int(row.iloc[0][col])
            return ("HER2+" if val == 1 else "HER2−"), val
    return "unknown", None


def read_patch_native(wsi_path: Path, x: int, y: int, ps0: int, display_px: int) -> np.ndarray:
    from openslide import OpenSlide

    with OpenSlide(str(wsi_path)) as slide:
        im = slide.read_region((int(x), int(y)), 0, (ps0, ps0)).convert("RGB")
    if im.size != (display_px, display_px):
        im = im.resize((display_px, display_px), Image.Resampling.LANCZOS)
    return np.asarray(im)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide_id", default="257")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels_xlsx", type=Path, default=_REPO / "herohe/Test (ground truth)(1).xlsx")
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--top_m", type=int, default=4, help="Exemplar columns per prototype row")
    ap.add_argument("--max_side", type=int, default=3200, help="Longest edge (px) for tissue WSI crop")
    ap.add_argument("--patch_display_px", type=int, default=140, help="Montage tile size (from native patch)")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    sid = str(args.slide_id)
    fdir = features_dir(args.split)
    wsi_path = resolve_wsi(sid, split=args.split)

    device = pick_device(args.device)
    model, cfg, seed = load_model(args.checkpoint, device)

    feats_full, coords_full, attrs = load_h5_full(fdir, sid)
    tissue_bbox = tissue_bbox_level0(coords_full, attrs)
    feats, coords = subsample_bag(feats_full, coords_full, args.max_patches, seed)

    tissue_rgb, scale, origin = read_wsi_crop(wsi_path, tissue_bbox, args.max_side)
    th, tw = tissue_rgb.shape[:2]

    x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    out = model(x)
    pred = model.predict(x)
    prob_pos = float(pred["prob_positive"][0])

    sa = out["soft_assign"][0].float().cpu().numpy()
    pa = out["patch_attn"][0].float().cpu().numpy()
    hard = sa.argmax(axis=1)
    pi_soft = sa.mean(axis=0)
    counts = np.bincount(hard, minlength=cfg.num_prototypes).astype(np.float64)
    pi_hard = counts / max(counts.sum(), 1.0)
    phen_attn = out.get("phen_token_attn")
    pi_attn = phen_attn[0].float().cpu().numpy() if phen_attn is not None else None

    label_tag, _ = load_test_label(sid, args.labels_xlsx) if args.split == "test" else ("train", None)

    assign_rgb = rasterize_assignment_crop(coords, hard, attrs, origin, scale, (th, tw))

    dom_k = int(pi_hard.argmax())
    proto_bbox = proto_roi_bbox(coords, hard, attrs, dom_k, tissue_bbox)
    proto_rgb, proto_scale, proto_origin = read_wsi_crop(wsi_path, proto_bbox, max_side=int(args.max_side * 0.55))
    pth, ptw = proto_rgb.shape[:2]
    proto_assign = rasterize_assignment_crop(
        coords, hard, attrs, proto_origin, proto_scale, (pth, ptw)
    )

    K = cfg.num_prototypes
    ps0 = int(float(attrs["patch_size_level0"]))
    score = sa * pa
    top_m = min(args.top_m, len(feats))
    montage = []
    for k in range(K):
        order = np.argsort(-score[:, k])[:top_m]
        row = []
        for pi in order:
            cx, cy = int(coords[pi, 0]), int(coords[pi, 1])
            try:
                row.append(read_patch_native(wsi_path, cx, cy, ps0, args.patch_display_px))
            except Exception:
                row.append(np.zeros((args.patch_display_px, args.patch_display_px, 3), dtype=np.uint8))
        while len(row) < top_m:
            row.append(row[-1] if row else np.zeros((args.patch_display_px, args.patch_display_px, 3), dtype=np.uint8))
        montage.append(row)

    tx0, ty0, tx1, ty1 = tissue_bbox
    crop_note = f"tissue crop {tx1 - tx0}×{ty1 - ty0} lv0 px → {tw}×{th} display"

    fig = plt.figure(figsize=(16, 13), facecolor="white")
    gs = GridSpec(
        3, 3,
        height_ratios=[1.25, 0.5, 1.15],
        width_ratios=[1, 1, 1],
        hspace=0.28,
        wspace=0.08,
    )

    ax_wsi = fig.add_subplot(gs[0, 0])
    ax_wsi.imshow(tissue_rgb, interpolation="lanczos")
    ax_wsi.set_title("(A) H&E — tissue crop (no slide background)", fontweight="bold", fontsize=11)
    ax_wsi.axis("off")

    ax_map = fig.add_subplot(gs[0, 1])
    ax_map.imshow(assign_rgb, interpolation="nearest")
    ax_map.set_title("(B) Prototypical assignment map", fontweight="bold", fontsize=11)
    ax_map.axis("off")

    ax_roi = fig.add_subplot(gs[0, 2])
    ax_roi.imshow(proto_rgb, interpolation="lanczos")
    ax_roi.imshow(proto_assign, alpha=0.62, interpolation="nearest")
    ax_roi.set_title(f"(C) ROI — dominant P{dom_k} region", fontweight="bold", fontsize=11)
    ax_roi.axis("off")

    ax_bar = fig.add_subplot(gs[1, :])
    xpos = np.arange(K)
    wbar = 0.35 if pi_attn is None else 0.25
    ax_bar.bar(xpos - wbar, pi_soft, width=wbar, color="#5B9BD5", label=r"$\pi$ soft (mean assign)", edgecolor="white")
    ax_bar.bar(xpos, pi_hard, width=wbar, color="#C0504D", label=r"$\pi$ hard (count/N)", edgecolor="white")
    if pi_attn is not None:
        ax_bar.bar(xpos + wbar, pi_attn, width=wbar, color="#4F8F5F", label=r"$\pi$ token-attn", edgecolor="white")
    ax_bar.axhline(1.0 / K, color="gray", ls="--", lw=1, alpha=0.6, label=f"uniform 1/{K}")
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels([f"P{k}" for k in range(K)])
    ax_bar.set_ylabel("Proportion")
    ymax = max(0.35, float(max(pi_soft.max(), pi_hard.max(), (pi_attn.max() if pi_attn is not None else 0))) * 1.2)
    ax_bar.set_ylim(0, ymax)
    ax_bar.legend(loc="upper right", fontsize=8, ncol=2)
    ax_bar.set_title("(D) Estimated prototype distribution (sampled bag)", fontweight="bold", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.25, ls="--")
    for i, v in enumerate(pi_hard):
        if v > 0.02:
            ax_bar.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color="#C0504D")

    ax_m = fig.add_subplot(gs[2, :])
    ax_m.set_xlim(0, top_m)
    ax_m.set_ylim(0, K)
    ax_m.invert_yaxis()
    ax_m.set_title(
        f"(E) Top exemplar patches per prototype (native {ps0}px crops, display {args.patch_display_px}px)",
        fontweight="bold",
        fontsize=11,
    )
    for k in range(K):
        for j in range(top_m):
            ax_m.imshow(montage[k][j], extent=(j, j + 1, k, k + 1), interpolation="lanczos")
            ax_m.add_patch(
                Rectangle((j, k), 1, 1, fill=False, edgecolor=PROTO_COLORS[k % len(PROTO_COLORS)], lw=2)
            )
    ax_m.set_yticks(np.arange(K) + 0.5)
    ax_m.set_yticklabels([f"P{k}" for k in range(K)], fontsize=10)
    ax_m.set_xticks([])
    for spine in ax_m.spines.values():
        spine.set_visible(False)

    split_note = "held-out TEST" if args.split == "test" else "TRAIN"
    fig.suptitle(
        f"Prototype-oriented interpretation (PANTHER-style) — slide {sid} ({split_note}, {label_tag})  "
        f"P(HER2+)={prob_pos:.3f}  |  {crop_note}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        "Test WSIs: herohe/wsi_test/  ·  Same numeric IDs in train/test are different specimens",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.out}  ({tw}×{th} tissue panel, dpi={args.dpi})")
    print(f"  WSI: {wsi_path}")
    print(f"  tissue bbox (lv0): {tissue_bbox}")
    print(f"  pi_soft:  {np.round(pi_soft, 3).tolist()}")
    print(f"  pi_hard:  {np.round(pi_hard, 3).tolist()}")


if __name__ == "__main__":
    main()
