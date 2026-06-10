"""Composite interpretability figure for one test slide (Fig G).

Panels:
  (a) Tissue-cropped thumbnail + patch-attention overlay (dominant active head)
  (b) Token ABMIL weights + hard-assign fractions (seaborn bars)
  (c) Top patches per active head (hard==k, within-cluster patch attn)

Usage:
    python herohe/gp2/scripts/render_interp_slide_figure.py \\
        --slide_id 304 --split test \\
        --out herohe/report/figures/interp_slide304_composite.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig
from herohe.gp2.scripts.herohe_wsi_paths import features_dir, resolve_wsi, trident_root
from herohe.gp2.scripts.interp_viz_utils import (
    active_heads,
    attention_heatmap_cropped,
    hard_assign,
    rank_patches_per_head,
)

PRIMARY_CKPT = _REPO / "herohe/gp2/runs/khead_token_abmil_hard_partition_ent0/fold_0/best.pt"
PALETTE = sns.color_palette("tab10", 10)


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
    model = PhenoHER2Binary(cfg).eval().to(device)
    model.load_state_dict(blob["model_state"], strict=True)
    return model, cfg, int(blob.get("seed", 0))


def read_patch(wsi: Path, x: int, y: int, attrs: dict, size: int) -> np.ndarray:
    from openslide import OpenSlide

    ps0 = int(float(attrs.get("patch_size_level0", 256)))
    with OpenSlide(str(wsi)) as slide:
        im = slide.read_region((x, y), 0, (ps0, ps0)).convert("RGB")
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im)


@torch.no_grad()
def render(
    checkpoint: Path,
    slide_id: str,
    split: str,
    out_path: Path,
    top_m: int = 4,
    tile_px: int = 112,
    max_patches: int = 4096,
    device_name: str = "mps",
    dpi: int = 200,
) -> dict:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    device = pick_device(device_name)
    model, cfg, seed = load_model(checkpoint, device)
    K = cfg.num_prototypes

    fdir = features_dir(split)
    wsi = resolve_wsi(slide_id, split=split)
    trident = trident_root(split)
    thumb = np.array(Image.open(trident / "thumbnails" / f"{slide_id}.jpg").convert("RGB"))

    with h5py.File(fdir / f"{slide_id}.h5") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        attrs = dict(f["coords"].attrs)

    idx = np.arange(len(feats))
    if len(feats) > max_patches:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(feats), max_patches, replace=False))
        feats, coords = feats[idx], coords[idx]

    out = model(torch.from_numpy(feats).unsqueeze(0).to(device))
    sa = out["soft_assign"][0].cpu().numpy()
    pa = out["patch_attn"][0].cpu().numpy()
    hard = hard_assign(sa)
    hard_frac = np.bincount(hard, minlength=K) / len(hard)
    pta = out.get("phen_token_attn")
    phen_w = pta[0].cpu().numpy() if pta is not None else np.zeros(K)
    prob = torch.softmax(out["logits_bin"][0].cpu(), dim=-1).numpy()
    phen_active = out.get("phenotype_active")
    if phen_active is not None:
        phen_active = phen_active[0].cpu().numpy().astype(bool)
    else:
        phen_active = np.ones(K, dtype=bool)

    heads = active_heads(phen_active, hard_frac, phen_w)
    hero_k = max(heads, key=lambda k: phen_w[k] if phen_w[k] > 0 else hard_frac[k])

    crop, heat, bbox = attention_heatmap_cropped(coords, pa[:, hero_k], attrs, thumb)
    x0, y0, x1, y1 = bbox

    fig = plt.figure(figsize=(14, 11), facecolor="white")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.15, 0.55, 1.35], hspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(crop)
    ax_a.imshow(heat, cmap="magma", alpha=0.55, vmin=0, vmax=1)
    ax_a.set_title(
        f"(a) Tissue crop · patch attention for P{hero_k} (hard-partition primary model, slide {slide_id})",
        loc="left", fontweight="semibold", fontsize=11,
    )
    ax_a.set_xlabel(f"$N={len(feats)}$ patches · $P(\\mathrm{{Pos}})={prob[1]:.2f}", fontsize=9)
    ax_a.axis("off")

    ax_b = fig.add_subplot(gs[1, 0])
    x = np.arange(K)
    width = 0.38
    ax_b.bar(x - width / 2, phen_w, width, label="Token weight $\\omega_k$", color="#2E5090", edgecolor="white")
    ax_b.bar(x + width / 2, hard_frac, width, label="Hard assign frac.", color="#C0504D", alpha=0.85, edgecolor="white")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"P{k}" for k in range(K)])
    ax_b.set_ylim(0, max(0.35, phen_w.max() * 1.15, hard_frac.max() * 1.15))
    ax_b.set_ylabel("Weight / fraction")
    ax_b.set_title("(b) Phenotype token ABMIL weights vs. hard-assign mass", loc="left", fontweight="semibold", fontsize=11)
    ax_b.legend(loc="upper right", fontsize=8)

    n_rows = len(heads)
    inner = gs[2, 0].subgridspec(n_rows, top_m, hspace=0.15, wspace=0.05)
    for row, k in enumerate(heads):
        order = rank_patches_per_head(hard, pa, k, top_m)
        for j in range(top_m):
            ax = fig.add_subplot(inner[row, j])
            if j >= len(order):
                ax.axis("off")
                continue
            pi = int(order[j])
            tile = read_patch(wsi, int(coords[pi, 0]), int(coords[pi, 1]), attrs, tile_px)
            ax.imshow(tile)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=PALETTE[k], lw=2.2))
            if j == 0:
                ax.set_ylabel(
                    f"P{k}\n{hard_frac[k]*100:.0f}%",
                    fontsize=9, fontweight="semibold", color=PALETTE[k],
                )
            if row == 0:
                ax.set_title(f"#{j+1}", fontsize=8)

    fig.suptitle(
        "Hard-partition khead interpretability (fold-0 checkpoint, token ABMIL readout)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.text(0.5, 0.005, "(c) Top patches per active head · ranked by within-cluster patch attention (hard == k)", ha="center", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "slide_id": slide_id,
        "split": split,
        "checkpoint": str(checkpoint),
        "P_pos": float(prob[1]),
        "n_patches": int(len(feats)),
        "active_heads": heads,
        "hero_head": hero_k,
        "tissue_bbox_thumb": [x0, y0, x1, y1],
        "output": str(out_path),
    }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=PRIMARY_CKPT)
    ap.add_argument("--slide_id", default="304")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", type=Path, default=_REPO / "herohe/report/figures/interp_slide304_composite.png")
    ap.add_argument("--top_m", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    if "304" in args.out.name and args.slide_id != "304":
        args.out = args.out.parent / f"interp_slide{args.slide_id}_composite.png"
    meta = render(args.checkpoint, args.slide_id, args.split, args.out, top_m=args.top_m, device_name=args.device)
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
