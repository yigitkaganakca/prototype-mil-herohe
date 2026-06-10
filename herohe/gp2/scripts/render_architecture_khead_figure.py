"""Publication-style khead architecture figure: real patches + attention + schematic.

Panels:
  (a) WSI tissue crop + 4px patch-attention heatmap
  (b) Top-scoring H&E patches per active head
  (c) Valieris-style architecture diagram with mini visualizations

Example:
    python herohe/gp2/scripts/render_architecture_khead_figure.py \\
        --checkpoint herohe/gp2/runs/khead_token_abmil_hard_partition_ent0/fold_0/best.pt \\
        --slide_id 304 --split test \\
        --out_dir herohe/report/figures
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
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
    read_wsi_patch,
)

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
    model = PhenoHER2Binary(cfg)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval().to(device)
    seed = int(blob.get("seed", 0))
    return model, cfg, seed


def load_bag(features_dir: Path, slide_id: str, max_patches: int, seed: int):
    with h5py.File(features_dir / f"{slide_id}.h5", "r") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        attrs = dict(f["coords"].attrs)
    idx = np.arange(len(feats))
    if len(feats) > max_patches:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(feats), max_patches, replace=False))
        feats, coords = feats[idx], coords[idx]
    return feats, coords, attrs, int(feats.shape[1])


def read_patch_rgb(wsi_path: Path, x: int, y: int, attrs: dict, out_px: int = 96) -> np.ndarray:
    tile = read_wsi_patch(wsi_path, x, y, attrs, out_px=out_px)
    if tile is not None:
        return tile
    return np.full((out_px, out_px, 3), 240, dtype=np.uint8)


@torch.no_grad()
def forward_khead(model, feats_np, device):
    x = torch.from_numpy(feats_np).unsqueeze(0).to(device)
    out = model(x)
    sa = out["soft_assign"][0].cpu().numpy()
    pa = out["patch_attn"][0].cpu().numpy()
    hard = hard_assign(sa)
    hard_frac = np.bincount(hard, minlength=pa.shape[1]) / len(hard)
    phen_w = out.get("phen_token_attn")
    phen_token_attn = phen_w.squeeze(0).cpu().numpy() if phen_w is not None else None
    phen_active = out.get("phenotype_active")
    if phen_active is not None:
        phen_active = phen_active[0].cpu().numpy().astype(bool)
    else:
        phen_active = np.ones(pa.shape[1], dtype=bool)
    dom_k = int(hard_frac.argmax())
    return pa, hard, phen_token_attn, phen_active, dom_k, hard_frac


# ---------------------------------------------------------------------------
# Panel (c): publication architecture schematic (NeurIPS / MICCAI style)
# ---------------------------------------------------------------------------

# Restrained palette — one accent hue, grayscale structure, muted proto tints.
T = {
    "ink": "#111827",
    "muted": "#6B7280",
    "line": "#CBD5E1",
    "flow": "#374151",
    "accent": "#1D4ED8",
    "fill": "#FFFFFF",
    "panel": "#F9FAFB",
    "train_edge": "#94A3B8",
    "proto": ["#94A3B8", "#A8A29E", "#78716C", "#64748B", "#71717A", "#6B7280", "#52525B", "#57534E"],
}


def _center(bx, by, bw, bh, cw, ch):
    return bx + (bw - cw) / 2, by + (bh - ch) / 2


def _flow_arrow(ax, x1, y1, x2, y2, *, dashed=False, color=None, lw=1.0, ms=11):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color or T["flow"],
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=5,
    ))


def _stage_box(ax, x, y, w, h, *, fill=None, edge=None, lw=0.85, dashed=False):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.06",
        facecolor=fill or T["fill"], edgecolor=edge or T["line"],
        linewidth=lw, linestyle=(0, (5, 4)) if dashed else "solid", zorder=2,
    )
    ax.add_patch(p)
    return p


def _stage_label(ax, cx, y_top, num: str, title: str, math: str | None = None, y_math: float | None = None):
    ax.text(cx, y_top + 0.28, num, ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            color=T["accent"], zorder=6)
    ax.text(cx, y_top + 0.06, title, ha="center", va="bottom", fontsize=8.5, fontweight="semibold",
            color=T["ink"], zorder=6)
    if math and y_math is not None:
        ax.text(cx, y_math, math, ha="center", va="top", fontsize=6.2, color=T["muted"], zorder=6,
                clip_on=True)


def _draw_patch_extraction(ax, bx, by, bw, bh):
    """Stage 1: WSI tile grid."""
    n, gap = 4, 0.04
    side = (min(bw, bh) * 0.52 - 3 * gap) / n
    total = n * side + (n - 1) * gap
    x0, y0 = _center(bx, by, bw, bh * 0.88, total, total)
    rng = np.random.default_rng(7)
    tints = [(0.88, 0.78, 0.82), (0.82, 0.72, 0.78), (0.90, 0.84, 0.86), (0.78, 0.68, 0.74)]
    for r in range(n):
        for c in range(n):
            col = tints[(r + c) % len(tints)]
            jitter = rng.uniform(-0.02, 0.02)
            ax.add_patch(Rectangle(
                (x0 + c * (side + gap), y0 + (n - 1 - r) * (side + gap)), side, side,
                facecolor=(col[0] + jitter, col[1] + jitter, col[2] + jitter),
                edgecolor="#D6D3D1", lw=0.45, zorder=3,
            ))
    ax.text(bx + bw / 2, by + bh * 0.06, r"$256{\times}256$ RGB", ha="center", fontsize=6.5,
            color=T["muted"], zorder=4)


def _draw_vstack_blocks(ax, cx, y0, bw, blocks: list[str], block_h: float, gap: float):
    """Draw labelled blocks top→bottom with arrows between them."""
    y = y0
    for i, label in enumerate(blocks):
        w = bw * 0.72
        ax.add_patch(Rectangle(
            (cx - w / 2, y - block_h), w, block_h,
            facecolor=T["panel"] if i % 2 == 0 else "#F3F4F6",
            edgecolor=T["line"], lw=0.55, zorder=3,
        ))
        fs = 5.4 if len(label) > 18 else 5.8
        ax.text(cx, y - block_h / 2, label, ha="center", va="center", fontsize=fs,
                color=T["ink"], zorder=4)
        if i < len(blocks) - 1:
            y_bot = y - block_h
            y_next_top = y_bot - gap
            _flow_arrow(ax, cx, y_bot - 0.01, cx, y_next_top + 0.01, lw=0.65, ms=7)
        y -= block_h + gap
    return y


def _draw_virchow2_encoder(ax, bx, by, bw, bh, feat_dim: int):
    """Stage 2: ViT encoder — separate blocks with visible vertical arrows."""
    cx = bx + bw / 2
    inner_y = by + bh * 0.08
    inner_h = bh * 0.80

    pw, ph = bw * 0.13, inner_h * 0.07
    y0 = inner_y + inner_h * 0.92
    ax.add_patch(Rectangle((cx - pw / 2, y0 - ph), pw, ph, facecolor="#E7E5E4",
                           edgecolor="#A8A29E", lw=0.5, zorder=3))
    ax.text(cx, y0 - ph / 2, "patch", ha="center", va="center", fontsize=5.5, color=T["muted"], zorder=4)
    _flow_arrow(ax, cx, y0 - ph, cx, y0 - ph - 0.05, lw=0.65, ms=7)

    block_h, gap = inner_h * 0.075, 0.055
    blocks = [
        "Patchify + pos. embed",
        "Multi-head self-attn",
        "SwiGLU MLP",
        "LayerScale + LN",
        r"$\times 32$ ViT-H/14 (frozen)",
    ]
    y_after = _draw_vstack_blocks(ax, cx, y0 - ph - 0.05, bw, blocks, block_h, gap)

    ew, eh = bw * 0.60, inner_h * 0.08
    y_emb = max(inner_y + inner_h * 0.06, y_after - eh - 0.06)
    _flow_arrow(ax, cx, y_after - 0.01, cx, y_emb + eh + 0.01, lw=0.65, ms=7)
    ax.add_patch(Rectangle((cx - ew / 2, y_emb), ew, eh, facecolor="#DBEAFE",
                           edgecolor=T["accent"], lw=0.65, zorder=3))
    ax.text(cx, y_emb + eh / 2, r"$\mathbf{h}_i=[\,\mathrm{CLS}\,\|\,\overline{\mathrm{patch}}\,]$",
            ha="center", va="center", fontsize=5.6, color=T["accent"], zorder=4)
    ax.text(bx + bw / 2, by + bh * 0.03, "Virchow2 encoder (frozen)", ha="center", fontsize=6.2,
            fontweight="semibold", color=T["ink"], zorder=4)


def _draw_linear_proj(ax, bx, by, bw, bh, d_in: int, d_out: int):
    """Stage 3: dimension reduction — labels below bars."""
    bar_h = bh * 0.28
    w_in, w_out = bw * 0.30, bw * 0.11
    gap = bw * 0.13
    total_w = w_in + gap + w_out
    x0, y0 = _center(bx, by + bh * 0.12, bw, bh * 0.55, total_w, bar_h)
    ax.add_patch(Rectangle((x0, y0), w_in, bar_h, facecolor="#E5E7EB", edgecolor=T["line"], lw=0.55, zorder=3))
    ax.add_patch(Rectangle((x0 + w_in + gap, y0 + bar_h * 0.12), w_out, bar_h * 0.76,
                           facecolor="#374151", edgecolor=T["ink"], lw=0.55, zorder=3))
    mid_y = y0 + bar_h / 2
    ax.text(x0 + w_in + gap / 2, mid_y + bar_h * 0.42, r"$W$", ha="center", fontsize=7, color=T["muted"], zorder=4)
    _flow_arrow(ax, x0 + w_in + 0.02, mid_y, x0 + w_in + gap - 0.02, mid_y, lw=0.75, ms=9)
    lbl_y = y0 - bar_h * 0.18
    ax.text(x0 + w_in / 2, lbl_y, rf"$D={d_in}$", ha="center", va="top", fontsize=6.5, color=T["ink"], zorder=4)
    ax.text(x0 + w_in + gap + w_out / 2, lbl_y, rf"$d={d_out}$", ha="center", va="top",
            fontsize=6.5, color=T["ink"], zorder=4)


def _draw_hard_assign(ax, bx, by, bw, bh, L: int):
    """Stage 4: partitioned patch cloud — one column per prototype."""
    rng = np.random.default_rng(3)
    n_pts = 48
    cols = L
    col_gap = 0.025
    usable_w = bw * 0.90
    col_w = (usable_w - (cols - 1) * col_gap) / cols
    x_start = bx + (bw - usable_w) / 2
    y_base = by + bh * 0.20
    h_pts = bh * 0.52
    r_dot = min(0.022, col_w * 0.22)
    for g in range(cols):
        mask = rng.integers(0, cols, n_pts) == g
        xs = rng.uniform(0.05, 0.95, n_pts)[mask]
        ys = rng.uniform(0.05, 0.95, n_pts)[mask]
        color = T["proto"][g % len(T["proto"])]
        x_col = x_start + g * (col_w + col_gap)
        for px, py in zip(xs, ys):
            ax.add_patch(plt.Circle(
                (x_col + px * col_w, y_base + py * h_pts),
                r_dot, facecolor=color, edgecolor="none", alpha=0.85, zorder=3,
            ))
        ax.add_patch(Rectangle(
            (x_col - 0.01, y_base - 0.01), col_w + 0.02, h_pts + 0.02,
            fill=False, edgecolor=color, lw=0.4, linestyle=(0, (2, 2)), zorder=2,
        ))
        ax.text(x_col + col_w / 2, y_base - 0.08, str(g), ha="center", va="top",
                fontsize=4.5, color=T["muted"], zorder=4)
    ax.text(bx + bw / 2, by + bh * 0.06, rf"$L={L}$ partitions", ha="center", fontsize=6.5,
            color=T["muted"], zorder=4)


def _draw_gated_pool(ax, bx, by, bw, bh):
    """Stage 5: within-cluster gated attention → token."""
    cx = bx + bw / 2
    cell, gap = bw * 0.13, bw * 0.035
    grid_w = 3 * cell + 2 * gap
    x0 = bx + bw * 0.08
    y0 = by + bh * 0.30
    for i in range(9):
        ax.add_patch(Rectangle(
            (x0 + (i % 3) * (cell + gap), y0 + (i // 3) * (cell + gap)), cell, cell,
            facecolor="#F3F4F6", edgecolor="#D1D5DB", lw=0.4, zorder=3,
        ))
    grid_h = 3 * cell + 2 * gap
    grid_cy = y0 + grid_h / 2
    ax.text(x0 + grid_w / 2, y0 - 0.12, r"$\alpha_{k,i}$", ha="center", fontsize=6.5, color=T["muted"], zorder=4)
    tok_w, tok_h = bw * 0.16, bh * 0.42
    tok_x = bx + bw - tok_w - bw * 0.10
    tok_y = grid_cy - tok_h / 2
    _flow_arrow(ax, x0 + grid_w + bw * 0.04, grid_cy, tok_x - bw * 0.04, grid_cy, lw=0.75, ms=9)
    ax.add_patch(Rectangle(
        (tok_x, tok_y), tok_w, tok_h, facecolor="#374151", edgecolor=T["ink"], lw=0.5, zorder=3,
    ))
    ax.text(tok_x + tok_w / 2, grid_cy, r"$\mathbf{t}_k$", ha="center", va="center",
            fontsize=7, color="white", zorder=4)


def _draw_token_abmil(ax, bx, by, bw, bh, weights, L: int):
    """Stage 6: gated attention over L tokens — bars above, pooled z below."""
    w_arr = np.asarray(weights[:L], dtype=float)
    w_arr = w_arr / (w_arr.sum() + 1e-8)
    n_show = min(L, 6)

    pad_x, pad_y = bw * 0.10, bh * 0.14
    ix, iy = bx + pad_x, by + pad_y
    iw, ih = bw - 2 * pad_x, bh - 2 * pad_y

    # Bottom: pooled slide vector
    zh = ih * 0.20
    zw = iw * 0.50
    zy = iy
    zx = ix + (iw - zw) / 2
    ax.add_patch(Rectangle((zx, zy), zw, zh, facecolor="#374151", edgecolor=T["ink"], lw=0.55, zorder=3))
    ax.text(zx + zw / 2, zy + zh / 2, r"$\mathbf{z}$", ha="center", va="center", fontsize=8,
            color="white", zorder=4)

    # Top: token weight bars (clear gap above z)
    bar_floor = zy + zh + ih * 0.22
    bar_ceiling = iy + ih
    bar_max_h = bar_ceiling - bar_floor
    bar_w = min(iw * 0.12, (iw - (n_show - 1) * iw * 0.03) / n_show)
    gap = iw * 0.03
    total_w = n_show * bar_w + (n_show - 1) * gap
    x0 = ix + (iw - total_w) / 2
    for i in range(n_show):
        h = bar_max_h * (0.30 + 0.70 * w_arr[i])
        ax.add_patch(Rectangle(
            (x0 + i * (bar_w + gap), bar_floor), bar_w, h,
            facecolor=T["proto"][i % len(T["proto"])], edgecolor=T["line"], lw=0.45, zorder=3,
        ))


def _draw_classifier(ax, bx, by, bw, bh, num_classes: int):
    """Stage 7: binary CE output."""
    labels = ["Neg", "Pos"] if num_classes == 2 else [f"c{i}" for i in range(num_classes)]
    probs = [0.35, 0.65][:num_classes]
    bar_w = bw * 0.62
    x0 = bx + (bw - bar_w) / 2
    for i, (lab, p) in enumerate(zip(labels, probs)):
        yy = by + bh * 0.22 + i * (bh * 0.28)
        ax.add_patch(Rectangle((x0, yy), bar_w * p, bh * 0.14, facecolor="#374151" if i else "#D1D5DB",
                               edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x0, yy), bar_w, bh * 0.14, fill=False, edgecolor=T["line"], lw=0.5, zorder=3))
        ax.text(x0 - 0.08, yy + bh * 0.07, lab, ha="right", va="center", fontsize=6.5, color=T["ink"], zorder=4)


def _draw_ap_train_branch(ax, L: int, d: int, x: float, y: float, w: float, h: float):
    """Training-only AP — compact horizontal strip."""
    _stage_box(ax, x, y, w, h, fill=T["panel"], edge=T["train_edge"], lw=1.0, dashed=True)

    cx = x + w / 2
    ax.text(cx, y + h - 0.08, "Training only · fold-wise AP",
            ha="center", va="top", fontsize=7.5, fontweight="semibold", color=T["muted"], zorder=4)

    mid_y = y + h * 0.42
    mat_s = min(h * 0.55, w * 0.055)
    mx = x + w * 0.03
    my = mid_y - mat_s / 2
    rng = np.random.default_rng(1)
    sim = rng.uniform(0.2, 1.0, (5, 5))
    sim = (sim + sim.T) / 2
    cell = mat_s / 5
    for i in range(5):
        for j in range(5):
            v = sim[i, j]
            ax.add_patch(Rectangle(
                (mx + j * cell, my + (4 - i) * cell), cell, cell,
                facecolor=(0.75 + 0.2 * v,) * 3, edgecolor="none", zorder=3,
            ))
    ax.add_patch(Rectangle((mx, my), mat_s, mat_s, fill=False, edgecolor=T["line"], lw=0.45, zorder=4))

    tx = mx + mat_s + w * 0.06
    ax.text(tx, mid_y, rf"$\{{\mathbf{{p}}_k\}}^{{{L}}}$", ha="left", va="center",
            fontsize=6.0, color=T["ink"], zorder=4)
    chip_start = tx + 0.55
    chip = min(h * 0.30, (x + w - chip_start - 0.04) / L * 0.90)
    chip_gap = max(0.022, chip * 0.10)
    gy0 = mid_y - chip / 2
    for k in range(L):
        cx_k = chip_start + k * (chip + chip_gap)
        ax.add_patch(Rectangle(
            (cx_k, gy0), chip, chip,
            facecolor=T["proto"][k % len(T["proto"])], edgecolor=T["line"], lw=0.4, zorder=3,
        ))
        ax.text(cx_k + chip / 2, gy0 + chip / 2, f"{k}", ha="center", va="center",
                fontsize=4.5, color=T["ink"], zorder=4)

    return cx, y


def _draw_centered_legend(ax, legend_x: float, legend_y: float, legend_w: float, legend_h: float,
                          route: str):
    """Legend row centred inside its box."""
    ax.add_patch(FancyBboxPatch(
        (legend_x, legend_y), legend_w, legend_h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=T["panel"], edgecolor=T["line"], lw=0.6, zorder=1,
    ))
    ly = legend_y + legend_h / 2
    lcx = legend_x + legend_w / 2
    caption = f"PhenoBIN · {route} routing · token-ABMIL readout"
    ax.text(lcx, ly, caption, ha="center", va="center", fontsize=7, color=T["muted"], zorder=2)

    # Inference / training key centred above caption line within box
    ky = ly + legend_h * 0.22
    key_w = 5.2
    kx = lcx - key_w / 2
    ax.text(kx + 0.55, ky, "Inference", ha="center", va="center", fontsize=6.5,
            fontweight="semibold", color=T["ink"], zorder=2)
    _flow_arrow(ax, kx + 1.05, ky, kx + 1.65, ky, lw=0.75, ms=7)
    ax.text(kx + 2.35, ky, "Training / offline", ha="center", va="center", fontsize=6.5,
            fontweight="semibold", color=T["muted"], zorder=2)
    _flow_arrow(ax, kx + 3.35, ky, kx + 3.95, ky, dashed=True, color=T["train_edge"], lw=0.75, ms=7)


def draw_schematic(
    ax,
    L: int,
    d: int,
    feat_dim: int,
    num_classes: int,
    khead_routing: str,
    phen_token_attn: np.ndarray | None,
):
    """Publication-quality left-to-right PhenoBIN pipeline (panel c), horizontally centered."""

    widths = [2.35, 2.95, 2.15, 2.85, 2.30, 2.50, 2.05]
    gap = 0.48
    h_mod = 2.85
    y_legend_y, y_legend_h = 0.16, 0.50
    y_math = y_legend_y + y_legend_h + 0.22
    y_pipe = y_math + 0.38
    y_top = y_pipe + h_mod
    ap_gap = 1.05
    ap_h = 1.05
    ap_y = y_top + ap_gap

    total_w = sum(widths) + gap * (len(widths) - 1)
    pad_x, pad_y = 0.18, 0.28
    x_start = pad_x
    CANVAS_W = total_w + 2 * pad_x

    stages = [
        ("1", "Patch extraction", r"$N$ tiles"),
        ("2", "Virchow2 encoder", rf"$\mathbf{{H}}\in\mathbb{{R}}^{{N\times {feat_dim}}}$"),
        ("3", "Linear projection", r"$\tilde{\mathbf{h}}_i = \mathrm{LN}(W\mathbf{h}_i)$"),
        ("4", "Hard assignment", r"$\hat{k}(i)=\arg\max_k \cos(\tilde{\mathbf{h}}_i,\mathbf{p}_k)$"),
        ("5", "Gated pool", r"$\mathbf{t}_k=\sum_{i\in\mathcal{P}_k}\alpha_{k,i}\tilde{\mathbf{h}}_i$"),
        ("6", "Token ABMIL", r"$\mathbf{z}=\sum_k \omega_k \mathbf{t}_k$"),
        ("7", "Classification", rf"CE ($C={num_classes}$)"),
    ]

    x = x_start
    centres = []
    boxes = []
    for w_i in widths:
        boxes.append((x, y_pipe, w_i, h_mod))
        centres.append(x + w_i / 2)
        x += w_i + gap

    ap_x = boxes[1][0] - 0.10
    ap_w = boxes[3][0] + boxes[3][2] - ap_x + 0.10
    ap_cx, ap_bot = _draw_ap_train_branch(ax, L, d, ap_x, ap_y, ap_w, ap_h)

    drawers = [
        _draw_patch_extraction,
        lambda a, bx, by, bw, bh: _draw_virchow2_encoder(a, bx, by, bw, bh, feat_dim),
        lambda a, bx, by, bw, bh: _draw_linear_proj(a, bx, by, bw, bh, feat_dim, d),
        lambda a, bx, by, bw, bh: _draw_hard_assign(a, bx, by, bw, bh, L),
        _draw_gated_pool,
        lambda a, bx, by, bw, bh: _draw_token_abmil(
            a, bx, by, bw, bh,
            phen_token_attn if phen_token_attn is not None else np.ones(L) / L,
            L,
        ),
        lambda a, bx, by, bw, bh: _draw_classifier(a, bx, by, bw, bh, num_classes),
    ]
    for (bx, by, bw, bh), draw_fn, (num, title, math) in zip(boxes, drawers, stages):
        _stage_box(ax, bx, by, bw, bh)
        draw_fn(ax, bx, by, bw, bh)
        _stage_label(ax, bx + bw / 2, y_top, num, title, math, y_math=y_math)

    mid = y_pipe + h_mod / 2
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        _flow_arrow(ax, x1 + 0.05, mid, x2 - 0.05, mid)

    # Annotations sit above the stage labels (which occupy ~y_top .. y_top+0.42).
    y_anno = y_top + 0.50
    y_arrow_start = y_top + 0.46

    enc_cx = centres[1]
    _flow_arrow(ax, enc_cx, y_arrow_start, ap_x + ap_w * 0.08, ap_bot,
                dashed=True, color=T["train_edge"], lw=0.9, ms=9)
    ax.text(enc_cx + 0.55, y_anno, r"train-fold $\{\mathbf{h}_i\}_{i=1}^{M}$",
            ha="center", va="bottom", fontsize=6.5, color=T["muted"], zorder=6)

    assign_cx = centres[3]
    _flow_arrow(ax, ap_cx, ap_bot, assign_cx, y_arrow_start,
                color=T["train_edge"], lw=0.9, ms=9)
    ax.text((ap_cx + assign_cx) / 2 + 0.45, y_anno, r"$\mathbf{p}_k$ (frozen)",
            ha="center", va="bottom", fontsize=7, color=T["muted"], zorder=6)

    route = khead_routing.replace("_", " ")
    _draw_centered_legend(ax, x_start, y_legend_y, total_w, y_legend_h, route)

    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, ap_y + ap_h + pad_y)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_facecolor(T["fill"])
    ax.set_anchor("C")

    ax.set_title(
        "(c) PhenoBIN architecture — phenotype discovery and hard-partition MIL readout",
        fontsize=11, fontweight="semibold", pad=12, loc="center",
    )


def render(
    checkpoint: Path,
    features_dir: Path,
    trident_dir: Path,
    wsi_path: Path,
    slide_id: str,
    out_dir: Path,
    max_patches: int,
    top_m: int,
    dpi: int,
    device_name: str,
    preview: bool = False,
) -> dict:
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    device = pick_device(device_name)
    model, cfg, seed = load_model(checkpoint, device)
    feats, coords, attrs, feat_dim = load_bag(features_dir, slide_id, max_patches, seed)
    patch_attn, hard, phen_token_attn, phen_active, dom_k, hard_frac = forward_khead(model, feats, device)
    L = patch_attn.shape[1]
    N = patch_attn.shape[0]
    heads = active_heads(phen_active, hard_frac, phen_token_attn)

    thumb_path = trident_dir / "thumbnails" / f"{slide_id}.jpg"
    thumb = np.array(Image.open(thumb_path).convert("RGB"))
    crop, heat, bbox = attention_heatmap_cropped(
        coords, patch_attn[:, dom_k], attrs, thumb, min_thumb_px=4,
    )
    heat_masked = np.ma.masked_where(heat < 0.05, heat)

    fig = plt.figure(figsize=(16, 18), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, figure=fig, height_ratios=[0.92, 1.48], hspace=0.08,
        left=0.05, right=0.95, top=0.93, bottom=0.05,
    )
    gs_top = gs[0].subgridspec(1, 2, wspace=0.12)

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.imshow(crop)
    im = ax_a.imshow(heat_masked, cmap="inferno", alpha=0.58, vmin=0.05, vmax=1.0, interpolation="nearest")
    ax_a.set_title(
        f"(a) Patch attention map (phenotype P{dom_k})",
        fontsize=11, fontweight="semibold", loc="center", pad=8,
    )
    ax_a.axis("off")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.02, shrink=0.82)
    cbar.set_label("norm. attn.", fontsize=8)

    ax_b = fig.add_subplot(gs_top[0, 1])
    tile_px = 118
    n_rows = len(heads)
    ax_b.set_xlim(0, top_m)
    ax_b.set_ylim(0, n_rows)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.invert_yaxis()
    ax_b.set_xticks([])
    ax_b.set_yticks(np.arange(n_rows) + 0.5)
    ax_b.set_yticklabels([f"P{k}" for k in heads], fontsize=10)
    ax_b.set_title(
        f"(b) Top-{top_m} patches per phenotype",
        fontsize=11, fontweight="semibold", loc="center", pad=8,
    )
    for row, k in enumerate(heads):
        order = rank_patches_per_head(hard, patch_attn, k, top_m)
        for j in range(top_m):
            if j >= len(order):
                continue
            pi = int(order[j])
            tile = read_patch_rgb(wsi_path, int(coords[pi, 0]), int(coords[pi, 1]), attrs, out_px=tile_px)
            ax_b.imshow(tile, extent=(j, j + 1, row, row + 1), aspect="equal")
            ax_b.add_patch(mpatches.Rectangle(
                (j, row), 1, 1, fill=False, edgecolor=PALETTE[k], lw=2.2,
            ))
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    ax_c = fig.add_subplot(gs[1])
    draw_schematic(
        ax_c, L=cfg.num_prototypes, d=cfg.hidden_dim, feat_dim=feat_dim,
        num_classes=cfg.num_classes,
        khead_routing=getattr(cfg, "khead_routing", "hard_partition"),
        phen_token_attn=phen_token_attn,
    )
    # Align panel (c) horizontally with the combined (a)+(b) row above.
    pos_a, pos_b, pos_c = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
    ax_c.set_position([pos_a.x0, pos_c.y0, pos_b.x1 - pos_a.x0, pos_c.height])

    if phen_token_attn is not None:
        omega_str = ", ".join(f"P{k}:{v:.2f}" for k, v in enumerate(phen_token_attn))
        fig.text(0.5, 0.03, f"Slide {slide_id} · $N={N}$ · Token ABMIL: {omega_str}",
                 ha="center", fontsize=8, color="#64748b")

    fig.suptitle(
        "PhenoBIN khead — hard-partition phenotype routing with token ABMIL readout",
        fontsize=13, fontweight="bold", y=0.985,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"arch_khead_slide{slide_id}"
    out_png = out_dir / f"{stem}_preview.png" if preview else out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)

    meta = {
        "slide_id": slide_id,
        "checkpoint": str(checkpoint),
        "wsi_path": str(wsi_path),
        "L": L,
        "N_bag": N,
        "dominant_proto": int(dom_k),
        "hard_frac": hard_frac.round(4).tolist(),
        "khead_pool": getattr(cfg, "khead_pool", None),
        "phen_token_attn": phen_token_attn.tolist() if phen_token_attn is not None else None,
        "outputs": {"combined": str(out_png), "preview": preview},
    }
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def render_schematic_only(
    checkpoint: Path,
    out_dir: Path,
    slide_id: str,
    feat_dim: int,
    dpi: int,
    device_name: str,
) -> dict:
    """Render only the architecture schematic (panel c) as a standalone figure.

    Used for the architecture chapter, where the figure must be purely
    mechanistic (no slide-specific results). Requires no WSI/feature access.
    """
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    device = pick_device(device_name)
    model, cfg, _ = load_model(checkpoint, device)

    fig = plt.figure(figsize=(16, 5.3), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    draw_schematic(
        ax, L=cfg.num_prototypes, d=cfg.hidden_dim, feat_dim=feat_dim,
        num_classes=cfg.num_classes,
        khead_routing=getattr(cfg, "khead_routing", "hard_partition"),
        phen_token_attn=None,
    )
    # Standalone figure: drop the "(c)" panel prefix.
    ax.set_title(
        "PhenoBIN architecture: phenotype discovery and hard-partition MIL readout",
        fontsize=12, fontweight="semibold", pad=12, loc="center",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"arch_khead_slide{slide_id}.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return {
        "mode": "schematic_only",
        "checkpoint": str(checkpoint),
        "L": int(cfg.num_prototypes),
        "outputs": {"schematic": str(out_png)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--slide_id", default="304")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--features_dir", type=Path, default=None)
    ap.add_argument("--trident_job_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=_REPO / "herohe/report/figures")
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--top_m", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=240)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--preview", action="store_true", help="Write *_preview.png only (for iteration)")
    ap.add_argument("--schematic_only", action="store_true",
                    help="Render only panel (c) schematic (no WSI / patch panels)")
    ap.add_argument("--feat_dim", type=int, default=2560,
                    help="Encoder embedding dim shown in the schematic (Virchow2=2560)")
    args = ap.parse_args()

    if args.schematic_only:
        meta = render_schematic_only(
            args.checkpoint, args.out_dir, args.slide_id, args.feat_dim,
            args.dpi, args.device,
        )
        print(json.dumps(meta, indent=2))
        return

    try:
        import openslide  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install openslide-python for RGB patch crops.") from e

    feat_dir = args.features_dir or features_dir(args.split)
    trident = args.trident_job_dir or trident_root(args.split)
    wsi = resolve_wsi(args.slide_id, split=args.split)

    meta = render(
        args.checkpoint, feat_dir, trident, wsi, args.slide_id, args.out_dir,
        args.max_patches, args.top_m, args.dpi, args.device, preview=args.preview,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
