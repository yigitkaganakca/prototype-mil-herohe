"""PANTHER-style pair figure: HER2+ slide 129 | HER2- slide 113.

Per slide (one column):
  (A) H&E tissue crop  (B) Hard assignment map  (C) Within-P_k patch-attention heatmap (readout)
  (D) pi_hard vs omega_k bar chart

Uses medoid hard-partition fold-0 checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.scripts.render_panther_fig3 import (
    PROTO_COLORS,
    load_h5_full,
    load_model,
    pick_device,
    proto_roi_bbox,
    rasterize_assignment_crop,
    read_wsi_crop,
    subsample_bag,
    tissue_bbox_level0,
)

MEDOID_CKPT = _REPO / "herohe/gp2/runs/khead_hard_partition_medoid_proto_control/fold_0/best.pt"
OUT = _REPO / "herohe/report/figures/interp_pair_129_113_composite.png"
POSTER_OUT = _REPO / "herohe/deliverables/figures/poster_interp_129.png"


def rasterize_attn_crop(
    coords: np.ndarray,
    hard: np.ndarray,
    patch_attn_k: np.ndarray,
    head_k: int,
    attrs: dict,
    origin: tuple[int, int],
    scale: float,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Within-head patch attention on crop pixels (0 outside P_k)."""
    x0o, y0o = origin
    ps_px = max(1, int(round(float(attrs["patch_size_level0"]) * scale)))
    th, tw = out_shape
    heat = np.zeros((th, tw), dtype=np.float32)
    for (cx, cy), k, a in zip(coords, hard, patch_attn_k):
        if int(k) != head_k or a <= 0:
            continue
        px = int(round((int(cx) - x0o) * scale))
        py = int(round((int(cy) - y0o) * scale))
        x1, y1 = min(tw, px + ps_px), min(th, py + ps_px)
        if x1 <= px or y1 <= py:
            continue
        heat[py:y1, px:x1] = np.maximum(heat[py:y1, px:x1], float(a))
    if heat.max() > 0:
        heat /= heat.max()
    return heat


@torch.no_grad()
def slide_bundle(model, cfg, seed, slide_id: str, split: str, device, max_patches: int, max_side: int):
    from herohe.gp2.scripts.herohe_wsi_paths import features_dir, resolve_wsi

    fdir = features_dir(split)
    wsi_path = resolve_wsi(slide_id, split=split)
    feats_full, coords_full, attrs = load_h5_full(fdir, slide_id)
    tissue_bbox = tissue_bbox_level0(coords_full, attrs)
    feats, coords = subsample_bag(feats_full, coords_full, max_patches, seed)

    tissue_rgb, scale, origin = read_wsi_crop(wsi_path, tissue_bbox, max_side)
    th, tw = tissue_rgb.shape[:2]

    out = model(torch.from_numpy(feats).float().unsqueeze(0).to(device))
    prob = float(torch.softmax(out["logits_bin"][0].cpu(), dim=-1)[1].item())
    sa = out["soft_assign"][0].float().cpu().numpy()
    hard = sa.argmax(axis=1)
    K = cfg.num_prototypes
    pi_hard = np.bincount(hard, minlength=K).astype(np.float64) / len(hard)
    pta = out.get("phen_token_attn")
    omega = pta[0].float().cpu().numpy() if pta is not None else np.zeros(K)
    patch_attn = out["patch_attn"][0].float().cpu().numpy()

    assign_rgb = rasterize_assignment_crop(coords, hard, attrs, origin, scale, (th, tw))

    dom_k = int(np.argmax(omega))
    proto_bbox = proto_roi_bbox(coords, hard, attrs, dom_k, tissue_bbox)
    proto_rgb, proto_scale, proto_origin = read_wsi_crop(
        wsi_path, proto_bbox, max_side=int(max_side * 0.55),
    )
    pth, ptw = proto_rgb.shape[:2]
    attn_k = patch_attn[:, dom_k].copy()
    attn_k[hard != dom_k] = 0.0
    proto_attn = rasterize_attn_crop(
        coords, hard, attn_k, dom_k, attrs, proto_origin, proto_scale, (pth, ptw),
    )

    return dict(
        slide_id=slide_id,
        tissue_rgb=tissue_rgb,
        assign_rgb=assign_rgb,
        proto_rgb=proto_rgb,
        proto_attn=proto_attn,
        pi_hard=pi_hard,
        omega=omega,
        dom_k=dom_k,
        prob_pos=prob,
        n_patches=len(feats),
    )


def render_poster(bundle, tag, K, out: Path, dpi: int):
    """Wide, short single-slide architecture-trace strip for the A0 poster.

    One row: A | B | C(+colourbar) | D bars. Kept short so it stacks under the
    architecture schematic in poster block 4 without overflowing the page.
    """
    b = bundle
    fig = plt.figure(figsize=(20.0, 5.2), facecolor="white")
    # A | B | C (inset cbar) | D — scaled down so D fits; wspace keeps C/D separated
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[0.88, 0.88, 0.92, 0.98],
        wspace=0.26,
        left=0.04, right=0.96, top=0.84, bottom=0.14,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(b["tissue_rgb"], aspect="auto", interpolation="lanczos")
    ax_a.set_title("(A) H&E tissue", fontsize=13, fontweight="semibold", pad=5)
    ax_a.axis("off")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(b["assign_rgb"], aspect="auto", interpolation="nearest")
    ax_b.set_title("(B) Hard routing", fontsize=13, fontweight="semibold", pad=5)
    ax_b.axis("off")

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(b["proto_rgb"], aspect="auto", interpolation="lanczos")
    attn = np.ma.masked_where(b["proto_attn"] <= 0, b["proto_attn"])
    readout_im = ax_c.imshow(
        attn, cmap="inferno", aspect="auto", alpha=0.78,
        vmin=0.05, vmax=1, interpolation="nearest",
    )
    ax_c.set_title(f"(C) Token P{b['dom_k']} attention", fontsize=13, fontweight="semibold", pad=5)
    ax_c.axis("off")
    cax = make_axes_locatable(ax_c).append_axes("right", size="4%", pad=0.10)
    cbar = fig.colorbar(readout_im, cax=cax, orientation="vertical")
    cbar.set_label("within-$P_k$ attn.", fontsize=9, labelpad=2)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.0, 0.5, 1.0])

    ax_d = fig.add_subplot(gs[0, 3])
    x = np.arange(K)
    bw = 0.40
    ax_d.bar(x - bw / 2, b["pi_hard"], bw, color="#C0504D",
             label=r"$\pi_{\mathrm{hard}}$ routing", edgecolor="white")
    ax_d.bar(x + bw / 2, b["omega"], bw, color="#2E5090",
             label=r"$\omega_k$ readout", edgecolor="white")
    ax_d.axhline(1.0 / K, color="#888888", ls="--", lw=1.1, alpha=0.7,
                 label=rf"uniform $1/{K}$")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"P{k}" for k in range(K)], fontsize=9)
    ax_d.set_ylim(0, max(0.4, float(b["pi_hard"].max()), float(b["omega"].max())) * 1.18)
    ax_d.set_ylabel("fraction / weight", fontsize=10)
    ax_d.set_title("(D) routing vs. token-ABMIL readout", fontsize=12, fontweight="semibold", pad=5)
    ax_d.tick_params(labelsize=9)
    ax_d.grid(axis="y", alpha=0.25, ls="--")
    ax_d.legend(loc="upper left", fontsize=9, frameon=False, ncol=1)

    fig.suptitle(
        f"PhenoBIN in action --- {tag} slide {b['slide_id']} "
        f"($P(\\mathrm{{Pos}})={b['prob_pos']:.2f}$):  hard routing (B) $\\rightarrow$ "
        f"within-cluster attention (C) $\\rightarrow$ token ABMIL (D)",
        fontsize=14, fontweight="bold", y=0.97,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[poster] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=MEDOID_CKPT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--layout", choices=["pair", "poster"], default="pair")
    ap.add_argument("--poster_slide", default="129")
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--max_side", type=int, default=2600)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg, seed = load_model(args.checkpoint, device)
    K = cfg.num_prototypes

    tag_by_id = {"129": "HER2$+$", "113": "HER2$-$"}
    if args.layout == "poster":
        out = args.out or POSTER_OUT
        sid = args.poster_slide
        tag = tag_by_id.get(sid, "")
        b = slide_bundle(model, cfg, seed, sid, "test", device, args.max_patches, args.max_side)
        render_poster(b, tag, K, out, args.dpi)
        return

    args.out = args.out or OUT
    specs = [
        ("129", "test", "HER2$+$"),
        ("113", "test", "HER2$-$"),
    ]
    bundles = [
        slide_bundle(model, cfg, seed, sid, split, device, args.max_patches, args.max_side)
        for sid, split, _ in specs
    ]

    fig = plt.figure(figsize=(19.5, 11.0), facecolor="white")
    # Row 0: headers | row 1: A,B,C + colorbar strip | row 2: P0--P7 legend | row 3: bars
    outer = fig.add_gridspec(
        4, 2,
        height_ratios=[0.05, 1.55, 0.07, 0.48],
        width_ratios=[1.0, 1.0],
        hspace=0.36,
        wspace=0.16,
        left=0.05,
        right=0.97,
        top=0.90,
        bottom=0.22,
    )

    top_axes: list = []
    cbar_axes: list = []
    for c, ((sid, _split, tag), b) in enumerate(zip(specs, bundles)):
        col_hdr = (
            f"{tag}  |  slide {sid}  |  $P(\\mathrm{{Pos}})={b['prob_pos']:.2f}$  |  $N={b['n_patches']}$"
        )
        ax_h = fig.add_subplot(outer[0, c])
        ax_h.axis("off")
        ax_h.text(
            0.5, 0.5, col_hdr,
            ha="center", va="center", fontsize=11, fontweight="bold",
            transform=ax_h.transAxes,
        )

        # A,B,C equal width; narrow 4th column is colorbar only (not part of tissue panels)
        gs_top = outer[1, c].subgridspec(
            1, 4, width_ratios=[1.0, 1.0, 1.0, 0.11], wspace=0.04,
        )

        panels = [
            (b["tissue_rgb"], "lanczos", "(A) H&E"),
            (b["assign_rgb"], "nearest", "(B) Hard assign."),
            (None, None, f"(C) P{b['dom_k']} readout attn."),
        ]
        readout_im = None
        for j, (img, interp, lbl) in enumerate(panels):
            ax = fig.add_subplot(gs_top[0, j])
            if j < 2:
                ax.imshow(img, aspect="auto", interpolation=interp)
            else:
                ax.imshow(b["proto_rgb"], aspect="auto", interpolation="lanczos")
                attn = np.ma.masked_where(b["proto_attn"] <= 0, b["proto_attn"])
                readout_im = ax.imshow(
                    attn, cmap="inferno", aspect="auto", alpha=0.78,
                    vmin=0.05, vmax=1, interpolation="nearest",
                )
            ax.set_title(lbl, fontsize=9.5, fontweight="semibold", pad=4)
            ax.axis("off")
            top_axes.append(ax)

        ax_cb = fig.add_subplot(gs_top[0, 3])
        cbar = fig.colorbar(readout_im, cax=ax_cb, orientation="vertical")
        cbar.set_label("Within-$P_k$ patch attn. (norm.)", fontsize=7.5, labelpad=3)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar_axes.append(ax_cb)

        ax_d = fig.add_subplot(outer[3, c])
        x = np.arange(K)
        bw = 0.34
        b1 = ax_d.bar(
            x - bw / 2, b["pi_hard"], bw,
            color="#C0504D", label=r"$\pi_{\mathrm{hard}}$",
            edgecolor="white",
        )
        b2 = ax_d.bar(
            x + bw / 2, b["omega"], bw,
            color="#2E5090", label=r"$\omega_k$",
            edgecolor="white",
        )
        ax_d.axhline(
            1.0 / K, color="#888888", ls="--", lw=0.9, alpha=0.7,
            label=rf"uniform $1/{K}$",
        )
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([f"P{k}" for k in range(K)], fontsize=9)
        ymax = max(0.4, float(b["pi_hard"].max()), float(b["omega"].max())) * 1.12
        ax_d.set_ylim(0, ymax)
        ax_d.set_title("(D) Routing vs.\\ readout", fontsize=9.5, fontweight="semibold", pad=6)
        ax_d.tick_params(labelsize=8)
        ax_d.grid(axis="y", alpha=0.22, ls="--")

    fig.canvas.draw()
    # Equalize A/B/C vertical extent (and colorbar strip height) across both slides
    y0 = min(ax.get_position().y0 for ax in top_axes)
    y1 = max(ax.get_position().y1 for ax in top_axes)
    for ax in top_axes:
        p = ax.get_position()
        ax.set_position([p.x0, y0, p.width, y1 - y0])
    for ax_cb in cbar_axes:
        p = ax_cb.get_position()
        ax_cb.set_position([p.x0, y0, p.width, y1 - y0])

    # (B) routing palette — shared across slides; colours are categorical, not morphology
    ax_proto_leg = fig.add_subplot(outer[2, :])
    ax_proto_leg.axis("off")
    proto_handles = [
        Patch(facecolor=PROTO_COLORS[k % len(PROTO_COLORS)], edgecolor="0.35", linewidth=0.6, label=f"P{k}")
        for k in range(K)
    ]
    ax_proto_leg.legend(
        handles=proto_handles,
        loc="center",
        ncol=K,
        fontsize=8.5,
        frameon=False,
        title="(B) Hard-routing colours ($\\pi_{\\mathrm{hard}}$: one medoid prototype per patch)",
        title_fontsize=8.5,
        handlelength=1.2,
        handleheight=1.0,
        columnspacing=1.1,
    )

    bar0 = outer[3, 0].get_position(fig)
    bar1 = outer[3, 1].get_position(fig)
    y_bar = 0.5 * (bar0.y0 + bar1.y1)
    fig.text(
        bar0.x0 - 0.022, y_bar, "Fraction / weight",
        ha="right", va="center", rotation=90, fontsize=9,
        transform=fig.transFigure,
    )

    fig.suptitle(
        "Medoid hard-partition interpretability: HER2$+$ (slide 129) vs. HER2$-$ (slide 113)",
        fontsize=12.5, fontweight="bold", y=0.96,
    )
    legend_elems = [
        Patch(
            facecolor="#C0504D", edgecolor="white",
            label=r"$\pi_{\mathrm{hard}}$ (patch routing)",
        ),
        Patch(
            facecolor="#2E5090", edgecolor="white",
            label=r"$\omega_k$ (token readout)",
        ),
        Line2D(
            [0], [0], color="#888888", ls="--", lw=1.5,
            label=rf"uniform $1/{K}$",
        ),
    ]
    fig.legend(
        handles=legend_elems,
        loc="center",
        bbox_to_anchor=(0.12, 0.01, 0.76, 0.05),
        bbox_transform=fig.transFigure,
        ncol=3,
        mode="expand",
        fontsize=9,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=1.0,
        borderaxespad=0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white", pad_inches=0.14)
    plt.close(fig)

    meta = {
        b["slide_id"]: {
            "P_pos": b["prob_pos"],
            "dom_k_token": int(b["dom_k"]),
            "pi_hard": b["pi_hard"].round(3).tolist(),
            "omega": b["omega"].round(3).tolist(),
            "n_patches": b["n_patches"],
        }
        for b in bundles
    }
    meta["checkpoint"] = str(args.checkpoint)
    meta["output"] = str(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
