"""Publication-style MIL baseline architecture comparison (paper-faithful schematics).

Shows the three comparison baselines only: ABMIL, CLAM, TransMIL. PhenoBIN (ours) is
presented separately in the architecture overview (Figure 4.1) and the aggregation-module
section, so it is intentionally omitted here to keep this figure about the comparators.
Schematics are redrawn from the original papers / the exact vendor code we run
(``herohe/gp2/vendor/adapters``).

Usage:
    python herohe/gp2/scripts/render_arch_baselines_figure.py \\
        --out herohe/report/figures/arch_baselines_mechanism.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

_REPO = Path(__file__).resolve().parents[3]

COL = {
    "input": "#E8EEF7",   # projection / input
    "op": "#FFFFFF",      # plain op
    "attn": "#FFF3E0",    # attention
    "pos": "#F3E8FF",     # positional encoding
    "head": "#FCE8E6",    # final classifier
    "aux": "#EEF7EC",     # auxiliary (training-only) branch
    "edge": "#333333",
}


def _cbox(ax, cy, w, h, text, fc=COL["op"], fs=7.5, bold=False, cx=5.0, ls="-"):
    """Rounded box centred at (cx, cy)."""
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc, edgecolor=COL["edge"], linewidth=1.0, linestyle=ls,
    )
    ax.add_patch(p)
    ax.text(
        cx, cy, text, ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal",
    )


def _arrow(ax, x0, y0, x1, y1, ls="-", color=None):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.0, color=color or COL["edge"], linestyle=ls,
    ))


def _varrow(ax, y0, y1, cx=5.0, ls="-", color=None):
    _arrow(ax, cx, y0, cx, y1, ls=ls, color=color)


def _instances(ax, cx, y0, n=7, label="Patch tokens"):
    total = n * 0.55 - 0.10
    x0 = cx - total / 2
    for i in range(n):
        ax.add_patch(Rectangle(
            (x0 + i * 0.55, y0), 0.45, 0.45,
            facecolor="#D0D0D0", edgecolor="#666", linewidth=0.8,
        ))
    ax.text(cx, y0 + 0.62, label, ha="center", fontsize=7.0, style="italic")


def _panel_tag(ax, tag, title):
    ax.text(0.2, 9.6, tag, ha="left", va="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 9.6, title, ha="center", va="center", fontsize=11, fontweight="bold")


def _frame(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")


def draw_abmil(ax):
    _frame(ax)
    _panel_tag(ax, "(a)", "ABMIL  (Ilse et al., 2018)")
    _instances(ax, 5.0, 8.5, n=7, label=r"$\mathbf{h}_1\ldots\mathbf{h}_N$  (Virchow2)")
    _cbox(ax, 7.55, 7.6, 0.85,
          r"Instance projection  $f_\theta$: Linear $D\!\rightarrow\!M{=}500$, ReLU + dropout",
          COL["input"], fs=7.0)
    _varrow(ax, 8.30, 8.00)
    _cbox(ax, 6.05, 7.8, 1.45,
          "Gated attention (per patch)\n"
          r"$a_i \propto \mathbf{w}^\top[\tanh(V\tilde{\mathbf{h}}_i)\odot\sigma(U\tilde{\mathbf{h}}_i)]$"
          "\n"
          r"$V,U\!:\!500\!\rightarrow\!L{=}128$",
          COL["attn"], fs=7.0)
    _varrow(ax, 7.13, 6.70)
    _cbox(ax, 4.65, 7.2, 0.80, r"Softmax over $N$  $\rightarrow$  weights $\alpha_i$",
          COL["op"], fs=7.0)
    _varrow(ax, 5.40, 5.05)
    _cbox(ax, 3.30, 7.8, 0.85,
          r"Bag vector  $\mathbf{z}=\sum_i \alpha_i \tilde{\mathbf{h}}_i$  (single pool)",
          COL["input"], fs=7.0)
    _varrow(ax, 3.95, 3.55)
    _cbox(ax, 2.05, 6.0, 0.85, r"Classifier  $g(\mathbf{z})\rightarrow C$ classes",
          COL["head"], fs=7.5, bold=True)
    _varrow(ax, 2.70, 2.35)


def draw_clam(ax):
    _frame(ax)
    _panel_tag(ax, "(b)", "CLAM-MB  (Lu et al., 2021)")
    MX, MW = 3.5, 5.2  # main-column centre and box width
    _instances(ax, MX, 8.5, n=6, label="Patch tokens")
    _cbox(ax, 7.55, MW, 0.85,
          "Shared trunk\n"
          r"Linear $D\!\rightarrow\!512$, ReLU + dropout",
          COL["input"], fs=7.0, cx=MX)
    _varrow(ax, 8.30, 8.00, cx=MX)
    _cbox(ax, 6.10, MW, 1.00,
          "Gated attention\n"
          r"per-class scores $A_{c,i}$",
          COL["attn"], fs=7.0, cx=MX)
    _varrow(ax, 7.13, 6.62, cx=MX)
    _cbox(ax, 4.45, MW, 1.05,
          r"Bag pooling  $\mathbf{z}_c=\sum_i A_{c,i}\mathbf{h}_i$"
          "\n"
          r"per-class classifier",
          COL["op"], fs=7.0, cx=MX)
    _varrow(ax, 5.55, 5.00, cx=MX)
    _cbox(ax, 2.55, MW, 0.95,
          r"Slide logits / class $c$  $\rightarrow$ softmax",
          COL["head"], fs=7.0, bold=True, cx=MX)
    _varrow(ax, 3.90, 3.50, cx=MX)
    # auxiliary, training-only instance-clustering branch off the shared attention
    _cbox(ax, 6.10, 3.5, 2.05,
          "Instance clustering\n(aux, train only)\n"
          r"top-/bottom-$k$ patches by $A_{c,i}$"
          "\n"
          r"$\rightarrow$ instance loss",
          COL["aux"], fs=6.5, cx=8.05, ls="--")
    _arrow(ax, MX + MW / 2, 6.10, 6.30, 6.10, ls="--")
    ax.text(8.05, 7.35, "shares attention", ha="center", fontsize=6.0, style="italic", color="#555")


def draw_transmil(ax):
    _frame(ax)
    _panel_tag(ax, "(c)", "TransMIL  (Shao et al., 2021)")
    _instances(ax, 5.0, 8.5, n=8, label="Unordered patch bag")
    _cbox(ax, 7.75, 7.6, 0.70, r"Linear projection  $D\!\rightarrow\!512$ (ReLU)",
          COL["input"], fs=7.0)
    _varrow(ax, 8.30, 8.10)
    _cbox(ax, 6.55, 7.8, 0.80,
          r"Pad to $\sqrt{N}\!\times\!\sqrt{N}$ grid  +  prepend [CLS]",
          COL["op"], fs=7.0)
    _varrow(ax, 7.40, 6.95)
    _cbox(ax, 5.40, 7.6, 0.70, "TransLayer 1  (Nystr\u00f6m self-attention)",
          COL["attn"], fs=7.0)
    _varrow(ax, 6.20, 5.75)
    _cbox(ax, 4.25, 7.8, 0.80,
          "PPEG positional encoding\n(depthwise conv on 2-D grid)",
          COL["pos"], fs=7.0)
    _varrow(ax, 5.05, 4.65)
    _cbox(ax, 3.05, 7.6, 0.80,
          "TransLayer 2  $\\rightarrow$  LayerNorm, take [CLS]",
          COL["attn"], fs=7.0)
    _varrow(ax, 3.85, 3.45)
    _cbox(ax, 1.95, 6.0, 0.80, r"Linear head  $\rightarrow C$ classes",
          COL["head"], fs=7.5, bold=True)
    _varrow(ax, 2.65, 2.30)


def render(out: Path, dpi: int = 200):
    fig = plt.figure(figsize=(15, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.06, left=0.01, right=0.99, top=0.86, bottom=0.04)
    draw_abmil(fig.add_subplot(gs[0, 0]))
    draw_clam(fig.add_subplot(gs[0, 1]))
    draw_transmil(fig.add_subplot(gs[0, 2]))
    fig.suptitle(
        "Comparison MIL baseline architectures\n"
        "(shared Virchow2 bags, identical 5-fold splits, val-loss checkpointing; $D{=}2560$)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "herohe/report/figures/arch_baselines_mechanism.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    render(args.out, args.dpi)


if __name__ == "__main__":
    main()
