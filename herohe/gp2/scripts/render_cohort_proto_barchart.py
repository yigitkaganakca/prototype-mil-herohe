#!/usr/bin/env python3
"""Academic seaborn re-style of the cohort prototype-label diagnostics figure.

Reads the medoid fold-0 diagnostics JSON and renders two stacked panels:
  (top)    Pearson r between per-slide prototype usage and the HER2+ label
  (bottom) HER2+ minus HER2- mean assignment gap (x 1000)

No morphological labels are assigned to any prototype.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[3]
DIAG = REPO / "herohe/gp2/runs/_medoid_vs_centroid/diag_medoid.json"
OUT = REPO / "herohe/report/figures/interp_cohort_proto_correlation_medoid.png"

POS = "#0D9488"   # teal  (positive)
NEG = "#C0504D"   # brick (negative)
MUTE = 0.38       # alpha for |r| below the hero threshold
HERO_T = 0.20     # |r| threshold to render at full saturation


def _bars(ax, vals, labels, ylabel, title, fmt):
    colors = [POS if v >= 0 else NEG for v in vals]
    alphas = [1.0 if abs(v) >= HERO_T else MUTE for v in vals]
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.8)
    for b, a in zip(bars, alphas):
        b.set_alpha(a)
    ax.axhline(0, color="#444444", linewidth=0.9)
    pad = 0.04 * max(1e-9, max(abs(v) for v in vals))
    for i, v in enumerate(vals):
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v + (pad if v >= 0 else -pad), fmt.format(v),
                ha="center", va=va, fontsize=9,
                fontweight="bold" if abs(v) >= HERO_T else "normal",
                color="#15233A")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=12, loc="left")
    sns.despine(ax=ax)


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    d = json.load(open(DIAG))
    K = int(d["K"])
    labels = [f"P{i}" for i in range(K)]
    r = np.asarray(d["label_correlation_pearson_per_proto"], dtype=float)
    gap = np.asarray(d["usage_pos_minus_neg_per_proto"], dtype=float) * 1000.0

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 6.6), facecolor="white", sharex=True,
        gridspec_kw={"hspace": 0.28},
    )
    _bars(ax0, r, labels,
          r"Pearson $r$ (usage vs HER2$+$)",
          r"(a) Per-prototype usage--label correlation ($n=360$ slides)",
          "{:+.2f}")
    _bars(ax1, gap, labels,
          r"HER2$+$ $-$ HER2$-$ usage ($\times10^{3}$)",
          r"(b) Mean assignment gap between HER2$+$ and HER2$-$ slides",
          "{:+.1f}")
    ax1.set_xlabel("Prototype")

    fig.suptitle(
        "Cohort prototype--label diagnostics (medoid hard-partition, fold-0)",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(top=0.91, left=0.10, right=0.98, bottom=0.09)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}  (offdiag cos={d['prototype_cosine_offdiag_mean']:.3f})")


if __name__ == "__main__":
    main()
