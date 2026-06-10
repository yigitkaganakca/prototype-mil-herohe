#!/usr/bin/env python3
"""Regenerate the three-class confusion grid (Fig 5.1) from the revised-pipeline metrics,
row-normalised to recall with raw counts overlaid."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
REGEN = REPO / "herohe/gp2/runs/report_regen/all_metrics.json"
OUT = REPO / "herohe/gp2/report/report_by_me/figures/results_confusion_3class_grid.png"

SPECS = [
    ("c3_primary_hard", "Our model"),
    ("c3_abmil", "ABMIL"),
    ("c3_clam", "CLAM"),
    ("c3_transmil", "TransMIL"),
]
CLASSES = ["Neg", "Low", "High"]


def main():
    d = json.load(open(REGEN))
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.8), facecolor="white",
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})
    axes = axes.flatten()
    for ax, (key, label) in zip(axes, SPECS):
        cm = np.array(d[key]["confusion_matrix"], dtype=float)
        row = cm.sum(axis=1, keepdims=True)
        rownorm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        im = ax.imshow(rownorm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                txt = f"{rownorm[i, j]*100:.0f}%\n({int(cm[i, j])})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if rownorm[i, j] >= 0.6 else "black")
        ax.set_title(label, fontweight="bold", fontsize=10)
    fig.suptitle("Three-class test confusion matrices ($n=149$): row-normalised recall (raw count)",
                 fontweight="bold", y=1.01)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04)
    cbar.set_label("Row-normalised recall")
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
