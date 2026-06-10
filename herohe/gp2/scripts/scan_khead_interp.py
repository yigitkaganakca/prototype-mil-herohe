#!/usr/bin/env python3
"""Cohort interpretability scan for PhenoBIN khead checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig
from herohe.gp2.scripts.compare_hard_partition_interp import score_slide, load_model
from herohe.gp2.scripts.herohe_wsi_paths import features_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    dev = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), dev)
    feat_root = features_dir(args.split)

    rows = []
    for fp in sorted(feat_root.glob("*.h5")):
        m = score_slide(model, fp, max_patches=args.max_patches)
        m["slide_id"] = fp.stem
        m["split"] = args.split
        rows.append(m)

    summary = {
        "checkpoint": args.checkpoint,
        "khead_routing": getattr(model.cfg, "khead_routing", None),
        "split": args.split,
        "n_slides": len(rows),
        "cohort_tok_ent_mean": float(np.nanmean([r["tok_ent"] for r in rows])),
        "cohort_tok_spread_mean": float(np.nanmean([r["tok_spread"] for r in rows if r["tok_spread"] is not None])),
        "cohort_tok_cos_off_mean": float(np.mean([r["tok_cos_off_mean"] for r in rows])),
        "cohort_hard_max_frac_mean": float(np.mean([r["hard_max_frac"] for r in rows])),
        "cohort_n_active_phen_mean": float(np.mean([r["n_active_phenotypes"] for r in rows])),
        "slide303": next((r for r in rows if r["slide_id"] == "303"), None),
        "best_tok_spread": max(rows, key=lambda r: r["tok_spread"] or 0.0) if rows else None,
        "slides": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    brief = {k: v for k, v in summary.items() if k != "slides"}
    print(json.dumps(brief, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
