#!/usr/bin/env python3
"""Rank HEROHE test slides for interpretability figures (primary model fold-0 ckpt).

Scores patch count, active prototypes, token-weight spread, token diversity,
and hard-assign balance. Writes JSON used by figure scripts.

Usage:
    python herohe/gp2/scripts/rank_interp_slides.py \\
        --scan herohe/gp2/data/khead_hard_partition_ent0_5fold_interp_scan.json \\
        --out herohe/gp2/data/interp_slide_rankings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_SCAN = _REPO / "herohe/gp2/data/khead_hard_partition_ent0_5fold_interp_scan.json"
DEFAULT_OUT = _REPO / "herohe/gp2/data/interp_slide_rankings.json"


def _enrich(v: dict) -> dict:
    hf = v.get("hard_frac") or []
    active_fracs = [f for f in hf if f > 0.01]
    hard_entropy = float(-sum(f * np.log(f + 1e-12) for f in active_fracs)) if active_fracs else 0.0
    w = v.get("phen_token_attn") or []
    story_heads = []
    for i in range(len(hf)):
        if i < len(w) and hf[i] >= 0.03 and w[i] >= 0.08:
            story_heads.append(i)
    return {
        "slide_id": v["slide_id"],
        "n_patches": int(v.get("n_patches", 0)),
        "n_active": int(v.get("n_active_phenotypes", 0)),
        "tok_spread": float(v.get("tok_spread") or 0.0),
        "tok_cos": float(v.get("tok_cos_off_mean", 0.0)),
        "hard_max_frac": float(v.get("hard_max_frac", 1.0)),
        "hard_entropy": hard_entropy,
        "P_pos": float(v.get("P_pos", 0.5)),
        "story_heads": story_heads,
        "n_story_heads": len(story_heads),
    }


def score_row(r: dict) -> float:
    patch_score = np.log1p(r["n_patches"]) / np.log1p(4096)
    dom_penalty = max(0.0, r["hard_max_frac"] - 0.70) * 1.5
    return float(
        0.22 * patch_score
        + 0.18 * (r["n_active"] / 8.0)
        + 0.22 * min(r["tok_spread"] / 0.35, 1.0)
        + 0.18 * min(r["tok_cos"] / 0.45, 1.0)
        + 0.20 * min(r["hard_entropy"] / 1.9, 1.0)
        - dom_penalty
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    data = json.loads(args.scan.read_text())
    rows = [_enrich(v) for v in data["slides"]]
    for r in rows:
        r["score"] = score_row(r)
    rows.sort(key=lambda x: x["score"], reverse=True)

    recommended = rows[0]["slide_id"]
    payload = {
        "checkpoint": data.get("checkpoint"),
        "khead_routing": data.get("khead_routing"),
        "recommended_slide_id": recommended,
        "rationale": (
            "Composite of patch count, active phenotype heads, token-weight spread, "
            "inter-token cosine diversity, and balanced hard-assign entropy."
        ),
        "previous_slide_292": next((r for r in rows if r["slide_id"] == "292"), None),
        "top_slides": rows[: args.top],
        "all_slides": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Recommended interpretability slide: {recommended}")
    print(f"Wrote {args.out}")
    for i, r in enumerate(rows[:5], 1):
        print(
            f"  {i}. slide {r['slide_id']:>3}  patches={r['n_patches']:>4}  "
            f"active={r['n_active']}  tok_sp={r['tok_spread']:.3f}  "
            f"story_heads={r['n_story_heads']}  score={r['score']:.3f}"
        )


if __name__ == "__main__":
    main()
