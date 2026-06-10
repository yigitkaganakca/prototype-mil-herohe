#!/usr/bin/env python3
"""Compare interpretability: independent ent0 vs hard_partition ent0 on one slide."""

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
from herohe.gp2.scripts.herohe_wsi_paths import features_dir


def load_model(ckpt: Path, device: torch.device):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(PhenoHER2BinaryConfig)}
    cfg = PhenoHER2BinaryConfig(**{k: blob["config"][k] for k in names if k in blob["config"]})
    m = PhenoHER2Binary(cfg).eval().to(device)
    m.load_state_dict(blob["model_state"], strict=True)
    return m


@torch.no_grad()
def score_slide(model, feat_path: Path, max_patches: int = 4096):
    with h5py.File(feat_path) as f:
        x = torch.from_numpy(f["features"][:max_patches]).float().unsqueeze(0)
    dev = next(model.parameters()).device
    out = model(x.to(dev))
    sa = out["soft_assign"][0].cpu().numpy()
    pa = out["patch_attn"][0].cpu().numpy()
    pta = out.get("phen_token_attn")
    pta = pta[0].cpu().numpy() if pta is not None else None
    active = out.get("phenotype_active")
    active = active[0].cpu().numpy().astype(bool) if active is not None else np.ones(sa.shape[1], dtype=bool)
    pt = out["phen_tokens"][0].cpu().numpy()
    prob = torch.softmax(out["logits_bin"][0].cpu(), dim=-1).numpy()

    sa_mean = sa.mean(0)
    hard = sa.argmax(1)
    hard_frac = np.bincount(hard, minlength=sa.shape[1]) / len(hard)
    pt_n = pt / (np.linalg.norm(pt, axis=1, keepdims=True) + 1e-8)
    gram = pt_n @ pt_n.T
    off = gram[~np.eye(len(pt), dtype=bool)]

    head_ent = []
    for i in range(pa.shape[1]):
        col = pa[:, i]
        s = col.sum()
        if s <= 1e-12:
            head_ent.append(float("nan"))
            continue
        p = col / s
        nz = p > 0
        head_ent.append(float(-(p[nz] * np.log(p[nz])).sum()))

    tok_ent = None
    tok_spread = None
    if pta is not None:
        nz = active & (pta > 1e-12)
        if nz.any():
            pw = pta[nz]
            pw = pw / pw.sum()
            tok_ent = float(-(pw * np.log(pw + 1e-12)).sum())
            tok_spread = float(pta[nz].max() - pta[nz].min())
        else:
            tok_ent = float("nan")
            tok_spread = float("nan")

    return {
        "n_patches": int(x.shape[1]),
        "P_pos": float(prob[1]),
        "khead_routing": getattr(model.cfg, "khead_routing", None),
        "n_active_phenotypes": int(active.sum()),
        "phenotype_active": active.astype(int).tolist(),
        "sa_ent": float(-(sa_mean * np.log(sa_mean + 1e-12)).sum()),
        "sa_max": float(sa_mean.max()),
        "hard_max_frac": float(hard_frac.max()),
        "hard_dom": int(hard_frac.argmax()),
        "hard_frac": hard_frac.round(4).tolist(),
        "phen_token_attn": pta.round(4).tolist() if pta is not None else None,
        "tok_ent": tok_ent,
        "tok_spread": tok_spread,
        "tok_cos_off_mean": float(off.mean()),
        "patch_attn_entropy_mean": float(np.nanmean(head_ent)),
        "patch_attn_entropy_heads": [round(x, 3) if np.isfinite(x) else None for x in head_ent],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_id", default="303")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--baseline_ckpt",
        type=Path,
        default=_REPO / "herohe/gp2/runs/khead_token_abmil_ls10_ent0/fold_0/best.pt",
    )
    ap.add_argument(
        "--hard_partition_ckpt",
        type=Path,
        default=_REPO / "herohe/gp2/runs/khead_token_abmil_hard_partition_ent0_fold0_probe/fold_0/best.pt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "herohe/gp2/data/hard_partition_ent0_fold0_interp_compare.json",
    )
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    dev = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    feat = features_dir(args.split) / f"{args.slide_id}.h5"

    results = {"slide_id": args.slide_id, "split": args.split}
    for name, ckpt in [
        ("independent_ent0", args.baseline_ckpt),
        ("hard_partition_ent0", args.hard_partition_ckpt),
    ]:
        model = load_model(ckpt, dev)
        results[name] = {"checkpoint": str(ckpt), **score_slide(model, feat)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
