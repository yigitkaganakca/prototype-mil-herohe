#!/usr/bin/env python3
"""Compare interpretability metrics: independent vs log_gate on one slide."""

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
    return m, cfg


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
    pt = out["phen_tokens"][0].cpu().numpy()
    logits = out["logits_bin"][0].cpu().numpy()
    prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    sa_mean = sa.mean(0)
    hard = sa.argmax(1)
    hard_frac = np.bincount(hard, minlength=sa.shape[1]) / len(hard)
    pt_n = pt / (np.linalg.norm(pt, axis=1, keepdims=True) + 1e-8)
    gram = pt_n @ pt_n.T
    off = gram[~np.eye(len(pt), dtype=bool)]

    pa_n = pa / (pa.sum(0, keepdims=True) + 1e-12)
    head_ent = (-(pa_n * np.log(pa_n + 1e-12)).sum(0)).tolist()

    return {
        "n_patches": int(x.shape[1]),
        "P_pos": float(prob[1]),
        "khead_routing": getattr(model.cfg, "khead_routing", None),
        "sa_ent": float(-(sa_mean * np.log(sa_mean + 1e-12)).sum()),
        "sa_max": float(sa_mean.max()),
        "hard_max_frac": float(hard_frac.max()),
        "hard_dom": int(hard_frac.argmax()),
        "hard_frac": hard_frac.round(4).tolist(),
        "phen_token_attn": pta.round(4).tolist() if pta is not None else None,
        "tok_ent": float(-(pta * np.log(pta + 1e-12)).sum()) if pta is not None else None,
        "tok_spread": float(pta.max() - pta.min()) if pta is not None else None,
        "tok_cos_off_mean": float(off.mean()),
        "patch_attn_entropy_mean": float(np.mean(head_ent)),
        "patch_attn_entropy_heads": [round(x, 3) for x in head_ent],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_id", default="303")
    ap.add_argument("--split", default="test")
    ap.add_argument("--baseline_ckpt", type=Path,
                    default=_REPO / "herohe/gp2/runs/khead_token_abmil_ls10_ent0/fold_0/best.pt")
    ap.add_argument("--log_gate_ckpt", type=Path,
                    default=_REPO / "herohe/gp2/runs/khead_token_abmil_log_gate_ent0_fold0_probe/fold_0/best.pt")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "herohe/gp2/data/log_gate_ent0_fold0_interp_compare.json")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    dev = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    feat = features_dir(args.split) / f"{args.slide_id}.h5"

    results = {"slide_id": args.slide_id, "split": args.split}
    for name, ckpt in [("independent_ent0", args.baseline_ckpt), ("log_gate_ent0", args.log_gate_ckpt)]:
        model, _ = load_model(ckpt, dev)
        results[name] = {"checkpoint": str(ckpt), **score_slide(model, feat)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
