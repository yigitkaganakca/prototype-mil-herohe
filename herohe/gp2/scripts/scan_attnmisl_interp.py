#!/usr/bin/env python3
"""AttnMISL interpretability: phenotype attention + MI-FCN embedding distinctiveness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.prototype_discovery import load_prototype_checkpoint
from herohe.gp2.scripts.herohe_wsi_paths import features_dir
from herohe.gp2.vendor.adapters.attnmisl import (
    AttnMISLClassifier,
    AttnMISLConfig,
    assign_patches_to_clusters,
    format_cluster_mifcn,
)
from herohe.gp2.vendor.factory import build_baseline_model


def ap_proto_cos(centers: torch.Tensor) -> dict:
    c = F.normalize(centers.float(), dim=-1)
    k = c.shape[0]
    g = (c @ c.T).cpu().numpy()
    mask = ~np.eye(k, dtype=bool)
    v = g[mask]
    return {"off_mean": float(v.mean()), "off_min": float(v.min()), "off_max": float(v.max())}


@torch.no_grad()
def attnmisl_slide_metrics(model: AttnMISLClassifier, x: torch.Tensor) -> dict:
    """Extract hard routing, phenotype attn, MI-FCN token cosines."""
    if x.ndim == 3:
        x = x.squeeze(0)
    proto = model._centers.to(x.device)
    clusters, mask = assign_patches_to_clusters(x, proto)
    k = model.cfg.cluster_num
    hard = torch.cat(
        [torch.full((clusters[i].shape[0],), i, device=x.device, dtype=torch.long) for i in range(k)]
    )
    hard_frac = torch.bincount(hard, minlength=k).float() / max(hard.numel(), 1)

    core = model.core
    embs = []
    for i in range(k):
        hh = format_cluster_mifcn(clusters[i])
        out = core.embedding_net(hh).view(-1)
        embs.append(out)
    h = torch.stack(embs, dim=0)  # K x 64
    h_n = F.normalize(h, dim=-1)
    gram = (h_n @ h_n.T).cpu().numpy()
    off = gram[~np.eye(k, dtype=bool)]

    attn_logits = core.attention(h).transpose(0, 1)  # 1 x K
    phen_w = core.masked_softmax(attn_logits, mask.unsqueeze(0))[0]
    pw = phen_w.cpu().numpy()
    ent = float(-(pw * np.log(pw + 1e-12)).sum())
    ln_k = float(np.log(max(int(mask.sum()), 1)))

    return {
        "n_patches": int(x.shape[0]),
        "hard_frac": hard_frac.cpu().numpy().round(4).tolist(),
        "hard_max_frac": float(hard_frac.max()),
        "hard_dom": int(hard_frac.argmax()),
        "n_active_phenotypes": int(mask.sum()),
        "phenotype_mask": mask.cpu().numpy().astype(int).tolist(),
        "phenotype_attn": pw.round(6).tolist(),
        "phen_ent": ent,
        "phen_ent_max_lnK": ln_k,
        "phen_max": float(pw.max()),
        "phen_spread": float(pw.max() - pw.min()),
        "mifcn_cos_off_mean": float(off.mean()) if off.size else 1.0,
        "mifcn_cos_off_min": float(off.min()) if off.size else 1.0,
    }


def load_attnmisl(ckpt: Path, device: torch.device) -> AttnMISLClassifier:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    cfg = AttnMISLConfig(
        in_dim=int(args.get("feature_dim", 2560)),
        cluster_num=int(args.get("attnmisl_cluster_num", 8)),
        num_classes=int(args.get("num_classes", 2)),
        dropout=float(args.get("attnmisl_dropout", 0.5)),
    )
    proto_path = args.get("prototypes")
    centers = load_prototype_checkpoint(proto_path)["centers"] if proto_path else None
    model = build_baseline_model(
        "attnmisl",
        num_classes=cfg.num_classes,
        feature_dim=cfg.in_dim,
        attnmisl_cluster_num=cfg.cluster_num,
        attnmisl_dropout=cfg.dropout,
        prototype_centers=centers,
    )
    model.load_state_dict(blob["model_state"], strict=True)
    return model.eval().to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--slide_id", default=None)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_attnmisl(Path(args.checkpoint), device)
    ap_cos = ap_proto_cos(model._centers)

    feat_root = features_dir(args.split)
    if args.slide_id:
        slides = [args.slide_id]
    else:
        slides = sorted(p.stem for p in feat_root.glob("*.h5"))

    rows = []
    for sid in slides:
        fp = feat_root / f"{sid}.h5"
        if not fp.is_file():
            continue
        with h5py.File(fp) as f:
            n = min(f["features"].shape[0], args.max_patches)
            x = torch.from_numpy(f["features"][:n]).float()
        m = attnmisl_slide_metrics(model, x.to(device))
        m["slide_id"] = sid
        m["split"] = args.split
        rows.append(m)

    phen_ents = [r["phen_ent"] for r in rows]
    phen_maxs = [r["phen_max"] for r in rows]
    hard_maxs = [r["hard_max_frac"] for r in rows]
    mifcn_cos = [r["mifcn_cos_off_mean"] for r in rows]

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_slides": len(rows),
        "ap_prototype_cos_offdiag_mean": ap_cos["off_mean"],
        "cohort_phen_ent_mean": float(np.mean(phen_ents)),
        "cohort_phen_ent_std": float(np.std(phen_ents)),
        "cohort_phen_max_mean": float(np.mean(phen_maxs)),
        "cohort_hard_max_frac_mean": float(np.mean(hard_maxs)),
        "cohort_mifcn_cos_off_mean": float(np.mean(mifcn_cos)),
        "slide303": next((r for r in rows if r["slide_id"] == "303"), None),
        "best_phen_spread": max(rows, key=lambda r: r["phen_spread"]) if rows else None,
        "slides": rows,
    }

    out = Path(args.out) if args.out else _REPO / "herohe/gp2/data/attnmisl_interp_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "slides"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
