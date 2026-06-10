"""Quantitative checks: are Pheno prototypes *used* and *specialised*, not path semantics.

Runs a trained checkpoint over many slides (Virchow2 .h5 only) and reports:

  - Mean / std of per-prototype patch assignment mass (soft_assign averaged over patches).
  - Cosine similarity matrix of learned prototype vectors P (post-training).
  - Optional: correlation (Pearson) between each prototype's mean assignment and slide label.
  - Optional: top patch indices per prototype on one slide (for later RGB cropping if you have WSIs).

This does **not** prove "HER2 phenotype" names; it tests whether the mechanism is non-collapsed
and whether any prototype tracks label signal enough to motivate qualitative figures.

Example (binary run):

    python herohe/gp2/scripts/prototype_diagnostics.py \\
        --checkpoint herohe/gp2/runs/phenobin_5fold_parallel/fold_0/best.pt \\
        --features_dir herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2 \\
        --labels_csv "herohe/Training (ground truth).csv" \\
        --label_mode gt_binary \\
        --max_slides 120 \\
        --device mps

Example (4-class Pheno):

    python herohe/gp2/scripts/prototype_diagnostics.py \\
        --checkpoint herohe/gp2/runs/some_pheno_run/fold_0/best.pt \\
        --features_dir .../features_virchow2 \\
        --labels_csv "herohe/Training (ground truth).csv" \\
        --label_mode ihc \\
        --max_slides 120 \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import (
    HerohePatchBagDataset,
    PhenoHER2,
    PhenoHER2Binary,
    PhenoHER2BinaryConfig,
    PhenoHER2Config,
)
from herohe.gp2.models.dataset import collate_single_bag


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_binary_ckpt(blob: dict, sd: dict) -> bool:
    if blob.get("model_type") == "PhenoHER2Binary":
        return True
    return any(k.startswith("head_bin_phen") for k in sd.keys())


def _cfg_from_blob(blob: dict, binary: bool):
    raw = blob["config"]
    if binary:
        names = {f.name for f in fields(PhenoHER2BinaryConfig)}
        return PhenoHER2BinaryConfig(**{k: raw[k] for k in names if k in raw})
    names = {f.name for f in fields(PhenoHER2Config)}
    return PhenoHER2Config(**{k: raw[k] for k in names if k in raw})


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--label_mode", choices=["ihc", "gt_binary"], required=True)
    ap.add_argument("--max_slides", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument(
        "--dump_top",
        type=int,
        default=0,
        help="If >0, also print top patch indices per prototype for the first slide only (first N).",
    )
    args = ap.parse_args()

    device = pick_device(args.device)
    blob = torch.load(args.checkpoint, map_location="cpu")
    sd = blob["model_state"]
    binary = _is_binary_ckpt(blob, sd)
    cfg = _cfg_from_blob(blob, binary)
    if binary:
        model = PhenoHER2Binary(cfg)
    else:
        model = PhenoHER2(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)

    K = cfg.num_prototypes
    P = model.prototypes.detach().float()
    Pn = F.normalize(P, dim=-1)
    gram = (Pn @ Pn.t()).cpu().numpy()
    off_diag = gram[~np.eye(K, dtype=bool)]

    ds = HerohePatchBagDataset(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        label_mode=args.label_mode,
        max_patches=args.max_patches,
        seed=0,
        return_coords=True,
    )
    n = min(len(ds), args.max_slides)
    loader = DataLoader(
        Subset(ds, list(range(n))),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single_bag,
    )

    usages: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    phen_token_w: list[np.ndarray] = []
    patch_entropies: list[np.ndarray] = []
    labels: list[int] = []
    proto_class_pref: list[torch.Tensor] = []

    first_batch = None
    for batch in loader:
        if first_batch is None:
            first_batch = batch
        x = batch["features"].to(device)
        c = batch["coords"].to(device) if batch["coords"] is not None else None
        out = model(x, coords=c)
        sa = out["soft_assign"][0].float()
        pa = out["patch_attn"][0].float()
        usages.append(sa.mean(dim=0).cpu().numpy())
        masses.append((sa * pa).sum(dim=0).cpu().numpy())
        pa_norm = pa / pa.sum(dim=0, keepdim=True).clamp_min(1e-12)
        patch_entropies.append((-(pa_norm * pa_norm.log()).sum(dim=0)).cpu().numpy())
        pta = out.get("phen_token_attn")
        if pta is not None:
            phen_token_w.append(pta[0].float().cpu().numpy())
        y = int(batch["label"].item())
        labels.append(y)
        pcl = out.get("proto_class_logits")
        if pcl is not None:
            proto_class_pref.append(F.softmax(pcl[0], dim=-1).cpu())
        elif out.get("phen_tokens") is not None and getattr(model, "head_proto", None) is not None:
            proto_class_pref.append(
                F.softmax(model.head_proto(out["phen_tokens"][0]), dim=-1).cpu()
            )
        else:
            proto_class_pref.append(torch.full((K, cfg.num_classes), float("nan")))

    U = np.stack(usages, axis=0)
    M = np.stack(masses, axis=0)
    yv = np.array(labels, dtype=np.float64)
    pc_mean = torch.stack(proto_class_pref, dim=0).mean(dim=0).numpy()

    # Entropy of mean assignment distribution per slide (high => spread across K)
    ent = -(U * np.log(U + 1e-12)).sum(axis=-1)

    # Pearson corr per prototype vs label
    corrs = []
    for k in range(K):
        xk = U[:, k]
        if np.std(xk) < 1e-8 or np.std(yv) < 1e-8:
            corrs.append(float("nan"))
        else:
            corrs.append(float(np.corrcoef(xk, yv)[0, 1]))

    neg_mask = yv < 0.5
    pos_mask = yv >= 0.5
    usage_neg_mean = U[neg_mask].mean(axis=0).tolist() if neg_mask.any() else [float("nan")] * K
    usage_pos_mean = U[pos_mask].mean(axis=0).tolist() if pos_mask.any() else [float("nan")] * K
    usage_pos_minus_neg = [
        float(p - n) if np.isfinite(p) and np.isfinite(n) else float("nan")
        for p, n in zip(usage_pos_mean, usage_neg_mean)
    ]

    report = {
        "checkpoint": str(args.checkpoint),
        "binary_head": binary,
        "khead_pool": getattr(cfg, "khead_pool", None),
        "slides_used": n,
        "K": K,
        "prototype_cosine_offdiag_mean": float(off_diag.mean()),
        "prototype_cosine_offdiag_std": float(off_diag.std()),
        "usage_per_proto_mean": U.mean(axis=0).tolist(),
        "usage_per_proto_std": U.std(axis=0).tolist(),
        "assign_attn_mass_per_proto_mean": M.mean(axis=0).tolist(),
        "slide_usage_entropy_mean": float(ent.mean()),
        "slide_usage_entropy_std": float(ent.std()),
        "proto_class_softmax_mean": pc_mean.tolist(),
        "label_correlation_pearson_per_proto": corrs,
        "usage_neg_mean_per_proto": usage_neg_mean,
        "usage_pos_mean_per_proto": usage_pos_mean,
        "usage_pos_minus_neg_per_proto": usage_pos_minus_neg,
        "n_neg_slides": int(neg_mask.sum()),
        "n_pos_slides": int(pos_mask.sum()),
    }

    if phen_token_w:
        W = np.stack(phen_token_w, axis=0)
        w_mean = W.mean(axis=0)
        w_std = W.std(axis=0)
        w_ent = -(W * np.log(W + 1e-12)).sum(axis=-1)
        report["phen_token_attn_mean"] = w_mean.tolist()
        report["phen_token_attn_std"] = w_std.tolist()
        report["phen_token_attn_entropy_mean"] = float(w_ent.mean())
        report["phen_token_attn_entropy_std"] = float(w_ent.std())
        report["phen_token_attn_max_proto"] = int(w_mean.argmax())
        report["phen_token_attn_min_proto"] = int(w_mean.argmin())

    pe = np.stack(patch_entropies, axis=0)
    report["patch_attn_entropy_per_proto_mean"] = pe.mean(axis=0).tolist()
    report["patch_attn_entropy_per_proto_std"] = pe.std(axis=0).tolist()

    if args.dump_top > 0 and first_batch is not None:
        x = first_batch["features"].to(device)
        c = first_batch["coords"].to(device) if first_batch["coords"] is not None else None
        out = model(x, coords=c)
        sa = out["soft_assign"][0]
        pa = out["patch_attn"][0]
        score = (sa * pa).cpu().numpy()
        N = score.shape[0]
        topn = min(args.dump_top, N)
        top_idx = {}
        for k in range(K):
            idx = np.argsort(-score[:, k])[:topn]
            top_idx[str(k)] = idx.tolist()
        report["first_slide_id"] = first_batch["slide_id"]
        report["top_patch_indices_first_slide"] = top_idx

    print(json.dumps(report, indent=2))

    print(
        "\n# How to read (short):\n"
        "- usage_*: if one prototype dominates all slides, others are dead.\n"
        "- prototype_cosine_offdiag_*: very high => redundant prototypes; very negative is rare (orth loss pushes apart).\n"
        "- slide_usage_entropy_*: higher => patches spread across K on average.\n"
        "- proto_class_softmax_mean: heuristic class preference per prototype (linear head, not ground truth).\n"
        "- label_correlation_*: exploratory only; weak is normal; strong suggests a thesis figure candidate.\n"
        "- usage_neg/pos_mean_*: PhiHER2-style — which prototypes get more patch mass on HER2+ vs Neg slides.\n"
        "- usage_pos_minus_neg_*: positive delta => prototype attends more on HER2+ slides (exploratory).\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
