"""Audit fusion gate and phen vs detail stream AUC from saved PhenoBIN checkpoints.

No retraining required. Loads fold_*/best.pt and evaluates each fold's val slides.

Example:
    python herohe/gp2/scripts/fusion_gate_audit.py \\
        --run_dir herohe/gp2/runs/phenobin_5fold_v2 \\
        --folds_csv herohe/gp2/data/folds_v1.csv \\
        --label_mode gt_binary
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import HerohePatchBagDataset, PhenoHER2Binary, PhenoHER2BinaryConfig, collate_single_bag


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def macro_auroc(y: np.ndarray, P: np.ndarray, num_classes: int) -> float:
    aucs = []
    for c in range(num_classes):
        yb = (y == c).astype(int)
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, P[:, c])))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def eval_fold(
    model: PhenoHER2Binary,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    model.eval()
    gates, y_true = [], []
    lp, ld, lf = [], [], []
    cal = model.calibration_temperature.clamp_min(1e-3)
    has_dual = model.fusion_gate is not None

    for batch in loader:
        x = batch["features"].to(device)
        c = batch["coords"].to(device) if batch["coords"] is not None else None
        out = model(x, coords=c)
        if has_dual:
            gates.append(float(out["fusion_gate"].item()))
        y_true.append(int(batch["label"].item()))
        lp.append(F.softmax(out["logits_phen"] / cal, -1).cpu().numpy()[0])
        if has_dual:
            ld.append(F.softmax(out["logits_detail"] / cal, -1).cpu().numpy()[0])
        lf.append(F.softmax(out["logits_bin"] / cal, -1).cpu().numpy()[0])

    y = np.array(y_true)
    lp = np.stack(lp)
    ld_arr = np.stack(ld) if ld else None
    lf = np.stack(lf)
    gates_arr = np.array(gates) if gates else None

    if num_classes == 2:
        score_phen = lp[:, 1]
        score_detail = ld_arr[:, 1] if ld_arr is not None else None
        score_fused = lf[:, 1]
        pred_fused = lf.argmax(axis=-1)
    else:
        score_phen = macro_auroc(y, lp, num_classes)
        score_detail = macro_auroc(y, ld_arr, num_classes) if ld_arr is not None else None
        score_fused = macro_auroc(y, lf, num_classes)
        pred_fused = lf.argmax(axis=-1)

    out_metrics = {
        "n": int(len(y)),
        "gate_mean": float(gates_arr.mean()) if gates_arr is not None else None,
        "gate_median": float(np.median(gates_arr)) if gates_arr is not None else None,
        "gate_frac_gt_0.5": float((gates_arr > 0.5).mean()) if gates_arr is not None else None,
        "auc_or_macro_auroc_phen": float(score_phen) if num_classes == 2 else score_phen,
        "auc_or_macro_auroc_fused": float(score_fused),
        "macro_f1_fused": float(
            f1_score(y, pred_fused, average="macro", labels=list(range(num_classes)), zero_division=0)
        ),
    }
    if score_detail is not None:
        key = "auc_or_macro_auroc_detail"
        out_metrics[key] = float(score_detail) if num_classes == 2 else score_detail
    return out_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--features_dir", type=Path, required=True)
    ap.add_argument("--labels_csv", type=Path, required=True)
    ap.add_argument("--folds_csv", type=Path, required=True)
    ap.add_argument("--label_mode", choices=["gt_binary", "valieris_3", "ihc"], required=True)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    num_classes = 2 if args.label_mode == "gt_binary" else 3 if args.label_mode == "valieris_3" else 4
    device = pick_device(args.device)
    cfg_names = {f.name for f in fields(PhenoHER2BinaryConfig)}

    fdf = pd.read_csv(args.folds_csv)
    fdf["slide_id"] = fdf["slide_id"].astype(str)
    ds = HerohePatchBagDataset(
        str(args.features_dir),
        str(args.labels_csv),
        label_mode=args.label_mode,
        max_patches=args.max_patches,
        subsample_mode="deterministic",
        seed=0,
        return_coords=True,
    )
    sid_to_i = {s: i for i, s in enumerate(ds.slide_ids())}

    fold_results = []
    for fold in range(int(fdf["fold"].max()) + 1):
        ckpt = args.run_dir / f"fold_{fold}" / "best.pt"
        if not ckpt.is_file():
            print(f"[skip] fold {fold}: no {ckpt}")
            continue
        blob = torch.load(ckpt, map_location="cpu")
        cfg = PhenoHER2BinaryConfig(**{k: blob["config"][k] for k in cfg_names if k in blob["config"]})
        model = PhenoHER2Binary(cfg)
        model.load_state_dict(blob["model_state"], strict=True)
        model.to(device)

        va = [sid_to_i[s] for s in fdf.loc[fdf.fold == fold, "slide_id"] if s in sid_to_i]
        loader = DataLoader(Subset(ds, va), batch_size=1, shuffle=False, collate_fn=collate_single_bag)
        m = eval_fold(model, loader, num_classes, device)
        m["fold"] = fold
        m["dual_stream"] = bool(cfg.use_dual_stream)
        fold_results.append(m)
        gm = m.get("gate_mean")
        gm_s = f"{gm:.4f}" if gm is not None else "n/a (phen-only)"
        print(
            f"fold {fold}: gate_mean={gm_s}  "
            f"phen={m['auc_or_macro_auroc_phen']:.4f}  "
            f"detail={m.get('auc_or_macro_auroc_detail', float('nan')):.4f}  "
            f"fused={m['auc_or_macro_auroc_fused']:.4f}"
        )

    summary = {
        "run_dir": str(args.run_dir),
        "label_mode": args.label_mode,
        "num_classes": num_classes,
        "folds": fold_results,
    }
    out_path = args.out_json or (args.run_dir / "fusion_gate_audit.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[fusion_gate_audit] wrote {out_path}")


if __name__ == "__main__":
    main()
