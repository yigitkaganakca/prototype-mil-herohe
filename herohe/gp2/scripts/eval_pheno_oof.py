"""Stack out-of-fold (OOF) predictions from a multi-fold PhenoHER2 v2 run.

Uses each fold's ``best.pt`` (post temperature calibration, as saved by
``train_phenotype_mil.py``) and runs ``model.predict`` on that fold's validation
slides only, then concatenates to 360 rows.

Example:

    python herohe/gp2/scripts/eval_pheno_oof.py \\
        --run_dir herohe/gp2/runs/phenoher2_v2 \\
        --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import HerohePatchBagDataset, PhenoHER2, PhenoHER2Config
from herohe.gp2.models.dataset import collate_single_bag


def pick_device(name: str) -> torch.device:
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


def metrics_from_probs(y: np.ndarray, p: np.ndarray, P: np.ndarray, num_classes: int = 4) -> dict:
    P = np.asarray(P, dtype=np.float64).copy()
    p = np.asarray(p, dtype=np.int64).copy()
    if not np.isfinite(P).all():
        bad = ~np.isfinite(P).all(axis=-1)
        P[bad] = 1.0 / num_classes
        p = np.where(bad, P.argmax(axis=-1), p)
    macro_f1 = f1_score(y, p, average="macro", labels=list(range(num_classes)), zero_division=0)
    mask01 = (y == 0) | (y == 1)
    auc01 = float("nan")
    if mask01.sum() >= 2 and len(np.unique(y[mask01])) == 2:
        denom = np.clip(P[mask01, 0] + P[mask01, 1], 1e-6, None)
        score_low = P[mask01, 1] / denom
        if np.isfinite(score_low).all():
            try:
                auc01 = float(roc_auc_score((y[mask01] == 1).astype(int), score_low))
            except ValueError:
                pass
    auc_3p = float("nan")
    if len(np.unique(y)) > 1 and np.isfinite(P[:, 3]).all():
        try:
            auc_3p = float(roc_auc_score((y == 3).astype(int), P[:, 3]))
        except ValueError:
            pass
    ce = float(
        np.mean([-np.log(np.clip(P[i, y[i]], 1e-12, 1.0)) for i in range(len(y))])
    )
    return {
        "macro_f1": float(macro_f1),
        "auc_0_vs_low": float(auc01),
        "auc_3p_vs_rest": float(auc_3p),
        "mean_log_loss": ce,
        "n": int(len(y)),
    }


def config_from_blob(blob: dict) -> PhenoHER2Config:
    raw = blob["config"]
    names = {f.name for f in fields(PhenoHER2Config)}
    return PhenoHER2Config(**{k: raw[k] for k in names if k in raw})


@torch.no_grad()
def collect_fold_predictions(
    model: PhenoHER2,
    val_loader: DataLoader,
    device: torch.device,
    fold_idx: int,
    slide_ids: list[str],
    val_idx: np.ndarray,
    apply_calibration: bool,
) -> list[dict]:
    model.eval()
    rows = []
    for batch, i in zip(val_loader, val_idx.tolist()):
        sid = slide_ids[i]
        y = int(batch["label"].item())
        x = batch["features"].to(device)
        c = batch["coords"].to(device) if batch["coords"] is not None else None
        out = model.predict(x, coords=c, apply_calibration=apply_calibration)
        probs = out["probs"][0].detach().cpu().numpy()
        pred = int(out["pred_class"].item())
        rows.append(
            {
                "slide_id": sid,
                "oof_fold": fold_idx,
                "y_true": y,
                "y_pred": pred,
                "p0": float(probs[0]),
                "p1": float(probs[1]),
                "p2": float(probs[2]),
                "p3": float(probs[3]),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument(
        "--apply_calibration",
        type=int,
        default=1,
        help="1 = use temperature from checkpoint in predict() (default); 0 = raw logits",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    use_calib = bool(args.apply_calibration)
    device = pick_device(args.device)

    fold_dirs = sorted(run_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* under {run_dir}")

    blob0 = torch.load(fold_dirs[0] / "best.pt", map_location="cpu")
    if "config" not in blob0:
        raise KeyError(
            f"{fold_dirs[0]}/best.pt has no 'config'; need a PhenoHER2 v2 checkpoint from train_phenotype_mil.py"
        )
    targs = blob0["args"]

    full = HerohePatchBagDataset(
        features_dir=targs["features_dir"],
        labels_csv=targs["labels_csv"],
        max_patches=targs.get("max_patches", 4096),
        seed=targs.get("seed", 0),
        return_coords=True,
    )
    folds_csv = targs.get("folds_csv")
    if not folds_csv:
        raise ValueError("Checkpoint args missing folds_csv; cannot build OOF splits.")

    fdf = pd.read_csv(folds_csv)
    fdf["slide_id"] = fdf["slide_id"].astype(str)
    sid_to_pos = {sid: i for i, sid in enumerate(full.slide_ids())}
    n_folds = int(fdf["fold"].max()) + 1
    splits = []
    for k in range(n_folds):
        va_sids = fdf.loc[fdf["fold"] == k, "slide_id"].tolist()
        va_idx = np.array([sid_to_pos[s] for s in va_sids], dtype=int)
        splits.append(va_idx)

    slide_ids = full.slide_ids()
    all_rows: list[dict] = []

    for fold_idx, va_idx in enumerate(splits):
        ckpt = run_dir / f"fold_{fold_idx}" / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing {ckpt}")
        blob = torch.load(ckpt, map_location=device)
        cfg = config_from_blob(blob)
        model = PhenoHER2(cfg).to(device)
        model.load_state_dict(blob["model_state"], strict=True)

        val_ds = Subset(full, va_idx.tolist())
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag)
        rows = collect_fold_predictions(
            model, val_loader, device, fold_idx, slide_ids, va_idx, use_calib
        )
        all_rows.extend(rows)
        ep = blob.get("metrics", {}).get("epoch", "?")
        print(f"[oof] fold {fold_idx}: {len(rows)} val preds (best epoch {ep}, calib={use_calib})")

    if len(all_rows) != len(full):
        raise RuntimeError(f"OOF row count {len(all_rows)} != dataset {len(full)}")

    all_rows.sort(key=lambda r: r["slide_id"])
    out_csv = run_dir / "oof_predictions.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["slide_id", "oof_fold", "y_true", "y_pred", "p0", "p1", "p2", "p3"],
        )
        w.writeheader()
        w.writerows(all_rows)

    y = np.array([r["y_true"] for r in all_rows], dtype=np.int64)
    p = np.array([r["y_pred"] for r in all_rows], dtype=np.int64)
    P = np.array([[r["p0"], r["p1"], r["p2"], r["p3"]] for r in all_rows], dtype=np.float64)
    oof_metrics = metrics_from_probs(y, p, P, num_classes=4)

    out_json = run_dir / "oof_metrics.json"
    payload = {
        "oof_metrics": oof_metrics,
        "run_dir": str(run_dir),
        "n_slides": len(all_rows),
        "apply_calibration": use_calib,
        "source_args": {k: targs[k] for k in sorted(targs.keys())},
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n[oof] wrote {out_csv}")
    print(f"[oof] wrote {out_json}")
    print(
        f"[oof] macro_f1={oof_metrics['macro_f1']:.4f}  "
        f"AUC(0vsLow)={oof_metrics['auc_0_vs_low']:.4f}  "
        f"AUC(3+vsRest)={oof_metrics['auc_3p_vs_rest']:.4f}  "
        f"mean_CE_nats={oof_metrics['mean_log_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
