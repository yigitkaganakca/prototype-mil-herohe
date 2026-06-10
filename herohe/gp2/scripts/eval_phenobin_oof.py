"""Stack OOF predictions from a multi-fold PhenoHER2Binary run (binary or Valieris 3-class)."""

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
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import HerohePatchBagDataset, PhenoHER2Binary, PhenoHER2BinaryConfig
from herohe.gp2.models.dataset import collate_single_bag
from herohe.gp2.scripts.metrics_utils import metrics_from_prob_matrix


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    fold_dirs = sorted(run_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* under {run_dir}")

    blob0 = torch.load(fold_dirs[0] / "best.pt", map_location="cpu")
    targs = blob0["args"]
    folds_csv = targs.get("folds_csv")
    if not folds_csv:
        raise ValueError("Checkpoint args missing folds_csv")

    num_classes = int(targs.get("num_classes", 2))
    label_mode = targs.get("label_mode", "gt_binary")
    device = pick_device(args.device)

    full = HerohePatchBagDataset(
        features_dir=targs["features_dir"],
        labels_csv=targs["labels_csv"],
        label_mode=label_mode,
        max_patches=targs.get("max_patches", 4096),
        subsample_mode="deterministic",
        seed=targs.get("seed", 0),
        return_coords=True,
    )
    fdf = pd.read_csv(folds_csv)
    fdf["slide_id"] = fdf["slide_id"].astype(str)
    sid_to_pos = {sid: i for i, sid in enumerate(full.slide_ids())}
    n_folds = int(fdf["fold"].max()) + 1

    prob_cols = [f"p{i}" for i in range(num_classes)]
    rows: list[dict] = []

    for fold_idx in range(n_folds):
        blob = torch.load(run_dir / f"fold_{fold_idx}" / "best.pt", map_location=device)
        raw = blob["config"]
        names = {f.name for f in fields(PhenoHER2BinaryConfig)}
        cfg = PhenoHER2BinaryConfig(**{k: raw[k] for k in names if k in raw})
        model = PhenoHER2Binary(cfg).to(device)
        model.load_state_dict(blob["model_state"], strict=True)
        if "calibration_temperature" in blob:
            model.calibration_temperature.data = torch.tensor(
                [float(blob["calibration_temperature"])], device=model.calibration_temperature.device
            )
        model.eval()

        va_sids = fdf.loc[fdf["fold"] == fold_idx, "slide_id"].tolist()
        va_idx = [sid_to_pos[s] for s in va_sids]
        val_loader = DataLoader(
            Subset(full, va_idx),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_single_bag,
        )

        with torch.no_grad():
            for batch, i in zip(val_loader, va_idx):
                x = batch["features"].to(device)
                c = batch["coords"].to(device) if batch["coords"] is not None else None
                out = model.predict(x, coords=c, apply_calibration=True)
                probs = out["probs"][0].detach().cpu().numpy()
                row = {
                    "slide_id": full.slide_ids()[i],
                    "oof_fold": fold_idx,
                    "y_true": int(batch["label"].item()),
                    "y_pred": int(out["pred_class"].item()),
                }
                for j, col in enumerate(prob_cols):
                    row[col] = float(probs[j])
                rows.append(row)
        print(f"[oof] fold {fold_idx}: {len(va_idx)} val preds")

    if len(rows) != len(full):
        raise RuntimeError(f"OOF row count {len(rows)} != dataset {len(full)}")

    out_csv = run_dir / "oof_predictions.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["slide_id", "oof_fold", "y_true", "y_pred"] + prob_cols,
        )
        writer.writeheader()
        writer.writerows(rows)

    y = np.array([r["y_true"] for r in rows], dtype=np.int64)
    p = np.array([r["y_pred"] for r in rows], dtype=np.int64)
    P = np.stack([[r[c] for c in prob_cols] for r in rows], axis=0)
    oof_metrics = metrics_from_prob_matrix(y, p, P, num_classes)

    out_json = run_dir / "oof_metrics.json"
    with open(out_json, "w") as fh:
        json.dump({"oof_metrics": oof_metrics, "num_classes": num_classes, "label_mode": label_mode}, fh, indent=2)

    print(f"\n[oof] wrote {out_csv}")
    print(f"[oof] wrote {out_json}")
    if num_classes == 2:
        print(
            f"[oof] macro_f1={oof_metrics['macro_f1']:.4f}  "
            f"AUC(pos)={oof_metrics['auc_positive']:.4f}  n={oof_metrics['n']}"
        )
    else:
        print(
            f"[oof] macro_f1={oof_metrics['macro_f1']:.4f}  "
            f"macro_AUROC={oof_metrics['macro_auroc']:.4f}  n={oof_metrics['n']}"
        )


if __name__ == "__main__":
    main()
