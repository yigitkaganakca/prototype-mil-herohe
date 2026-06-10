#!/usr/bin/env python3
"""Evaluate PhenoBIN / PhenoHER2-Binary checkpoints on official HEROHE test (150 slides).

Supports single checkpoint or multi-checkpoint probability averaging (5-fold ensemble).
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
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import HerohePatchBagDataset, PhenoHER2Binary, PhenoHER2BinaryConfig
from herohe.gp2.models.dataset import collate_single_bag
from herohe.gp2.scripts.metrics_utils import metrics_from_prob_matrix


def brier_score(y: np.ndarray, P: np.ndarray, num_classes: int) -> float:
    onehot = np.eye(num_classes)[y]
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def expected_calibration_error(y: np.ndarray, P: np.ndarray, n_bins: int = 15) -> float:
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def labels_csv_path(raw: str) -> str:
    p = Path(raw).expanduser()
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
        out = p.with_suffix(".eval_tmp.csv")
        df.to_csv(out, sep=";", index=False)
        return str(out)
    return str(p)


def load_model(checkpoint: Path, device: torch.device) -> PhenoHER2Binary:
    blob = torch.load(checkpoint, map_location="cpu")
    raw = blob["config"]
    names = {f.name for f in fields(PhenoHER2BinaryConfig)}
    cfg = PhenoHER2BinaryConfig(**{k: raw[k] for k in names if k in raw})
    model = PhenoHER2Binary(cfg)
    model.load_state_dict(blob["model_state"], strict=True)
    T = float(blob.get("calibration_temperature", 1.0))
    if isinstance(T, torch.Tensor):
        T = float(T.item())
    model.calibration_temperature.data = torch.tensor([T])
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def evaluate_checkpoints(
    checkpoints: list[Path],
    features_dir: str,
    labels_csv: str,
    label_mode: str,
    device: torch.device,
    max_patches: int | None,
    out_dir: Path | None,
    tag: str,
    apply_calibration: bool = False,
) -> dict:
    models = [load_model(c, device) for c in checkpoints]
    num_classes = models[0].cfg.num_classes
    if num_classes == 2 and label_mode != "gt_binary":
        raise ValueError(f"2-class checkpoint requires label_mode=gt_binary, got {label_mode}")
    if num_classes == 3 and label_mode != "valieris_3":
        raise ValueError(f"3-class checkpoint requires label_mode=valieris_3, got {label_mode}")

    ds = HerohePatchBagDataset(
        features_dir=features_dir,
        labels_csv=labels_csv,
        label_mode=label_mode,
        max_patches=max_patches,
        subsample_mode="deterministic",
        seed=0,
        return_coords=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag)

    rows = []
    all_y, all_p, all_probs = [], [], []
    for batch in loader:
        x = batch["features"].to(device)
        c = batch["coords"].to(device) if batch["coords"] is not None else None
        sid = batch["slide_id"]
        if isinstance(sid, (list, tuple)):
            sid = sid[0]
        y = int(batch["label"].item())
        fold_probs = []
        for model in models:
            out = model.predict(x, coords=c, apply_calibration=apply_calibration)
            fold_probs.append(out["probs"][0].detach().cpu().numpy())
        probs = np.mean(fold_probs, axis=0)
        pred = int(probs.argmax())
        all_y.append(y)
        all_p.append(pred)
        all_probs.append(probs)
        row = {"slide_id": sid, "label": y, "pred": pred}
        for cidx in range(num_classes):
            row[f"prob_{cidx}"] = float(probs[cidx])
        rows.append(row)

    y = np.array(all_y)
    p = np.array(all_p)
    P = np.stack(all_probs, axis=0)
    metrics = metrics_from_prob_matrix(y, p, P, num_classes)
    rep = classification_report(y, p, labels=list(range(num_classes)), zero_division=0, output_dict=True)
    metrics["checkpoints"] = [str(c) for c in checkpoints]
    metrics["n_models"] = len(checkpoints)
    metrics["dual_stream"] = bool(models[0].cfg.use_dual_stream)
    metrics["num_prototypes"] = int(models[0].cfg.num_prototypes)
    metrics["n_labeled_slides"] = int(len(ds))
    metrics["apply_calibration"] = bool(apply_calibration)
    metrics["max_patches"] = (-1 if max_patches is None else int(max_patches))
    metrics["confusion_matrix"] = confusion_matrix(y, p, labels=list(range(num_classes))).tolist()
    metrics["classification_report"] = rep
    if num_classes == 2:
        metrics["posF1"] = float(rep["1"]["f1-score"])
        metrics["wPRC"] = float(rep["weighted avg"]["precision"])
        metrics["wREC"] = float(rep["weighted avg"]["recall"])
        metrics["wF1"] = float(rep["weighted avg"]["f1-score"])
        metrics["wACC"] = float(rep["accuracy"])
        metrics["bACC"] = float(balanced_accuracy_score(y, p))
        metrics["AUPRC"] = float(average_precision_score(y, P[:, 1]))
        metrics["AUC"] = float(roc_auc_score(y, P[:, 1]))
    metrics["brier"] = brier_score(y, P, num_classes)
    metrics["ECE"] = expected_calibration_error(y, P)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_csv = out_dir / f"predictions_{tag}.csv"
        with open(pred_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        metrics["predictions_csv"] = str(pred_csv)
        json_path = out_dir / f"metrics_{tag}.json"
        json_path.write_text(json.dumps(metrics, indent=2))
        metrics["metrics_json"] = str(json_path)

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument(
        "--labels_csv",
        default=str(_REPO / "herohe" / "Test (ground truth)(1).xlsx"),
    )
    ap.add_argument("--label_mode", choices=["gt_binary", "valieris_3"], required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max_patches", type=int, default=-1,
                    help="cap on patches per slide; <=0 means no cap (full bag, default for test)")
    ap.add_argument("--apply_calibration", action="store_true",
                    help="apply per-fold temperature scaling (default OFF; temperature scaling removed from reported pipeline)")
    ap.add_argument("--tag", default="phenobin")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_REPO / "herohe" / "gp2" / "runs" / "test_eval_phenonly",
    )
    args = ap.parse_args()

    labels_csv = labels_csv_path(args.labels_csv)
    device = pick_device(args.device)
    ckpts = [c.expanduser().resolve() for c in args.checkpoint]
    max_patches = None if args.max_patches is None or args.max_patches <= 0 else int(args.max_patches)
    print(f"\n=== PhenoBIN test ({args.label_mode}, {len(ckpts)} model(s), tag={args.tag}, "
          f"max_patches={max_patches}, calibration={args.apply_calibration}) ===")
    m = evaluate_checkpoints(
        ckpts,
        args.features_dir,
        labels_csv,
        args.label_mode,
        device,
        max_patches,
        args.out_dir,
        args.tag,
        apply_calibration=args.apply_calibration,
    )
    if args.label_mode == "gt_binary":
        print(
            f"n={m['n']}  AUC={m['AUC']:.4f}  posF1={m['posF1']:.4f}  wF1={m['wF1']:.4f}  "
            f"wACC={m['wACC']:.4f}  bACC={m['bACC']:.4f}  dual_stream={m['dual_stream']}"
        )
    else:
        print(
            f"n={m['n']}  macro-AUROC={m.get('macro_auroc', float('nan')):.4f}  "
            f"macro-F1={m['macro_f1']:.4f}  dual_stream={m['dual_stream']}"
        )
    print("confusion_matrix:", m["confusion_matrix"])

    summary = {"label_mode": args.label_mode, "tag": args.tag, "results": [m]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"summary_{args.tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
