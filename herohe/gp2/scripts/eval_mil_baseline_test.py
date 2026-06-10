#!/usr/bin/env python3
"""Evaluate ABMIL / CLAM / TransMIL checkpoints on official HEROHE test set.

Supports single checkpoint or multi-checkpoint probability averaging (5-fold ensemble).
Reads ``aggregator`` and ``calibration_temperature`` from each checkpoint blob.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models.dataset import HerohePatchBagDataset, collate_single_bag
from herohe.gp2.prototype_discovery import load_prototype_checkpoint
from herohe.gp2.vendor.factory import build_baseline_model
from herohe.gp2.scripts.metrics_utils import metrics_from_prob_matrix
from herohe.gp2.scripts.mil_calibration import apply_temperature, forward_logits


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


def load_mil_checkpoint(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, str, float, int]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    a = blob.get("args") or {}
    agg = str(blob.get("aggregator") or a.get("aggregator", "abmil")).lower()
    num_classes = int(a.get("num_classes", 2))
    feature_dim = int(a.get("feature_dim", 2560))
    T = blob.get("calibration_temperature", 1.0)
    if isinstance(T, torch.Tensor):
        T = float(T.item())
    else:
        T = float(T)

    if agg == "abmil":
        model = build_baseline_model(
            agg,
            num_classes,
            feature_dim,
            abmil_hidden=int(a.get("abmil_hidden", 512)),
            abmil_attn=int(a.get("abmil_attn", 256)),
            abmil_dropout=float(a.get("abmil_dropout", 0.4)),
        )
    elif agg == "clam":
        model = build_baseline_model(
            agg,
            num_classes,
            feature_dim,
            clam_dropout=float(a.get("clam_dropout", 0.4)),
            clam_k_sample=int(a.get("k_sample", 8)),
        )
    elif agg == "transmil":
        model = build_baseline_model(
            agg,
            num_classes,
            feature_dim,
            trans_d_model=int(a.get("trans_d_model", 512)),
            trans_layers=int(a.get("trans_layers", 2)),
            trans_heads=int(a.get("trans_heads", 8)),
            trans_dropout=float(a.get("trans_dropout", 0.25)),
        )
    elif agg == "attnmisl":
        proto_path = a.get("prototypes")
        proto_centers = None
        if proto_path:
            proto_centers = load_prototype_checkpoint(proto_path)["centers"]
        model = build_baseline_model(
            agg,
            num_classes,
            feature_dim,
            attnmisl_cluster_num=int(a.get("attnmisl_cluster_num", 8)),
            attnmisl_dropout=float(a.get("attnmisl_dropout", 0.5)),
            prototype_centers=proto_centers,
        )
    else:
        raise ValueError(f"Unknown aggregator {agg!r} in {checkpoint}")

    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    model.to(device)
    return model, agg, T, num_classes


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
    loaded = [load_mil_checkpoint(c, device) for c in checkpoints]
    num_classes = loaded[0][3]
    aggregator = loaded[0][1]
    if any(l[1] != aggregator for l in loaded):
        raise ValueError("All checkpoints must share the same aggregator for ensemble eval")
    if num_classes == 2 and label_mode != "gt_binary":
        raise ValueError("2-class checkpoint requires label_mode=gt_binary")
    if num_classes == 3 and label_mode != "valieris_3":
        raise ValueError("3-class checkpoint requires label_mode=valieris_3")

    return_coords = aggregator == "transmil"
    ds = HerohePatchBagDataset(
        features_dir=features_dir,
        labels_csv=labels_csv,
        label_mode=label_mode,
        max_patches=max_patches,
        subsample_mode="deterministic",
        seed=0,
        return_coords=return_coords,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag)

    rows = []
    all_y, all_p, all_probs = [], [], []
    for batch in loader:
        y = int(batch["label"].item())
        sid = batch["slide_id"]
        if isinstance(sid, (list, tuple)):
            sid = sid[0]
        fold_probs = []
        for model, agg, T, _ in loaded:
            logits = forward_logits(model, batch, device, agg)
            if apply_calibration:
                logits = apply_temperature(logits, T)
            fold_probs.append(F.softmax(logits, dim=-1)[0].detach().cpu().numpy())
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
    metrics["aggregator"] = aggregator
    metrics["checkpoints"] = [str(c) for c in checkpoints]
    metrics["n_models"] = len(checkpoints)
    metrics["num_classes"] = num_classes
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
                    help="apply per-fold temperature scaling (default OFF; removed from reported pipeline)")
    ap.add_argument("--tag", default="mil_baseline")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_REPO / "herohe" / "gp2" / "runs" / "test_eval_mil",
    )
    args = ap.parse_args()

    labels_csv = labels_csv_path(args.labels_csv)
    device = pick_device(args.device)
    ckpts = [c.expanduser().resolve() for c in args.checkpoint]
    max_patches = None if args.max_patches is None or args.max_patches <= 0 else int(args.max_patches)
    print(f"\n=== MIL baseline test ({args.label_mode}, {len(ckpts)} model(s), tag={args.tag}, "
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
            f"aggregator={m['aggregator']}  n={m['n']}  AUC={m['AUC']:.4f}  "
            f"posF1={m['posF1']:.4f}  wF1={m['wF1']:.4f}  bACC={m['bACC']:.4f}"
        )
    else:
        print(
            f"aggregator={m['aggregator']}  n={m['n']}  macro-AUROC={m.get('macro_auroc', float('nan')):.4f}  "
            f"macro-F1={m['macro_f1']:.4f}"
        )
    print("confusion_matrix:", m["confusion_matrix"])

    summary = {"label_mode": args.label_mode, "tag": args.tag, "results": [m]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"summary_{args.tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
