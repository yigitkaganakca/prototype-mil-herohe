#!/usr/bin/env python3
"""Evaluate AB-MIL checkpoints on official HEROHE test set (Virchow2 features).

Supports single checkpoint or multi-checkpoint probability averaging (5-fold ensemble).

Example (binary 5-fold ensemble):

    python herohe/gp2/scripts/eval_abmil_test.py \\
        --checkpoint herohe/gp2/runs/abmil_binary_virchow2_5fold/fold_{0..4}/best.pt \\
        --features_dir herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2 \\
        --labels_csv "herohe/Test (ground truth)(1).xlsx" \\
        --label_mode gt_binary \\
        --device mps
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
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models.abmil import ABMIL, ABMILConfig
from herohe.gp2.models.dataset import HerohePatchBagDataset, collate_single_bag
from herohe.gp2.scripts.metrics_utils import metrics_from_prob_matrix


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


def load_abmil(checkpoint: Path, device: torch.device) -> ABMIL:
    blob = torch.load(checkpoint, map_location="cpu")
    a = blob.get("args") or {}
    cfg = ABMILConfig(
        in_dim=int(a.get("feature_dim", 2560)),
        hidden_dim=int(a.get("abmil_hidden", 512)),
        attn_dim=int(a.get("abmil_attn", 256)),
        num_classes=int(a.get("num_classes", 2)),
        dropout=float(a.get("abmil_dropout", 0.25)),
    )
    model = ABMIL(cfg)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def evaluate(
    checkpoints: list[Path],
    features_dir: str,
    labels_csv: str,
    label_mode: str,
    device: torch.device,
    max_patches: int,
    out_dir: Path | None,
    tag: str,
) -> dict:
    models = [load_abmil(ck, device) for ck in checkpoints]
    num_classes = models[0].cfg.num_classes
    if num_classes == 2 and label_mode != "gt_binary":
        raise ValueError("2-class ABMIL requires label_mode=gt_binary")
    if num_classes == 3 and label_mode != "valieris_3":
        raise ValueError("3-class ABMIL requires label_mode=valieris_3")

    ds = HerohePatchBagDataset(
        features_dir=features_dir,
        labels_csv=labels_csv,
        label_mode=label_mode,
        max_patches=max_patches,
        subsample_mode="deterministic",
        seed=0,
        return_coords=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag)

    rows = []
    all_y, all_p, all_probs = [], [], []
    for batch in loader:
        x = batch["features"].to(device)
        sid = batch["slide_id"][0]
        y = int(batch["label"].item())
        fold_probs = []
        for model in models:
            out = model(x)
            probs = F.softmax(out["logits"], dim=-1)[0].detach().cpu().numpy()
            fold_probs.append(probs)
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
    metrics["checkpoints"] = [str(c) for c in checkpoints]
    metrics["n_models"] = len(checkpoints)
    metrics["num_classes"] = num_classes
    metrics["n_labeled_slides"] = int(len(ds))
    metrics["confusion_matrix"] = confusion_matrix(
        y, p, labels=list(range(num_classes))
    ).tolist()
    metrics["classification_report"] = classification_report(
        y, p, labels=list(range(num_classes)), zero_division=0, output_dict=True
    )

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
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--tag", default="abmil")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_REPO / "herohe" / "gp2" / "runs" / "test_eval_abmil",
    )
    args = ap.parse_args()

    labels_csv = labels_csv_path(args.labels_csv)
    device = pick_device(args.device)
    ckpts = [c.expanduser().resolve() for c in args.checkpoint]
    print(f"\n=== ABMIL test ({args.label_mode}, {len(ckpts)} model(s)) ===")
    m = evaluate(
        ckpts,
        args.features_dir,
        labels_csv,
        args.label_mode,
        device,
        args.max_patches,
        args.out_dir,
        args.tag,
    )
    if args.label_mode == "gt_binary":
        print(
            f"n={m['n']}  AUC(positive)={m.get('auc_positive', float('nan')):.4f}  "
            f"macro-F1={m['macro_f1']:.4f}  models={m['n_models']}"
        )
    else:
        print(
            f"n={m['n']}  macro-AUROC={m.get('macro_auroc', float('nan')):.4f}  "
            f"macro-F1={m['macro_f1']:.4f}  models={m['n_models']}"
        )
    print("confusion_matrix:", m["confusion_matrix"])

    summary = {"label_mode": args.label_mode, "tag": args.tag, "results": [m]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"summary_{args.tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
