"""Compare fold-0 val metrics: new matched ABMIL/khead vs prior runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Row:
    label: str
    select_on: str
    epoch: int | None
    val_auc: float | None
    val_loss: float | None
    notes: str = ""


def from_ckpt(path: Path, label: str, notes: str = "") -> Row | None:
    if not path.is_file():
        return None
    b = torch.load(path, map_location="cpu", weights_only=False)
    m = b.get("metrics", {})
    a = b.get("args", {})
    cfg = b.get("config", {})
    readout = cfg.get("readout", a.get("readout", a.get("aggregator", "?")))
    sel = a.get("select_on", "?")
    note = notes or f"readout={readout} lr={a.get('lr')} wd={a.get('weight_decay')} mixup={a.get('mixup_alpha', 0)}"
    return Row(
        label=label,
        select_on=str(sel),
        epoch=m.get("epoch"),
        val_auc=m.get("auc_positive"),
        val_loss=m.get("val_loss"),
        notes=note,
    )


def from_log_val_loss_best(log: Path, label: str, notes: str) -> Row | None:
    if not log.is_file():
        return None
    with log.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return Row(
        label=label,
        select_on="val_loss (log counterfactual)",
        epoch=int(best["epoch"]),
        val_auc=float(best["val_auc_positive"]),
        val_loss=float(best["val_loss"]),
        notes=notes,
    )


def from_log_val_auc_best(log: Path, label: str, notes: str) -> Row | None:
    if not log.is_file():
        return None
    with log.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["val_auc_positive"]))
    return Row(
        label=label,
        select_on="val_auc_positive (log counterfactual)",
        epoch=int(best["epoch"]),
        val_auc=float(best["val_auc_positive"]),
        val_loss=float(best["val_loss"]),
        notes=notes,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_abmil", type=Path, required=True)
    ap.add_argument("--new_khead", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--tag", default="", help="Suffix for NEW run labels, e.g. v2hparams")
    args = ap.parse_args()
    r = args.repo
    new_tag = f" ({args.tag})" if args.tag else ""

    rows: list[Row] = []

    rows.append(from_ckpt(args.new_abmil / "fold_0/best.pt", f"NEW matched ABMIL{new_tag}", f"shared recipe, val_loss{new_tag}"))
    rows.append(from_ckpt(args.new_khead / "fold_0/best.pt", f"NEW matched khead{new_tag}", f"shared recipe, val_loss{new_tag}"))

    prev = r / "herohe/gp2/runs/fold0_matched_valloss"
    if prev.exists() and args.tag:
        rows.append(from_ckpt(prev / "abmil/fold_0/best.pt", "PREV matched ABMIL (lr2e-4)", "lr=2e-4 wd=5e-4 dropout=0.2"))
        rows.append(from_ckpt(prev / "khead/fold_0/best.pt", "PREV matched khead (lr2e-4)", "lr=2e-4 wd=5e-4 dropout=0.2"))

    rows.append(
        from_ckpt(
            r / "herohe/gp2/runs/abmil_binary_virchow2_v2/fold_0/best.pt",
            "OLD ABMIL v2",
            "native hparams lr=1e-4 wd=1e-3 dropout=0.4 epochs=50",
        )
    )
    rows.append(
        from_log_val_loss_best(
            r / "herohe/gp2/runs/abmil_binary_virchow2_v2/fold_0/log.csv",
            "OLD ABMIL v2 (reselect val_loss)",
            "same train as v2, pick min val_loss epoch from log",
        )
    )

    rows.append(
        from_ckpt(
            r / "herohe/gp2/runs/phenobin_binary_ap_L16_khead_5fold_valloss/fold_0/best.pt",
            "OLD khead 5fold_valloss",
            "mixup=0.4 phenobin hparams",
        )
    )
    rows.append(
        from_ckpt(
            r / "herohe/gp2/runs/phenobin_binary_ap_L16_5fold/fold_0/best.pt",
            "OLD AP-L16 full",
            "readout=full select_on=val_auc mixup=0.4",
        )
    )
    rows.append(
        from_ckpt(
            r / "herohe/gp2/runs/phenobin_binary_ap_L16_5fold_valloss/fold_0/best.pt",
            "OLD AP-L16 full valloss",
            "readout=full select_on=val_loss mixup=0.4",
        )
    )

    rows = [x for x in rows if x is not None]

    print("\n=== Fold 0 validation comparison ===")
    print(f"{'run':<32} {'select_on':<28} {'ep':>3} {'val_auc':>8} {'val_loss':>8}")
    print("-" * 88)
    for row in rows:
        print(
            f"{row.label:<32} {row.select_on:<28} {row.epoch or '-':>3} "
            f"{row.val_auc or 0:8.4f} {row.val_loss or 0:8.4f}  {row.notes}"
        )

    new_ab = next(x for x in rows if x.label.startswith("NEW matched ABMIL"))
    new_kh = next(x for x in rows if x.label.startswith("NEW matched khead"))
    if new_ab.val_auc and new_kh.val_auc:
        print(
            f"\nMatched head-to-head: khead vs ABMIL val_auc delta "
            f"{new_kh.val_auc - new_ab.val_auc:+.4f}  val_loss delta {new_kh.val_loss - new_ab.val_loss:+.4f}"
        )

    payload = {
        "rows": [row.__dict__ for row in rows],
        "head_to_head": {
            "abmil_val_auc": new_ab.val_auc,
            "khead_val_auc": new_kh.val_auc,
            "delta_auc": (new_kh.val_auc - new_ab.val_auc) if new_ab.val_auc and new_kh.val_auc else None,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
