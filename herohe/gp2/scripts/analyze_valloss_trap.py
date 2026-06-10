#!/usr/bin/env python3
"""Analyze val_loss checkpoint trapping vs val_auc peak from training logs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def analyze_fold(run_dir: Path, fold: int, min_ep: int = 8, max_miss: float = 0.02) -> dict:
    log = run_dir / f"fold_{fold}" / "log.csv"
    ckpt = run_dir / f"fold_{fold}" / "best.pt"
    if not log.exists():
        raise FileNotFoundError(log)

    rows = list(csv.DictReader(open(log)))
    tr = np.array([float(r["train_loss"]) for r in rows])
    vl = np.array([float(r["val_loss"]) for r in rows])
    auc = np.array([float(r["val_auc_positive"]) for r in rows])

    best_vl_i = int(vl.argmin())
    best_auc_i = int(auc.argmax())
    saved_ep = best_vl_i + 1
    saved_auc = float(auc[best_vl_i])
    peak_auc = float(auc.max())

    gap = None
    if ckpt.exists():
        b = torch.load(ckpt, map_location="cpu", weights_only=False)
        ep = int(b["metrics"]["epoch"])
        saved_ep = ep
        saved_auc = float(b["metrics"].get("auc_positive", auc[ep - 1]))
        gap = float(b["metrics"]["val_loss"]) - tr[ep - 1]

    miss = peak_auc - saved_auc
    reasons = []
    if saved_ep < min_ep:
        reasons.append(f"saved ep {saved_ep} < {min_ep}")
    if miss > max_miss:
        reasons.append(f"auc miss {miss:.3f} > {max_miss} (peak {peak_auc:.3f} @ ep{best_auc_i + 1})")
    if gap is not None and gap < 0:
        reasons.append(f"negative train-val gap {gap:+.3f}")

    post = slice(3, len(vl))
    dtr = np.diff(tr[post]) if len(tr[post]) > 1 else np.array([])
    dvl = np.diff(vl[post]) if len(vl[post]) > 1 else np.array([])
    corr = float(np.corrcoef(np.diff(tr), np.diff(vl))[0, 1]) if len(tr) > 2 else None

    return {
        "fold": fold,
        "epochs_ran": len(rows),
        "saved_ep": saved_ep,
        "val_loss_min_ep": best_vl_i + 1,
        "val_loss_min": round(float(vl.min()), 4),
        "val_loss_at_save": round(float(vl[saved_ep - 1]), 4),
        "val_auc_at_save": round(saved_auc, 4),
        "val_auc_peak": round(peak_auc, 4),
        "val_auc_peak_ep": best_auc_i + 1,
        "auc_lag_epochs": best_auc_i - best_vl_i,
        "peak_minus_saved_auc": round(miss, 4),
        "delta_corr": round(corr, 3) if corr is not None else None,
        "post_ep3_train_down_rate": round(float(np.mean(dtr < 0)), 3) if len(dtr) else None,
        "post_ep3_val_up_rate": round(float(np.mean(dvl > 0)), 3) if len(dvl) else None,
        "trapped_early": (best_vl_i + 1) <= 4 and (best_auc_i - best_vl_i) >= 3,
        "stable": not reasons,
        "reasons": reasons,
    }


def analyze_run(run_dir: Path, label: str, folds: list[int] | None = None) -> dict:
    folds = folds if folds is not None else list(range(5))
    per_fold = [analyze_fold(run_dir, f) for f in folds]
    trapped = sum(1 for x in per_fold if x["trapped_early"])
    return {
        "label": label,
        "run_dir": str(run_dir),
        "folds": per_fold,
        "n_folds_trapped_early": trapped,
        "mean_saved_auc": round(float(np.mean([x["val_auc_at_save"] for x in per_fold])), 4),
        "mean_peak_auc": round(float(np.mean([x["val_auc_peak"] for x in per_fold])), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=None)
    ap.add_argument("--run_dirs", nargs="*", default=None)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    reports = []
    if args.run_dir:
        reports.append(analyze_run(args.run_dir, args.label or args.run_dir.name))
    if args.run_dirs:
        labels = args.labels or [Path(p).name for p in args.run_dirs]
        for p, lab in zip(args.run_dirs, labels):
            folds = [1] if "fold1" in str(p) or "head2head" in str(p) else None
            if folds is None and Path(p).joinpath("fold_1/log.csv").exists() and not Path(p).joinpath("fold_0/log.csv").exists():
                folds = [1]
            reports.append(analyze_run(Path(p), lab, folds=folds))

    out = {"runs": reports}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    print(f"Wrote {args.out_json}\n")
    for r in reports:
        print(f"=== {r['label']} ===")
        for f in r["folds"]:
            trap = "TRAP" if f["trapped_early"] else "ok"
            print(
                f"  fold{f['fold']}: saved ep{f['saved_ep']} auc={f['val_auc_at_save']:.3f} "
                f"peak={f['val_auc_peak']:.3f}@ep{f['val_auc_peak_ep']} "
                f"vl_min@ep{f['val_loss_min_ep']} lag={f['auc_lag_epochs']} [{trap}]"
            )


if __name__ == "__main__":
    main()
