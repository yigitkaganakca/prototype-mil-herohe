#!/usr/bin/env python3
"""Compare fold-1 khead vs ABMIL logs and split stats; write khead_fold1_investigation.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fold_stats(merged: pd.DataFrame, f: int) -> dict:
    val = merged[merged["fold"] == f]
    train = merged[merged["fold"] != f]
    return {
        "fold": f,
        "val_n": len(val),
        "val_pos": int(val["gt_binary"].sum()),
        "val_neg": int((val["gt_binary"] == 0).sum()),
        "val_pos_rate": float(val["gt_binary"].mean()),
        "train_pos_rate": float(train["gt_binary"].mean()),
    }


def analyze_log(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    tr = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    auc = [float(r["val_auc_positive"]) for r in rows]
    best_vl_i = min(range(len(vl)), key=lambda i: vl[i])
    best_auc_i = max(range(len(auc)), key=lambda i: auc[i])
    dtr = [tr[i] - tr[i - 1] for i in range(1, len(tr))]
    dvl = [vl[i] - vl[i - 1] for i in range(1, len(vl))]
    corr = float(np.corrcoef(dtr, dvl)[0, 1]) if len(dtr) > 2 else None
    return {
        "epochs": len(rows),
        "val_loss_min": round(vl[best_vl_i], 4),
        "val_loss_min_ep": best_vl_i + 1,
        "val_auc_max": round(auc[best_auc_i], 4),
        "val_auc_max_ep": best_auc_i + 1,
        "auc_at_min_vl": round(auc[best_vl_i], 4),
        "vl_at_max_auc": round(vl[best_auc_i], 4),
        "auc_lag_epochs": best_auc_i - best_vl_i,
        "delta_corr": round(corr, 3) if corr is not None else None,
        "saved_ep_val_loss": best_vl_i + 1,
        "saved_auc_val_loss": round(auc[best_vl_i], 4),
        "peak_minus_saved_auc": round(auc[best_auc_i] - auc[best_vl_i], 4),
    }


def check_fold1_stable(run_dir: Path, min_ep: int = 8, max_miss: float = 0.02) -> dict:
    log = run_dir / "fold_1" / "log.csv"
    ckpt = run_dir / "fold_1" / "best.pt"
    if not log.exists() or not ckpt.exists():
        return {"stable": False, "reason": "missing fold_1 log or checkpoint"}

    import torch

    rows = list(csv.DictReader(open(log)))
    vl = [float(r["val_loss"]) for r in rows]
    auc = [float(r["val_auc_positive"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = b["metrics"]
    ep = int(m["epoch"])
    saved_auc = float(m["auc_positive"])
    peak_auc = max(auc)
    peak_ep = auc.index(peak_auc) + 1
    gap = float(m["val_loss"]) - tr[ep - 1]
    miss = peak_auc - saved_auc
    stable = (ep >= min_ep) and (miss <= max_miss) and (gap >= 0.0)
    reasons = []
    if ep < min_ep:
        reasons.append(f"saved ep {ep} < {min_ep}")
    if miss > max_miss:
        reasons.append(f"auc miss {miss:.3f} > {max_miss} (peak {peak_auc:.3f} @ ep{peak_ep})")
    if gap < 0:
        reasons.append(f"negative train-val gap {gap:+.3f}")
    return {
        "stable": stable,
        "epoch": ep,
        "val_auc": saved_auc,
        "peak_auc": peak_auc,
        "peak_ep": peak_ep,
        "gap": gap,
        "reasons": reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    repo = args.repo
    out = args.out or repo / "data/khead_fold1_investigation.json"

    merged = pd.read_csv(repo / "data/folds_v1.csv")
    run_names = [
        "phenobin_khead_iterate_nobias_mean_fullval",
        "phenobin_khead_iterate_mean_fullval",
        "phenobin_khead_iterate_nobias_mean",
        "abmil_binary_virchow2_v2",
        "phenobin_binary_ap_L16_khead_mean_5fold_valloss",
    ]

    report: dict = {"fold_stats": [fold_stats(merged, f) for f in range(5)], "runs": {}}
    for name in run_names:
        run = repo / "runs" / name
        if not run.exists():
            continue
        report["runs"][name] = {"folds": {}, "fold_1_gate": check_fold1_stable(run)}
        for f in range(5):
            p = run / f"fold_{f}" / "log.csv"
            if p.exists():
                report["runs"][name]["folds"][f"fold_{f}"] = analyze_log(p)

    f1 = report["runs"].get("phenobin_khead_iterate_nobias_mean_fullval", {}).get("folds", {}).get("fold_1", {})
    f0 = report["runs"].get("phenobin_khead_iterate_nobias_mean_fullval", {}).get("folds", {}).get("fold_0", {})
    report["diagnosis"] = {
        "fold1_val_pos_rate": report["fold_stats"][1]["val_pos_rate"],
        "fold0_val_pos_rate": report["fold_stats"][0]["val_pos_rate"],
        "khead_f1_auc_lag_epochs": f1.get("auc_lag_epochs"),
        "khead_f0_auc_lag_epochs": f0.get("auc_lag_epochs"),
        "khead_f1_delta_corr": f1.get("delta_corr"),
        "khead_f0_delta_corr": f0.get("delta_corr"),
        "root_causes": [
            "val_loss minimum at ep2-3 on folds 1-2 while val_auc peaks 4-8 epochs later",
            "khead multi-head routing makes val CE volatile (ep6 spikes) even with mean-pool readout",
            "fold split balance is not the driver — fold1 pos_rate ~ fold0",
            "ABMIL on same fold1 reaches higher val_auc when selected on val_auc, not val_loss",
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    print(json.dumps(report["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
