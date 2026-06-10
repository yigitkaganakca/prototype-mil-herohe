#!/usr/bin/env python3
"""Sweep ABMIL anti-overfit settings on fold 0; pick first config with no overfitting."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
FEAT = ROOT / "herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
CSV = ROOT / "herohe/Training (ground truth).csv"
FOLDS = ROOT / "herohe/gp2/data/folds_v1.csv"
TRAIN = ROOT / "herohe/gp2/scripts/train_mil_baseline.py"
OUT_BASE = ROOT / "herohe/gp2/runs/abmil_antioverfit_sweep"

CONFIGS = [
    {"name": "v1_defaults", "extra": []},
    {"name": "v2_strong_reg", "extra": ["--abmil_dropout", "0.5", "--weight_decay", "0.002"]},
    {"name": "v3_mp2048", "extra": ["--max_patches", "2048"]},
    {
        "name": "v4_strong_mp2048",
        "extra": [
            "--abmil_dropout",
            "0.5",
            "--weight_decay",
            "0.002",
            "--max_patches",
            "2048",
            "--label_smoothing",
            "0.15",
        ],
    },
    {
        "name": "v5_low_lr",
        "extra": ["--lr", "5e-5", "--abmil_dropout", "0.45", "--weight_decay", "0.0015"],
    },
]


def run_one(name: str, extra: list[str]) -> dict:
    out = OUT_BASE / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(TRAIN),
        "--aggregator",
        "abmil",
        "--num_classes",
        "2",
        "--label_mode",
        "gt_binary",
        "--features_dir",
        str(FEAT),
        "--labels_csv",
        str(CSV),
        "--folds_csv",
        str(FOLDS),
        "--out_dir",
        str(out),
        "--device",
        "mps",
        "--only_fold",
        "0",
        "--epochs",
        "50",
        *extra,
    ]
    print(f"\n>>> RUN {name}\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    summary = json.loads((out / "summary.json").read_text())
    fold = summary["folds"][0]
    diag = fold.get("overfit_diag", {})
    return {
        "name": name,
        "auc": fold.get("auc_positive"),
        "best_epoch": fold.get("best_epoch"),
        "overfit_diag": diag,
        "ok": not diag.get("overfitting_detected", True),
    }


def main() -> None:
    results = []
    winner = None
    for cfg in CONFIGS:
        r = run_one(cfg["name"], cfg["extra"])
        results.append(r)
        print(f"[result] {r}")
        if winner is None and r["ok"]:
            winner = r

    report = {"results": results, "winner": winner}
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "sweep_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if winner is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
