#!/usr/bin/env python3
"""Regenerate the BINARY ensemble + per-fold prediction CSVs for the MIL
baselines (ABMIL, CLAM, TransMIL) into the task-namespaced uncertainty layout.

Needed because an earlier directory-collision (config-name-only output dirs) let
the three-class run overwrite the binary baseline prediction CSVs. The binary
baselines have no per-run test_eval/, so they must be re-evaluated from their
fold checkpoints. This writes:

    runs/uncertainty/ensemble/binary/<key>/predictions_<key>.csv
    runs/uncertainty/perfold/binary/<key>/predictions_<key>_f{f}.csv

and does NOT modify uncertainty.json. After running, verify with
``recompute_report_stats.py --task binary``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from herohe.gp2.scripts import eval_mil_baseline_test as B
from herohe.gp2.scripts import eval_phenobin_test as P

DEVICE = P.pick_device("mps")
VIRCHOW = str(REPO / "herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2")
TESTLAB = P.labels_csv_path(str(REPO / "herohe/Test (ground truth)(1).xlsx"))
RUNS = REPO / "herohe/gp2/runs"
OUT = RUNS / "uncertainty"

BASELINES = [
    ("ABMIL", RUNS / "abmil_phiher2fold_valloss"),
    ("CLAM", RUNS / "clam_phiher2fold_valloss"),
    ("TransMIL", RUNS / "transmil_phiher2fold_valloss"),
]


def main():
    for key, run in BASELINES:
        folds = [run / f"fold_{f}/best.pt" for f in range(5)]
        missing = [c for c in folds if not c.exists()]
        if missing:
            print(f"[skip] {key}: missing {missing}")
            continue
        # per-fold (for headline mean+/-std)
        for f in range(5):
            B.evaluate_checkpoints([folds[f]], VIRCHOW, TESTLAB, "gt_binary", DEVICE,
                                   None, OUT / "perfold" / "binary" / key, f"{key}_f{f}",
                                   apply_calibration=False)
        # 5-fold ensemble
        em = B.evaluate_checkpoints(folds, VIRCHOW, TESTLAB, "gt_binary", DEVICE,
                                    None, OUT / "ensemble" / "binary" / key, key,
                                    apply_calibration=False)
        print(f"[done] {key}: ensemble AUC={em.get('AUC'):.4f} "
              f"macroF1={em.get('macro_f1'):.4f} bACC={em.get('bACC'):.4f}")


if __name__ == "__main__":
    main()
