#!/usr/bin/env python3
"""Re-evaluate every checkpoint set that feeds a report table under the revised
inference pipeline (no temperature scaling, no patch cap / full bags at test).

Writes per-run metrics under runs/report_regen/<key>/ and a single
report_regen/all_metrics.json plus a console comparison against the old
(temperature-calibrated, 4096-capped) report values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from herohe.gp2.scripts import eval_phenobin_test as P
from herohe.gp2.scripts import eval_mil_baseline_test as B

DEVICE = P.pick_device("mps")
VIRCHOW = str(REPO / "herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2")
RESNET = str(REPO / "herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_resnet50")
TESTLAB = P.labels_csv_path(str(REPO / "herohe/Test (ground truth)(1).xlsx"))
RUNS = REPO / "herohe/gp2/runs"
OUT = RUNS / "report_regen"
OUT.mkdir(parents=True, exist_ok=True)


def ck(run: str) -> list[Path]:
    return [RUNS / run / f"fold_{f}/best.pt" for f in range(5)]


# key, run_dir, label_mode, features, old_report_value (for sanity diff)
PHENO = [
    ("bin_primary_hard", "khead_hard_partition_medoid_proto_control", "gt_binary", VIRCHOW, "AUC .826/mF1 .732/bACC .750"),
    ("bin_indep_token", "medoid_benchmark/bin_indep_token_L8", "gt_binary", VIRCHOW, "AUC .829/mF1 .758/bACC .775"),
    ("bin_mean", "medoid_benchmark/bin_hard_mean_L8", "gt_binary", VIRCHOW, "AUC .769/mF1 .743/bACC .753"),
    ("bin_concat", "medoid_benchmark/bin_hard_concat_L8", "gt_binary", VIRCHOW, "AUC .762/mF1 .720/bACC .725"),
    ("bin_k4", "medoid_benchmark/bin_hard_token_L4", "gt_binary", VIRCHOW, "AUC .819/mF1 .756/bACC .764"),
    ("bin_k16", "medoid_benchmark/bin_hard_token_L16", "gt_binary", VIRCHOW, "AUC .786/mF1 .704/bACC .717"),
    ("bin_resnet_primary", "khead_token_abmil_hard_partition_ent0_resnet50", "gt_binary", RESNET, "AUC .569/posF1 .27"),
    ("c3_primary_hard", "medoid_benchmark/tri_hard_token_L8", "valieris_3", VIRCHOW, "mAU .818/mF1 .622/acc .698/negF1 .414"),
    ("c3_indep_token", "medoid_benchmark/tri_indep_token_L8", "valieris_3", VIRCHOW, "mAU .797/mF1 .603/acc .698/negF1 .333"),
    ("c3_indep_mean", "medoid_benchmark/tri_indep_mean_L8", "valieris_3", VIRCHOW, "mAU .778/mF1 .607/acc .685/negF1 .400"),
    ("c3_hard_mean", "medoid_benchmark/tri_hard_mean_L8", "valieris_3", VIRCHOW, "pending"),
    ("c3_hard_concat", "medoid_benchmark/tri_hard_concat_L8", "valieris_3", VIRCHOW, "pending"),
]

BASE = [
    ("bin_abmil", "abmil_phiher2fold_valloss", "gt_binary", VIRCHOW, "AUC .762/mF1 .726/bACC .731"),
    ("bin_clam", "clam_phiher2fold_valloss", "gt_binary", VIRCHOW, "AUC .769/mF1 .719/bACC .722"),
    ("bin_transmil", "transmil_phiher2fold_valloss", "gt_binary", VIRCHOW, "AUC .765/mF1 .690/bACC .700"),
    ("c3_abmil", "abmil_valieris3_5fold_s42_valloss", "valieris_3", VIRCHOW, "mAU .773/mF1 .573/acc .671/negF1 .308"),
    ("c3_clam", "clam_valieris3_5fold_s42_valloss", "valieris_3", VIRCHOW, "mAU .762/mF1 .586/acc .685/negF1 .320"),
    ("c3_transmil", "transmil_valieris3_5fold_s42_valloss", "valieris_3", VIRCHOW, "mAU .748/mF1 .557/acc .671/negF1 .261"),
    ("bin_resnet_abmil", "abmil_resnet50_5fold_s42_valloss", "gt_binary", RESNET, "AUC .552"),
    ("bin_resnet_clam", "clam_resnet50_5fold_s42_valloss", "gt_binary", RESNET, "AUC .562"),
    ("bin_resnet_transmil", "transmil_resnet50_5fold_s42_valloss", "gt_binary", RESNET, "AUC .487"),
]


def fnum(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else " - "


def summarize(key, m, old):
    is_binary = m.get("AUC") is not None
    if is_binary:
        line = (f"AUC={fnum(m.get('AUC'))} mF1={fnum(m.get('macro_f1'))} posF1={fnum(m.get('posF1'))} "
                f"bACC={fnum(m.get('bACC'))} AUPRC={fnum(m.get('AUPRC'))} ECE={fnum(m.get('ECE'))}")
    else:
        negf1 = (m.get('classification_report') or {}).get('0', {}).get('f1-score')
        line = (f"mAUROC={fnum(m.get('macro_auroc'))} mF1={fnum(m.get('macro_f1'))} "
                f"acc={fnum(m.get('wACC'))} negF1={fnum(negf1)} ECE={fnum(m.get('ECE'))}")
    print(f"  {key:22s} NEW: {line}")
    print(f"  {'':22s} OLD: {old}")


def main():
    results = {}
    print("\n========== PHENOBIN ==========")
    for key, run, lm, feat, old in PHENO:
        print(f"[run] {key} ({run})")
        m = P.evaluate_checkpoints(ck(run), feat, TESTLAB, lm, DEVICE, None, OUT / key, key, apply_calibration=False)
        results[key] = m
        summarize(key, m, old)
    print("\n========== BASELINES ==========")
    for key, run, lm, feat, old in BASE:
        print(f"[run] {key} ({run})")
        m = B.evaluate_checkpoints(ck(run), feat, TESTLAB, lm, DEVICE, None, OUT / key, key, apply_calibration=False)
        results[key] = m
        summarize(key, m, old)

    compact = {}
    for k, m in results.items():
        compact[k] = {
            "AUC": m.get("AUC"), "macro_f1": m.get("macro_f1"), "macro_auroc": m.get("macro_auroc"),
            "posF1": m.get("posF1"), "bACC": m.get("bACC"), "wACC": m.get("wACC"),
            "AUPRC": m.get("AUPRC"), "ECE": m.get("ECE"), "brier": m.get("brier"),
            "negF1": (m.get("classification_report") or {}).get("0", {}).get("f1-score"),
            "confusion_matrix": m.get("confusion_matrix"), "n": m.get("n"),
        }
    (OUT / "all_metrics.json").write_text(json.dumps(compact, indent=2))
    print(f"\nWrote {OUT/'all_metrics.json'}")


if __name__ == "__main__":
    main()
