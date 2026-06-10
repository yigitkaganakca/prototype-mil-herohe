#!/usr/bin/env python3
"""Insert three-class hard-partition mean/concat metrics into results.tex."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "herohe/gp2/report/report_by_me/results.tex"
MB = REPO / "herohe/gp2/runs/medoid_benchmark"
UNC = REPO / "herohe/gp2/runs/uncertainty/uncertainty.json"


def rnd(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def acc(m: dict) -> float:
    return float(m["classification_report"]["accuracy"])


def neg_f1(m: dict) -> float:
    return float(m["classification_report"]["0"]["f1-score"])


def main() -> None:
    unc = json.loads(UNC.read_text())["three"]["configs"]
    mean_m = json.loads(
        (MB / "tri_hard_mean_L8/test_eval/metrics_tri_hard_mean_L8_5fold.json").read_text()
    )
    concat_m = json.loads(
        (MB / "tri_hard_concat_L8/test_eval/metrics_tri_hard_concat_L8_5fold.json").read_text()
    )
    um, uc = unc["PhenoBIN_hard_mean"], unc["PhenoBIN_hard_concat"]

    def row(label: str, cfg: dict, met: dict) -> str:
        mu = cfg["fold_mean"]["macro_auroc"]
        sd = cfg["fold_std"]["macro_auroc"]
        ens = cfg["ensemble"]["macro_auroc"]
        lo, hi = cfg["ensemble"]["macroAUROC_CI95"]
        return (
            f"{label} & ${mu:.3f}{{\\pm}}{sd:.3f}$ & "
            f"{rnd(ens)} [{lo:.2f}--{hi:.2f}] & "
            f"{rnd(met['macro_f1'])} & {rnd(acc(met))} & {rnd(neg_f1(met))} \\\\"
        )

    mean_row = row("Our model (hard + mean)", um, mean_m)
    concat_row = row("Our model (hard + concat)", uc, concat_m)
    insert_block = "    " + mean_row + "\n    " + concat_row + "\n"

    text = RESULTS.read_text()
    if "Our model (hard + mean)" not in text:
        anchor = "    Our model (indep.\\ + mean) & $0.758{\\pm}0.033$ & 0.770 [0.70--0.84] & 0.596 & 0.671 & 0.375 \\\\\n"
        if anchor not in text:
            raise SystemExit("anchor row not found in results.tex")
        text = text.replace(anchor, anchor + insert_block)

    text = text.replace(
        "All three of our model variants lead",
        "All five of our model variants lead",
    )
    old_para = (
        "On this task the hard-partition primary is the strongest variant: highest ensemble macro-AUROC\n"
        "(0.818), macro-F1 (0.622), and HER2-negative F1 (0.414)."
    )
    new_para = (
        "On this task the hard-partition primary with token-level ABMIL remains the strongest overall "
        "variant: highest ensemble macro-AUROC (0.818) and macro-F1 (0.622). Hard-partition mean and "
        "concat readouts reach macro-AUROC 0.798 and 0.801 respectively but lower macro-F1 (0.608, "
        "0.598), confirming that token-level competition over disjoint phenotype compartments is still "
        "preferred for full three-way stratification despite concat's strong binary ranking."
    )
    text = text.replace(old_para, new_para)

    readout_old = (
        "Under hard partition, mean pooling reaches ensemble AUC 0.792 and concat 0.844, "
        "compared with token ABMIL at 0.826 (Table~\\ref{tab:routing-readout})."
    )
    readout_new = (
        "Under hard partition, mean pooling reaches ensemble AUC 0.792 and concat 0.844 on binary "
        "classification, compared with token ABMIL at 0.826 (Table~\\ref{tab:routing-readout}); "
        "on three-class stratification the same readouts reach macro-AUROC 0.798 and 0.801 "
        "(Table~\\ref{tab:threeclass-results}) but do not exceed the primary token-ABMIL model "
        "(0.818)."
    )
    text = text.replace(readout_old, readout_new)

    RESULTS.write_text(text)
    print(f"Patched {RESULTS}")
    print(f"  hard+mean:  mAUROC={mean_m['macro_auroc']:.3f} mF1={mean_m['macro_f1']:.3f}")
    print(f"  hard+concat: mAUROC={concat_m['macro_auroc']:.3f} mF1={concat_m['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
