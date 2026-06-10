#!/usr/bin/env python3
"""Uncertainty reporting for the medoid PhenoBIN benchmark (inference-only).

For every config (medoid PhenoBIN variants + baselines), this:
  1. Evaluates each of the 5 CV fold-checkpoints INDIVIDUALLY on the official
     150-slide test set -> per-fold metric -> HEADLINE mean +/- std
     (PhiHER2-style "5 runs, mean +/- std").
  2. Keeps the 5-fold probability ENSEMBLE as a secondary number.
  3. Bootstrap 95% CI (slide resampling) on the ensemble metric.
  4. Paired DeLong test between key binary configs (correlated ROC-AUC).

Writes runs/uncertainty/uncertainty.json and prints a console summary.
Run with --task binary | three | both.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from herohe.gp2.scripts import eval_phenobin_test as P
from herohe.gp2.scripts import eval_mil_baseline_test as B

DEVICE = P.pick_device("mps")
VIRCHOW = str(REPO / "herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2")
RESNET = str(REPO / "herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_resnet50")
TESTLAB = P.labels_csv_path(str(REPO / "herohe/Test (ground truth)(1).xlsx"))
RUNS = REPO / "herohe/gp2/runs"
MB = RUNS / "medoid_benchmark"
OUT = RUNS / "uncertainty"
OUT.mkdir(parents=True, exist_ok=True)

# key, run_dir, label_mode, features, engine ("pheno"|"base")
BINARY = [
    ("PhenoBIN_primary_hard", RUNS / "khead_hard_partition_medoid_proto_control", "gt_binary", VIRCHOW, "pheno"),
    ("PhenoBIN_indep_token", MB / "bin_indep_token_L8", "gt_binary", VIRCHOW, "pheno"),
    ("PhenoBIN_indep_mean", MB / "bin_indep_mean_L8", "gt_binary", VIRCHOW, "pheno"),
    ("PhenoBIN_hard_mean", MB / "bin_hard_mean_L8", "gt_binary", VIRCHOW, "pheno"),
    ("PhenoBIN_hard_concat", MB / "bin_hard_concat_L8", "gt_binary", VIRCHOW, "pheno"),
    ("PhenoBIN_L4", MB / "bin_hard_token_L4", "gt_binary", VIRCHOW, "pheno"),
    ("PhenoBIN_L16", MB / "bin_hard_token_L16", "gt_binary", VIRCHOW, "pheno"),
    ("ABMIL", RUNS / "abmil_phiher2fold_valloss", "gt_binary", VIRCHOW, "base"),
    ("CLAM", RUNS / "clam_phiher2fold_valloss", "gt_binary", VIRCHOW, "base"),
    ("TransMIL", RUNS / "transmil_phiher2fold_valloss", "gt_binary", VIRCHOW, "base"),
    ("PhenoBIN_resnet", RUNS / "khead_token_abmil_hard_partition_ent0_resnet50", "gt_binary", RESNET, "pheno"),
    ("ABMIL_resnet", RUNS / "abmil_resnet50_5fold_s42_valloss", "gt_binary", RESNET, "base"),
    ("CLAM_resnet", RUNS / "clam_resnet50_5fold_s42_valloss", "gt_binary", RESNET, "base"),
    ("TransMIL_resnet", RUNS / "transmil_resnet50_5fold_s42_valloss", "gt_binary", RESNET, "base"),
]

THREE = [
    ("PhenoBIN_primary_hard", MB / "tri_hard_token_L8", "valieris_3", VIRCHOW, "pheno"),
    ("PhenoBIN_indep_token", MB / "tri_indep_token_L8", "valieris_3", VIRCHOW, "pheno"),
    ("PhenoBIN_indep_mean", MB / "tri_indep_mean_L8", "valieris_3", VIRCHOW, "pheno"),
    ("PhenoBIN_hard_mean", MB / "tri_hard_mean_L8", "valieris_3", VIRCHOW, "pheno"),
    ("PhenoBIN_hard_concat", MB / "tri_hard_concat_L8", "valieris_3", VIRCHOW, "pheno"),
    ("ABMIL", RUNS / "abmil_valieris3_5fold_s42_valloss", "valieris_3", VIRCHOW, "base"),
    ("CLAM", RUNS / "clam_valieris3_5fold_s42_valloss", "valieris_3", VIRCHOW, "base"),
    ("TransMIL", RUNS / "transmil_valieris3_5fold_s42_valloss", "valieris_3", VIRCHOW, "base"),
]

# DeLong pairs (binary, by key)
DELONG_PAIRS = [
    ("PhenoBIN_primary_hard", "PhenoBIN_indep_token"),
    ("PhenoBIN_primary_hard", "ABMIL"),
    ("PhenoBIN_primary_hard", "CLAM"),
    ("PhenoBIN_primary_hard", "TransMIL"),
    ("PhenoBIN_L4", "PhenoBIN_primary_hard"),
    ("PhenoBIN_L16", "PhenoBIN_primary_hard"),
]


def engine(e):
    return P if e == "pheno" else B


def folds(run: Path) -> list[Path]:
    return [run / f"fold_{f}/best.pt" for f in range(5)]


def read_preds(csv_path: Path):
    sid, y, probs = [], [], []
    with open(csv_path) as fh:
        r = csv.DictReader(fh)
        cols = [c for c in r.fieldnames if c.startswith("prob_")]
        cols.sort(key=lambda c: int(c.split("_")[1]))
        for row in r:
            sid.append(row["slide_id"])
            y.append(int(row["label"]))
            probs.append([float(row[c]) for c in cols])
    return sid, np.array(y), np.array(probs)


def headline_metric(m: dict, label_mode: str) -> float:
    return m["AUC"] if label_mode == "gt_binary" else m["macro_auroc"]


def collect_per_fold(key, run, lm, feat, eng, tag):
    mod = engine(eng)
    per = {"AUC": [], "macro_f1": [], "macro_auroc": [], "bACC": [], "wACC": [], "negF1": []}
    for f in range(5):
        ck = run / f"fold_{f}/best.pt"
        if not ck.exists():
            return None
        m = mod.evaluate_checkpoints([ck], feat, TESTLAB, lm, DEVICE, None,
                                     OUT / "perfold" / tag / key, f"{key}_f{f}", apply_calibration=False)
        per["AUC"].append(m.get("AUC"))
        per["macro_f1"].append(m.get("macro_f1"))
        per["macro_auroc"].append(m.get("macro_auroc"))
        per["bACC"].append(m.get("bACC"))
        per["wACC"].append(m.get("wACC"))
        negf1 = (m.get("classification_report") or {}).get("0", {}).get("f1-score")
        per["negF1"].append(negf1)
    return per


def msd(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=0))


def bootstrap_ci(y, p1, metric_fn, B_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(B_iter):
        idx = rng.integers(0, n, n)
        try:
            stats.append(metric_fn(y[idx], p1[idx]))
        except Exception:
            continue
    stats = np.array([s for s in stats if np.isfinite(s)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ---- DeLong (fast, Sun & Xu 2014) ----
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_var(y, preds):
    """preds: (k, n) array of scores for k predictors. Returns AUCs and covariance."""
    order = (-y).argsort(kind="mergesort")  # positives first (label 1)
    y = y[order]
    preds = preds[:, order]
    m = int((y == 1).sum())
    n = int((y == 0).sum())
    k = preds.shape[0]
    pos = preds[:, :m]
    neg = preds[:, m:]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, np.atleast_2d(s)


def delong_test(y, p_a, p_b):
    from scipy import stats as st
    preds = np.vstack([p_a, p_b])
    aucs, s = delong_var(y.astype(float), preds)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    pval = 2.0 * st.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(pval)


def run_task(configs, label_is_binary: bool, tag: str):
    out = {}
    ens_probs = {}  # key -> (sid, y, P)
    for key, run, lm, feat, eng in configs:
        if not (run / "fold_0/best.pt").exists():
            print(f"[skip] {key}: no checkpoints at {run}")
            continue
        print(f"[per-fold] {key} ({run.name})")
        per = collect_per_fold(key, run, lm, feat, eng, tag)
        if per is None:
            print(f"[skip] {key}: incomplete folds")
            continue
        mod = engine(eng)
        edir = OUT / "ensemble" / tag / key
        em = mod.evaluate_checkpoints(folds(run), feat, TESTLAB, lm, DEVICE, None, edir, key, apply_calibration=False)
        sid, y, Pmat = read_preds(edir / f"predictions_{key}.csv")
        ens_probs[key] = (sid, y, Pmat)

        rec = {"run": str(run), "label_mode": lm, "per_fold": per, "fold_mean": {}, "fold_std": {}}
        for metric in ["AUC", "macro_f1", "macro_auroc", "bACC", "wACC", "negF1"]:
            mu, sd = msd(per[metric])
            if mu is not None:
                rec["fold_mean"][metric] = mu
                rec["fold_std"][metric] = sd
        rec["ensemble"] = {
            "AUC": em.get("AUC"), "macro_f1": em.get("macro_f1"), "macro_auroc": em.get("macro_auroc"),
            "bACC": em.get("bACC"), "wACC": em.get("wACC"), "AUPRC": em.get("AUPRC"),
            "ECE": em.get("ECE"), "brier": em.get("brier"),
            "negF1": (em.get("classification_report") or {}).get("0", {}).get("f1-score"),
        }
        # bootstrap CI on ensemble headline
        if label_is_binary:
            from sklearn.metrics import roc_auc_score
            lo, hi = bootstrap_ci(y, Pmat[:, 1], lambda yy, pp: roc_auc_score(yy, pp))
            rec["ensemble"]["AUC_CI95"] = [lo, hi]
        else:
            from herohe.gp2.scripts.metrics_utils import macro_ovr_auc
            # bootstrap macro-AUROC needs full prob matrix
            rng = np.random.default_rng(0)
            n = len(y); stats = []
            for _ in range(2000):
                idx = rng.integers(0, n, n)
                stats.append(macro_ovr_auc(y[idx], Pmat[idx], Pmat.shape[1]))
            stats = np.array([s for s in stats if np.isfinite(s)])
            rec["ensemble"]["macroAUROC_CI95"] = [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]
        out[key] = rec
        m_, s_ = (rec["fold_mean"].get("AUC"), rec["fold_std"].get("AUC")) if label_is_binary \
            else (rec["fold_mean"].get("macro_auroc"), rec["fold_std"].get("macro_auroc"))
        hl = "AUC" if label_is_binary else "macroAUROC"
        print(f"    {key:24s} {hl} fold {m_:.3f}+/-{s_:.3f}  ens={headline_metric(em, lm):.3f}")

    # DeLong (binary only)
    delong = {}
    if label_is_binary:
        for a, b in DELONG_PAIRS:
            if a in ens_probs and b in ens_probs:
                sid_a, ya, Pa = ens_probs[a]
                sid_b, yb, Pb = ens_probs[b]
                # align by slide id
                idx_b = {s: i for i, s in enumerate(sid_b)}
                order = [idx_b[s] for s in sid_a]
                auc_a, auc_b, pval = delong_test(ya, Pa[:, 1], Pb[np.array(order), 1])
                delong[f"{a}__vs__{b}"] = {"auc_a": auc_a, "auc_b": auc_b, "delong_p": pval}
                print(f"    DeLong {a} ({auc_a:.3f}) vs {b} ({auc_b:.3f}): p={pval:.3f}")
    return {"configs": out, "delong": delong}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "three", "both"], default="both")
    args = ap.parse_args()
    result = {}
    if args.task in ("binary", "both"):
        print("\n========== BINARY ==========")
        result["binary"] = run_task(BINARY, True, "binary")
    if args.task in ("three", "both"):
        print("\n========== THREE-CLASS ==========")
        result["three"] = run_task(THREE, False, "three")
    (OUT / "uncertainty.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT/'uncertainty.json'}")


if __name__ == "__main__":
    main()
