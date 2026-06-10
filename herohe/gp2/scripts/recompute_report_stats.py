#!/usr/bin/env python3
"""Recompute and VERIFY every cheap report statistic from saved prediction CSVs.

This is an inference-free audit: it never re-runs a model. It reads the
per-fold and ensemble prediction CSVs already written under
``runs/uncertainty/{perfold,ensemble}/<key>/`` by ``compute_uncertainty.py``
and recomputes, from scratch and with the exact same formulas:

  * headline mean+/-std of the fold-wise metric (ddof=0),
  * ensemble metrics (AUC / macro-AUROC, PR-AUC, macro-F1, bACC, acc, negF1,
    ECE, Brier),
  * bootstrap 95% CI on the ensemble headline (2000 resamples, seed 0),
  * paired DeLong p-values between the binary DeLong pairs,
  * paired-bootstrap p-values on the three-class macro-AUROC gap
    (the multiclass analogue of DeLong; not stored in uncertainty.json).

It then diffs the recomputed values against ``uncertainty.json`` and prints a
PASS/FAIL table so we can confirm the numbers quoted in the report.

Usage:
    python herohe/gp2/scripts/recompute_report_stats.py [--task binary|three|both]
                                                         [--tol 1e-6]
                                                         [--paired-B 5000]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "herohe/gp2/runs"
UNC = RUNS / "uncertainty"
MB = RUNS / "medoid_benchmark"

# Authoritative per-run BINARY ensemble predictions (never overwritten by the
# three-class run, unlike runs/uncertainty/ensemble/<shared_key>/). Used to
# verify the binary shared keys whose uncertainty/ensemble CSV was clobbered.
BIN_SOURCE = {
    "PhenoBIN_primary_hard": RUNS / "khead_hard_partition_medoid_proto_control/test_eval/predictions_medoid_proto_5fold.csv",
    "PhenoBIN_indep_token": MB / "bin_indep_token_L8/test_eval/predictions_bin_indep_token_L8_5fold.csv",
    "PhenoBIN_indep_mean": MB / "bin_indep_mean_L8/test_eval/predictions_bin_indep_mean_L8_5fold.csv",
    "PhenoBIN_hard_mean": MB / "bin_hard_mean_L8/test_eval/predictions_bin_hard_mean_L8_5fold.csv",
    "PhenoBIN_hard_concat": MB / "bin_hard_concat_L8/test_eval/predictions_bin_hard_concat_L8_5fold.csv",
}

# Binary DeLong pairs and three-class paired-bootstrap pairs (match the report).
DELONG_PAIRS = [
    ("PhenoBIN_primary_hard", "PhenoBIN_indep_token"),
    ("PhenoBIN_primary_hard", "ABMIL"),
    ("PhenoBIN_primary_hard", "CLAM"),
    ("PhenoBIN_primary_hard", "TransMIL"),
    ("PhenoBIN_L4", "PhenoBIN_primary_hard"),
    ("PhenoBIN_L16", "PhenoBIN_primary_hard"),
]
THREE_PAIRED = [
    ("PhenoBIN_primary_hard", "PhenoBIN_indep_token"),
    ("PhenoBIN_primary_hard", "PhenoBIN_indep_mean"),
    ("PhenoBIN_primary_hard", "ABMIL"),
    ("PhenoBIN_primary_hard", "CLAM"),
    ("PhenoBIN_primary_hard", "TransMIL"),
]


# --------------------------------------------------------------------------- #
# Metric formulas (copied verbatim from eval_phenobin_test.py / metrics_utils) #
# --------------------------------------------------------------------------- #
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


def macro_ovr_auc(y: np.ndarray, P: np.ndarray, num_classes: int) -> float:
    aucs = []
    for c in range(num_classes):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2 or not np.isfinite(P[:, c]).all():
            continue
        try:
            aucs.append(float(roc_auc_score(y_bin, P[:, c])))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


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
    order = (-y).argsort(kind="mergesort")
    y = y[order]
    preds = preds[:, order]
    m = int((y == 1).sum())
    n = int((y == 0).sum())
    k = preds.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(preds[r, :m])
        ty[r] = _compute_midrank(preds[r, m:])
        tz[r] = _compute_midrank(preds[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(s)


def delong_test(y, p_a, p_b):
    from scipy import stats as st
    aucs, s = delong_var(y.astype(float), np.vstack([p_a, p_b]))
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2.0 * st.norm.sf(abs(z)))


def paired_bootstrap_macro_auroc(y, Pa, Pb, num_classes, B=5000, seed=0):
    """Multiclass analogue of DeLong: resample slides, recompute the
    macro-AUROC gap (A-B) on the *same* resample, two-sided p that the gap=0."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        da = macro_ovr_auc(y[idx], Pa[idx], num_classes)
        db = macro_ovr_auc(y[idx], Pb[idx], num_classes)
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    delta = macro_ovr_auc(y, Pa, num_classes) - macro_ovr_auc(y, Pb, num_classes)
    p_le = float(np.mean(diffs <= 0.0))
    p_ge = float(np.mean(diffs >= 0.0))
    pval = float(min(1.0, 2.0 * min(p_le, p_ge)))
    return float(delta), pval


def bootstrap_ci(metric_fn, y, P, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            stats.append(metric_fn(y[idx], P[idx]))
        except Exception:
            continue
    stats = np.array([s for s in stats if np.isfinite(s)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# --------------------------------------------------------------------------- #
def read_preds(csv_path: Path):
    sid, y, probs = [], [], []
    with open(csv_path) as fh:
        r = csv.DictReader(fh)
        cols = sorted((c for c in r.fieldnames if c.startswith("prob_")),
                      key=lambda c: int(c.split("_")[1]))
        for row in r:
            sid.append(row["slide_id"])
            y.append(int(row["label"]))
            probs.append([float(row[c]) for c in cols])
    return sid, np.array(y), np.array(probs, dtype=float)


def ensemble_metrics(y, P, num_classes, binary):
    p = P.argmax(axis=1)
    out = {
        "macro_f1": float(f1_score(y, p, average="macro",
                                   labels=list(range(num_classes)), zero_division=0)),
        "brier": brier_score(y, P, num_classes),
        "ECE": expected_calibration_error(y, P),
    }
    from sklearn.metrics import classification_report
    rep = classification_report(y, p, labels=list(range(num_classes)),
                                zero_division=0, output_dict=True)
    out["negF1"] = float(rep["0"]["f1-score"])
    out["wACC"] = float(rep["accuracy"])
    if binary:
        out["AUC"] = float(roc_auc_score(y, P[:, 1]))
        out["AUPRC"] = float(average_precision_score(y, P[:, 1]))
        out["bACC"] = float(balanced_accuracy_score(y, p))
    else:
        out["macro_auroc"] = macro_ovr_auc(y, P, num_classes)
    return out


def fold_headline(key, binary, num_classes, task):
    vals = []
    for f in range(5):
        c = UNC / "perfold" / task / key / f"predictions_{key}_f{f}.csv"   # namespaced
        if not c.exists():
            c = UNC / "perfold" / key / f"predictions_{key}_f{f}.csv"       # legacy flat
        if not c.exists():
            return None
        _, y, P = read_preds(c)
        if P.shape[1] != num_classes:  # dir was overwritten by the other task
            return None
        vals.append(roc_auc_score(y, P[:, 1]) if binary else macro_ovr_auc(y, P, num_classes))
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), [round(float(v), 3) for v in vals]


def fmt(x):
    return "----" if x is None else f"{x:.4f}"


def check(name, got, exp, tol, rows):
    if got is None or exp is None:
        rows.append((name, fmt(got), fmt(exp), "n/a"))
        return
    ok = abs(got - exp) <= tol
    rows.append((name, fmt(got), fmt(exp), "PASS" if ok else f"FAIL d={got-exp:+.2e}"))


def run(task, configs_keys, binary, num_classes, stored, tol, paired_B):
    print(f"\n{'='*78}\n{task.upper()}\n{'='*78}")
    cfg_store = stored.get("configs", {})
    ens = {}
    rows = []
    overwritten = []
    for key in configs_keys:
        epath = UNC / "ensemble" / task / key / f"predictions_{key}.csv"     # namespaced
        src = f"uncertainty/ensemble/{task}"
        if not epath.exists():
            epath = UNC / "ensemble" / key / f"predictions_{key}.csv"         # legacy flat
            src = "uncertainty/ensemble"
        # For binary shared keys whose legacy CSV was clobbered, prefer the
        # authoritative per-run test_eval prediction file.
        if binary and key in BIN_SOURCE and BIN_SOURCE[key].exists() and not (
                UNC / "ensemble" / task / key / f"predictions_{key}.csv").exists():
            epath = BIN_SOURCE[key]
            src = "per-run test_eval"
        if not epath.exists():
            print(f"[skip] {key}: no ensemble csv")
            continue
        sid, y, P = read_preds(epath)
        if P.shape[1] != num_classes:
            overwritten.append(key)
            print(f"[OVERWRITTEN] {key}: on-disk preds are {P.shape[1]}-class; the "
                  f"{'three-class' if num_classes == 2 else 'binary'} run clobbered this "
                  f"{'binary' if num_classes == 2 else 'three-class'} CSV \u2014 not recoverable from disk")
            continue
        if src != "uncertainty/ensemble":
            print(f"[src] {key}: verifying from {src}")
        ens[key] = (sid, y, P)
        em = ensemble_metrics(y, P, num_classes, binary)
        st = cfg_store.get(key, {}).get("ensemble", {})
        hl_fn = (lambda yy, pp: roc_auc_score(yy, pp[:, 1] if pp.ndim > 1 else pp)) if binary \
            else (lambda yy, pp: macro_ovr_auc(yy, pp, num_classes))
        if binary:
            lo, hi = bootstrap_ci(lambda yy, pp: roc_auc_score(yy, pp), y, P[:, 1])
            ci_key = "AUC_CI95"
        else:
            lo, hi = bootstrap_ci(lambda yy, pp: macro_ovr_auc(yy, pp, num_classes), y, P)
            ci_key = "macroAUROC_CI95"
        fh = fold_headline(key, binary, num_classes, task)
        print(f"\n--- {key} ---")
        if fh:
            fm = cfg_store.get(key, {}).get("fold_mean", {})
            fs = cfg_store.get(key, {}).get("fold_std", {})
            hlname = "AUC" if binary else "macro_auroc"
            print(f"  per-fold {fh[2]}  mean={fh[0]:.4f} std={fh[1]:.4f}  "
                  f"(stored mean={fm.get(hlname)} std={fs.get(hlname)})")
        for mk in (["AUC", "AUPRC", "macro_f1", "bACC", "negF1", "ECE", "brier"] if binary
                   else ["macro_auroc", "macro_f1", "wACC", "negF1", "ECE", "brier"]):
            check(f"{key}.{mk}", em.get(mk), st.get(mk), tol, rows)
        stored_ci = st.get(ci_key)
        if stored_ci:
            check(f"{key}.{ci_key}[lo]", lo, stored_ci[0], 5e-3, rows)
            check(f"{key}.{ci_key}[hi]", hi, stored_ci[1], 5e-3, rows)

    # significance
    print(f"\n--- significance ({'DeLong' if binary else 'paired bootstrap, B='+str(paired_B)}) ---")
    if binary:
        st_del = stored.get("delong", {})
        for a, b in DELONG_PAIRS:
            if a in ens and b in ens:
                (sa, ya, Pa), (sb, yb, Pb) = ens[a], ens[b]
                order = [{s: i for i, s in enumerate(sb)}[s] for s in sa]
                auc_a, auc_b, p = delong_test(ya, Pa[:, 1], Pb[np.array(order), 1])
                stored_p = st_del.get(f"{a}__vs__{b}", {}).get("delong_p")
                tag = "" if stored_p is None else (
                    "  PASS" if abs(p - stored_p) <= 1e-4 else f"  FAIL (stored {stored_p:.5f})")
                print(f"  {a} ({auc_a:.3f}) vs {b} ({auc_b:.3f}): p={p:.5f}{tag}")
    else:
        for a, b in THREE_PAIRED:
            if a in ens and b in ens:
                (sa, ya, Pa), (sb, yb, Pb) = ens[a], ens[b]
                order = [{s: i for i, s in enumerate(sb)}[s] for s in sa]
                delta, p = paired_bootstrap_macro_auroc(ya, Pa, Pb[np.array(order)],
                                                        num_classes, B=paired_B)
                print(f"  {a} vs {b}: dmAUROC={delta:+.4f}  p={p:.4f}")

    # summary
    fails = [r for r in rows if r[3].startswith("FAIL")]
    print(f"\n{'metric':40s} {'recomputed':>11s} {'stored':>9s}  status")
    for nm, g, e, s in rows:
        print(f"{nm:40s} {g:>11s} {e:>9s}  {s}")
    print(f"\n{len(rows)-len(fails)}/{len(rows)} checks PASS; {len(fails)} FAIL")
    if overwritten:
        print(f"NOT VERIFIABLE FROM DISK (CSV overwritten by the other task): {overwritten}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "three", "both"], default="both")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--paired-B", type=int, default=5000)
    args = ap.parse_args()
    stored = json.loads((UNC / "uncertainty.json").read_text())

    binary_keys = list((stored.get("binary", {}).get("configs", {})).keys())
    three_keys = list((stored.get("three", {}).get("configs", {})).keys())
    total_fail = []
    if args.task in ("binary", "both"):
        total_fail += run("binary", binary_keys, True, 2, stored.get("binary", {}),
                          args.tol, args.paired_B)
    if args.task in ("three", "both"):
        total_fail += run("three", three_keys, False, 3, stored.get("three", {}),
                          args.tol, args.paired_B)
    print(f"\n{'#'*78}\nTOTAL FAILS: {len(total_fail)}")


if __name__ == "__main__":
    main()
