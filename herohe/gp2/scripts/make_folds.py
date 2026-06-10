"""Generate a reproducible stratified k-fold split for HEROHE training.

Output schema:  one row per slide with columns
    slide_id, ihc, gt_binary, valieris_class, laboratory, fold

`fold` is the index of the validation fold the slide belongs to (0..k-1).
Training fold k consists of all rows where `fold != k`.

Stratification key (``--stratify``):
  composite — (IHC, GT_binary) [default; folds_v1]
  binary    — gt_binary only [PhiHER2-style random stratified CV]
  valieris  — Valieris 3-class {0,1,2} [folds_v2 for M6]

We exclude any slide that does not have a corresponding feature file under
`--features_dir`, so the resulting fold file is guaranteed to be 1:1 with
trainable slides.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels_csv", required=True,
                    help='Path to "Training (ground truth).csv".')
    ap.add_argument("--features_dir", required=True,
                    help="Directory containing per-slide feature .h5 files (TRIDENT layout).")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stratify",
        choices=["composite", "binary", "valieris"],
        default="composite",
        help="composite=(ihc,gt_binary); binary=gt_binary only (PhiHER2); valieris=3-class",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    df["slide_id"] = df["Case"].astype(str).str.strip()

    # IHC: 0/1/2/3, integer.
    df["ihc"] = pd.to_numeric(df["Immunohistochemistry"], errors="coerce").astype("Int64")

    # gt_binary: HEROHE Final Result. Map "Negative" -> 0, "Positive" -> 1.
    gt_raw = df["Final Result (Ground truth)"].astype(str).str.strip().str.lower()
    df["gt_binary"] = gt_raw.map({"negative": 0, "positive": 1}).astype("Int64")

    # Valieris 3-class: neg / low / high (ASCO/CAP-aligned)
    valieris = []
    for ihc_val, fin in zip(df["ihc"], gt_raw):
        if pd.isna(ihc_val):
            valieris.append(np.nan)
            continue
        ihc_int = int(ihc_val)
        if ihc_int == 0:
            valieris.append(0)
        elif ihc_int == 1:
            valieris.append(1)
        elif ihc_int == 2:
            valieris.append(1 if fin == "negative" else 2 if fin == "positive" else np.nan)
        elif ihc_int == 3:
            valieris.append(2)
        else:
            valieris.append(np.nan)
    df["valieris_class"] = valieris

    df["laboratory"] = df["Laboratory"].astype(str).str.strip()

    # Drop rows without a usable IHC or GT label (defensive; HEROHE training set
    # has all of these populated, so this should be a no-op).
    df = df.dropna(subset=["ihc", "gt_binary", "valieris_class"]).copy()
    df["ihc"] = df["ihc"].astype(int)
    df["gt_binary"] = df["gt_binary"].astype(int)
    df["valieris_class"] = df["valieris_class"].astype(int)

    # Restrict to slides that actually have features on disk.
    feat_dir = Path(args.features_dir)
    have_feat = {p.stem for p in feat_dir.glob("*.h5")}
    missing = sorted(set(df["slide_id"]) - have_feat,
                     key=lambda s: int(s) if s.isdigit() else 10**9)
    if missing:
        print(f"[make_folds] {len(missing)} labeled slides have no feature file -> excluded:",
              missing[:20], "..." if len(missing) > 20 else "")
    df = df[df["slide_id"].isin(have_feat)].reset_index(drop=True)

    # Stratification key.
    if args.stratify == "valieris":
        df["strat_key"] = df["valieris_class"].astype(str)
        print("[make_folds] Valieris 3-class stratification counts:")
        for k, v in sorted(Counter(df["strat_key"]).items()):
            print(f"   class {k}: {v}")
    elif args.stratify == "binary":
        df["strat_key"] = df["gt_binary"].astype(str)
        print("[make_folds] binary (gt_binary) stratification counts:")
        for k, v in sorted(Counter(df["strat_key"]).items()):
            print(f"   class {k}: {v}")
    else:
        df["strat_key"] = df["ihc"].astype(str) + "_" + df["gt_binary"].astype(str)
        print("[make_folds] composite (IHC, GT) stratification key counts:")
        for k, v in sorted(Counter(df["strat_key"]).items()):
            print(f"   {k}: {v}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    df["fold"] = -1
    for fold_idx, (_, va_idx) in enumerate(skf.split(np.zeros(len(df)), df["strat_key"])):
        df.loc[va_idx, "fold"] = fold_idx
    assert (df["fold"] >= 0).all()

    df_out = df[["slide_id", "ihc", "gt_binary", "valieris_class", "laboratory", "fold"]].copy()
    df_out["slide_id_int"] = pd.to_numeric(df_out["slide_id"], errors="coerce")
    df_out = df_out.sort_values(["fold", "slide_id_int", "slide_id"], na_position="last").drop(
        columns=["slide_id_int"]
    )
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\n[make_folds] wrote {out_path}  rows={len(df_out)}")

    # Sanity report: per-fold composition.
    print("\n=== per-fold distribution ===")
    header = (
        f"{'fold':>4} {'val_n':>5}   IHC={{0,1,2,3}}            "
        f"GT={{Neg,Pos}}   Valieris={{0,1,2}}   labs"
    )
    print(header)
    for fold_idx in range(args.n_folds):
        sub = df_out[df_out["fold"] == fold_idx]
        ihc_cnt = Counter(sub["ihc"])
        ihc_str = "[" + ",".join(f"{ihc_cnt.get(c, 0):>3}" for c in (0, 1, 2, 3)) + "]"
        gt_cnt = Counter(sub["gt_binary"])
        gt_str = f"[Neg={gt_cnt.get(0, 0):>3}, Pos={gt_cnt.get(1, 0):>3}]"
        v_cnt = Counter(sub["valieris_class"])
        v_str = "[" + ",".join(f"{v_cnt.get(c, 0):>3}" for c in (0, 1, 2)) + "]"
        n_labs = sub["laboratory"].nunique()
        print(f"{fold_idx:>4} {len(sub):>5}   {ihc_str}  {gt_str}  {v_str}  {n_labs} labs")

    print("\n=== overall ===")
    print("IHC counts:", dict(sorted(Counter(df_out['ihc']).items())))
    print("GT counts :", dict(sorted(Counter(df_out['gt_binary']).items())))
    print("Valieris  :", dict(sorted(Counter(df_out['valieris_class']).items())))


if __name__ == "__main__":
    main()
