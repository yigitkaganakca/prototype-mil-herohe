"""Generate PhiHER2 S2-style 5× random 80/20 stratified holdout splits (HEROHE 360).

Each repeat writes one CSV compatible with ``train_phenobin_mil.py`` and
``init_prototypes_ap.py --val_fold 0``:

  fold=0  → validation (~72 slides, 20%)
  fold=1  → training  (~288 slides, 80%)

Train with ``--only_fold 0``; fit AP with ``--val_fold 0`` (288 train slides only).
Test set (150 slides) is never included.

Output layout (default):
  {out_dir}/repeat_{r}.csv   for r in 0..n_repeats-1
  {out_dir}/manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _load_labeled_slides(labels_csv: str, features_dir: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["slide_id"] = df["Case"].astype(str).str.strip()
    gt_raw = df["Final Result (Ground truth)"].astype(str).str.strip().str.lower()
    df["gt_binary"] = gt_raw.map({"negative": 0, "positive": 1}).astype("Int64")
    df["ihc"] = pd.to_numeric(df["Immunohistochemistry"], errors="coerce").astype("Int64")
    df["laboratory"] = df["Laboratory"].astype(str).str.strip()
    df = df.dropna(subset=["ihc", "gt_binary"]).copy()
    df["ihc"] = df["ihc"].astype(int)
    df["gt_binary"] = df["gt_binary"].astype(int)

    feat_dir = Path(features_dir)
    have_feat = {p.stem for p in feat_dir.glob("*.h5")}
    missing = sorted(set(df["slide_id"]) - have_feat, key=lambda s: int(s) if s.isdigit() else 10**9)
    if missing:
        print(f"[holdout] {len(missing)} labeled slides missing features → excluded: {missing[:10]}")
    df = df[df["slide_id"].isin(have_feat)].reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument(
        "--out_dir",
        default="herohe/gp2/data/phiher2_holdout_s42",
        help="Directory for repeat_*.csv files",
    )
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stratify",
        choices=["binary", "composite"],
        default="binary",
        help="binary=gt_binary (PhiHER2 HEROHE); composite=(ihc,gt_binary)",
    )
    args = ap.parse_args()

    df = _load_labeled_slides(args.labels_csv, args.features_dir)
    if args.stratify == "binary":
        y = df["gt_binary"].to_numpy()
        strat_label = "gt_binary"
    else:
        y = (df["ihc"].astype(str) + "_" + df["gt_binary"].astype(str)).to_numpy()
        strat_label = "ihc_gt_binary"

    print(f"[holdout] n={len(df)}  stratify={strat_label}  val_fraction={args.val_fraction}")
    print("  gt counts:", dict(sorted(Counter(df["gt_binary"]).items())))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sss = StratifiedShuffleSplit(
        n_splits=args.n_repeats,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    X = np.zeros(len(df))
    manifest = {
        "protocol": "phiher2_s2_holdout",
        "n_repeats": args.n_repeats,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "stratify": strat_label,
        "n_slides": len(df),
        "repeats": [],
    }

    for repeat, (_, va_idx) in enumerate(sss.split(X, y)):
        fold_col = np.ones(len(df), dtype=int)  # train = fold 1
        fold_col[va_idx] = 0  # val = fold 0
        out_df = df[["slide_id", "ihc", "gt_binary", "laboratory"]].copy()
        out_df["fold"] = fold_col
        out_df["repeat"] = repeat
        out_df["slide_id_int"] = pd.to_numeric(out_df["slide_id"], errors="coerce")
        out_df = out_df.sort_values(["fold", "slide_id_int", "slide_id"], na_position="last").drop(
            columns=["slide_id_int"]
        )

        csv_path = out_dir / f"repeat_{repeat}.csv"
        out_df.to_csv(csv_path, index=False)

        n_val = int((out_df["fold"] == 0).sum())
        n_train = int((out_df["fold"] == 1).sum())
        val_gt = Counter(out_df.loc[out_df["fold"] == 0, "gt_binary"])
        train_gt = Counter(out_df.loc[out_df["fold"] == 1, "gt_binary"])
        entry = {
            "repeat": repeat,
            "csv": str(csv_path),
            "n_train": n_train,
            "n_val": n_val,
            "train_gt": {str(k): v for k, v in sorted(train_gt.items())},
            "val_gt": {str(k): v for k, v in sorted(val_gt.items())},
        }
        manifest["repeats"].append(entry)
        print(
            f"  repeat {repeat}: train={n_train} val={n_val}  "
            f"val GT={dict(val_gt)}  → {csv_path.name}"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[holdout] wrote {args.n_repeats} splits under {out_dir}")
    print(f"[holdout] manifest → {manifest_path}")


if __name__ == "__main__":
    main()
