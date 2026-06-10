"""Re-score OOF predictions after excluding low-patch slides.

Uses per-slide patch counts from Virchow2 ``.h5`` feature bags and existing
``oof_predictions.csv`` (or generates binary OOF from ``best.pt`` checkpoints).

Example (existing OOF only, no torch needed for CSV path)::

    python herohe/gp2/scripts/patch_filter_oof_ablation.py \\
        --oof_csv herohe/gp2/runs/clam_virchow2_5fold/oof_predictions.csv \\
        --features_dir herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2 \\
        --thresholds 0 256 500 1000

Generate binary OOF from PhenoBIN / ABMIL run dir (requires torch)::

    python herohe/gp2/scripts/patch_filter_oof_ablation.py \\
        --run_dir herohe/gp2/runs/phenobin_5fold_parallel \\
        --features_dir herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2 \\
        --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))


def patch_counts_from_h5(features_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fp in sorted(features_dir.glob("*.h5")):
        sid = fp.stem
        with h5py.File(fp, "r") as f:
            counts[sid] = int(f["features"].shape[0])
    return counts


def metrics_4class(y: np.ndarray, p: np.ndarray, P: np.ndarray) -> dict:
    macro_f1 = f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    acc = accuracy_score(y, p)
    out = {"macro_f1": float(macro_f1), "acc": float(acc), "n": int(len(y))}
    if len(np.unique(y)) > 1 and P.shape[1] >= 4:
        mask01 = (y == 0) | (y == 1)
        if mask01.sum() >= 2 and len(np.unique(y[mask01])) == 2:
            denom = np.clip(P[mask01, 0] + P[mask01, 1], 1e-6, None)
            score = P[mask01, 1] / denom
            try:
                out["auc_0_vs_low"] = float(roc_auc_score((y[mask01] == 1).astype(int), score))
            except ValueError:
                out["auc_0_vs_low"] = float("nan")
    return out


def metrics_binary(y: np.ndarray, p: np.ndarray, prob_pos: np.ndarray) -> dict:
    macro_f1 = f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)
    acc = accuracy_score(y, p)
    auc = float("nan")
    if len(np.unique(y)) == 2 and np.isfinite(prob_pos).all():
        try:
            auc = float(roc_auc_score(y, prob_pos))
        except ValueError:
            pass
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(macro_f1),
        "acc": float(acc),
        "auc_positive": auc,
        "n": int(len(y)),
        "n_neg": int((y == 0).sum()),
        "n_pos": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_oof_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["slide_id"] = df["slide_id"].astype(str)
    return df


def filter_and_score(
    df: pd.DataFrame,
    patch_counts: dict[str, int],
    min_patches: int,
    task: str,
    gt_binary: pd.Series | None = None,
) -> dict:
    df = df.copy()
    df["n_patches"] = df["slide_id"].map(lambda s: patch_counts.get(str(s), 0))
    sub = df[df["n_patches"] >= min_patches].copy()
    if sub.empty:
        return {"min_patches": min_patches, "n": 0}

    y = sub["y_true"].astype(int).to_numpy()
    p = sub["y_pred"].astype(int).to_numpy()

    if task == "binary":
        if "p1" in sub.columns and sub["p1"].notna().all():
            prob = sub["p1"].to_numpy(dtype=np.float64)
        elif {"p0", "p1"}.issubset(sub.columns):
            prob = sub["p1"].to_numpy(dtype=np.float64)
        else:
            raise ValueError("Binary task needs p1 column in OOF CSV")
        m = metrics_binary(y, p, prob)
    elif task == "4class":
        P = sub[["p0", "p1", "p2", "p3"]].to_numpy(dtype=np.float64)
        m = metrics_4class(y, p, P)
    else:
        raise ValueError(task)

    m["min_patches"] = min_patches
    m["excluded"] = int(len(df) - len(sub))
    if gt_binary is not None:
        sub = sub.merge(gt_binary, on="slide_id", how="left")
        yb = sub["gt_binary"].astype(int).to_numpy()
        if "p1" in sub.columns and len(np.unique(yb)) == 2:
            if sub["p1"].max() <= 1.0 and sub[["p0", "p1"]].shape[1] == 2 and "p2" not in sub.columns:
                prob = sub["p1"].to_numpy(dtype=np.float64)
            else:
                prob = (sub["p2"] + sub["p3"]).to_numpy(dtype=np.float64)
            m["binary_hint"] = metrics_binary(yb, (prob >= 0.5).astype(int), prob)
    return m


def generate_binary_oof(run_dir: Path, device_name: str) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader, Subset

    from herohe.gp2.models.dataset import HerohePatchBagDataset, collate_single_bag

    fold_dirs = sorted(run_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* under {run_dir}")

    blob0 = torch.load(fold_dirs[0] / "best.pt", map_location="cpu")
    targs = blob0["args"]
    model_type = blob0.get("model_type", "mil_baseline")
    folds_csv = targs.get("folds_csv")
    if not folds_csv:
        raise ValueError("Checkpoint args missing folds_csv")

    label_mode = targs.get("label_mode", "gt_binary")
    full = HerohePatchBagDataset(
        features_dir=targs["features_dir"],
        labels_csv=targs["labels_csv"],
        label_mode=label_mode,
        max_patches=targs.get("max_patches", 4096),
        seed=targs.get("seed", 0),
        return_coords=True,
    )
    fdf = pd.read_csv(folds_csv)
    fdf["slide_id"] = fdf["slide_id"].astype(str)
    sid_to_pos = {sid: i for i, sid in enumerate(full.slide_ids())}
    n_folds = int(fdf["fold"].max()) + 1

    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows: list[dict] = []
    for fold_idx in range(n_folds):
        ckpt_path = run_dir / f"fold_{fold_idx}" / "best.pt"
        blob = torch.load(ckpt_path, map_location=device)
        targs = blob["args"]

        if model_type == "PhenoHER2Binary" or "PhenoHER2Binary" in str(type(blob.get("config"))):
            from dataclasses import fields

            from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig

            raw = blob["config"]
            names = {f.name for f in fields(PhenoHER2BinaryConfig)}
            cfg = PhenoHER2BinaryConfig(**{k: raw[k] for k in names if k in raw})
            model = PhenoHER2Binary(cfg).to(device)
            model.load_state_dict(blob["model_state"], strict=True)
            model.eval()

            def predict_batch(batch):
                x = batch["features"].to(device)
                c = batch["coords"].to(device) if batch["coords"] is not None else None
                out = model.predict(x, coords=c, apply_calibration=True)
                probs = out["probs"][0].detach().cpu().numpy()
                pred = int(out["pred_class"].item())
                return pred, probs

        else:
            from herohe.gp2.models.abmil import ABMIL, ABMILConfig
            from herohe.gp2.models.clam_mb import CLAM_MB
            from herohe.gp2.models.transmil import TransMIL, TransMILConfig

            agg = blob.get("aggregator", targs.get("aggregator", "abmil")).lower()
            nc = int(targs.get("num_classes", 2))
            fd = int(targs.get("feature_dim", 2560))
            if agg == "abmil":
                model = ABMIL(
                    ABMILConfig(
                        in_dim=fd,
                        hidden_dim=int(targs.get("abmil_hidden", 512)),
                        attn_dim=int(targs.get("abmil_attn", 256)),
                        num_classes=nc,
                        dropout=float(targs.get("abmil_dropout", 0.4)),
                    )
                ).to(device)
            elif agg == "clam":
                model = CLAM_MB(
                    gate=True,
                    size_arg="small",
                    dropout=float(targs.get("clam_dropout", 0.4)),
                    k_sample=int(targs.get("k_sample", 8)),
                    n_classes=nc,
                    subtyping=True,
                    embed_dim=fd,
                ).to(device)
            elif agg == "transmil":
                model = TransMIL(
                    TransMILConfig(
                        in_dim=fd,
                        d_model=int(targs.get("trans_d_model", 512)),
                        n_heads=int(targs.get("trans_heads", 8)),
                        n_layers=int(targs.get("trans_layers", 2)),
                        dropout=float(targs.get("trans_dropout", 0.25)),
                        num_classes=nc,
                        use_coord_pe=True,
                    )
                ).to(device)
            else:
                raise ValueError(f"Unknown aggregator {agg}")
            model.load_state_dict(blob["model_state"], strict=True)
            model.eval()

            def predict_batch(batch):
                x = batch["features"].to(device)
                y = batch["label"]
                if agg == "clam":
                    h = x.squeeze(0)
                    logits, _, _, _, _ = model(h, label=y, instance_eval=False)
                elif agg == "transmil":
                    c = batch["coords"].to(device) if batch["coords"] is not None else None
                    logits = model(x, coords=c)["logits"]
                else:
                    logits = model(x)["logits"]
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                pred = int(probs.argmax())
                return pred, probs

        va_sids = fdf.loc[fdf["fold"] == fold_idx, "slide_id"].tolist()
        va_idx = [sid_to_pos[s] for s in va_sids]
        val_ds = Subset(full, va_idx)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag)

        with torch.no_grad():
            for batch, i in zip(val_loader, va_idx):
                sid = full.slide_ids()[i]
                y = int(batch["label"].item())
                pred, probs = predict_batch(batch)
                row = {
                    "slide_id": sid,
                    "oof_fold": fold_idx,
                    "y_true": y,
                    "y_pred": pred,
                    "p0": float(probs[0]),
                    "p1": float(probs[1]),
                }
                if len(probs) > 2:
                    row["p2"] = float(probs[2])
                    row["p3"] = float(probs[3])
                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof_csv", type=Path, default=None)
    ap.add_argument("--run_dir", type=Path, default=None, help="Generate OOF from best.pt if --oof_csv omitted")
    ap.add_argument("--features_dir", type=Path, required=True)
    ap.add_argument("--folds_csv", type=Path, default=Path("herohe/gp2/data/folds_v1.csv"))
    ap.add_argument("--thresholds", type=int, nargs="+", default=[0, 256, 500, 1000])
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    if args.oof_csv is None and args.run_dir is None:
        ap.error("Provide --oof_csv and/or --run_dir")

    patch_counts = patch_counts_from_h5(args.features_dir.resolve())

    gt_binary = None
    if args.folds_csv.is_file():
        fdf = pd.read_csv(args.folds_csv)
        fdf["slide_id"] = fdf["slide_id"].astype(str)
        if "gt_binary" in fdf.columns:
            gt_binary = fdf[["slide_id", "gt_binary"]]

    sources: list[tuple[str, pd.DataFrame, str]] = []
    if args.oof_csv is not None:
        df = load_oof_csv(args.oof_csv.resolve())
        task = "binary" if "p2" not in df.columns else "4class"
        sources.append((str(args.oof_csv), df, task))

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        oof_path = run_dir / "oof_predictions.csv"
        if oof_path.is_file() and args.oof_csv is None:
            df = load_oof_csv(oof_path)
            task = "binary" if "p2" not in df.columns else "4class"
            sources.append((str(oof_path), df, task))
        elif args.oof_csv is None:
            print(f"[gen] Generating OOF from {run_dir} ...")
            df = generate_binary_oof(run_dir, args.device)
            out_csv = run_dir / "oof_predictions.csv"
            df.to_csv(out_csv, index=False)
            print(f"[gen] Wrote {out_csv} ({len(df)} rows)")
            task = "binary" if "p2" not in df.columns else "4class"
            sources.append((str(run_dir), df, task))

    all_results = {}
    for name, df, task in sources:
        print(f"\n=== {name} ({task}) ===")
        res = []
        for thr in args.thresholds:
            m = filter_and_score(df, patch_counts, thr, task, gt_binary)
            res.append(m)
            if m.get("n", 0) == 0:
                print(f"  >={thr}: n=0 (excluded all)")
                continue
            line = (
                f"  >={thr}: n={m['n']} excluded={m['excluded']} "
                f"macroF1={m['macro_f1']:.3f} acc={m['acc']:.3f}"
            )
            if task == "binary":
                line += f" AUC={m['auc_positive']:.3f}"
            else:
                if "auc_0_vs_low" in m:
                    line += f" AUC01={m['auc_0_vs_low']:.3f}"
            if "binary_hint" in m:
                bh = m["binary_hint"]
                line += (
                    f" | binary_hint: AUC={bh['auc_positive']:.3f} "
                    f"macroF1={bh['macro_f1']:.3f} (n={bh['n']})"
                )
            print(line)
        all_results[name] = res

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
