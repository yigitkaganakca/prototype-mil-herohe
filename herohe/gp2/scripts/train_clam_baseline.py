"""Train CLAM-MB (Valieris-style) on Virchow2 feature bags with the same folds as PhenoHER2.

Loss: ``bag_weight * CE(bag) + (1 - bag_weight) * instance_loss`` (default bag_weight=0.7).
Early stopping and best checkpoint use ``val_loss`` = mean bag-level cross-entropy on val
unless ``--select_on val_macro_f1``.

Example (fold 0 only, MPS):

    python herohe/gp2/scripts/train_clam_baseline.py \\
        --features_dir herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2 \\
        --labels_csv "herohe/Training (ground truth).csv" \\
        --folds_csv herohe/gp2/data/folds_v1.csv \\
        --out_dir herohe/gp2/runs/clam_virchow2_fold0 \\
        --only_fold 0 --device mps --epochs 200 --patience 15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models.clam_mb import CLAM_MB
from herohe.gp2.models.dataset import HerohePatchBagDataset, collate_single_bag


def pick_device(name: str) -> torch.device:
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[clam] MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[clam] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def _evaluate_clam(
    model: CLAM_MB,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> dict:
    model.eval()
    all_y, all_p, all_probs = [], [], []
    val_ce_sum = 0.0
    val_ce_n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            h = x.squeeze(0)
            logits, Y_prob, Y_hat, _, _ = model(h, label=None, instance_eval=False)
            ce = F.cross_entropy(logits, y.view(-1))
            if torch.isfinite(ce):
                val_ce_sum += float(ce.item())
                val_ce_n += 1
            probs = Y_prob[0].detach().cpu().numpy()
            pred = int(Y_hat[0, 0].item())
            all_y.append(int(y.item()))
            all_p.append(pred)
            all_probs.append(probs)
    y = np.array(all_y)
    p = np.array(all_p)
    P = np.stack(all_probs, axis=0)
    if not np.isfinite(P).all():
        n_bad = int((~np.isfinite(P).all(axis=-1)).sum())
        print(
            f"[eval] WARNING: {n_bad}/{len(P)} val slides produced non-finite probs; "
            f"replacing with uniform 1/{num_classes}."
        )
        bad_rows = ~np.isfinite(P).all(axis=-1)
        P[bad_rows] = 1.0 / num_classes
        p = np.where(bad_rows, P.argmax(axis=-1), p)
    macro_f1 = f1_score(y, p, average="macro", labels=list(range(num_classes)), zero_division=0)
    mask01 = (y == 0) | (y == 1)
    auc01 = float("nan")
    if mask01.sum() >= 2 and len(np.unique(y[mask01])) == 2:
        denom = np.clip(P[mask01, 0] + P[mask01, 1], 1e-6, None)
        score_low = P[mask01, 1] / denom
        if np.isfinite(score_low).all():
            try:
                auc01 = float(roc_auc_score((y[mask01] == 1).astype(int), score_low))
            except ValueError:
                pass
    auc_3p = float("nan")
    if len(np.unique(y)) > 1 and np.isfinite(P[:, 3]).all():
        try:
            auc_3p = float(roc_auc_score((y == 3).astype(int), P[:, 3]))
        except ValueError:
            pass
    return {
        "macro_f1": float(macro_f1),
        "auc_0_vs_low": float(auc01),
        "auc_3p_vs_rest": float(auc_3p),
        "val_loss": float(val_ce_sum / val_ce_n) if val_ce_n > 0 else float("nan"),
        "n": int(len(y)),
    }


def train_one_fold(
    fold_idx: int,
    full_dataset: HerohePatchBagDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    out_dir: Path,
    device: torch.device,
) -> dict:
    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds = Subset(full_dataset, val_idx.tolist())

    model = CLAM_MB(
        gate=True,
        size_arg="small",
        dropout=args.dropout,
        k_sample=args.k_sample,
        n_classes=args.n_classes,
        subtyping=True,
        embed_dim=args.feature_dim,
    ).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single_bag)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag)

    log_path = fold_dir / "log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_macro_f1",
                "val_auc_0_vs_low",
                "val_auc_3p_vs_rest",
                "lr",
            ]
        )

    select_on = args.select_on
    best_score = float("inf") if select_on == "val_loss" else -float("inf")
    best_is_lower_better = select_on == "val_loss"
    best_metrics: dict = {}
    best_epoch = -1
    epochs_since_best = 0
    bw = args.bag_weight

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            h = x.squeeze(0)
            optim.zero_grad(set_to_none=True)
            logits, _, _, _, results = model(h, label=y, instance_eval=True)
            ce_bag = F.cross_entropy(logits, y.view(-1))
            inst = results.get("instance_loss", 0.0)
            if isinstance(inst, torch.Tensor):
                total = bw * ce_bag + (1.0 - bw) * inst
            else:
                total = bw * ce_bag
            if not torch.isfinite(total):
                print(f"[fold {fold_idx}] epoch {epoch}: non-finite loss, skipping step")
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running += float(total.item())
            n_seen += 1
        sched.step()
        train_loss = running / max(n_seen, 1)
        metrics = _evaluate_clam(model, val_loader, device, num_classes=args.n_classes)
        with open(log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{metrics['val_loss']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    f"{metrics['auc_0_vs_low']:.4f}",
                    f"{metrics['auc_3p_vs_rest']:.4f}",
                    f"{optim.param_groups[0]['lr']:.2e}",
                ]
            )
        print(
            f"[fold {fold_idx}] epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={metrics['val_loss']:.4f}  "
            f"val_macroF1={metrics['macro_f1']:.4f}  "
            f"val_AUC(0vsLow)={metrics['auc_0_vs_low']:.4f}  "
            f"val_AUC(3+)={metrics['auc_3p_vs_rest']:.4f}"
        )
        score = metrics["val_loss"] if select_on == "val_loss" else metrics["macro_f1"]
        improved = (score < best_score) if best_is_lower_better else (score > best_score)
        if improved and np.isfinite(score):
            best_score = score
            best_metrics = {**metrics, "epoch": epoch}
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metrics": best_metrics,
                    "args": vars(args),
                },
                fold_dir / "best.pt",
            )
        else:
            epochs_since_best += 1
            if args.patience > 0 and epochs_since_best >= args.patience:
                print(
                    f"[fold {fold_idx}] early stopping at epoch {epoch}: "
                    f"no {select_on} improvement for {args.patience} epochs "
                    f"(best={best_score:.4f} @ epoch {best_epoch})"
                )
                break

    print(
        f"[fold {fold_idx}] best_{select_on}={best_score:.4f}  "
        f"best_metrics={best_metrics}"
    )
    return {
        "macro_f1": best_metrics.get("macro_f1", float("nan")),
        "auc_0_vs_low": best_metrics.get("auc_0_vs_low", float("nan")),
        "auc_3p_vs_rest": best_metrics.get("auc_3p_vs_rest", float("nan")),
        "val_loss": best_metrics.get("val_loss", float("nan")),
        "best_epoch": best_epoch,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument(
        "--only_fold",
        type=int,
        default=None,
        help="Train only this fold index (0-based).",
    )
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument(
        "--select_on",
        choices=["val_loss", "val_macro_f1"],
        default="val_loss",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--feature_dim", type=int, default=2560)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--k_sample", type=int, default=8)
    ap.add_argument("--bag_weight", type=float, default=0.7)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[clam] device={device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    full = HerohePatchBagDataset(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        max_patches=args.max_patches,
        seed=args.seed,
        return_coords=False,
    )
    print(
        f"[clam] dataset size={len(full)}; class counts={full.class_counts().tolist()}"
    )
    y = full.labels()

    if args.folds_csv is not None:
        import pandas as pd

        fdf = pd.read_csv(args.folds_csv)
        fdf["slide_id"] = fdf["slide_id"].astype(str)
        sid_to_pos = {sid: i for i, sid in enumerate(full.slide_ids())}
        unknown = [sid for sid in fdf["slide_id"] if sid not in sid_to_pos]
        if unknown:
            raise ValueError(
                f"folds_csv lists {len(unknown)} slide_id(s) not in dataset, e.g. {unknown[:5]}"
            )
        missing_in_folds = sorted(set(full.slide_ids()) - set(fdf["slide_id"]))
        if missing_in_folds:
            raise ValueError(
                f"{len(missing_in_folds)} dataset slides missing from {args.folds_csv}; "
                f"e.g. {missing_in_folds[:5]}"
            )
        n_folds = int(fdf["fold"].max()) + 1
        if args.n_folds != n_folds:
            print(f"[clam] overriding --n_folds={args.n_folds} with {n_folds} from folds_csv")
        splits = []
        for k in range(n_folds):
            va_sids = fdf.loc[fdf["fold"] == k, "slide_id"].tolist()
            va_idx = np.array([sid_to_pos[s] for s in va_sids], dtype=int)
            tr_idx = np.array(
                [i for i in range(len(full)) if i not in set(va_idx.tolist())],
                dtype=int,
            )
            splits.append((tr_idx, va_idx))
        print(f"[clam] fold assignments from {args.folds_csv} ({n_folds} folds)")
    else:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.zeros(len(y)), y))

    if args.only_fold is not None:
        if not (0 <= args.only_fold < len(splits)):
            raise ValueError(f"--only_fold={args.only_fold} out of range [0, {len(splits)})")
        _splits_iter = [(args.only_fold, splits[args.only_fold])]
        print(f"[clam] running ONLY fold {args.only_fold}")
    else:
        _splits_iter = list(enumerate(splits))

    fold_metrics: list[dict] = []
    for fold_idx, (tr_idx, va_idx) in _splits_iter:
        print(f"\n=== CLAM-MB fold {fold_idx}: train={len(tr_idx)} val={len(va_idx)} ===")
        m = train_one_fold(
            fold_idx=fold_idx,
            full_dataset=full,
            train_idx=tr_idx,
            val_idx=va_idx,
            args=args,
            out_dir=out_dir,
            device=device,
        )
        fold_metrics.append({"fold": fold_idx, **m})

    summary = {
        "folds": fold_metrics,
        "macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "macro_f1_std": float(np.std([m["macro_f1"] for m in fold_metrics])),
        "auc_0_vs_low_mean": float(np.nanmean([m["auc_0_vs_low"] for m in fold_metrics])),
        "auc_3p_vs_rest_mean": float(np.nanmean([m["auc_3p_vs_rest"] for m in fold_metrics])),
        "args": vars(args),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[clam] summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
