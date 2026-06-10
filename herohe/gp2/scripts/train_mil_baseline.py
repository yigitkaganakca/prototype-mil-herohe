"""Train AB-MIL, CLAM-MB, or TransMIL on Virchow2 (or ResNet50) feature bags — 2-class or 4-class.

Uses the same ``HerohePatchBagDataset`` protocol as ``train_phenobin_mil.py``:
``--folds_csv`` with ``slide_id``, ``fold``, optional ``StratifiedKFold`` on labels.

Aggregators
-----------
* **abmil** — Ilse et al. 2018 gated attention (`AttentionDeepMIL` adapter in ``herohe.gp2.vendor``).
* **clam** — Lu et al. 2021 CLAM multi-branch (`CLAM_MB`` from ``mahmoodlab/CLAM``).
* **transmil** — Shao et al. 2021 TransMIL (`szc19990412/TransMIL`` TransLayer + PPEG).
* **attnmisl** — Yao et al. 2020 DeepAttnMISL: MI-FCN per phenotype + attention over C phenotypes.

All aggregators import official upstream model code via ``herohe.gp2.vendor.factory``;
see ``gp2/vendor/VENDOR_VERSIONS.md`` for pinned commits.

**Class imbalance (bag CE):** by default uses the same **Effective Number of Samples**
weights (Cui et al., 2019) as ``PhenoHER2Loss`` / ``PhenoHER2BinaryLoss``, computed from
**training-fold** label counts only, applied to **bag-level** ``cross_entropy`` for ABMIL,
TransMIL, and CLAM’s bag term (instance branch unchanged). Disable with
``--ce_class_weights none``.

Examples
--------
4-class IHC, CLAM (same as legacy ``train_clam_baseline.py``):

    python herohe/gp2/scripts/train_mil_baseline.py --aggregator clam --num_classes 4 \\
        --label_mode ihc --features_dir .../features_virchow2 \\
        --labels_csv "herohe/Training (ground truth).csv" --folds_csv herohe/gp2/data/folds_v1.csv \\
        --out_dir herohe/gp2/runs/clam_grid --device mps

Binary ISH (gt_binary), AB-MIL:

    python herohe/gp2/scripts/train_mil_baseline.py --aggregator abmil --num_classes 2 \\
        --label_mode gt_binary --features_dir .../features_virchow2 \\
        --labels_csv "herohe/Training (ground truth).csv" --folds_csv herohe/gp2/data/folds_v1.csv \\
        --out_dir herohe/gp2/runs/abmil_bin --device mps

TransMIL (needs coords in h5):

    python herohe/gp2/scripts/train_mil_baseline.py --aggregator transmil --num_classes 4 \\
        --label_mode ihc --features_dir .../features_virchow2 ... --out_dir .../transmil_4cls

TRIDENT **ResNet50** bags (binary): same as Virchow2 but point ``--features_dir`` to
``.../features_resnet50`` and set ``--feature_dim 1024`` (Virchow2 remains 2560).
See ``gp2/scripts/prep_binary_resnet50_runs.sh`` for a local path check and command templates.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.scripts.mil_calibration import apply_temperature, fit_temperature_mil
from herohe.gp2.scripts.training_diagnostics import analyze_fold_log
from herohe.gp2.scripts.metrics_utils import macro_ovr_auc
from herohe.gp2.models.dataset import HerohePatchBagDataset, collate_single_bag
from herohe.gp2.models.losses import effective_number_class_weights
from herohe.gp2.prototype_discovery import load_prototype_checkpoint
from herohe.gp2.vendor.adapters.abmil import attention_entropy
from herohe.gp2.vendor.factory import build_baseline_model


def pick_device(name: str) -> torch.device:
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[mil] MPS unavailable; CPU.")
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[mil] CUDA unavailable; CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def make_model(
    aggregator: str,
    num_classes: int,
    feature_dim: int,
    *,
    clam_dropout: float,
    clam_k_sample: int,
    bag_weight: float,
    trans_d_model: int,
    trans_layers: int,
    trans_heads: int,
    trans_dropout: float,
    abmil_hidden: int,
    abmil_attn: int,
    abmil_dropout: float,
    attnmisl_cluster_num: int = 8,
    attnmisl_dropout: float = 0.5,
    prototype_centers=None,
) -> nn.Module:
    del bag_weight  # CLAM training loop only
    return build_baseline_model(
        aggregator,
        num_classes,
        feature_dim,
        clam_dropout=clam_dropout,
        clam_k_sample=clam_k_sample,
        trans_d_model=trans_d_model,
        trans_layers=trans_layers,
        trans_heads=trans_heads,
        trans_dropout=trans_dropout,
        abmil_hidden=abmil_hidden,
        abmil_attn=abmil_attn,
        abmil_dropout=abmil_dropout,
        attnmisl_cluster_num=attnmisl_cluster_num,
        attnmisl_dropout=attnmisl_dropout,
        prototype_centers=prototype_centers,
    )


def _ce_kw(ce_weight: torch.Tensor | None, label_smoothing: float = 0.0) -> dict:
    kw: dict = {}
    if ce_weight is not None:
        kw["weight"] = ce_weight
    if label_smoothing > 0.0:
        kw["label_smoothing"] = label_smoothing
    return kw


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    aggregator: str,
    num_classes: int,
    ce_weight: torch.Tensor | None,
    label_smoothing: float = 0.0,
    temperature: float = 1.0,
) -> dict:
    model.eval()
    all_y, all_p, all_probs = [], [], []
    val_ce_sum = 0.0
    val_ce_n = 0
    agg = aggregator.lower()
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            h = x.squeeze(0)
            if agg == "clam":
                logits, _, Y_hat, _, _ = model(h, label=None, instance_eval=False)
                logits = apply_temperature(logits, temperature)
                probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                pred = int(logits.argmax(dim=-1).item())
                ce = F.cross_entropy(logits, y.view(-1), **_ce_kw(ce_weight, label_smoothing))
            elif agg == "abmil" or agg == "attnmisl":
                out = model(x)
                logits = apply_temperature(out["logits"], temperature)
                probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                pred = int(logits.argmax(dim=-1).item())
                ce = F.cross_entropy(logits, y.view(-1), **_ce_kw(ce_weight, label_smoothing))
            else:
                c = batch["coords"].to(device) if batch["coords"] is not None else None
                out = model(x, coords=c)
                logits = apply_temperature(out["logits"], temperature)
                probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                pred = int(logits.argmax(dim=-1).item())
                ce = F.cross_entropy(logits, y.view(-1), **_ce_kw(ce_weight, label_smoothing))
            if torch.isfinite(ce):
                val_ce_sum += float(ce.item())
                val_ce_n += 1
            all_y.append(int(y.item()))
            all_p.append(pred)
            all_probs.append(probs)
    y = np.array(all_y)
    p = np.array(all_p)
    P = np.stack(all_probs, axis=0)
    if not np.isfinite(P).all():
        bad = ~np.isfinite(P).all(axis=-1)
        P[bad] = 1.0 / num_classes
        p = np.where(bad, P.argmax(axis=-1), p)
    macro_f1 = f1_score(
        y, p, average="macro", labels=list(range(num_classes)), zero_division=0
    )
    out_metrics: dict = {
        "macro_f1": float(macro_f1),
        "val_loss": float(val_ce_sum / val_ce_n) if val_ce_n > 0 else float("nan"),
        "n": int(len(y)),
    }
    if num_classes == 2:
        auc = float("nan")
        if len(np.unique(y)) == 2 and np.isfinite(P[:, 1]).all():
            try:
                auc = float(roc_auc_score(y, P[:, 1]))
            except ValueError:
                pass
        out_metrics["auc_positive"] = auc
    elif num_classes == 3:
        out_metrics["macro_auroc"] = macro_ovr_auc(y, P, num_classes)
    else:
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
        out_metrics["auc_0_vs_low"] = auc01
        auc_3p = float("nan")
        if len(np.unique(y)) > 1 and np.isfinite(P[:, 3]).all():
            try:
                auc_3p = float(roc_auc_score((y == 3).astype(int), P[:, 3]))
            except ValueError:
                pass
        out_metrics["auc_3p_vs_rest"] = auc_3p
    return out_metrics


def train_one_fold(
    fold_idx: int,
    train_dataset: HerohePatchBagDataset,
    val_dataset: HerohePatchBagDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    out_dir: Path,
    device: torch.device,
) -> dict:
    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ds = Subset(train_dataset, train_idx.tolist())
    val_ds = Subset(val_dataset, val_idx.tolist())

    proto_centers = None
    if args.aggregator.lower() == "attnmisl":
        if not args.prototypes:
            raise ValueError("--prototypes is required when --aggregator attnmisl")
        proto_blob = load_prototype_checkpoint(args.prototypes)
        proto_centers = proto_blob["centers"]
        print(
            f"[fold {fold_idx}] AttnMISL prototypes: {args.prototypes} "
            f"K={proto_blob.get('K', proto_centers.shape[0])}"
        )

    model = make_model(
        args.aggregator,
        args.num_classes,
        args.feature_dim,
        clam_dropout=args.clam_dropout,
        clam_k_sample=args.k_sample,
        bag_weight=args.bag_weight,
        trans_d_model=args.trans_d_model,
        trans_layers=args.trans_layers,
        trans_heads=args.trans_heads,
        trans_dropout=args.trans_dropout,
        abmil_hidden=args.abmil_hidden,
        abmil_attn=args.abmil_attn,
        abmil_dropout=args.abmil_dropout,
        attnmisl_cluster_num=args.attnmisl_cluster_num,
        attnmisl_dropout=args.attnmisl_dropout,
        prototype_centers=proto_centers,
    ).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_single_bag
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag
    )

    train_labels = train_dataset.labels()[train_idx]
    class_counts = np.bincount(train_labels, minlength=args.num_classes)
    if args.ce_class_weights == "effective":
        ce_w = effective_number_class_weights(class_counts, beta=args.cb_beta).to(device)
        print(
            f"[fold {fold_idx}] CE weights (effective-n, β={args.cb_beta}): "
            f"{ce_w.detach().cpu().tolist()}  train_counts={class_counts.tolist()}"
        )
    else:
        ce_w = None

    if args.num_classes == 2:
        log_cols = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_auc_positive", "lr"]
    elif args.num_classes == 3:
        log_cols = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_macro_f1",
            "val_macro_auroc",
            "lr",
        ]
    else:
        log_cols = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_macro_f1",
            "val_auc_0_vs_low",
            "val_auc_3p_vs_rest",
            "lr",
        ]
    log_path = fold_dir / "log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(log_cols)

    select_on = args.select_on
    print(f"[fold {fold_idx}] checkpoint selection: select_on={select_on}")
    best_score = float("inf") if select_on == "val_loss" else -float("inf")
    best_is_lower = select_on == "val_loss"
    best_metrics: dict = {}
    best_epoch = -1
    epochs_since_best = 0
    min_sel_ep = int(getattr(args, "min_epochs_for_selection", 0) or 0)
    agg = args.aggregator.lower()
    bw = args.bag_weight

    ce_kw = _ce_kw(ce_w, args.label_smoothing)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            h = x.squeeze(0)
            optim.zero_grad(set_to_none=True)
            if agg == "clam":
                logits, _, _, _, results = model(h, label=y, instance_eval=True)
                ce_bag = F.cross_entropy(logits, y.view(-1), **ce_kw)
                inst = results.get("instance_loss", 0.0)
                if isinstance(inst, torch.Tensor):
                    total = bw * ce_bag + (1.0 - bw) * inst
                else:
                    total = bw * ce_bag
            elif agg == "abmil" or agg == "attnmisl":
                out = model(x)
                ce = F.cross_entropy(out["logits"], y.view(-1), **ce_kw)
                if agg == "abmil" and args.w_attn_entropy > 0.0:
                    ent = attention_entropy(out["attn"])
                    total = ce - args.w_attn_entropy * ent
                else:
                    total = ce
            else:
                c = batch["coords"].to(device) if batch["coords"] is not None else None
                out = model(x, coords=c)
                total = F.cross_entropy(out["logits"], y.view(-1), **ce_kw)
            if not torch.isfinite(total):
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running += float(total.item())
            n_seen += 1
        sched.step()
        train_loss = running / max(n_seen, 1)
        metrics = evaluate(
            model, val_loader, device, agg, args.num_classes, ce_w, args.label_smoothing
        )
        row = [
            epoch,
            f"{train_loss:.4f}",
            f"{metrics['val_loss']:.4f}",
            f"{metrics['macro_f1']:.4f}",
        ]
        if args.num_classes == 2:
            row.append(f"{metrics['auc_positive']:.4f}")
        elif args.num_classes == 3:
            row.append(f"{metrics['macro_auroc']:.4f}")
        else:
            row.extend(
                [
                    f"{metrics['auc_0_vs_low']:.4f}",
                    f"{metrics['auc_3p_vs_rest']:.4f}",
                ]
            )
        row.append(f"{optim.param_groups[0]['lr']:.2e}")
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        if args.num_classes == 2:
            print(
                f"[fold {fold_idx}] ep {epoch}/{args.epochs} train={train_loss:.4f} "
                f"val_loss={metrics['val_loss']:.4f} macroF1={metrics['macro_f1']:.4f} "
                f"AUCpos={metrics['auc_positive']:.4f}"
            )
        elif args.num_classes == 3:
            print(
                f"[fold {fold_idx}] ep {epoch}/{args.epochs} train={train_loss:.4f} "
                f"val_loss={metrics['val_loss']:.4f} macroF1={metrics['macro_f1']:.4f} "
                f"macroAUROC={metrics['macro_auroc']:.4f}"
            )
        else:
            print(
                f"[fold {fold_idx}] ep {epoch}/{args.epochs} train={train_loss:.4f} "
                f"val_loss={metrics['val_loss']:.4f} macroF1={metrics['macro_f1']:.4f} "
                f"AUC01={metrics['auc_0_vs_low']:.4f} AUC3p={metrics['auc_3p_vs_rest']:.4f}"
            )
        if select_on == "val_loss":
            score = metrics["val_loss"]
        elif select_on == "val_auc_positive":
            score = metrics.get("auc_positive", float("nan"))
        elif select_on == "val_macro_auroc":
            score = metrics.get("macro_auroc", float("nan"))
        else:
            score = metrics["macro_f1"]
        improved = (score < best_score) if best_is_lower else (score > best_score)
        eligible = min_sel_ep <= 0 or epoch >= min_sel_ep
        if improved and np.isfinite(score) and eligible:
            best_score = score
            best_metrics = {**metrics, "epoch": epoch}
            best_epoch = epoch
            epochs_since_best = 0
            print(f"[fold {fold_idx}] *** new best {select_on}={score:.4f} @ ep {epoch} ***")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metrics": best_metrics,
                    "args": vars(args),
                    "aggregator": agg,
                    "class_counts_train": class_counts.tolist(),
                    "ce_weights": ce_w.detach().cpu() if ce_w is not None else None,
                },
                fold_dir / "best.pt",
            )
        else:
            epochs_since_best += 1
            if (
                args.patience > 0
                and epoch >= args.min_epochs
                and epochs_since_best >= args.patience
            ):
                print(f"[fold {fold_idx}] early stop @ {epoch} (best ep {best_epoch})")
                break

    if args.num_classes == 2:
        auc_col = "val_auc_positive"
    elif args.num_classes == 3:
        auc_col = "val_macro_auroc"
    else:
        auc_col = "val_auc_3p_vs_rest"
    diag = analyze_fold_log(
        log_path,
        auc_col=auc_col,
        num_classes=args.num_classes,
        min_epochs_for_selection=min_sel_ep,
        select_on=select_on,
    )
    print(f"[fold {fold_idx}] overfit_diag={diag}")

    ckpt_path = fold_dir / "best.pt"
    calib_T = 1.0
    metrics_calib: dict = {}
    if ckpt_path.is_file() and not args.no_calibrate and args.num_classes in (2, 3):
        blob = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(blob["model_state"])
        print(f"[fold {fold_idx}] reloading best (epoch {best_epoch}) for calibration")
        calib_T = fit_temperature_mil(model, val_loader, device, agg)
        print(f"[fold {fold_idx}] fitted temperature T={calib_T:.4f}")
        metrics_calib = evaluate(
            model,
            val_loader,
            device,
            agg,
            args.num_classes,
            ce_w,
            args.label_smoothing,
            temperature=calib_T,
        )
        blob["calibration_temperature"] = calib_T
        blob["metrics_calibrated"] = metrics_calib
        torch.save(blob, ckpt_path)

    print(f"[fold {fold_idx}] best_{select_on}={best_score:.4f} best={best_metrics}")
    out: dict = {
        "best_epoch": best_epoch,
        "val_loss": best_metrics.get("val_loss", float("nan")),
        "macro_f1": best_metrics.get("macro_f1", float("nan")),
        "overfit_diag": diag,
        "calibration_temperature": calib_T,
    }
    if metrics_calib:
        out["metrics_calibrated"] = metrics_calib
    if args.num_classes == 2:
        out["auc_positive"] = best_metrics.get("auc_positive", float("nan"))
        if metrics_calib:
            out["auc_positive_calibrated"] = metrics_calib.get("auc_positive", float("nan"))
    elif args.num_classes == 3:
        out["macro_auroc"] = best_metrics.get("macro_auroc", float("nan"))
        if metrics_calib:
            out["macro_auroc_calibrated"] = metrics_calib.get("macro_auroc", float("nan"))
    else:
        out["auc_0_vs_low"] = best_metrics.get("auc_0_vs_low", float("nan"))
        out["auc_3p_vs_rest"] = best_metrics.get("auc_3p_vs_rest", float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregator", choices=["abmil", "clam", "transmil", "attnmisl"], required=True)
    ap.add_argument("--num_classes", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument("--label_mode", choices=["ihc", "gt_binary", "valieris_3"], required=True)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--only_fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_epochs", type=int, default=5, help="No early stop before this epoch.")
    ap.add_argument(
        "--min_epochs_for_selection",
        type=int,
        default=5,
        help="Do not save checkpoints before this epoch (val_loss or ranking metric).",
    )
    ap.add_argument(
        "--select_on",
        choices=["val_loss", "val_macro_f1", "val_auc_positive", "val_macro_auroc"],
        default="val_auc_positive",
        help="Checkpoint metric (binary: val_auc_positive; 3-class: val_macro_auroc).",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--feature_dim", type=int, default=2560)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument(
        "--val_subsample",
        choices=["random", "fixed"],
        default="fixed",
        help="Validation bag subsampling when max_patches caps N. "
        "'fixed' = deterministic per slide (recommended); "
        "'random' = new subset each val epoch (legacy).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clam_dropout", type=float, default=0.4)
    ap.add_argument("--k_sample", type=int, default=8)
    ap.add_argument("--bag_weight", type=float, default=0.7, help="CLAM only: bag vs instance loss")
    ap.add_argument("--trans_d_model", type=int, default=512)
    ap.add_argument("--trans_layers", type=int, default=2)
    ap.add_argument("--trans_heads", type=int, default=8)
    ap.add_argument("--trans_dropout", type=float, default=0.25)
    ap.add_argument("--abmil_hidden", type=int, default=512)
    ap.add_argument("--abmil_attn", type=int, default=256)
    ap.add_argument("--abmil_dropout", type=float, default=0.4)
    ap.add_argument(
        "--w_attn_entropy",
        type=float,
        default=0.0,
        help="ABMIL only: subtract w*H(attn) from bag CE (w>0 encourages smoother attention)",
    )
    ap.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Binary/4-class bag CE label smoothing (0 = off).",
    )
    ap.add_argument(
        "--ce_class_weights",
        choices=["effective", "none"],
        default="effective",
        help="effective = Cui et al. effective-number weights on bag CE (train-fold counts), "
        "same β as PhenoHER2; none = unweighted CE.",
    )
    ap.add_argument(
        "--cb_beta",
        type=float,
        default=0.999,
        help="Beta for effective-number class weights (matches PhenoHER2 --cb_beta).",
    )
    ap.add_argument(
        "--prototypes",
        default=None,
        help="Fold-wise prototype .pt (required for attnmisl; same AP files as khead).",
    )
    ap.add_argument("--attnmisl_cluster_num", type=int, default=8)
    ap.add_argument("--attnmisl_dropout", type=float, default=0.5)
    ap.add_argument(
        "--no_calibrate",
        action="store_true",
        help="Skip post-hoc temperature scaling on validation (khead uses calibration by default).",
    )
    args = ap.parse_args()

    if args.num_classes == 4 and args.label_mode != "ihc":
        raise ValueError("num_classes=4 requires label_mode=ihc")
    if args.num_classes == 3 and args.label_mode != "valieris_3":
        raise ValueError("num_classes=3 requires label_mode=valieris_3")
    if args.num_classes == 2 and args.label_mode != "gt_binary":
        raise ValueError("num_classes=2 requires label_mode=gt_binary (ISH slide label)")
    if args.num_classes == 3 and args.select_on == "val_auc_positive":
        args.select_on = "val_macro_auroc"
    if args.aggregator == "attnmisl" and not args.prototypes:
        raise ValueError("--prototypes is required when --aggregator attnmisl")

    return_coords = args.aggregator.lower() == "transmil"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    val_subsample_mode = "deterministic" if args.val_subsample == "fixed" else "random"

    train_dataset = HerohePatchBagDataset(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        label_mode=args.label_mode,
        max_patches=args.max_patches,
        subsample_mode="random",
        seed=args.seed,
        return_coords=return_coords,
    )
    val_dataset = HerohePatchBagDataset(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        label_mode=args.label_mode,
        max_patches=args.max_patches,
        subsample_mode=val_subsample_mode,
        seed=args.seed,
        return_coords=return_coords,
    )
    full = train_dataset
    print(
        f"[mil] aggregator={args.aggregator} num_classes={args.num_classes} "
        f"device={device} slides={len(full)} counts={full.class_counts().tolist()} "
        f"val_subsample={args.val_subsample}"
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
        splits = []
        for k in range(n_folds):
            va_sids = fdf.loc[fdf["fold"] == k, "slide_id"].tolist()
            va_idx = np.array([sid_to_pos[s] for s in va_sids], dtype=int)
            tr_idx = np.array(
                [i for i in range(len(full)) if i not in set(va_idx.tolist())],
                dtype=int,
            )
            splits.append((tr_idx, va_idx))
    else:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.zeros(len(y)), y))

    if args.only_fold is not None:
        _iter = [(args.only_fold, splits[args.only_fold])]
    else:
        _iter = list(enumerate(splits))

    fold_metrics = []
    for fold_idx, (tr_idx, va_idx) in _iter:
        print(f"\n=== {args.aggregator} fold {fold_idx} train={len(tr_idx)} val={len(va_idx)} ===")
        m = train_one_fold(
            fold_idx, train_dataset, val_dataset, tr_idx, va_idx, args, out_dir, device
        )
        fold_metrics.append({"fold": fold_idx, **m})

    keys_mean = ["macro_f1", "val_loss"]
    if args.num_classes == 2:
        keys_mean.append("auc_positive")
    elif args.num_classes == 3:
        keys_mean.append("macro_auroc")
    else:
        keys_mean.extend(["auc_0_vs_low", "auc_3p_vs_rest"])
    summary = {
        "folds": fold_metrics,
        "args": vars(args),
        "overfit_any_fold": any(
            fm.get("overfit_diag", {}).get("overfitting_detected", True) for fm in fold_metrics
        ),
        "overfit_flags": [{"fold": fm["fold"], **fm.get("overfit_diag", {})} for fm in fold_metrics],
    }
    for k in keys_mean:
        vs = [fm[k] for fm in fold_metrics if k in fm and np.isfinite(fm[k])]
        if vs:
            summary[f"{k}_mean"] = float(np.mean(vs))
            summary[f"{k}_std"] = float(np.std(vs))
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
