"""Train PhenoHER2 v2 with stratified 5-fold cross-validation.

v2 additions vs v1:
    - LossWeights uses `emd` and `balance` (CORN/entropy reg are gone).
    - Optional bag-level Mixup at training time (--mixup_alpha, --mixup_p).
    - Optional patch dropout passed to the model config (--patch_dropout).
    - Post-training temperature scaling: after each fold's best epoch we
      reload that checkpoint, fit T on the validation set, and re-save.
    - Logs `train_loss`, `val_macro_f1`, `val_auc_0_vs_low`, `val_auc_3p_vs_rest`.
    - Anti–small-val overfit hooks: ``--min_epochs`` (no early stop before then) and
      optional ``--val_loss_ema_alpha`` (logs an EMA of val loss). Use
      ``--val_loss_select ema`` to drive checkpointing/patience from that EMA; default
      ``raw`` compares **per-epoch** val loss so epoch 1 does not get an unfair advantage
      (EMA otherwise remembers later spikes while epoch 1’s EMA is only one point).

Reads:
    - Virchow2 feature bags from <features_dir>/<slide_id>.h5
    - HEROHE labels from <labels_csv>
    - Pre-computed prototypes (.pt) from init_prototypes.py (optional but recommended)

Writes per fold:
    - <out_dir>/fold_k/best.pt    (best validation macro-F1 checkpoint, calibrated)
    - <out_dir>/fold_k/log.csv    (epoch-level metrics)
    - <out_dir>/summary.json      (aggregate metrics across folds)

Run on the M4 Max via:

    python herohe/gp2/scripts/train_phenotype_mil.py \\
        --features_dir herohe/gp2/results_smoke_trident_mac/20x_256px_0px_overlap/features_virchow2 \\
        --labels_csv "herohe/Training (ground truth).csv" \\
        --prototypes herohe/gp2/data/prototypes_K16.pt \\
        --out_dir herohe/gp2/runs/phenoher2_v2 \\
        --K 16 --epochs 60 --device mps \\
        --mixup_alpha 0.4 --mixup_p 0.3 --patch_dropout 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

# Make the repo importable when running from anywhere
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]  # .../gradCode
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import (
    HerohePatchBagDataset,
    PhenoHER2,
    PhenoHER2Config,
    PhenoHER2Loss,
)
from herohe.gp2.models.dataset import collate_single_bag
from herohe.gp2.models.losses import LossWeights, soft_ordinal_target


def pick_device(name: str) -> torch.device:
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[train] MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[train] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def make_model(args, class_counts) -> tuple[PhenoHER2, PhenoHER2Loss]:
    cfg = PhenoHER2Config(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_prototypes=args.K,
        num_classes=4,
        dropout=args.dropout,
        use_spatial_block=bool(args.spatial_block),
        use_cls_pool=bool(args.cls_pool),
        patch_dropout=args.patch_dropout,
        use_dual_stream=bool(args.dual_stream),
    )
    model = PhenoHER2(cfg)
    if args.prototypes:
        blob = torch.load(args.prototypes, map_location="cpu")
        centers = blob["centers"] if isinstance(blob, dict) else blob
        if centers.shape[0] != args.K:
            raise ValueError(
                f"--prototypes has K={centers.shape[0]}, but --K={args.K}. "
                "Re-run init_prototypes.py with the right K."
            )
        model.load_prototypes_from_kmeans(centers)
        print(f"[train] loaded prototypes from {args.prototypes}: shape={tuple(centers.shape)}")
    weights = LossWeights(
        ce=args.w_ce,
        emd=args.w_emd,
        aux01=args.w_aux01,
        high3p=args.w_3p,
        balance=args.w_balance,
        orthogonality=args.w_orth,
    )
    loss_fn = PhenoHER2Loss(
        class_counts=class_counts,
        prototype_param=model.prototypes,
        weights=weights,
        beta=args.cb_beta,
        soft_label_smoothing=args.label_smoothing,
        sinkhorn_iter=args.sinkhorn_iter,
        sinkhorn_epsilon=args.sinkhorn_eps,
    )
    return model, loss_fn


# --------------------------------------------------------------------------------------
# Bag Mixup
# --------------------------------------------------------------------------------------


def bag_mixup(
    feats_a: torch.Tensor,
    feats_b: torch.Tensor,
    coords_a,
    coords_b,
    y_a: int,
    y_b: int,
    num_classes: int,
    alpha: float,
    rng: np.random.Generator,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, "torch.Tensor | None", torch.Tensor]:
    """Bag-level Mixup: sample lambda~Beta(a,a), keep lambda*N from A and
    (1-lambda)*N from B (where N = mean of |A|, |B|), concatenate as one bag,
    and produce a soft target = lambda*y_a + (1-lambda)*y_b (smoothed).

    feats_*: (1, N_*, D)
    coords_*: (1, N_*, 2) or None
    Returns: (mixed_feats (1, N, D), mixed_coords (1, N, 2) or None, soft_target (1, K)).
    """
    Na = feats_a.shape[1]
    Nb = feats_b.shape[1]
    n_target = max(64, (Na + Nb) // 2)
    lam = float(rng.beta(alpha, alpha))
    lam = max(0.05, min(0.95, lam))
    n_a = max(1, int(round(lam * n_target)))
    n_b = max(1, n_target - n_a)
    if n_a > Na:
        n_a = Na
    if n_b > Nb:
        n_b = Nb

    idx_a = torch.from_numpy(rng.choice(Na, size=n_a, replace=(n_a > Na))).long()
    idx_b = torch.from_numpy(rng.choice(Nb, size=n_b, replace=(n_b > Nb))).long()

    fa = feats_a[:, idx_a]
    fb = feats_b[:, idx_b]
    mixed_feats = torch.cat([fa, fb], dim=1)

    if coords_a is not None and coords_b is not None:
        ca = coords_a[:, idx_a]
        cb = coords_b[:, idx_b]
        mixed_coords = torch.cat([ca, cb], dim=1)
    else:
        mixed_coords = None

    # Soft label: lam * one_hot(y_a) + (1-lam) * one_hot(y_b), then optionally
    # smear over ordinal neighbours so EMD^2 sees a coherent target.
    if label_smoothing > 0:
        ta = soft_ordinal_target(torch.tensor([y_a]), num_classes, label_smoothing)
        tb = soft_ordinal_target(torch.tensor([y_b]), num_classes, label_smoothing)
    else:
        ta = F.one_hot(torch.tensor([y_a]), num_classes).float()
        tb = F.one_hot(torch.tensor([y_b]), num_classes).float()
    soft_target = lam * ta + (1.0 - lam) * tb
    return mixed_feats, mixed_coords, soft_target


def soft_target_loss(out: dict, soft_target: torch.Tensor, ce_weights: torch.Tensor) -> torch.Tensor:
    """CE + EMD^2 against a soft target distribution. Used for Mixup steps
    where we don't have a hard integer label.

    Skips aux01, 3p, and balance/orth (they need hard labels or are computed
    inside the main loss). Mixup is purely a regulariser on the main 4-class
    head.
    """
    logits = out["logits_4cls"]
    log_probs = F.log_softmax(logits, dim=-1)
    # Class-weighted CE with soft target
    sample_weights = (soft_target * ce_weights.unsqueeze(0)).sum(dim=-1)
    ce = -(soft_target * log_probs).sum(dim=-1)
    ce = (ce * sample_weights).mean()
    # EMD^2
    probs = F.softmax(logits, dim=-1)
    cdf_p = torch.cumsum(probs, dim=-1)
    cdf_q = torch.cumsum(soft_target, dim=-1)
    diff = cdf_p[:, :-1] - cdf_q[:, :-1]
    K = probs.shape[1]
    emd = (diff.pow(2).sum(dim=-1) / max(K - 1, 1)).mean()
    return ce + 0.5 * emd


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _evaluate(
    model: PhenoHER2,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    apply_calibration: bool = False,
    loss_fn: PhenoHER2Loss | None = None,
) -> dict:
    model.eval()
    all_y, all_p, all_probs = [], [], []
    val_loss_sum = 0.0
    val_loss_n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            c = batch["coords"].to(device) if batch["coords"] is not None else None
            y = int(batch["label"].item())
            if loss_fn is not None:
                out_train = model(x, coords=c)
                total, _ = loss_fn(out_train, torch.tensor([y], device=device))
                if torch.isfinite(total):
                    val_loss_sum += float(total.item())
                    val_loss_n += 1
            out = model.predict(x, coords=c, apply_calibration=apply_calibration)
            probs = out["probs"][0].detach().cpu().numpy()
            pred = int(out["pred_class"].item())
            all_y.append(y)
            all_p.append(pred)
            all_probs.append(probs)
    y = np.array(all_y)
    p = np.array(all_p)
    P = np.stack(all_probs, axis=0)
    # Sanitize: NaN/Inf in predicted probs (can happen if the model diverges late
    # in training) must not kill a multi-hour run. Replace with the no-information
    # uniform 1/C distribution and warn.
    if not np.isfinite(P).all():
        n_bad = int((~np.isfinite(P).all(axis=-1)).sum())
        print(f"[eval] WARNING: {n_bad}/{len(P)} val slides produced non-finite probs; "
              f"replacing with uniform 1/{num_classes}.")
        bad_rows = ~np.isfinite(P).all(axis=-1)
        P[bad_rows] = 1.0 / num_classes
        # Re-derive predicted class for any sanitized rows.
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
        "val_loss": float(val_loss_sum / val_loss_n) if val_loss_n > 0 else float("nan"),
        "n": int(len(y)),
    }


# --------------------------------------------------------------------------------------
# Single fold trainer
# --------------------------------------------------------------------------------------


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

    train_labels = np.array([full_dataset.entries[i]["label"] for i in train_idx], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=4)
    print(f"[fold {fold_idx}] class counts (train): {class_counts.tolist()}")

    model, loss_fn = make_model(args, class_counts)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single_bag)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single_bag)

    select_on = args.select_on
    use_val_ema = select_on == "val_loss" and args.val_loss_ema_alpha > 0.0
    use_ema_for_selection = use_val_ema and getattr(args, "val_loss_select", "raw") == "ema"
    log_cols = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_macro_f1",
        "val_auc_0_vs_low",
        "val_auc_3p_vs_rest",
        "lr",
    ]
    if use_val_ema:
        log_cols.insert(3, "val_loss_ema")

    log_path = fold_dir / "log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(log_cols)

    rng = np.random.default_rng(args.seed + fold_idx)

    # Selection metric: 'val_loss' (lower is better) is much more stable than
    # macro-F1 on small validation sets and matches the early-stopping convention
    # used by Valieris et al. 2024. 'val_macro_f1' (higher is better) is kept
    # available for ablation.
    best_score = float("inf") if select_on == "val_loss" else -float("inf")
    best_is_lower_better = select_on == "val_loss"
    best_metrics: dict = {}
    best_epoch = -1
    epochs_since_best = 0
    val_loss_ema: float | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            c = batch["coords"].to(device) if batch["coords"] is not None else None
            y = batch["label"].to(device)

            do_mixup = (
                args.mixup_alpha > 0
                and rng.random() < args.mixup_p
                and len(train_idx) >= 2
            )
            if do_mixup:
                # Sample a second slide uniformly from the training fold
                j = int(rng.integers(0, len(train_ds)))
                other = collate_single_bag([train_ds[j]])
                xb = other["features"].to(device)
                cb = other["coords"].to(device) if other["coords"] is not None else None
                yb = int(other["label"].item())
                mixed_x, mixed_c, soft_t = bag_mixup(
                    feats_a=x,
                    feats_b=xb,
                    coords_a=c,
                    coords_b=cb,
                    y_a=int(y.item()),
                    y_b=yb,
                    num_classes=4,
                    alpha=args.mixup_alpha,
                    rng=rng,
                    label_smoothing=args.label_smoothing,
                )
                soft_t = soft_t.to(device)
                out = model(mixed_x, coords=mixed_c)
                total = soft_target_loss(out, soft_t, loss_fn.ce_weights)
            else:
                out = model(x, coords=c)
                total, parts = loss_fn(out, y)

            if not torch.isfinite(total):
                print(f"[fold {fold_idx}] epoch {epoch}: non-finite loss, skipping step")
                optim.zero_grad(set_to_none=True)
                continue
            optim.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running += float(total.item())
            n_seen += 1
        sched.step()
        train_loss = running / max(n_seen, 1)
        metrics = _evaluate(
            model, val_loader, device, apply_calibration=False, loss_fn=loss_fn
        )
        raw_val = float(metrics["val_loss"])
        if use_val_ema:
            a = float(args.val_loss_ema_alpha)
            if val_loss_ema is None:
                val_loss_ema = raw_val
            else:
                val_loss_ema = a * raw_val + (1.0 - a) * val_loss_ema
        # Fair comparison: raw val_loss each epoch vs raw; EMA can still be logged (--val_loss_select ema for old behaviour).
        if select_on == "val_loss":
            selection_val = float(val_loss_ema) if use_ema_for_selection else raw_val
        else:
            selection_val = raw_val

        row = [
            epoch,
            f"{train_loss:.4f}",
            f"{raw_val:.4f}",
            f"{metrics['macro_f1']:.4f}",
            f"{metrics['auc_0_vs_low']:.4f}",
            f"{metrics['auc_3p_vs_rest']:.4f}",
            f"{optim.param_groups[0]['lr']:.2e}",
        ]
        if use_val_ema:
            row.insert(3, f"{val_loss_ema:.4f}")

        with open(log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(row)
        ema_s = f" val_loss_ema={val_loss_ema:.4f}" if use_val_ema else ""
        print(
            f"[fold {fold_idx}] epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={raw_val:.4f}{ema_s}  "
            f"val_macroF1={metrics['macro_f1']:.4f}  "
            f"val_AUC(0vsLow)={metrics['auc_0_vs_low']:.4f}  "
            f"val_AUC(3+)={metrics['auc_3p_vs_rest']:.4f}"
        )
        score = selection_val if select_on == "val_loss" else metrics["macro_f1"]
        improved = (score < best_score) if best_is_lower_better else (score > best_score)
        if improved and np.isfinite(score):
            best_score = score
            best_metrics = {
                **metrics,
                "epoch": epoch,
                "selection_score": score,
                "val_loss_ema": val_loss_ema if use_val_ema else None,
                "val_loss_select": args.val_loss_select,
            }
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(model.cfg),
                    "metrics": best_metrics,
                    "args": vars(args),
                    "calibration_temperature": float(model.calibration_temperature.item()),
                },
                fold_dir / "best.pt",
            )
        else:
            epochs_since_best += 1
            if (
                args.patience > 0
                and epochs_since_best >= args.patience
                and epoch >= args.min_epochs
            ):
                if select_on == "val_loss":
                    if use_ema_for_selection:
                        sel = "val_loss_ema"
                    elif use_val_ema:
                        sel = "val_loss (raw; EMA logged)"
                    else:
                        sel = "val_loss"
                else:
                    sel = select_on
                print(
                    f"[fold {fold_idx}] early stopping at epoch {epoch}: "
                    f"no improvement on {sel} for {args.patience} epochs "
                    f"(best={best_score:.4f} @ epoch {best_epoch}; min_epochs={args.min_epochs})"
                )
                break

    # ------------------------------------------------------------------
    # Post-training temperature calibration on the validation set
    # ------------------------------------------------------------------
    print(f"[fold {fold_idx}] reloading best (epoch {best_epoch}) for calibration")
    blob = torch.load(fold_dir / "best.pt", map_location=device)
    model.load_state_dict(blob["model_state"])
    T = model.fit_temperature(val_loader, device=device)
    print(f"[fold {fold_idx}] fitted temperature T={T:.4f}")
    metrics_calib = _evaluate(
        model, val_loader, device, apply_calibration=True, loss_fn=loss_fn
    )
    blob["calibration_temperature"] = T
    blob["metrics_calibrated"] = metrics_calib
    blob["model_state"] = model.state_dict()
    torch.save(blob, fold_dir / "best.pt")
    print(
        f"[fold {fold_idx}] post-calib  "
        f"val_macroF1={metrics_calib['macro_f1']:.4f}  "
        f"val_AUC(0vsLow)={metrics_calib['auc_0_vs_low']:.4f}  "
        f"val_AUC(3+)={metrics_calib['auc_3p_vs_rest']:.4f}"
    )
    best_metrics_out = {**best_metrics, "T": T, "metrics_calibrated": metrics_calib}
    if select_on == "val_loss":
        if use_ema_for_selection:
            sel_label = "val_loss_ema"
        elif use_val_ema:
            sel_label = "val_loss_raw"
        else:
            sel_label = "val_loss"
    else:
        sel_label = select_on
    print(f"[fold {fold_idx}] best_{sel_label}={best_score:.4f}  best_metrics={best_metrics_out}")
    return best_metrics_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--prototypes", default=None, help="optional .pt from init_prototypes.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument(
        "--folds_csv",
        default=None,
        help="Optional path to a pre-computed folds CSV (e.g. data/folds_v1.csv). "
             "Must contain columns slide_id and fold. If provided, --n_folds is "
             "inferred from the file and StratifiedKFold is bypassed.",
    )
    ap.add_argument(
        "--only_fold",
        type=int,
        default=None,
        help="If set, train only this single fold index (0-based). Useful for smoke tests and ablations.",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stop patience (epochs without improvement on --select_on). 0 disables.",
    )
    ap.add_argument(
        "--min_epochs",
        type=int,
        default=12,
        help="Do not early-stop before this epoch (reduces noisy val_loss spikes on small val).",
    )
    ap.add_argument(
        "--val_loss_ema_alpha",
        type=float,
        default=0.25,
        help="If >0 and --select_on val_loss, also track EMA: ema = alpha*val_loss + (1-alpha)*ema. "
        "Whether EMA drives checkpointing is set by --val_loss_select. Set 0 to disable EMA entirely.",
    )
    ap.add_argument(
        "--val_loss_select",
        choices=["raw", "ema"],
        default="raw",
        help="When --select_on val_loss: use per-epoch raw val_loss (fair vs epoch 1) or EMA for "
        "best checkpoint + early stopping. EMA is still logged when val_loss_ema_alpha>0 and "
        "select is raw.",
    )
    ap.add_argument(
        "--select_on",
        choices=["val_loss", "val_macro_f1"],
        default="val_loss",
        help="Metric used for best-checkpoint selection and early stopping. "
             "val_loss is more stable on small validation sets and matches "
             "Valieris et al. 2024.",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--weight_decay",
        type=float,
        default=2e-4,
        help="AdamW weight decay (stronger default to reduce slide-level overfitting).",
    )
    ap.add_argument("--feature_dim", type=int, default=2560)
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--max_patches", type=int, default=4096, help="cap N during training")
    ap.add_argument("--cb_beta", type=float, default=0.999)
    # Loss weights
    ap.add_argument("--w_ce", type=float, default=1.0)
    ap.add_argument("--w_emd", type=float, default=0.5, help="EMD^2 ordinal weight (replaces CORN)")
    ap.add_argument("--w_aux01", type=float, default=0.5)
    ap.add_argument("--w_3p", type=float, default=0.25)
    ap.add_argument("--w_balance", type=float, default=0.05, help="Sinkhorn balance reg")
    ap.add_argument("--w_orth", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.1, help="ordinal-neighbour smear for EMD^2 target")
    ap.add_argument("--sinkhorn_iter", type=int, default=3)
    ap.add_argument("--sinkhorn_eps", type=float, default=0.05)
    # v2 architecture / training extras
    ap.add_argument("--patch_dropout", type=float, default=0.0)
    ap.add_argument("--dual_stream", type=int, default=1, help="set 0 to disable dual stream")
    ap.add_argument("--mixup_alpha", type=float, default=0.0, help="bag Mixup Beta(alpha, alpha); 0 disables")
    ap.add_argument("--mixup_p", type=float, default=0.3, help="prob of doing mixup per training step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial_block", type=int, default=0, help="set 1 to enable")
    ap.add_argument("--cls_pool", type=int, default=0, help="set 1 to enable [CLS] pooling")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[train] device={device}")
    print(
        f"[train] anti-overfit: min_epochs={args.min_epochs}  "
        f"val_loss_ema_alpha={args.val_loss_ema_alpha}  val_loss_select={args.val_loss_select}  "
        f"weight_decay={args.weight_decay}  dropout={args.dropout}"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    full = HerohePatchBagDataset(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        max_patches=args.max_patches,
        seed=args.seed,
        return_coords=True,
    )
    print(
        f"[train] dataset size={len(full)}; "
        f"class counts={full.class_counts().tolist()}"
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
                f"folds_csv lists {len(unknown)} slide_id(s) not present in dataset, "
                f"e.g. {unknown[:5]}"
            )
        missing_in_folds = sorted(set(full.slide_ids()) - set(fdf["slide_id"]))
        if missing_in_folds:
            raise ValueError(
                f"{len(missing_in_folds)} dataset slides have no fold assignment in "
                f"{args.folds_csv}; e.g. {missing_in_folds[:5]}"
            )
        n_folds = int(fdf["fold"].max()) + 1
        if args.n_folds != n_folds:
            print(f"[train] overriding --n_folds={args.n_folds} with {n_folds} from {args.folds_csv}")
        splits = []
        for k in range(n_folds):
            va_sids = fdf.loc[fdf["fold"] == k, "slide_id"].tolist()
            va_idx = np.array([sid_to_pos[s] for s in va_sids], dtype=int)
            tr_idx = np.array([i for i in range(len(full)) if i not in set(va_idx.tolist())], dtype=int)
            splits.append((tr_idx, va_idx))
        print(f"[train] using fold assignments from {args.folds_csv} ({n_folds} folds)")
    else:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.zeros(len(y)), y))

    if args.only_fold is not None:
        if not (0 <= args.only_fold < len(splits)):
            raise ValueError(
                f"--only_fold={args.only_fold} out of range [0, {len(splits)})"
            )
        splits = [(args.only_fold, splits[args.only_fold])]
        splits = [(idx, tr_va) for idx, tr_va in splits]
        print(f"[train] running ONLY fold {args.only_fold}")
        _splits_iter = splits
    else:
        _splits_iter = list(enumerate(splits))

    fold_metrics: list[dict] = []
    for fold_idx, (tr_idx, va_idx) in _splits_iter:
        print(f"\n=== Fold {fold_idx} : train={len(tr_idx)} val={len(va_idx)} ===")
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
    print(f"\n[train] summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
