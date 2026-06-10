"""Train PhenoHER2-Binary on Virchow2 bags: HEROHE ISH label (Negative vs Positive).

Uses the same fold protocol as ``train_phenotype_mil.py`` (optional ``folds_v1.csv``).
Labels come from ``Final Result (Ground truth)`` via ``HerohePatchBagDataset(...,
label_mode="gt_binary")``.

Example:

    python herohe/gp2/scripts/train_phenobin_mil.py \\
        --features_dir <path>/features_virchow2 \\
        --labels_csv "herohe/Training (ground truth).csv" \\
        --folds_csv herohe/gp2/data/folds_v1.csv \\
        --prototypes herohe/gp2/data/prototypes_K16.pt \\
        --out_dir herohe/gp2/runs/phenobin_v1 \\
        --device mps

ResNet50 bags (TRIDENT ``features_resnet50``, 1024-D): use ``--feature_dim 1024``,
``--features_dir .../features_resnet50``, and a matching prototype file, e.g.
``init_prototypes.py`` output ``prototypes_K16_resnet50.pt`` (do not load the
Virchow2 ``prototypes_K16.pt`` — shape mismatch). Template commands:
``gp2/scripts/prep_binary_resnet50_runs.sh``.
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

from herohe.gp2.scripts.training_diagnostics import analyze_fold_log
from herohe.gp2.scripts.metrics_utils import metrics_from_prob_matrix
from herohe.gp2.models import (
    BinaryLossWeights,
    HerohePatchBagDataset,
    PhenoHER2Binary,
    PhenoHER2BinaryConfig,
    PhenoHER2BinaryLoss,
)
from herohe.gp2.models.dataset import collate_single_bag
from herohe.gp2.models.losses import soft_ordinal_target
from herohe.gp2.prototype_discovery import load_prototype_checkpoint


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


def make_model(args, class_counts: np.ndarray) -> tuple[PhenoHER2Binary, PhenoHER2BinaryLoss]:
    proto_K = args.K
    if args.prototypes:
        blob = load_prototype_checkpoint(args.prototypes)
        file_K = int(blob.get("K", blob["centers"].shape[0]))
        method = blob.get("method", "unknown")
        if file_K != args.K:
            if method == "hierarchical_ap":
                print(
                    f"[train] AP prototypes L={file_K} overrides --K={args.K} "
                    f"(method={method})"
                )
                proto_K = file_K
                args.K = file_K
            else:
                raise ValueError(
                    f"--prototypes has K={file_K}, but --K={args.K}. "
                    "Re-run init_prototypes.py with the right K."
                )

    readout = args.readout
    use_dual = bool(args.dual_stream) and readout == "full"
    if readout != "full" and args.dual_stream:
        print(f"[train] readout={readout}: ignoring --dual_stream (not used outside full readout)")

    cfg = PhenoHER2BinaryConfig(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_prototypes=proto_K,
        dropout=args.dropout,
        use_spatial_block=bool(args.spatial_block),
        use_cls_pool=bool(args.cls_pool),
        patch_dropout=args.patch_dropout,
        use_dual_stream=use_dual,
        num_classes=args.num_classes,
        readout=readout,
        proto_attn_bias=bool(args.proto_attn_bias),
        khead_pool=args.khead_pool,
        khead_routing=args.khead_routing,
        patch_attn_temperature=args.patch_attn_temperature,
        init_temperature=args.proto_temperature,
        stkim_p=args.stkim_p,
        stkim_k=args.stkim_k,
        stkim_frac=args.stkim_frac,
        mine_patches=args.mine_patches,
        mine_on_val=bool(args.mine_on_val),
    )
    model = PhenoHER2Binary(cfg)
    if args.mine_patches > 0:
        print(
            f"[train] PhiHER2 inst mining: random max_patches → top {args.mine_patches} "
            f"(mine_on_val={bool(args.mine_on_val)})"
        )
    if args.prototypes:
        centers = blob["centers"]
        model.load_prototypes_from_kmeans(centers)
        freeze_ap = method == "hierarchical_ap" and not args.no_freeze_ap_prototypes
        if args.freeze_prototypes or freeze_ap:
            model.set_prototypes_trainable(False)
            print(f"[train] prototypes FROZEN (method={method})")
        elif method == "hierarchical_ap":
            print("[train] AP prototypes loaded; fine-tuning enabled (--no_freeze_ap_prototypes)")
        print(
            f"[train] loaded prototypes from {args.prototypes}: "
            f"shape={tuple(centers.shape)} method={method}"
        )
    weights = BinaryLossWeights(
        ce=args.w_ce,
        balance=args.w_balance,
        orthogonality=args.w_orth,
    )
    loss_fn = PhenoHER2BinaryLoss(
        class_counts=class_counts,
        prototype_param=model.prototypes,
        weights=weights,
        beta=args.cb_beta,
        sinkhorn_iter=args.sinkhorn_iter,
        sinkhorn_epsilon=args.sinkhorn_eps,
        label_smoothing=args.label_smoothing,
        use_class_weights=not bool(getattr(args, "plain_ce", 0)),
    )
    if getattr(args, "plain_ce", 0):
        print("[train] plain CE: no class weights, no label smoothing")
    return model, loss_fn


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
):
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

    if label_smoothing > 0:
        ta = soft_ordinal_target(torch.tensor([y_a]), num_classes, label_smoothing)
        tb = soft_ordinal_target(torch.tensor([y_b]), num_classes, label_smoothing)
    else:
        ta = F.one_hot(torch.tensor([y_a]), num_classes).float()
        tb = F.one_hot(torch.tensor([y_b]), num_classes).float()
    soft_target = lam * ta + (1.0 - lam) * tb
    return mixed_feats, mixed_coords, soft_target


def soft_target_loss_mc(
    out: dict, soft_target: torch.Tensor, ce_weights: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Mixup-only: class-weighted soft CE; 2-class adds tiny ordinal EMD."""
    logits = out["logits_bin"]
    log_probs = F.log_softmax(logits, dim=-1)
    sample_weights = (soft_target * ce_weights.unsqueeze(0)).sum(dim=-1)
    ce = -(soft_target * log_probs).sum(dim=-1)
    ce = (ce * sample_weights).mean()
    if num_classes == 2:
        probs = F.softmax(logits, dim=-1)
        cdf_p = torch.cumsum(probs, dim=-1)
        cdf_q = torch.cumsum(soft_target, dim=-1)
        diff = cdf_p[:, :-1] - cdf_q[:, :-1]
        emd = (diff.pow(2).sum(dim=-1) / max(num_classes - 1, 1)).mean()
        return ce + 0.5 * emd
    return ce


def soft_target_loss_bin(out: dict, soft_target: torch.Tensor, ce_weights: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for 2-class mixup."""
    return soft_target_loss_mc(out, soft_target, ce_weights, num_classes=2)


def _evaluate(
    model: PhenoHER2Binary,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    apply_calibration: bool = False,
    loss_fn: PhenoHER2BinaryLoss | None = None,
) -> dict:
    model.eval()
    all_y, all_p, all_probs = [], [], []
    val_loss_sum = 0.0
    val_ce_sum = 0.0
    val_bal_sum = 0.0
    val_orth_sum = 0.0
    val_loss_n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            c = batch["coords"].to(device) if batch["coords"] is not None else None
            y = int(batch["label"].item())
            if loss_fn is not None:
                out_train = model(x, coords=c)
                total, parts = loss_fn(out_train, torch.tensor([y], device=device))
                if torch.isfinite(total):
                    val_loss_sum += float(total.item())
                    val_ce_sum += float(parts["ce"].item())
                    if "balance" in parts:
                        val_bal_sum += float(parts["balance"].item())
                    if "orthogonality" in parts:
                        val_orth_sum += float(parts["orthogonality"].item())
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
    if not np.isfinite(P).all():
        n_bad = int((~np.isfinite(P).all(axis=-1)).sum())
        print(f"[eval] WARNING: {n_bad}/{len(P)} val slides non-finite probs; uniform 1/K.")
        bad_rows = ~np.isfinite(P).all(axis=-1)
        P[bad_rows] = 1.0 / num_classes
        p = np.where(bad_rows, P.argmax(axis=-1), p)
    metrics = metrics_from_prob_matrix(y, p, P, num_classes)
    if val_loss_n > 0:
        metrics["val_loss"] = float(val_loss_sum / val_loss_n)
        metrics["val_ce"] = float(val_ce_sum / val_loss_n)
        if loss_fn is not None and loss_fn.weights.balance > 0.0:
            metrics["val_balance"] = float(val_bal_sum / val_loss_n)
        if loss_fn is not None and loss_fn.weights.orthogonality > 0.0:
            metrics["val_orth"] = float(val_orth_sum / val_loss_n)
    else:
        metrics["val_loss"] = float("nan")
        metrics["val_ce"] = float("nan")
    return metrics


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

    train_labels = np.array([train_dataset.entries[i]["label"] for i in train_idx], dtype=np.int64)
    num_classes = args.num_classes
    class_counts = np.bincount(train_labels, minlength=num_classes)
    if num_classes == 2:
        print(f"[fold {fold_idx}] binary class counts (train): neg={class_counts[0]} pos={class_counts[1]}")
    else:
        print(
            f"[fold {fold_idx}] valieris class counts (train): "
            f"neg={class_counts[0]} low={class_counts[1]} high={class_counts[2]}"
        )

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
    if num_classes == 2:
        log_cols = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_auc_positive", "lr"]
        score_key = "auc_positive"
    else:
        log_cols = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_ce",
            "val_balance",
            "val_macro_f1",
            "val_macro_auroc",
            "lr",
        ]
        score_key = "macro_auroc"
    if use_val_ema:
        log_cols.insert(3, "val_loss_ema")

    log_path = fold_dir / "log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(log_cols)

    rng = np.random.default_rng(args.seed + fold_idx)

    best_score = float("inf") if select_on == "val_loss" else -float("inf")
    best_is_lower_better = select_on == "val_loss"
    best_metrics: dict = {}
    best_epoch = -1
    epochs_since_best = 0
    val_loss_ema: float | None = None
    running_min_val_loss = float("inf")
    ranking_select = select_on in ("val_auc_positive", "val_macro_auroc", "val_macro_f1")
    min_sel_ep = int(getattr(args, "min_epochs_for_selection", 0) or 0)
    val_loss_ratio = float(getattr(args, "selection_val_loss_ratio", 0.0) or 0.0)
    if not ranking_select:
        val_loss_ratio = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1) or 1))
        optim.zero_grad(set_to_none=True)
        accum_count = 0
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
                    num_classes=num_classes,
                    alpha=args.mixup_alpha,
                    rng=rng,
                    label_smoothing=args.label_smoothing,
                )
                soft_t = soft_t.to(device)
                out = model(mixed_x, coords=mixed_c)
                total = soft_target_loss_mc(out, soft_t, loss_fn.ce_weights, num_classes)
            else:
                out = model(x, coords=c)
                total, parts = loss_fn(out, y)
                if (
                    args.w_attn_entropy > 0.0
                    and args.readout in ("khead", "khead_abmil")
                    and out.get("patch_attn") is not None
                ):
                    ent = PhenoHER2Binary.mean_patch_attention_entropy(out["patch_attn"])
                    total = total - args.w_attn_entropy * ent

            if not torch.isfinite(total):
                print(f"[fold {fold_idx}] epoch {epoch}: non-finite loss, skipping step")
                optim.zero_grad(set_to_none=True)
                accum_count = 0
                continue
            scaled = total / grad_accum
            scaled.backward()
            accum_count += 1
            running += float(total.item())
            n_seen += 1
            if accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum_count = 0
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
        sched.step()
        train_loss = running / max(n_seen, 1)
        metrics = _evaluate(
            model, val_loader, device, num_classes, apply_calibration=False, loss_fn=loss_fn
        )
        raw_val = float(metrics["val_loss"])
        running_min_val_loss = min(running_min_val_loss, raw_val)
        if use_val_ema:
            a = float(args.val_loss_ema_alpha)
            if val_loss_ema is None:
                val_loss_ema = raw_val
            else:
                val_loss_ema = a * raw_val + (1.0 - a) * val_loss_ema
        if select_on == "val_loss":
            selection_val = float(val_loss_ema) if use_ema_for_selection else raw_val
            score = selection_val
            score_raw = score
        elif select_on == "val_auc_positive":
            score_raw = float(metrics.get("auc_positive", float("nan")))
            score = score_raw
        elif select_on == "val_macro_auroc":
            score_raw = float(metrics.get("macro_auroc", float("nan")))
            score = score_raw
        else:
            score_raw = float(metrics["macro_f1"])
            score = score_raw

        row = [
            epoch,
            f"{train_loss:.4f}",
            f"{raw_val:.4f}",
        ]
        if num_classes == 3:
            row.extend(
                [
                    f"{metrics.get('val_ce', float('nan')):.4f}",
                    f"{metrics.get('val_balance', float('nan')):.4f}",
                ]
            )
        row.extend(
            [
                f"{metrics['macro_f1']:.4f}",
                f"{metrics.get(score_key, float('nan')):.4f}",
                f"{optim.param_groups[0]['lr']:.2e}",
            ]
        )
        if use_val_ema:
            row.insert(3, f"{val_loss_ema:.4f}")

        with open(log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(row)
        ema_s = f" val_loss_ema={val_loss_ema:.4f}" if use_val_ema else ""
        print(
            f"[fold {fold_idx}] epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={raw_val:.4f}{ema_s}  "
            + (
                f"val_ce={metrics.get('val_ce', float('nan')):.4f}  "
                f"val_bal={metrics.get('val_balance', float('nan')):.4f}  "
                if num_classes == 3
                else ""
            )
            + f"val_macroF1={metrics['macro_f1']:.4f}  "
            f"val_{score_key}={metrics.get(score_key, float('nan')):.4f}"
        )
        improved = (score < best_score) if best_is_lower_better else (score > best_score)
        eligible = min_sel_ep <= 0 or epoch >= min_sel_ep
        if ranking_select and val_loss_ratio > 0:
            ceiling = val_loss_ratio * running_min_val_loss
            if raw_val > ceiling:
                eligible = False
        if improved and np.isfinite(score) and eligible:
            best_score = score
            best_metrics = {
                **metrics,
                "epoch": epoch,
                "selection_score": score,
                "selection_score_raw": score_raw,
                "val_loss_ema": val_loss_ema if use_val_ema else None,
                "val_loss_select": args.val_loss_select,
            }
            best_epoch = epoch
            epochs_since_best = 0
            print(
                f"[fold {fold_idx}] *** saved checkpoint @ epoch {epoch} "
                f"({score_key}_raw={score_raw:.4f}, selection_score={score:.4f}) ***"
            )
            torch.save(
                {
                    "model_type": "PhenoHER2Binary",
                    "model_state": model.state_dict(),
                    "config": vars(model.cfg),
                    "metrics": best_metrics,
                    "args": vars(args),
                    "calibration_temperature": float(model.calibration_temperature.item()),
                },
                fold_dir / "best.pt",
            )
        elif improved and np.isfinite(score) and not eligible:
            print(
                f"[fold {fold_idx}] epoch {epoch}: ranking metric improved "
                f"({score_key}={score:.4f}) but checkpoint skipped "
                f"(epoch<{min_sel_ep} or val_loss {raw_val:.3f} > "
                f"{val_loss_ratio:.2f}x min {running_min_val_loss:.3f})"
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

    if best_epoch < 0:
        raise RuntimeError(
            f"[fold {fold_idx}] no checkpoint passed selection gates; "
            f"train longer or relax min_epochs_for_selection / selection_val_loss_ratio"
        )

    print(f"[fold {fold_idx}] reloading best (epoch {best_epoch}) for calibration")
    blob = torch.load(fold_dir / "best.pt", map_location=device)
    model.load_state_dict(blob["model_state"])
    T = model.fit_temperature(val_loader, device=device)
    print(f"[fold {fold_idx}] fitted temperature T={T:.4f}")
    metrics_calib = _evaluate(
        model, val_loader, device, num_classes, apply_calibration=True, loss_fn=loss_fn
    )
    blob["calibration_temperature"] = T
    blob["metrics_calibrated"] = metrics_calib
    blob["model_state"] = model.state_dict()
    torch.save(blob, fold_dir / "best.pt")
    print(
        f"[fold {fold_idx}] post-calib  val_macroF1={metrics_calib['macro_f1']:.4f}  "
        f"val_{score_key}={metrics_calib.get(score_key, float('nan')):.4f}"
    )
    auc_col = "val_auc_positive" if num_classes == 2 else "val_macro_auroc"
    diag = analyze_fold_log(
        log_path,
        auc_col=auc_col,
        num_classes=num_classes,
        min_epochs_for_selection=min_sel_ep,
        selection_val_loss_ratio=val_loss_ratio,
        select_on=select_on,
    )
    print(f"[fold {fold_idx}] overfit_diag={diag}")
    best_metrics_out = {
        **best_metrics,
        "T": T,
        "metrics_calibrated": metrics_calib,
        "overfit_diag": diag,
    }
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
    ap.add_argument("--labels_csv", required=True, help="HEROHE Training (ground truth).csv")
    ap.add_argument("--num_classes", type=int, choices=[2, 3], default=2)
    ap.add_argument(
        "--label_mode",
        choices=["gt_binary", "valieris_3"],
        default="gt_binary",
        help="gt_binary=ISH Negative/Positive; valieris_3=ASCO/CAP neg/low/high (M6).",
    )
    ap.add_argument("--prototypes", default=None)
    ap.add_argument(
        "--freeze_prototypes",
        action="store_true",
        help="Freeze prototype vectors during MIL (always; also default for hierarchical AP)",
    )
    ap.add_argument(
        "--no_freeze_ap_prototypes",
        action="store_true",
        help="Fine-tune AP-discovered prototypes (PhiHER2 Cluster-PT default is frozen)",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument(
        "--folds_csv",
        default=None,
        help="Pre-computed folds (slide_id, fold). Same as 4-class PhenoHER2.",
    )
    ap.add_argument("--only_fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument(
        "--min_epochs",
        type=int,
        default=12,
        help="No early stop before this epoch.",
    )
    ap.add_argument(
        "--min_epochs_for_selection",
        type=int,
        default=None,
        help="When selecting on AUC/F1, do not save checkpoints before this epoch "
        "(default: 5 for 3-class, 0 for binary).",
    )
    ap.add_argument(
        "--selection_val_loss_ratio",
        type=float,
        default=None,
        help="Ranking-only: save only if val_loss <= ratio * running min (default 1.15 for "
        "3-class AUROC/F1 selection; off when select_on=val_loss).",
    )
    ap.add_argument("--val_loss_ema_alpha", type=float, default=0.25)
    ap.add_argument(
        "--val_loss_select",
        choices=["raw", "ema"],
        default="raw",
        help="When --select_on val_loss: per-epoch raw val_loss (default, fair) vs EMA for "
        "checkpoint + early stop. EMA still logged if val_loss_ema_alpha>0.",
    )
    ap.add_argument(
        "--select_on",
        choices=["val_loss", "val_macro_f1", "val_auc_positive", "val_macro_auroc"],
        default="val_auc_positive",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--feature_dim", type=int, default=2560)
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument(
        "--val_subsample",
        choices=["random", "fixed"],
        default="fixed",
        help="Validation bag subsampling when max_patches caps N.",
    )
    ap.add_argument("--cb_beta", type=float, default=0.999)
    ap.add_argument("--w_ce", type=float, default=1.0)
    ap.add_argument("--w_balance", type=float, default=0.05)
    ap.add_argument("--w_orth", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.05, help="Bag CE label smoothing + mixup smear")
    ap.add_argument(
        "--plain_ce",
        type=int,
        default=0,
        help="Plain unweighted CE (PhiHER2-style): disables effective-number class weights",
    )
    ap.add_argument("--sinkhorn_iter", type=int, default=3)
    ap.add_argument("--sinkhorn_eps", type=float, default=0.05)
    ap.add_argument("--patch_dropout", type=float, default=0.1)
    ap.add_argument("--dual_stream", type=int, default=1)
    ap.add_argument(
        "--readout",
        default="full",
        choices=["full", "khead", "khead_abmil"],
        help="full=Transformer proto pool; khead=K parallel MIL heads; khead_abmil=+ABMIL residual",
    )
    ap.add_argument(
        "--proto_attn_bias",
        type=int,
        default=1,
        help="For khead readouts: add cosine proto similarity to attention logits (1) or off (0)",
    )
    ap.add_argument(
        "--khead_pool",
        default="concat",
        choices=["concat", "mean", "token_abmil"],
        help="khead readout: concat | mean-pool | token_abmil (ABMIL over K phenotype tokens)",
    )
    ap.add_argument(
        "--khead_routing",
        default="independent",
        choices=["independent", "log_gate", "hard_partition"],
        help="khead patch pooling: independent | log_gate | hard_partition (within-cluster pool)",
    )
    ap.add_argument(
        "--patch_attn_temperature",
        type=float,
        default=1.0,
        help="Temperature on patch softmax per khead (higher = softer pooling, less CE volatility)",
    )
    ap.add_argument(
        "--proto_temperature",
        type=float,
        default=1.0,
        help="Initial cosine-sim temperature for prototype assignment (higher = softer routing)",
    )
    ap.add_argument(
        "--w_attn_entropy",
        type=float,
        default=0.0,
        help="khead/khead_abmil: subtract w*mean H(patch_attn) from bag CE (w>0 = smoother attn)",
    )
    ap.add_argument(
        "--stkim_p",
        type=float,
        default=0.0,
        help="khead STKIM (ACMIL): per-head mask top-k attn logits with prob p (train only; 0=off)",
    )
    ap.add_argument(
        "--stkim_k",
        type=int,
        default=10,
        help="STKIM: min top-k patches to consider per head (before stkim_frac)",
    )
    ap.add_argument(
        "--stkim_frac",
        type=float,
        default=0.0,
        help="STKIM: if >0, k=max(stkim_k, frac*valid_patches) per head",
    )
    ap.add_argument(
        "--mine_patches",
        type=int,
        default=0,
        help="PhiHER2 inst_selector: keep top-k patches by P(pos) after bag subsample (0=off)",
    )
    ap.add_argument(
        "--mine_on_val",
        type=int,
        default=0,
        help="Apply inst_selector mining at validation/inference (PhiHER2 uses 1)",
    )
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Optimizer steps every N slides (PhiHER2 uses ~32)",
    )
    ap.add_argument(
        "--val_max_patches",
        type=int,
        default=-1,
        help="Val bag cap: -1 = same as --max_patches, 0 = full bag (no subsample)",
    )
    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    ap.add_argument("--mixup_p", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial_block", type=int, default=0)
    ap.add_argument("--cls_pool", type=int, default=0)
    args = ap.parse_args()

    if args.num_classes == 3 and args.label_mode != "valieris_3":
        raise ValueError("num_classes=3 requires label_mode=valieris_3")
    if args.num_classes == 2 and args.label_mode != "gt_binary":
        raise ValueError("num_classes=2 requires label_mode=gt_binary")
    if args.num_classes == 3 and args.select_on == "val_auc_positive":
        args.select_on = "val_macro_auroc"

    if args.min_epochs_for_selection is None:
        args.min_epochs_for_selection = 5 if args.num_classes == 3 else 0
    if args.selection_val_loss_ratio is None:
        args.selection_val_loss_ratio = (
            1.15 if args.num_classes == 3 and args.select_on != "val_loss" else 0.0
        )
    # 3-class: same training recipe as binary v2 (phenobin_5fold_v2); only checkpoint metric differs.
    if args.num_classes == 3 and args.select_on != "val_loss":
        if args.mixup_alpha == 0.0:
            args.mixup_alpha = 0.4
        args.min_epochs = max(args.min_epochs, 12)
        args.patience = max(args.patience, 7)
    if args.select_on != "val_loss" and args.val_loss_ema_alpha > 0.0:
        args.val_loss_ema_alpha = 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    loss_desc = (
        "CE-only"
        if args.w_balance == 0.0 and args.w_orth == 0.0
        else f"w_ce={args.w_ce} w_balance={args.w_balance} w_orth={args.w_orth}"
    )
    print(
        f"[train] PhenoHER2-Binary device={device}  num_classes={args.num_classes} "
        f"label_mode={args.label_mode}  readout={args.readout}  select_on={args.select_on}  "
        f"min_epochs_for_selection={args.min_epochs_for_selection}  "
        f"selection_val_loss_ratio={args.selection_val_loss_ratio}  "
        f"mixup_alpha={args.mixup_alpha}  label_smoothing={args.label_smoothing}  "
        f"dropout={args.dropout}  patch_dropout={args.patch_dropout}  wd={args.weight_decay}  "
        f"lr={args.lr}  loss={loss_desc}"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    val_subsample_mode = "deterministic" if args.val_subsample == "fixed" else "random"
    if args.val_max_patches == 0:
        val_max_patches = None
    elif args.val_max_patches > 0:
        val_max_patches = args.val_max_patches
    else:
        val_max_patches = args.max_patches
    train_dataset = HerohePatchBagDataset(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        label_mode=args.label_mode,
        max_patches=args.max_patches,
        subsample_mode="random",
        seed=args.seed,
        return_coords=True,
    )
    val_dataset = HerohePatchBagDataset(
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        label_mode=args.label_mode,
        max_patches=val_max_patches,
        subsample_mode=val_subsample_mode,
        seed=args.seed,
        return_coords=True,
    )
    full = train_dataset
    cc = full.class_counts(num_classes=args.num_classes)
    if args.num_classes == 2:
        print(
            f"[train] dataset size={len(full)}; neg={cc[0]} pos={cc[1]}; "
            f"select_on={args.select_on} val_subsample={args.val_subsample} "
            f"val_max_patches={val_max_patches}"
        )
    else:
        print(
            f"[train] dataset size={len(full)}; neg={cc[0]} low={cc[1]} high={cc[2]}; "
            f"select_on={args.select_on} val_subsample={args.val_subsample}"
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
                f"{len(missing_in_folds)} dataset slides missing from folds_csv; e.g. {missing_in_folds[:5]}"
            )
        n_folds = int(fdf["fold"].max()) + 1
        if args.n_folds != n_folds:
            print(f"[train] overriding --n_folds={args.n_folds} with {n_folds} from folds_csv")
        splits = []
        for k in range(n_folds):
            va_sids = fdf.loc[fdf["fold"] == k, "slide_id"].tolist()
            va_idx = np.array([sid_to_pos[s] for s in va_sids], dtype=int)
            tr_idx = np.array([i for i in range(len(full)) if i not in set(va_idx.tolist())], dtype=int)
            splits.append((tr_idx, va_idx))
        print(f"[train] fold assignments from {args.folds_csv}")
    else:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.zeros(len(y)), y))

    if args.only_fold is not None:
        if not (0 <= args.only_fold < len(splits)):
            raise ValueError(f"--only_fold={args.only_fold} out of range [0, {len(splits)})")
        _splits_iter = [(args.only_fold, splits[args.only_fold])]
        print(f"[train] ONLY fold {args.only_fold}")
    else:
        _splits_iter = list(enumerate(splits))

    fold_metrics: list[dict] = []
    for fold_idx, (tr_idx, va_idx) in _splits_iter:
        print(f"\n=== Fold {fold_idx} : train={len(tr_idx)} val={len(va_idx)} ===")
        m = train_one_fold(
            fold_idx=fold_idx,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_idx=tr_idx,
            val_idx=va_idx,
            args=args,
            out_dir=out_dir,
            device=device,
        )
        fold_metrics.append({"fold": fold_idx, **m})

    summary: dict = {
        "folds": fold_metrics,
        "macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "macro_f1_std": float(np.std([m["macro_f1"] for m in fold_metrics])),
        "args": vars(args),
        "overfit_any_fold": any(
            m.get("overfit_diag", {}).get("overfitting_detected", True) for m in fold_metrics
        ),
        "overfit_flags": [{"fold": m["fold"], **m.get("overfit_diag", {})} for m in fold_metrics],
    }
    if args.num_classes == 2:
        summary["auc_positive_mean"] = float(
            np.nanmean([m.get("metrics_calibrated", {}).get("auc_positive", m.get("auc_positive", float("nan"))) for m in fold_metrics])
        )
    else:
        summary["macro_auroc_mean"] = float(
            np.nanmean([m.get("metrics_calibrated", {}).get("macro_auroc", m.get("macro_auroc", float("nan"))) for m in fold_metrics])
        )
        summary["macro_auroc_std"] = float(
            np.nanstd([m.get("metrics_calibrated", {}).get("macro_auroc", m.get("macro_auroc", float("nan"))) for m in fold_metrics])
        )
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[train] summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
