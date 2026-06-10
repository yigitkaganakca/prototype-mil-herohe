"""Training-log diagnostics for detecting slide-level overfitting."""

from __future__ import annotations

import csv
from pathlib import Path


def analyze_fold_log(
    log_path: Path | str,
    auc_col: str = "val_auc_positive",
    num_classes: int = 2,
    min_epochs_for_selection: int = 0,
    selection_val_loss_ratio: float = 0.0,
    select_on: str = "val_auc_positive",
) -> dict:
    """Summarize whether the saved checkpoint looks overfit.

    When ``select_on == "val_loss"``, the reference checkpoint epoch is the
    minimum ``val_loss`` row (matching selection). Otherwise the reference is
    the best ranking metric row on/after ``min_epochs_for_selection``.
    """
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        return {"overfitting_detected": True, "overfitting_flags": ["empty log"]}

    if auc_col not in rows[0]:
        auc_col = next((c for c in rows[0] if "auc" in c.lower()), None)

    last = rows[-1]
    min_val_loss = min(float(r["val_loss"]) for r in rows)
    log_peak_row = (
        max(rows, key=lambda r: float(r[auc_col])) if auc_col and auc_col in rows[0] else rows[0]
    )
    log_peak_ep = int(log_peak_row["epoch"]) if auc_col else -1

    if select_on == "val_loss":
        eligible_rows = rows
        if min_epochs_for_selection > 0:
            eligible_rows = [r for r in rows if int(r["epoch"]) >= min_epochs_for_selection]
        best_row = min(eligible_rows or rows, key=lambda r: float(r["val_loss"]))
    else:
        eligible_rows = rows
        if min_epochs_for_selection > 0:
            eligible_rows = [r for r in rows if int(r["epoch"]) >= min_epochs_for_selection]
        if auc_col and eligible_rows:
            best_row = max(eligible_rows, key=lambda r: float(r[auc_col]))
        elif auc_col:
            best_row = log_peak_row
        else:
            best_row = min(rows, key=lambda r: float(r["val_loss"]))

    best_ep = int(best_row["epoch"])
    train_at_best = float(best_row["train_loss"])
    train_final = float(last["train_loss"])
    val_loss_best = float(best_row["val_loss"])
    val_loss_final = float(last["val_loss"])
    auc_best = float(best_row[auc_col]) if auc_col and auc_col in best_row else float("nan")

    train_floor = 0.08 if num_classes <= 2 else 0.15
    rel_ratio = selection_val_loss_ratio if selection_val_loss_ratio > 0 else (
        1.15 if num_classes >= 3 else 1.25
    )
    val_loss_ceiling = rel_ratio * min_val_loss

    flags = []
    notes = []
    if select_on != "val_loss" and min_epochs_for_selection > 0 and log_peak_ep < min_epochs_for_selection:
        notes.append(
            f"log_peak_auc_epoch={log_peak_ep} (warmup; ranking selection starts @ {min_epochs_for_selection})"
        )

    if train_at_best < train_floor:
        flags.append(f"train_loss_at_best={train_at_best:.4f}<{train_floor}")

    if select_on == "val_loss":
        if val_loss_final > val_loss_best * 1.15:
            flags.append(
                f"val_loss_final={val_loss_final:.3f}>{val_loss_best:.3f} after best-val checkpoint"
            )
        if train_final < train_at_best * 0.85 and val_loss_final > val_loss_best * 1.05:
            flags.append(
                "train_val_divergence: train_loss fell but val_loss rose vs best-val epoch"
            )
    else:
        if val_loss_best > val_loss_ceiling:
            flags.append(
                f"val_loss_at_best_auc={val_loss_best:.3f}>{val_loss_ceiling:.3f} "
                f"({rel_ratio:.2f}x min_val_loss={min_val_loss:.3f})"
            )

    gap_at_best = val_loss_best - train_at_best
    gap_final = val_loss_final - train_final

    return {
        "n_epochs": len(rows),
        "select_on": select_on,
        "best_epoch": best_ep,
        "best_auc_epoch": log_peak_ep if select_on == "val_loss" else best_ep,
        "log_peak_auc_epoch": log_peak_ep,
        "log_peak_auc": float(log_peak_row[auc_col]) if auc_col else float("nan"),
        "best_auc": auc_best,
        "best_val_loss": val_loss_best,
        "train_loss_at_best": train_at_best,
        "train_loss_final": train_final,
        "val_loss_at_best": val_loss_best,
        "val_loss_final": val_loss_final,
        "min_val_loss": min_val_loss,
        "train_val_gap_at_best": gap_at_best,
        "train_val_gap_final": gap_final,
        "val_loss_ratio_final_vs_best": val_loss_final / max(val_loss_best, 1e-6),
        "val_loss_ratio_best_vs_min": val_loss_best / max(min_val_loss, 1e-6),
        "selection_notes": notes,
        "overfitting_detected": bool(flags),
        "overfitting_flags": flags,
    }
