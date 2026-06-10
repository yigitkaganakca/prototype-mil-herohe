"""Analyze khead fold-0 sweep logs for training stability vs ABMIL reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def read_log(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def metrics_from_log(rows: list[dict]) -> dict:
    if not rows:
        return {}
    tr = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    auc = [float(r["val_auc_positive"]) for r in rows]
    best_vl_i = min(range(len(vl)), key=lambda i: vl[i])
    best_auc_i = max(range(len(auc)), key=lambda i: auc[i])

    ep2_spike = vl[1] - vl[0] if len(vl) >= 2 else 0.0
    auc_at_saved = auc[best_vl_i]
    peak_auc = auc[best_auc_i]
    auc_missed = peak_auc - auc_at_saved

    post = vl[best_vl_i:]
    post_improve = any(v < vl[best_vl_i] - 1e-6 for v in post[1:]) if len(post) > 1 else False

    gap_saved = vl[best_vl_i] - tr[best_vl_i]
    gap_final = vl[-1] - tr[-1]

    # Stability heuristics (ABMIL-like)
    stable_ep2 = ep2_spike < 0.05
    stable_late = best_vl_i >= 4
    stable_auc = auc_missed <= 0.015
    stable_gap = gap_saved >= 0.0
    stable_score = sum([stable_ep2, stable_late, stable_auc, stable_gap, not post_improve])

    return {
        "n_epochs": len(rows),
        "saved_epoch": best_vl_i + 1,
        "val_loss_min": vl[best_vl_i],
        "val_loss_final": vl[-1],
        "val_loss_drift": vl[-1] - vl[best_vl_i],
        "val_auc_saved": auc_at_saved,
        "val_auc_peak": peak_auc,
        "peak_auc_epoch": best_auc_i + 1,
        "auc_missed": auc_missed,
        "ep2_val_spike": ep2_spike,
        "train_val_gap_saved": gap_saved,
        "train_val_gap_final": gap_final,
        "val_loss_improved_after_saved": post_improve,
        "stable_score": stable_score,
        "flags": {
            "ep2_ok": stable_ep2,
            "saved_epoch_ge5": stable_late,
            "auc_near_peak": stable_auc,
            "gap_ok_at_saved": stable_gap,
        },
    }


def from_ckpt(run_dir: Path) -> dict:
    p = run_dir / "fold_0/best.pt"
    if not p.is_file():
        return {}
    b = torch.load(p, map_location="cpu", weights_only=False)
    m = b.get("metrics", {})
    a = b.get("args", {})
    return {
        "checkpoint_epoch": m.get("epoch"),
        "checkpoint_auc": m.get("auc_positive"),
        "checkpoint_val_loss": m.get("val_loss"),
        "readout": a.get("readout"),
        "lr": a.get("lr"),
        "wd": a.get("weight_decay"),
        "dropout": a.get("dropout"),
        "patch_dropout": a.get("patch_dropout"),
        "min_epochs_for_selection": a.get("min_epochs_for_selection"),
        "selection_val_loss_ratio": a.get("selection_val_loss_ratio"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=Path, required=True)
    ap.add_argument("--abmil_ref", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    abmil_rows = read_log(args.abmil_ref)
    abmil = metrics_from_log(abmil_rows)
    abmil["name"] = "ABMIL v2 ref"

    results = [abmil]
    for run_dir in sorted(args.sweep_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        log = run_dir / "fold_0/log.csv"
        if not log.is_file():
            continue
        row = metrics_from_log(read_log(log))
        row["name"] = run_dir.name
        row.update(from_ckpt(run_dir))
        results.append(row)

    results.sort(key=lambda r: (-(r.get("stable_score") or 0), -(r.get("val_auc_saved") or 0)))

    print("\n=== Fold-0 stability sweep (sorted by stable_score, then val_auc @ saved) ===")
    print(
        f"{'name':<22} {'st':>2} {'ep':>3} {'val_auc':>7} {'peak':>7} "
        f"{'miss':>5} {'ep2Δ':>6} {'vl_drift':>8} {'gap@sv':>7}"
    )
    print("-" * 78)
    for r in results:
        print(
            f"{r['name']:<22} {r.get('stable_score', '-'):>2} "
            f"{r.get('saved_epoch', r.get('checkpoint_epoch', '-')):>3} "
            f"{r.get('val_auc_saved', r.get('checkpoint_auc', 0)) or 0:7.4f} "
            f"{r.get('val_auc_peak', 0) or 0:7.4f} "
            f"{r.get('auc_missed', 0) or 0:5.3f} "
            f"{r.get('ep2_val_spike', 0) or 0:+6.3f} "
            f"{r.get('val_loss_drift', 0) or 0:8.3f} "
            f"{r.get('train_val_gap_saved', 0) or 0:+7.3f}"
        )

    best_khead = next((r for r in results if r["name"] != "ABMIL v2 ref"), None)
    if best_khead and abmil:
        print(
            f"\nBest khead variant: {best_khead['name']} "
            f"(stable_score={best_khead.get('stable_score')} "
            f"val_auc={best_khead.get('val_auc_saved'):.4f} vs ABMIL {abmil.get('val_auc_saved'):.4f})"
        )

    args.out_json.write_text(json.dumps({"abmil_ref": abmil, "runs": results}, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
