#!/usr/bin/env python3
"""Collect validation + test metrics for min_epochs_for_selection comparison."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "herohe/gp2/runs"
MIN_EP = 5


@dataclass
class ModelSpec:
    task: str
    model: str
    run_dir: str
    test_summary: str
    val_auc_key: str  # fold-level key in ckpt metrics
    test_auc_key: str
    is_khead: bool = False


SPECS = [
    ModelSpec(
        "binary Virchow2",
        "ABMIL",
        "abmil_phiher2fold_valloss",
        "test_eval_mil/binary_phiher2fold/summary_abmil_phiher2fold_valloss_5fold.json",
        "auc_positive",
        "auc_positive",
    ),
    ModelSpec(
        "binary Virchow2",
        "CLAM",
        "clam_phiher2fold_valloss",
        "test_eval_mil/binary_phiher2fold/summary_clam_phiher2fold_valloss_5fold.json",
        "auc_positive",
        "auc_positive",
    ),
    ModelSpec(
        "binary Virchow2",
        "TransMIL",
        "transmil_phiher2fold_valloss",
        "test_eval_mil/binary_phiher2fold/summary_transmil_phiher2fold_valloss_5fold.json",
        "auc_positive",
        "auc_positive",
    ),
    ModelSpec(
        "binary Virchow2",
        "khead",
        "khead_reg_sweep/reg_d06_wd4e3_ent15_pd20",
        "test_eval_phenonly/khead_reg_tuned/summary_khead_reg_tuned_5fold.json",
        "auc_positive",
        "AUC",
        True,
    ),
    ModelSpec(
        "3-class Virchow2",
        "ABMIL",
        "abmil_valieris3_5fold_s42_valloss",
        "test_eval_mil/valieris3_5fold_s42/summary_abmil_valieris3_5fold_s42_valloss_5fold.json",
        "macro_auroc",
        "macro_auroc",
    ),
    ModelSpec(
        "3-class Virchow2",
        "CLAM",
        "clam_valieris3_5fold_s42_valloss",
        "test_eval_mil/valieris3_5fold_s42/summary_clam_valieris3_5fold_s42_valloss_5fold.json",
        "macro_auroc",
        "macro_auroc",
    ),
    ModelSpec(
        "3-class Virchow2",
        "TransMIL",
        "transmil_valieris3_5fold_s42_valloss",
        "test_eval_mil/valieris3_5fold_s42/summary_transmil_valieris3_5fold_s42_valloss_5fold.json",
        "macro_auroc",
        "macro_auroc",
    ),
    ModelSpec(
        "3-class Virchow2",
        "khead",
        "khead_valieris3_tuned_5fold_s42",
        "test_eval_phenonly/khead_valieris3_tuned/summary_khead_valieris3_tuned_5fold.json",
        "macro_auroc",
        "macro_auroc",
        True,
    ),
    ModelSpec(
        "binary ResNet50",
        "ABMIL",
        "abmil_resnet50_5fold_s42_valloss",
        "test_eval_mil/resnet50_binary_5fold_s42/summary_abmil_resnet50_5fold_s42_valloss_5fold.json",
        "auc_positive",
        "auc_positive",
    ),
    ModelSpec(
        "binary ResNet50",
        "CLAM",
        "clam_resnet50_5fold_s42_valloss",
        "test_eval_mil/resnet50_binary_5fold_s42/summary_clam_resnet50_5fold_s42_valloss_5fold.json",
        "auc_positive",
        "auc_positive",
    ),
    ModelSpec(
        "binary ResNet50",
        "TransMIL",
        "transmil_resnet50_5fold_s42_valloss",
        "test_eval_mil/resnet50_binary_5fold_s42/summary_transmil_resnet50_5fold_s42_valloss_5fold.json",
        "auc_positive",
        "auc_positive",
    ),
    ModelSpec(
        "binary ResNet50",
        "khead",
        "khead_resnet50_tuned_5fold_s42",
        "test_eval_phenonly/khead_resnet50_tuned/summary_khead_resnet50_tuned_5fold.json",
        "auc_positive",
        "AUC",
        True,
    ),
]


def best_val_loss_epoch(log_path: Path, min_ep: int) -> int | None:
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        return None
    eligible = [r for r in rows if int(r["epoch"]) >= min_ep]
    pool = eligible or rows
    return int(min(pool, key=lambda r: float(r["val_loss"]))["epoch"])


def fold_would_change(log_path: Path, ckpt_path: Path, min_ep: int = MIN_EP) -> bool:
    if not log_path.is_file() or not ckpt_path.is_file():
        return False
    ep0 = best_val_loss_epoch(log_path, 1)
    ep5 = best_val_loss_epoch(log_path, min_ep)
    return ep0 != ep5


def collect_val_metrics(spec: ModelSpec) -> dict:
    run = RUNS / spec.run_dir
    fold_rows = []
    for f in range(5):
        ckpt = run / f"fold_{f}" / "best.pt"
        log = run / f"fold_{f}" / "log.csv"
        if not ckpt.is_file():
            continue
        b = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = b["metrics"]
        auc = m.get(spec.val_auc_key)
        if auc is None and spec.val_auc_key == "auc_positive":
            auc = m.get("AUC")
        fold_rows.append(
            {
                "fold": f,
                "epoch": int(m["epoch"]),
                "val_loss": float(m.get("val_loss", float("nan"))),
                "val_macro_f1": float(m.get("macro_f1", float("nan"))),
                "val_auc": float(auc) if auc is not None else float("nan"),
                "would_change_min5": fold_would_change(log, ckpt),
            }
        )
    out = {
        "folds": fold_rows,
        "n_folds": len(fold_rows),
        "val_auc_mean": float(np.mean([x["val_auc"] for x in fold_rows])) if fold_rows else None,
        "val_auc_std": float(np.std([x["val_auc"] for x in fold_rows])) if fold_rows else None,
        "val_macro_f1_mean": float(np.mean([x["val_macro_f1"] for x in fold_rows]))
        if fold_rows
        else None,
        "val_loss_mean": float(np.mean([x["val_loss"] for x in fold_rows])) if fold_rows else None,
        "n_would_change": sum(1 for x in fold_rows if x["would_change_min5"]),
    }
    return out


def collect_test_metrics(spec: ModelSpec) -> dict | None:
    p = RUNS / spec.test_summary
    if not p.is_file():
        return None
    r = json.loads(p.read_text())["results"][0]
    auc = r.get(spec.test_auc_key, r.get("macro_auroc", r.get("auc_positive")))
    return {
        "test_auc": float(auc),
        "test_macro_f1": float(r.get("macro_f1", float("nan"))),
        "n": int(r.get("n", r.get("n_labeled_slides", 0))),
        "path": str(p),
    }


def collect_all() -> list[dict]:
    rows = []
    for spec in SPECS:
        val = collect_val_metrics(spec)
        test = collect_test_metrics(spec)
        rows.append(
            {
                "task": spec.task,
                "model": spec.model,
                "run_dir": spec.run_dir,
                "validation": val,
                "test": test,
            }
        )
    return rows


def print_table(before: list[dict], after: list[dict] | None = None) -> None:
    after_map = {}
    if after:
        after_map = {(r["task"], r["model"]): r for r in after}

    print("\n" + "=" * 120)
    if after:
        print("COMPARISON: min_epochs_for_selection=0 (before) vs =5 (after)")
    else:
        print("METRICS SNAPSHOT (current checkpoints)")
    print("=" * 120)
    hdr = (
        f"{'Task':<18} {'Model':<10} "
        f"{'Val AUC':>8} {'Val F1':>7} "
        f"{'Test AUC':>9} {'Test F1':>8} "
        + (f"{'ΔVal AUC':>9} {'ΔTest':>9} " if after else "")
        + f"{'Aff':>4}"
    )
    print(hdr)
    print("-" * len(hdr))

    for b in before:
        key = (b["task"], b["model"])
        a = after_map.get(key)
        v = b["validation"]
        t = b.get("test") or {}
        val_auc = v.get("val_auc_mean")
        val_f1 = v.get("val_macro_f1_mean")
        test_auc = t.get("test_auc") if t else None
        test_f1 = t.get("test_macro_f1") if t else None
        aff = v.get("n_would_change", 0)

        line = (
            f"{b['task']:<18} {b['model']:<10} "
            f"{val_auc:>8.4f} {val_f1:>7.4f} "
            if val_auc is not None
            else f"{b['task']:<18} {b['model']:<10} {'—':>8} {'—':>7} "
        )
        if test_auc is not None:
            line += f"{test_auc:>9.4f} {test_f1:>8.4f} "
        else:
            line += f"{'—':>9} {'—':>8} "

        if after and a:
            va = a["validation"].get("val_auc_mean")
            ta = (a.get("test") or {}).get("test_auc")
            if val_auc is not None and va is not None:
                line += f"{va - val_auc:>+9.4f} "
            else:
                line += f"{'—':>9} "
            if test_auc is not None and ta is not None:
                line += f"{ta - test_auc:>+9.4f} "
            else:
                line += f"{'—':>9} "
        elif after:
            line += f"{'—':>9} {'—':>9} "

        line += f"{aff:>4}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="Write JSON snapshot")
    ap.add_argument("--compare", type=Path, help="Compare against prior snapshot JSON")
    ap.add_argument("--print-table", action="store_true")
    args = ap.parse_args()

    current = collect_all()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(current, indent=2))

    if args.compare:
        before = json.loads(args.compare.read_text())
        print_table(before, current)
    elif args.print_table:
        print_table(current)

    return current


if __name__ == "__main__":
    main()
