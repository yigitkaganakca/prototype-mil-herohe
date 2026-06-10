#!/usr/bin/env bash
# Binary Virchow2 khead ablation: readout pool mean vs concat.
# Fixed ls10 recipe, L=8, min_sel=5. Mean reuses existing ls10 run unless FORCE=1.
#
# Usage:
#   run_khead_pool_ablation.sh all
#   run_khead_pool_ablation.sh train mean|concat
#   run_khead_pool_ablation.sh eval mean|concat
#   run_khead_pool_ablation.sh summarize
set -euo pipefail

export PYTHONUNBUFFERED=1

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
export REPO
PY="${PY:-python}"
FEAT="$REPO/herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
TEST_FEAT="$REPO/herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2"
LABELS="$REPO/herohe/Training (ground truth).csv"
TEST_LABELS="$REPO/herohe/Test (ground truth)(1).xlsx"
FOLDS="$REPO/herohe/gp2/data/folds_phiher2_binary_s42.csv"
ABL_ROOT="$REPO/herohe/gp2/runs/khead_pool_ablation"
MEAN_SRC="$REPO/herohe/gp2/runs/khead_reg_sweep/reg_d06_wd4e3_ent15_pd20_ls10"
LOG="$REPO/herohe/gp2/data/khead_pool_ablation.log"
SUMMARY_JSON="$REPO/herohe/gp2/data/khead_pool_ablation_summary.json"
L=8
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

DROPOUT="0.60"
WD="0.004"
ENT="0.15"
PATCH_DROP="0.20"
LS="0.10"

run_dir_for() {
  echo "$ABL_ROOT/${1}_ls10"
}

ensure_protos() {
  local FOLD="$1"
  local PROTO="$REPO/herohe/gp2/data/prototypes_ap_phiher2fold_fold${FOLD}_train_L${L}.pt"
  if [[ ! -f "$PROTO" ]]; then
    echo "[protos] building L=${L} fold-${FOLD}..." >&2
    "$PY" "$REPO/herohe/gp2/scripts/init_prototypes_ap.py" \
      --features_dir "$FEAT" \
      --folds_csv "$FOLDS" \
      --val_fold "$FOLD" \
      --stage2_method kmeans \
      --target_L "$L" \
      --output "$PROTO" >&2
  fi
  echo "$PROTO"
}

train_fold() {
  local POOL="$1"
  local FOLD="$2"
  local RUN
  RUN="$(run_dir_for "$POOL")"
  local CKPT="$RUN/fold_${FOLD}/best.pt"
  if [[ -f "$CKPT" && "$FORCE" != "1" ]]; then
    echo "[skip] pool=${POOL} fold ${FOLD}"
    return 0
  fi
  if [[ "$FORCE" == "1" && -d "$RUN/fold_${FOLD}" ]]; then
    mv "$RUN/fold_${FOLD}" "$RUN/fold_${FOLD}_prev_$(date +%Y%m%d_%H%M%S)"
  fi
  local PROTO
  PROTO="$(ensure_protos "$FOLD")"
  echo "========== $(date) POOL-ABLATION train pool=${POOL} fold-${FOLD} L=${L} ls=${LS} =========="
  "$PY" "$REPO/herohe/gp2/scripts/train_phenobin_mil.py" \
    --features_dir "$FEAT" \
    --labels_csv "$LABELS" \
    --folds_csv "$FOLDS" \
    --prototypes "$PROTO" \
    --out_dir "$RUN" \
    --device mps \
    --only_fold "$FOLD" \
    --K "$L" \
    --num_classes 2 \
    --label_mode gt_binary \
    --readout khead \
    --khead_pool "$POOL" \
    --proto_attn_bias 0 \
    --dual_stream 0 \
    --w_balance 0 \
    --w_orth 0 \
    --epochs 50 \
    --patience 10 \
    --min_epochs 5 \
    --min_epochs_for_selection 5 \
    --max_patches 4096 \
    --val_max_patches -1 \
    --val_subsample fixed \
    --select_on val_loss \
    --hidden_dim 384 \
    --lr 1e-4 \
    --weight_decay "$WD" \
    --dropout "$DROPOUT" \
    --patch_dropout "$PATCH_DROP" \
    --label_smoothing "$LS" \
    --w_attn_entropy "$ENT" \
    --seed "$SEED" \
    --cb_beta 0.999
}

train_pool() {
  local POOL="$1"
  if [[ "$POOL" == "mean" && "$FORCE" != "1" ]]; then
    echo "[mean] using existing ls10 run at $MEAN_SRC"
    for f in 0 1 2 3 4; do
      [[ -f "$MEAN_SRC/fold_${f}/best.pt" ]] || { echo "missing $MEAN_SRC/fold_${f}/best.pt"; return 1; }
    done
    return 0
  fi
  for f in 0 1 2 3 4; do
    train_fold "$POOL" "$f"
  done
}

eval_pool() {
  local POOL="$1"
  local RUN TAG
  if [[ "$POOL" == "mean" && "$FORCE" != "1" ]]; then
    RUN="$MEAN_SRC"
    TAG="reg_d06_wd4e3_ent15_pd20_ls10_5fold"
    TEST_OUT="$MEAN_SRC/test_eval"
  else
    RUN="$(run_dir_for "$POOL")"
    TAG="${POOL}_ls10_5fold"
    TEST_OUT="$ABL_ROOT/${POOL}_ls10/test_eval"
  fi
  local CKPTS=()
  for f in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${f}/best.pt"); done
  for c in "${CKPTS[@]}"; do [[ -f "$c" ]] || { echo "[eval] missing $c"; return 1; }; done
  mkdir -p "$TEST_OUT"
  echo "========== $(date) POOL-ABLATION eval pool=${POOL} =========="
  "$PY" "$REPO/herohe/gp2/scripts/eval_phenobin_test.py" \
    --checkpoint "${CKPTS[@]}" \
    --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" \
    --label_mode gt_binary \
    --device mps \
    --max_patches 4096 \
    --tag "$TAG" \
    --out_dir "$TEST_OUT"
}

summarize() {
  "$PY" << 'PY'
import csv
import json
import os
from pathlib import Path
import numpy as np

repo = Path(os.environ.get("REPO", "."))
abl = repo / "herohe/gp2/runs/khead_pool_ablation"
mean_src = repo / "herohe/gp2/runs/khead_reg_sweep/reg_d06_wd4e3_ent15_pd20_ls10"
rows = []

for pool in ("mean", "concat"):
    if pool == "mean":
        summ = mean_src / "test_eval" / "summary_reg_d06_wd4e3_ent15_pd20_ls10_5fold.json"
        run = mean_src
        note = "existing ls10 tuned run (khead_pool=mean)"
    else:
        summ = abl / f"{pool}_ls10/test_eval/summary_{pool}_ls10_5fold.json"
        run = abl / f"{pool}_ls10"
        note = ""
    rec = {"pool": pool, "tag": f"{pool}_ls10", "run_dir": str(run), "note": note}
    if summ.is_file():
        m = json.loads(summ.read_text())["results"][0]
        rec.update({
            "test_auc": round(m.get("AUC", m.get("auc_positive", float("nan"))), 4),
            "test_macro_f1": round(m["macro_f1"], 4),
            "test_bacc": round(m["bACC"], 4),
            "test_pos_f1": round(m.get("posF1", float("nan")), 4),
            "summary_path": str(summ),
        })
    else:
        rec["status"] = "pending"

    val_aucs = []
    for fold in range(5):
        log = run / f"fold_{fold}/log.csv"
        if not log.is_file():
            continue
        lr = list(csv.DictReader(log.open()))
        eligible = [r for r in lr if int(float(r["epoch"])) >= 5]
        if not eligible:
            continue
        best = min(eligible, key=lambda r: float(r["val_loss"]))
        val_aucs.append(float(best.get("val_auc_positive", best.get("val_auc", 0))))
    if val_aucs:
        rec["val_auc_mean"] = round(float(np.mean(val_aucs)), 4)
        rec["val_auc_std"] = round(float(np.std(val_aucs)), 4)
    rows.append(rec)

out = repo / "herohe/gp2/data/khead_pool_ablation_summary.json"
out.write_text(json.dumps({
    "recipe": "L=8 d0.60 wd4e-3 ent0.15 pd0.20 ls0.10 min_sel5",
    "results": rows,
}, indent=2))
print("\n=== khead pool ablation (binary Virchow2) ===")
for r in rows:
    if "test_auc" in r:
        print(f"  {r['pool']:6s}: AUC={r['test_auc']:.4f}  F1={r['test_macro_f1']:.4f}  bACC={r['test_bacc']:.4f}")
    else:
        print(f"  {r['pool']:6s}: PENDING")
print(f"\nWrote {out}")
PY
}

cmd="${1:-all}"
shift || true

mkdir -p "$ABL_ROOT"
exec >> >(tee -a "$LOG") 2>&1
echo "========== $(date) khead_pool_ablation $cmd =========="

case "$cmd" in
  train)
    [[ -n "${1:-}" ]] || { echo "usage: train mean|concat"; exit 1; }
    train_pool "$1"
    ;;
  eval)
    [[ -n "${1:-}" ]] || { echo "usage: eval mean|concat"; exit 1; }
    eval_pool "$1"
    summarize
    ;;
  all)
    train_pool concat
    eval_pool concat
    eval_pool mean
    summarize
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "usage: $0 {all|train POOL|eval POOL|summarize}"
    exit 1
    ;;
esac

echo "========== $(date) khead_pool_ablation DONE =========="
