#!/usr/bin/env bash
# Binary K ablation (L=4/8/16) for primary model: hard_partition + token_abmil ent0.
# Reuses fold-wise AP prototypes at each L; K=8 from existing hard_partition run.
#
# Usage:
#   run_khead_hard_partition_token_abmil_k_ablation.sh all
#   run_khead_hard_partition_token_abmil_k_ablation.sh train K
#   run_khead_hard_partition_token_abmil_k_ablation.sh eval K
#   run_khead_hard_partition_token_abmil_k_ablation.sh summarize
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
ABL_ROOT="$REPO/herohe/gp2/runs/khead_hard_partition_token_abmil_k_ablation"
K8_SRC="$REPO/herohe/gp2/runs/khead_token_abmil_hard_partition_ent0"
LOG="$REPO/herohe/gp2/data/khead_hard_partition_token_abmil_k_ablation.log"
SUMMARY_JSON="$REPO/herohe/gp2/data/khead_hard_partition_token_abmil_k_ablation_summary.json"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

run_dir_for() {
  echo "$ABL_ROOT/k${1}_hard_partition_ent0"
}

ensure_protos() {
  local K="$1"
  local FOLD="$2"
  local PROTO="$REPO/herohe/gp2/data/prototypes_ap_phiher2fold_fold${FOLD}_train_L${K}.pt"
  if [[ ! -f "$PROTO" ]]; then
    echo "[protos] building L=${K} fold-${FOLD}..." >&2
    "$PY" "$REPO/herohe/gp2/scripts/init_prototypes_ap.py" \
      --features_dir "$FEAT" --folds_csv "$FOLDS" --val_fold "$FOLD" \
      --stage2_method kmeans --target_L "$K" --output "$PROTO" >&2
  fi
  echo "$PROTO"
}

train_fold() {
  local K="$1"
  local FOLD="$2"
  local RUN
  RUN="$(run_dir_for "$K")"
  local CKPT="$RUN/fold_${FOLD}/best.pt"
  if [[ -f "$CKPT" && "$FORCE" != "1" ]]; then
    echo "[skip] K=${K} fold ${FOLD}"
    return 0
  fi
  if [[ "$FORCE" == "1" && -d "$RUN/fold_${FOLD}" ]]; then
    mv "$RUN/fold_${FOLD}" "$RUN/fold_${FOLD}_prev_$(date +%Y%m%d_%H%M%S)"
  fi
  local PROTO
  PROTO="$(ensure_protos "$K" "$FOLD")"
  echo "========== $(date) K-ABL hard_partition token_abmil K=${K} fold-${FOLD} =========="
  "$PY" "$REPO/herohe/gp2/scripts/train_phenobin_mil.py" \
    --features_dir "$FEAT" --labels_csv "$LABELS" --folds_csv "$FOLDS" \
    --prototypes "$PROTO" --out_dir "$RUN" --device mps --only_fold "$FOLD" \
    --K "$K" --num_classes 2 --label_mode gt_binary \
    --readout khead --khead_pool token_abmil --khead_routing hard_partition \
    --proto_attn_bias 0 --dual_stream 0 --w_balance 0 --w_orth 0 \
    --epochs 50 --patience 10 --min_epochs 5 --min_epochs_for_selection 5 \
    --max_patches 4096 --val_max_patches -1 --val_subsample fixed \
    --select_on val_loss --hidden_dim 384 --lr 1e-4 --weight_decay 0.004 \
    --dropout 0.60 --patch_dropout 0.20 --label_smoothing 0.10 \
    --w_attn_entropy 0.0 --seed "$SEED" --cb_beta 0.999
}

train_k() {
  local K="$1"
  if [[ "$K" == "8" && "$FORCE" != "1" ]]; then
    echo "[K=8] using existing hard_partition ent0 run at $K8_SRC"
    for f in 0 1 2 3 4; do
      [[ -f "$K8_SRC/fold_${f}/best.pt" ]] || { echo "missing $K8_SRC/fold_${f}/best.pt"; return 1; }
    done
    return 0
  fi
  for f in 0 1 2 3 4; do train_fold "$K" "$f"; done
}

eval_k() {
  local K="$1"
  local RUN
  if [[ "$K" == "8" && "$FORCE" != "1" ]]; then
    RUN="$K8_SRC"
  else
    RUN="$(run_dir_for "$K")"
  fi
  local CKPTS=()
  for f in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${f}/best.pt"); done
  for c in "${CKPTS[@]}"; do [[ -f "$c" ]] || { echo "[eval] missing $c"; return 1; }; done
  local TEST_OUT="$ABL_ROOT/k${K}_hard_partition_ent0/test_eval"
  mkdir -p "$TEST_OUT"
  echo "========== $(date) K-ABL eval K=${K} hard_partition token_abmil =========="
  "$PY" "$REPO/herohe/gp2/scripts/eval_phenobin_test.py" \
    --checkpoint "${CKPTS[@]}" --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" --label_mode gt_binary --device mps \
    --max_patches -1 --tag "k${K}_hard_partition_ent0_5fold" \
    --out_dir "$TEST_OUT"
}

summarize() {
  "$PY" << 'PY'
import csv, json, os
from pathlib import Path
import numpy as np

repo = Path(os.environ.get("REPO", "."))
abl = repo / "herohe/gp2/runs/khead_hard_partition_token_abmil_k_ablation"
k8_src = repo / "herohe/gp2/runs/khead_token_abmil_hard_partition_ent0"
rows = []
for K in (4, 8, 16):
    if K == 8:
        summ = k8_src / "test_eval/metrics_khead_token_abmil_hard_partition_ent0_5fold.json"
        run = k8_src
    else:
        summ = abl / f"k{K}_hard_partition_ent0/test_eval/metrics_k{K}_hard_partition_ent0_5fold.json"
        run = abl / f"k{K}_hard_partition_ent0"
    rec = {"K": K, "run_dir": str(run), "recipe": "hard_partition+token_abmil ent0"}
    if summ.is_file():
        m = json.loads(summ.read_text())
        rec.update({
            "test_auc": round(m.get("AUC", m.get("auc_positive", float("nan"))), 4),
            "test_macro_f1": round(m["macro_f1"], 4),
            "test_bacc": round(m["bACC"], 4),
            "summary_path": str(summ),
        })
    else:
        rec["status"] = "pending"
    val_aucs = []
    for fold in range(5):
        log = run / f"fold_{fold}/log.csv"
        if not log.is_file():
            continue
        with log.open() as fh:
            lr = list(csv.DictReader(fh))
        eligible = [r for r in lr if int(float(r["epoch"])) >= 5]
        if eligible:
            best = min(eligible, key=lambda r: float(r["val_loss"]))
            val_aucs.append(float(best.get("val_auc_positive", best.get("val_auc", 0))))
    if val_aucs:
        rec["val_auc_mean"] = round(float(np.mean(val_aucs)), 4)
        rec["val_auc_std"] = round(float(np.std(val_aucs)), 4)
    rows.append(rec)

out = repo / "herohe/gp2/data/khead_hard_partition_token_abmil_k_ablation_summary.json"
out.write_text(json.dumps({"recipe": "hard_partition token_abmil ent0", "results": rows}, indent=2))
print("\n=== K ablation: hard_partition + token_abmil (binary) ===")
for r in sorted(rows, key=lambda x: x.get("test_auc", 0), reverse=True):
    if "test_auc" in r:
        print(f"  K={r['K']:2d}: AUC={r['test_auc']:.4f}  F1={r['test_macro_f1']:.4f}  bACC={r['test_bacc']:.4f}")
    else:
        print(f"  K={r['K']:2d}: PENDING")
print(f"\nWrote {out}")
PY
}

cmd="${1:-all}"
mkdir -p "$ABL_ROOT"
exec >> "$LOG" 2>&1
echo "========== $(date) hard_partition token_abmil K ablation: $cmd =========="

case "$cmd" in
  train) [[ -n "${2:-}" ]] || { echo "usage: train K"; exit 1; }; train_k "$2" ;;
  eval)  [[ -n "${2:-}" ]] || { echo "usage: eval K"; exit 1; }; eval_k "$2"; summarize ;;
  all)
    for K in 4 16; do train_k "$K"; eval_k "$K"; done
    eval_k 8
    summarize
    ;;
  summarize) summarize ;;
  *) echo "usage: $0 {all|train K|eval K|summarize}"; exit 1 ;;
esac

echo "========== $(date) DONE =========="
