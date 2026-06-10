#!/usr/bin/env bash
# AttnMISL binary baseline on ResNet-50 bags (1024-D).
# Same HEROHE protocol as run_attnmisl_binary_5fold_s42.sh; fold-wise AP prototypes (L=8).
#
# Usage:
#   run_attnmisl_binary_resnet50_5fold_s42.sh [fold|all|eval]
set -euo pipefail

export PYTHONUNBUFFERED=1

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
FEAT="$REPO/herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_resnet50"
TEST_FEAT="$REPO/herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_resnet50"
LABELS="$REPO/herohe/Training (ground truth).csv"
TEST_LABELS="$REPO/herohe/Test (ground truth)(1).xlsx"
FOLDS="$REPO/herohe/gp2/data/folds_phiher2_binary_s42.csv"
RUN="$REPO/herohe/gp2/runs/attnmisl_paper_binary_resnet50_5fold_s42_valloss"
TEST_OUT="$REPO/herohe/gp2/runs/test_eval_mil/resnet50_binary_5fold_s42"
LOG="$REPO/herohe/gp2/data/attnmisl_paper_binary_resnet50_5fold_s42.log"
L=8
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

FOLD="${1:-all}"

COMMON_TRAIN=(
  --aggregator attnmisl
  --num_classes 2
  --label_mode gt_binary
  --features_dir "$FEAT"
  --labels_csv "$LABELS"
  --folds_csv "$FOLDS"
  --device mps
  --epochs 50
  --patience 10
  --min_epochs 5
  --min_epochs_for_selection 5
  --select_on val_loss
  --lr 1e-4
  --weight_decay 0.002
  --feature_dim 1024
  --max_patches 4096
  --val_subsample fixed
  --label_smoothing 0.15
  --ce_class_weights effective
  --cb_beta 0.999
  --seed "$SEED"
  --attnmisl_cluster_num "$L"
  --attnmisl_dropout 0.5
)

ensure_protos() {
  local FOLD_ID="$1"
  local PROTO="$REPO/herohe/gp2/data/prototypes_ap_resnet50_fold${FOLD_ID}_train_L${L}.pt"
  if [[ ! -f "$PROTO" ]]; then
    echo "[proto] building $PROTO"
    "$PY" "$REPO/herohe/gp2/scripts/init_prototypes_ap.py" \
      --features_dir "$FEAT" \
      --folds_csv "$FOLDS" \
      --val_fold "$FOLD_ID" \
      --stage2_method kmeans \
      --target_L "$L" \
      --output "$PROTO" >&2
  fi
  echo "$PROTO"
}

train_one() {
  local F="$1"
  local CKPT="$RUN/fold_${F}/best.pt"
  if [[ -f "$CKPT" && "$FORCE" != "1" ]]; then
    echo "[skip] attnmisl resnet50 fold $F"
    return 0
  fi
  if [[ "$FORCE" == "1" && -d "$RUN/fold_${F}" ]]; then
    mv "$RUN/fold_${F}" "$RUN/fold_${F}_prev_$(date +%Y%m%d_%H%M%S)"
  fi
  local PROTO
  PROTO="$(ensure_protos "$F")"
  echo "========== $(date) attnmisl resnet50 fold-$F START proto=$PROTO =========="
  "$PY" "$REPO/herohe/gp2/scripts/train_mil_baseline.py" \
    --out_dir "$RUN" \
    --only_fold "$F" \
    --prototypes "$PROTO" \
    "${COMMON_TRAIN[@]}"
}

eval_all() {
  local TAG="attnmisl_paper_binary_resnet50_5fold_s42_valloss_5fold"
  local CKPTS=()
  for f in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${f}/best.pt"); done
  for c in "${CKPTS[@]}"; do [[ -f "$c" ]] || { echo "[eval] missing $c"; return 1; }; done
  "$PY" "$REPO/herohe/gp2/scripts/eval_mil_baseline_test.py" \
    --checkpoint "${CKPTS[@]}" \
    --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" \
    --label_mode gt_binary \
    --device mps \
    --max_patches 4096 \
    --tag "$TAG" \
    --out_dir "$TEST_OUT"
}

exec >> "$LOG" 2>&1
echo "========== $(date) attnmisl resnet50 binary 5-fold-s42 fold=$FOLD =========="

if [[ "$FOLD" == "eval" ]]; then
  eval_all
  exit 0
fi

if [[ "$FOLD" == "all" ]]; then
  for f in 0 1 2 3 4; do train_one "$f"; done
  eval_all
else
  train_one "$FOLD"
fi

echo "========== $(date) attnmisl resnet50 binary DONE =========="
