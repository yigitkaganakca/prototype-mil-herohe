#!/usr/bin/env bash
# ResNet50 MIL baselines (ISH neg/pos) — matched protocol on TRIDENT ResNet50 bags (1024-D).
# Models: ABMIL, CLAM, TransMIL only.
#
# ResNet PhenoBIN (our model) is trained separately:
#   run_khead_token_abmil_hard_partition_ent0_resnet50.sh
#
# Usage:
#   run_resnet50_binary_5fold_s42.sh abmil|clam|transmil|all [fold|all]
#   run_resnet50_binary_5fold_s42.sh eval
set -euo pipefail

export PYTHONUNBUFFERED=1

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
FEAT="$REPO/herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_resnet50"
TEST_FEAT="$REPO/herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_resnet50"
LABELS="$REPO/herohe/Training (ground truth).csv"
TEST_LABELS="$REPO/herohe/Test (ground truth)(1).xlsx"
FOLDS="$REPO/herohe/gp2/data/folds_phiher2_binary_s42.csv"
TEST_OUT="$REPO/herohe/gp2/runs/test_eval_mil/resnet50_binary_5fold_s42"
LOG="$REPO/herohe/gp2/data/resnet50_binary_5fold_s42.log"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

AGG="${1:-all}"
FOLD="${2:-all}"

COMMON_TRAIN=(
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
)

run_dir_for() {
  case "$1" in
    abmil) echo "$REPO/herohe/gp2/runs/abmil_resnet50_5fold_s42_valloss" ;;
    clam) echo "$REPO/herohe/gp2/runs/clam_resnet50_5fold_s42_valloss" ;;
    transmil) echo "$REPO/herohe/gp2/runs/transmil_resnet50_5fold_s42_valloss" ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
}

extra_args_for() {
  case "$1" in
    abmil) echo --abmil_hidden 512 --abmil_attn 256 --abmil_dropout 0.5 --w_attn_entropy 0.05 ;;
    clam) echo --clam_dropout 0.5 --k_sample 8 --bag_weight 0.7 ;;
    transmil) echo --trans_d_model 512 --trans_layers 2 --trans_heads 8 --trans_dropout 0.25 ;;
  esac
}

train_one() {
  local A="$1" F="$2"
  local RUN CKPT
  RUN="$(run_dir_for "$A")"
  CKPT="$RUN/fold_${F}/best.pt"
  if [[ -f "$CKPT" && "$FORCE" != "1" ]]; then
    echo "[skip] resnet50 $A fold $F"
    return 0
  fi
  if [[ "$FORCE" == "1" && -d "$RUN/fold_${F}" ]]; then
    mv "$RUN/fold_${F}" "$RUN/fold_${F}_prev_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "========== $(date) resnet50 $A fold-$F START =========="
  # shellcheck disable=SC2046
  "$PY" "$REPO/herohe/gp2/scripts/train_mil_baseline.py" \
    --aggregator "$A" \
    --out_dir "$RUN" \
    --only_fold "$F" \
    "${COMMON_TRAIN[@]}" \
    $(extra_args_for "$A")
}

train_agg() {
  local A="$1" FS="$2"
  if [[ "$FS" == "all" ]]; then
    for f in 0 1 2 3 4; do train_one "$A" "$f"; done
  else
    train_one "$A" "$FS"
  fi
}

eval_baseline() {
  local A="$1"
  local RUN TAG
  RUN="$(run_dir_for "$A")"
  TAG="${A}_resnet50_5fold_s42_valloss_5fold"
  local CKPTS=()
  for f in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${f}/best.pt"); done
  for c in "${CKPTS[@]}"; do [[ -f "$c" ]] || { echo "[eval] missing $c"; return 1; }; done
  "$PY" "$REPO/herohe/gp2/scripts/eval_mil_baseline_test.py" \
    --checkpoint "${CKPTS[@]}" \
    --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" \
    --label_mode gt_binary \
    --device mps \
    --max_patches -1 \
    --tag "$TAG" \
    --out_dir "$TEST_OUT"
}

eval_all() {
  for A in abmil clam transmil; do eval_baseline "$A" || true; done
}

exec > >(tee -a "$LOG") 2>&1
echo "========== $(date) resnet50_binary agg=$AGG fold=$FOLD =========="

NTEST=$(find "$TEST_FEAT" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$AGG" == "eval" || "$FOLD" == "all" ]] && [[ "$NTEST" -lt 150 ]]; then
  echo "ERROR: need 150 ResNet50 test h5 (have $NTEST). Run run_trident_test_resnet50_feat.sh first."
  exit 1
fi

if [[ "$AGG" == "eval" ]]; then
  eval_all
  exit 0
fi

if [[ "$AGG" == "all" ]]; then
  for A in abmil clam transmil; do
    train_agg "$A" "$FOLD"
    [[ "$FOLD" == "all" ]] && eval_baseline "$A"
  done
else
  train_agg "$AGG" "$FOLD"
  [[ "$FOLD" == "all" ]] && eval_baseline "$AGG"
fi

echo "========== $(date) resnet50_binary DONE =========="
