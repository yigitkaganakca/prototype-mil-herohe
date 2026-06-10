#!/usr/bin/env bash
# Binary token_abmil + hard_partition routing — ls10 ent0 recipe.
#
# Usage:
#   run_khead_token_abmil_hard_partition_ent0.sh [fold|all|eval]
set -euo pipefail

export PYTHONUNBUFFERED=1

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
FEAT="$REPO/herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
TEST_FEAT="$REPO/herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2"
LABELS="$REPO/herohe/Training (ground truth).csv"
TEST_LABELS="$REPO/herohe/Test (ground truth)(1).xlsx"
FOLDS="$REPO/herohe/gp2/data/folds_phiher2_binary_s42.csv"
RUN="$REPO/herohe/gp2/runs/khead_token_abmil_hard_partition_ent0"
LOG="$REPO/herohe/gp2/data/khead_token_abmil_hard_partition_ent0.log"
L=8
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

FOLD="${1:-all}"

ensure_protos() {
  local FOLD_ID="$1"
  local PROTO="$REPO/herohe/gp2/data/prototypes_ap_phiher2fold_fold${FOLD_ID}_train_L${L}.pt"
  if [[ ! -f "$PROTO" ]]; then
    "$PY" "$REPO/herohe/gp2/scripts/init_prototypes_ap.py" \
      --features_dir "$FEAT" --folds_csv "$FOLDS" --val_fold "$FOLD_ID" \
      --stage2_method kmeans --target_L "$L" --output "$PROTO" >&2
  fi
  echo "$PROTO"
}

train_one() {
  local F="$1"
  local CKPT="$RUN/fold_${F}/best.pt"
  if [[ -f "$CKPT" && "$FORCE" != "1" ]]; then
    echo "[skip] hard_partition ent0 binary fold $F → $CKPT"
    return 0
  fi
  if [[ "$FORCE" == "1" && -d "$RUN/fold_${F}" ]]; then
    mv "$RUN/fold_${F}" "$RUN/fold_${F}_prev_$(date +%Y%m%d_%H%M%S)"
  fi
  local PROTO
  PROTO="$(ensure_protos "$F")"
  echo "========== $(date) hard_partition ent0 binary fold-$F START =========="
  "$PY" "$REPO/herohe/gp2/scripts/train_phenobin_mil.py" \
    --features_dir "$FEAT" --labels_csv "$LABELS" --folds_csv "$FOLDS" \
    --prototypes "$PROTO" --out_dir "$RUN" --device mps --only_fold "$F" \
    --K "$L" --num_classes 2 --label_mode gt_binary \
    --readout khead --khead_pool token_abmil --khead_routing hard_partition \
    --proto_attn_bias 0 --dual_stream 0 --w_balance 0 --w_orth 0 \
    --epochs 50 --patience 10 --min_epochs 5 --min_epochs_for_selection 5 \
    --max_patches 4096 --val_max_patches -1 --val_subsample fixed \
    --select_on val_loss --hidden_dim 384 --lr 1e-4 --weight_decay 0.004 \
    --dropout 0.60 --patch_dropout 0.20 --label_smoothing 0.10 \
    --w_attn_entropy 0.0 --seed "$SEED" --cb_beta 0.999
}

eval_all() {
  local TAG="khead_token_abmil_hard_partition_ent0_5fold"
  local CKPTS=()
  for f in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${f}/best.pt"); done
  for c in "${CKPTS[@]}"; do [[ -f "$c" ]] || { echo "[eval] missing $c"; return 1; }; done
  mkdir -p "$RUN/test_eval"
  "$PY" "$REPO/herohe/gp2/scripts/eval_phenobin_test.py" \
    --checkpoint "${CKPTS[@]}" --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" --label_mode gt_binary --device mps \
    --max_patches -1 --tag "$TAG" --out_dir "$RUN/test_eval"
}

exec >> "$LOG" 2>&1
echo "========== $(date) hard_partition ent0 binary fold=$FOLD =========="

if [[ "$FOLD" == "eval" ]]; then eval_all; exit 0; fi
if [[ "$FOLD" == "all" ]]; then
  for f in 0 1 2 3 4; do train_one "$f"; done
  eval_all
else
  train_one "$FOLD"
fi
echo "========== $(date) hard_partition ent0 binary DONE =========="
