#!/usr/bin/env bash
# PRIMARY binary model (reported): prototype-guided multi-head MIL with
#   hard-partition routing + token-level ABMIL readout, L=8, real-patch MEDOID
#   prototypes, d=384, no attention-entropy penalty (ent0).
# This is the run behind the headline binary numbers (test AUC 0.826, macro-F1 0.732)
# and the interpretability figures. Requires the medoid prototype files built by
# make_medoid_prototypes.py (see README step 2).
#
# The output directory is kept as khead_hard_partition_medoid_proto_control/ because the
# metrics scripts (compute_uncertainty.py, recompute_report_stats.py) read that path.
#
# Usage: run_medoid_primary_binary.sh [fold|all|eval]
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
FEAT="$REPO/herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
TEST_FEAT="$REPO/herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2"
LABELS="$REPO/herohe/Training (ground truth).csv"
TEST_LABELS="$REPO/herohe/Test (ground truth)(1).xlsx"
FOLDS="$REPO/herohe/gp2/data/folds_phiher2_binary_s42.csv"
RUN="$REPO/herohe/gp2/runs/khead_hard_partition_medoid_proto_control"
LOG="$REPO/herohe/gp2/data/medoid_primary_binary.log"
L=8
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

FOLD="${1:-all}"

proto() { echo "$REPO/herohe/gp2/data/prototypes_medoid_phiher2fold_fold${1}_train_L${L}.pt"; }

train_one() {
  local F="$1"
  local CKPT="$RUN/fold_${F}/best.pt"
  if [[ -f "$CKPT" && "$FORCE" != "1" ]]; then
    echo "[skip] medoid primary binary fold $F -> $CKPT"
    return 0
  fi
  if [[ "$FORCE" == "1" && -d "$RUN/fold_${F}" ]]; then
    mv "$RUN/fold_${F}" "$RUN/fold_${F}_prev_$(date +%Y%m%d_%H%M%S)"
  fi
  local PROTO; PROTO="$(proto "$F")"
  [[ -f "$PROTO" ]] || { echo "[ERR] missing $PROTO (run make_medoid_prototypes.py first)"; return 1; }
  echo "========== $(date) medoid primary binary fold-$F START =========="
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
  local CKPTS=()
  for f in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${f}/best.pt"); done
  for c in "${CKPTS[@]}"; do [[ -f "$c" ]] || { echo "[eval] missing $c"; return 1; }; done
  mkdir -p "$RUN/test_eval"
  "$PY" "$REPO/herohe/gp2/scripts/eval_phenobin_test.py" \
    --checkpoint "${CKPTS[@]}" --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" --label_mode gt_binary --device mps \
    --max_patches -1 --tag medoid_proto_5fold --out_dir "$RUN/test_eval"
}

exec >> "$LOG" 2>&1
echo "========== $(date) medoid primary binary fold=$FOLD =========="

if [[ "$FOLD" == "eval" ]]; then eval_all; exit 0; fi
if [[ "$FOLD" == "all" ]]; then
  for f in 0 1 2 3 4; do train_one "$f"; done
  eval_all
else
  train_one "$FOLD"
fi
echo "========== $(date) medoid primary binary DONE =========="
