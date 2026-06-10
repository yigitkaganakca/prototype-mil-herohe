#!/usr/bin/env bash
# Medoid benchmark: all reported PhenoBIN ablation + three-class configs under the
# SINGLE primary recipe (d=384, ent0, dropout=0.60, patch_dropout=0.20, wd=4e-3,
# label_smoothing 0.10 binary / 0.15 three-class, select_on=val_loss, train cap 4096,
# full-bag test eval), varying ONLY routing / readout pool / L / task. Real-patch MEDOID
# prototypes throughout (see make_medoid_prototypes.py).
#
# These produce the report's binary ablation table, prototype-count (L) ablation, the
# three-class headline, and the three-class ablations. The primary binary config
# (hard x token x L8) is trained separately by run_medoid_primary_binary.sh and is NOT
# repeated here. Skips folds whose best.pt already exists.
#
# Usage: run_medoid_benchmark.sh
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
FEAT="$REPO/herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
TEST_FEAT="$REPO/herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2"
LABELS="$REPO/herohe/Training (ground truth).csv"
TEST_LABELS="$REPO/herohe/Test (ground truth)(1).xlsx"
FOLDS="$REPO/herohe/gp2/data/folds_phiher2_binary_s42.csv"
ROOT="$REPO/herohe/gp2/runs/medoid_benchmark"
TRAIN="$REPO/herohe/gp2/scripts/train_phenobin_mil.py"
EVAL="$REPO/herohe/gp2/scripts/eval_phenobin_test.py"
mkdir -p "$ROOT"

proto() { echo "$REPO/herohe/gp2/data/prototypes_medoid_phiher2fold_fold${1}_train_L${2}.pt"; }

# train_eval NAME L ROUTING POOL TASK   (TASK = binary | three)
train_eval() {
  local NAME="$1" L="$2" ROUTING="$3" POOL="$4" TASK="$5"
  local RUN="$ROOT/$NAME"
  local NUMC LABELMODE LS EXTRA
  if [[ "$TASK" == "three" ]]; then
    NUMC=3; LABELMODE="valieris_3"; LS="0.15"
    EXTRA=(--val_loss_ema_alpha 0 --mixup_alpha 0 --mixup_p 0)
  else
    NUMC=2; LABELMODE="gt_binary"; LS="0.10"; EXTRA=()
  fi
  echo "########## $(date) CONFIG $NAME (L=$L routing=$ROUTING pool=$POOL task=$TASK) ##########"
  for F in 0 1 2 3 4; do
    if [[ -f "$RUN/fold_${F}/best.pt" ]]; then echo "[skip] $NAME fold $F"; continue; fi
    local PROTO; PROTO="$(proto "$F" "$L")"
    [[ -f "$PROTO" ]] || { echo "[ERR] missing $PROTO (run make_medoid_prototypes.py first)"; return 1; }
    echo "===== $(date) train $NAME fold-$F ====="
    "$PY" "$TRAIN" \
      --features_dir "$FEAT" --labels_csv "$LABELS" --folds_csv "$FOLDS" \
      --prototypes "$PROTO" --out_dir "$RUN" --device mps --only_fold "$F" \
      --K "$L" --num_classes "$NUMC" --label_mode "$LABELMODE" \
      --readout khead --khead_pool "$POOL" --khead_routing "$ROUTING" \
      --proto_attn_bias 0 --dual_stream 0 --w_balance 0 --w_orth 0 \
      --epochs 50 --patience 10 --min_epochs 5 --min_epochs_for_selection 5 \
      --max_patches 4096 --val_max_patches -1 --val_subsample fixed \
      --select_on val_loss --hidden_dim 384 --lr 1e-4 --weight_decay 0.004 \
      --dropout 0.60 --patch_dropout 0.20 --label_smoothing "$LS" \
      --w_attn_entropy 0.0 --seed 0 --cb_beta 0.999 ${EXTRA[@]+"${EXTRA[@]}"}
  done
  # ensemble test eval, full bag
  local CKPTS=(); for F in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${F}/best.pt"); done
  for c in "${CKPTS[@]}"; do [[ -f "$c" ]] || { echo "[eval] missing $c, skip eval"; return 0; }; done
  echo "===== $(date) test-eval $NAME (full bag) ====="
  "$PY" "$EVAL" --checkpoint "${CKPTS[@]}" --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" --label_mode "$LABELMODE" --device mps \
    --max_patches -1 --tag "${NAME}_5fold" --out_dir "$RUN/test_eval"
}

echo "========== $(date) MEDOID BENCHMARK START =========="
# Binary readout / routing ablations (L8) under the primary recipe
train_eval "bin_indep_token_L8"  8  independent     token_abmil binary
train_eval "bin_indep_mean_L8"   8  independent     mean        binary
train_eval "bin_hard_mean_L8"    8  hard_partition  mean        binary
train_eval "bin_hard_concat_L8"  8  hard_partition  concat      binary
# Binary prototype-count (L) ablation (hard x token)
train_eval "bin_hard_token_L4"   4  hard_partition  token_abmil binary
train_eval "bin_hard_token_L16" 16  hard_partition  token_abmil binary
# Three-class (valieris_3): primary + ablations
train_eval "tri_hard_token_L8"   8  hard_partition  token_abmil three
train_eval "tri_indep_token_L8"  8  independent     token_abmil three
train_eval "tri_indep_mean_L8"   8  independent     mean        three
train_eval "tri_hard_mean_L8"    8  hard_partition  mean        three
train_eval "tri_hard_concat_L8"  8  hard_partition  concat      three
echo "========== $(date) MEDOID BENCHMARK DONE =========="
