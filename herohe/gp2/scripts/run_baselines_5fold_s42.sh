#!/usr/bin/env bash
# Virchow2 MIL baselines — matched protocol for binary and three-class HEROHE:
#   5-fold stratified CV (folds_phiher2_binary_s42.csv) + val_loss + 5-fold test ensemble.
#
# PhenoBIN (our model) is NOT trained here:
#   binary  → run_medoid_primary_binary.sh
#   3-class → run_medoid_benchmark.sh (tri_hard_token_L8)
#
# Usage:
#   run_baselines_5fold_s42.sh binary  abmil|clam|transmil|all [fold|all]
#   run_baselines_5fold_s42.sh threeclass abmil|clam|transmil|all [fold|all]
#   run_baselines_5fold_s42.sh binary eval
#   run_baselines_5fold_s42.sh threeclass eval
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
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

TASK="${1:-}"
AGG="${2:-all}"
FOLD="${3:-all}"

if [[ -z "$TASK" || "$TASK" == "--help" || "$TASK" == "-h" ]]; then
  sed -n '2,14p' "$0" | sed 's/^# \?//'
  exit 0
fi

case "$TASK" in
  binary)
    NUM_CLASSES=2
    LABEL_MODE="gt_binary"
    TEST_OUT="$REPO/herohe/gp2/runs/test_eval_mil/binary_5fold_s42"
    LOG="$REPO/herohe/gp2/data/binary_baselines_5fold_s42.log"
    EVAL_TAG_SUFFIX="binary_5fold_s42_valloss_5fold"
    ;;
  threeclass)
    NUM_CLASSES=3
    LABEL_MODE="valieris_3"
    TEST_OUT="$REPO/herohe/gp2/runs/test_eval_mil/valieris3_5fold_s42"
    LOG="$REPO/herohe/gp2/data/threeclass_baselines_5fold_s42.log"
    EVAL_TAG_SUFFIX="valieris3_5fold_s42_valloss_5fold"
    ;;
  *)
    echo "unknown task: $TASK (expected binary|threeclass)" >&2
    exit 1
    ;;
esac

COMMON_TRAIN=(
  --num_classes "$NUM_CLASSES"
  --label_mode "$LABEL_MODE"
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
  --feature_dim 2560
  --max_patches 4096
  --val_subsample fixed
  --label_smoothing 0.15
  --ce_class_weights effective
  --cb_beta 0.999
  --seed "$SEED"
)

run_dir_for() {
  local A="$1"
  local canonical legacy
  case "$TASK" in
    binary)
      case "$A" in
        abmil)
          canonical="$REPO/herohe/gp2/runs/abmil_binary_5fold_s42_valloss"
          legacy="$REPO/herohe/gp2/runs/abmil_phiher2fold_valloss"
          ;;
        clam)
          canonical="$REPO/herohe/gp2/runs/clam_binary_5fold_s42_valloss"
          legacy="$REPO/herohe/gp2/runs/clam_phiher2fold_valloss"
          ;;
        transmil)
          canonical="$REPO/herohe/gp2/runs/transmil_binary_5fold_s42_valloss"
          legacy="$REPO/herohe/gp2/runs/transmil_phiher2fold_valloss"
          ;;
        *) echo "unknown aggregator: $A" >&2; exit 1 ;;
      esac
      ;;
    threeclass)
      case "$A" in
        abmil) echo "$REPO/herohe/gp2/runs/abmil_valieris3_5fold_s42_valloss"; return ;;
        clam) echo "$REPO/herohe/gp2/runs/clam_valieris3_5fold_s42_valloss"; return ;;
        transmil) echo "$REPO/herohe/gp2/runs/transmil_valieris3_5fold_s42_valloss"; return ;;
        *) echo "unknown aggregator: $A" >&2; exit 1 ;;
      esac
      ;;
  esac
  if [[ -d "$canonical" ]]; then
    echo "$canonical"
  elif [[ -d "$legacy" ]]; then
    echo "$legacy"
  else
    echo "$canonical"
  fi
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
    echo "[skip] $TASK $A fold $F → $CKPT"
    return 0
  fi
  if [[ "$FORCE" == "1" && -d "$RUN/fold_${F}" ]]; then
    mv "$RUN/fold_${F}" "$RUN/fold_${F}_prev_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "========== $(date) $TASK $A fold-$F START =========="
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

eval_agg() {
  local A="$1"
  local RUN TAG
  RUN="$(run_dir_for "$A")"
  TAG="${A}_${EVAL_TAG_SUFFIX}"
  local CKPTS=()
  for f in 0 1 2 3 4; do CKPTS+=("$RUN/fold_${f}/best.pt"); done
  for c in "${CKPTS[@]}"; do
    if [[ ! -f "$c" ]]; then
      echo "[eval] $A missing $c — skip"
      return 1
    fi
  done
  "$PY" "$REPO/herohe/gp2/scripts/eval_mil_baseline_test.py" \
    --checkpoint "${CKPTS[@]}" \
    --features_dir "$TEST_FEAT" \
    --labels_csv "$TEST_LABELS" \
    --label_mode "$LABEL_MODE" \
    --device mps \
    --max_patches -1 \
    --tag "$TAG" \
    --out_dir "$TEST_OUT"
}

stability_summary() {
  [[ "$TASK" == "binary" ]] || return 0
  local A="$1"
  local RUN
  RUN="$(run_dir_for "$A")"
  "$PY" << PY
import csv, json, torch
from pathlib import Path

run = Path("$RUN")
rows_out = []
for f in range(5):
    log = run / f"fold_{f}" / "log.csv"
    ckpt = run / f"fold_{f}" / "best.pt"
    if not ckpt.is_file():
        continue
    r = list(csv.DictReader(open(log)))
    vl = [float(x["val_loss"]) for x in r]
    auc = [float(x["val_auc_positive"]) for x in r]
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    ep = int(b["metrics"]["epoch"])
    saved_auc = float(b["metrics"]["auc_positive"])
    peak_auc = max(auc) if auc else float("nan")
    miss = peak_auc - saved_auc
    rows_out.append({
        "fold": f, "epoch": ep, "val_auc": round(saved_auc, 4),
        "peak_auc": round(peak_auc, 4), "val_loss_min_ep": vl.index(min(vl)) + 1 if vl else -1,
        "T": round(float(b.get("calibration_temperature", 1.0)), 4),
        "stable": ep >= 8 and miss <= 0.02,
    })
if rows_out:
    out = {
        "aggregator": "$A",
        "protocol": "5fold_stratified_cv_s42_ensemble",
        "folds": rows_out,
        "val_auc_mean": round(sum(x["val_auc"] for x in rows_out) / len(rows_out), 4),
        "all_stable": all(x["stable"] for x in rows_out),
    }
    (run / "stability_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
PY
}

eval_all() {
  for A in abmil clam transmil; do
    eval_agg "$A" || true
    stability_summary "$A" || true
  done
  if [[ "$TASK" == "binary" ]]; then
    "$PY" << 'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ.get("REPO", "."))
test_out = repo / "herohe/gp2/runs/test_eval_mil/binary_5fold_s42"
rows = []
for name, tag in [
    ("abmil", "abmil_binary_5fold_s42_valloss_5fold"),
    ("clam", "clam_binary_5fold_s42_valloss_5fold"),
    ("transmil", "transmil_binary_5fold_s42_valloss_5fold"),
]:
    p = test_out / f"summary_{tag}.json"
    if not p.is_file():
        rows.append({"model": name, "status": "missing", "path": str(p)})
        continue
    m = json.loads(p.read_text())["results"][0]
    rows.append({
        "model": name,
        "AUC": round(m.get("AUC", m.get("auc_positive", float("nan"))), 4),
        "posF1": round(m.get("posF1", float("nan")), 4),
        "wF1": round(m.get("wF1", float("nan")), 4),
        "bACC": round(m.get("bACC", float("nan")), 4),
        "AUPRC": round(m.get("AUPRC", float("nan")), 4),
    })
out = {
    "protocol": "5fold_stratified_cv_s42_val_loss_test_ensemble",
    "folds_csv": "folds_phiher2_binary_s42.csv",
    "rows": rows,
}
out_path = test_out / "table1_binary_baselines.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print("\n=== Table 1 draft (5-fold CV + ensemble) ===")
for r in rows:
    print(r)
print(f"\nWrote {out_path}")
PY
  fi
}

exec > >(tee -a "$LOG") 2>&1
echo "========== $(date) baselines task=$TASK agg=$AGG fold=$FOLD =========="

if [[ "$AGG" == "eval" ]]; then
  eval_all
  exit 0
fi

if [[ "$AGG" == "all" ]]; then
  for A in abmil clam transmil; do
    train_agg "$A" "$FOLD"
    if [[ "$FOLD" == "all" ]]; then
      eval_agg "$A"
      stability_summary "$A"
    fi
  done
else
  train_agg "$AGG" "$FOLD"
  if [[ "$FOLD" == "all" ]]; then
    eval_agg "$AGG"
    stability_summary "$AGG"
  fi
fi

echo "========== $(date) baselines task=$TASK DONE =========="
