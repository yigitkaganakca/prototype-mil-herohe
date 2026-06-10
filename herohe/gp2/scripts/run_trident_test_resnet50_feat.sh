#!/usr/bin/env bash
# ResNet50 features only on official 150-slide test set (seg/coords already from Virchow2 test run).
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
WSIDIR="$REPO/herohe/wsi_test"
JOB="$REPO/herohe/gp2/results_trident_test"
LIST="$REPO/herohe/gp2/data/wsi_list_test_150.csv"
TRIDENT="$REPO/TRIDENT"
LOG="$REPO/herohe/gp2/data/trident_test_resnet50_feat.log"
FEAT_OUT="$JOB/20x_256px_0px_overlap/features_resnet50"

mkdir -p "$FEAT_OUT"
exec > >(tee -a "$LOG") 2>&1

N_EXIST=$(find "$FEAT_OUT" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$N_EXIST" -ge 150 && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] ResNet50 test features already complete: $N_EXIST h5 in $FEAT_OUT"
  exit 0
fi

cd "$TRIDENT"
echo "[$(date '+%F %T')] === FEAT (resnet50) test 150 ==="
"$PY" run_batch_of_slides.py \
  --task feat \
  --wsi_dir "$WSIDIR" \
  --custom_list_of_wsis "$LIST" \
  --job_dir "$JOB" \
  --patch_encoder resnet50 \
  --coords_dir 20x_256px_0px_overlap \
  --gpu 0

N=$(find "$FEAT_OUT" -maxdepth 1 -name '*.h5' | wc -l | tr -d ' ')
echo "========== $(date) DONE ResNet50 test features: $N / 150 =========="
