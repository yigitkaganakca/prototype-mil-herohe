#!/usr/bin/env bash
# TRIDENT: seg -> coords -> Virchow2 features for official HEROHE test (150 slides).
# Prerequisite: herohe/gp2/scripts/verify_wsi_test_mirax.py exits 0
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
WSIDIR="$REPO/herohe/wsi_test"
JOB="$REPO/herohe/gp2/results_trident_test"
LIST="$REPO/herohe/gp2/data/wsi_list_test_150.csv"
TRIDENT="$REPO/TRIDENT"
LOG="$REPO/herohe/gp2/data/trident_test_150.log"
# Source of the official HEROHE test WSIs (obtain from the challenge organizers).
# Set GC_TEST to that directory before running, e.g. GC_TEST=/path/to/HEROHE/Test
GC_TEST="${GC_TEST:-}"

mkdir -p "$JOB"
exec > >(tee -a "$LOG") 2>&1

echo "========== $(date) TRIDENT test 150: preflight =========="
"$PY" "$REPO/herohe/gp2/scripts/verify_wsi_test_mirax.py" \
  --wsi_dir "$WSIDIR" \
  --ref_dir "$GC_TEST" \
  --out_json "$REPO/herohe/gp2/data/wsi_test_mirax_verify.json"

cd "$TRIDENT"

echo "[$(date '+%F %T')] === SEG (otsu) test 150 ==="
"$PY" run_batch_of_slides.py \
  --task seg \
  --wsi_dir "$WSIDIR" \
  --custom_list_of_wsis "$LIST" \
  --job_dir "$JOB" \
  --segmenter otsu \
  --gpu 0

echo "[$(date '+%F %T')] === COORDS test 150 ==="
"$PY" run_batch_of_slides.py \
  --task coords \
  --wsi_dir "$WSIDIR" \
  --custom_list_of_wsis "$LIST" \
  --job_dir "$JOB" \
  --mag 20 \
  --patch_size 256 \
  --overlap 0 \
  --gpu 0

echo "[$(date '+%F %T')] === FEAT (virchow2) test 150 ==="
"$PY" run_batch_of_slides.py \
  --task feat \
  --wsi_dir "$WSIDIR" \
  --custom_list_of_wsis "$LIST" \
  --job_dir "$JOB" \
  --patch_encoder virchow2 \
  --coords_dir 20x_256px_0px_overlap \
  --gpu 0

N=$(ls "$JOB/20x_256px_0px_overlap/features_virchow2"/*.h5 2>/dev/null | wc -l | tr -d ' ')
echo "========== $(date) DONE test features: $N / 150 h5 in $JOB =========="
