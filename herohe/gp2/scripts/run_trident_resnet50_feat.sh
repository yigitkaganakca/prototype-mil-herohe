#!/usr/bin/env bash
# ResNet50 feature extraction only.
# Reuses the existing seg/coords artefacts from the Virchow2 run; outputs go to
#   $JOB/20x_256px_0px_overlap/features_resnet50/<slide_id>.h5
# Logs to herohe/gp2/data/trident_resnet50_run.log

set -euo pipefail

PY="${PY:-python}"
WSIDIR="${WSIDIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
JOB=$WSIDIR/herohe/gp2/results_trident_mac_full
LIST=$WSIDIR/herohe/gp2/data/wsi_list_full.csv
TRIDENT=$WSIDIR/TRIDENT

cd "$TRIDENT"

echo "[$(date '+%F %T')] === FEAT (resnet50) ==="
"$PY" run_batch_of_slides.py \
    --task feat \
    --wsi_dir "$WSIDIR" \
    --custom_list_of_wsis "$LIST" \
    --job_dir "$JOB" \
    --patch_encoder resnet50 \
    --coords_dir 20x_256px_0px_overlap

echo "[$(date '+%F %T')] === DONE resnet50 ==="
