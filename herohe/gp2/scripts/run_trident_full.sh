#!/usr/bin/env bash
# Full TRIDENT run: seg -> coords -> feat (Virchow2) on the verified WSI list.
# Logs to herohe/gp2/data/trident_full_run.log

set -euo pipefail

PY="${PY:-python}"
WSIDIR="${WSIDIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
JOB=$WSIDIR/herohe/gp2/results_trident_mac_full
LIST=$WSIDIR/herohe/gp2/data/wsi_list_full.csv
TRIDENT=$WSIDIR/TRIDENT

mkdir -p "$JOB"

cd "$TRIDENT"

echo "[$(date '+%F %T')] === SEG (otsu) ==="
"$PY" run_batch_of_slides.py \
    --task seg \
    --wsi_dir "$WSIDIR" \
    --custom_list_of_wsis "$LIST" \
    --job_dir "$JOB" \
    --segmenter otsu \
    --gpu 0

echo "[$(date '+%F %T')] === COORDS ==="
"$PY" run_batch_of_slides.py \
    --task coords \
    --wsi_dir "$WSIDIR" \
    --custom_list_of_wsis "$LIST" \
    --job_dir "$JOB" \
    --mag 20 \
    --patch_size 256 \
    --overlap 0

echo "[$(date '+%F %T')] === FEAT (virchow2) ==="
"$PY" run_batch_of_slides.py \
    --task feat \
    --wsi_dir "$WSIDIR" \
    --custom_list_of_wsis "$LIST" \
    --job_dir "$JOB" \
    --patch_encoder virchow2 \
    --coords_dir 20x_256px_0px_overlap

echo "[$(date '+%F %T')] === DONE ==="
