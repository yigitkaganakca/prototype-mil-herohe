#!/usr/bin/env bash
# Generate §4 khead architecture figure for a HEROHE slide.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
PY="${PY:-python}"
OUT="$REPO/herohe/report/figures"
CKPT="${CKPT:-$REPO/herohe/gp2/runs/khead_token_abmil_hard_partition_ent0/fold_0/best.pt}"
SLIDE_ID="${SLIDE_ID:-304}"
SPLIT="${SPLIT:-test}"

mkdir -p "$OUT"

echo "[$(date '+%F %T')] Rendering khead architecture figure (slide $SLIDE_ID, split=$SPLIT)..."
"$PY" "$REPO/herohe/gp2/scripts/render_architecture_khead_figure.py" \
  --checkpoint "$CKPT" \
  --slide_id "$SLIDE_ID" \
  --split "$SPLIT" \
  --out_dir "$OUT" \
  --dpi 220 \
  --device mps

echo "Done. Output: $OUT/arch_khead_slide${SLIDE_ID}.png"
