#!/usr/bin/env bash
# Full 5-fold binary + 3-class hard_partition ent0, then figures + report prep.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
bash "$REPO/herohe/gp2/scripts/run_khead_token_abmil_hard_partition_ent0.sh" all
bash "$REPO/herohe/gp2/scripts/run_khead_token_abmil_hard_partition_valieris3_ent0.sh" all
echo "========== $(date) ALL TRAINING COMPLETE =========="
