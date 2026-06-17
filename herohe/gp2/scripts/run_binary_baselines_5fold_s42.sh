#!/usr/bin/env bash
# Back-compat wrapper — use run_baselines_5fold_s42.sh binary ...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
exec "$SCRIPT_DIR/run_baselines_5fold_s42.sh" binary "$@"
