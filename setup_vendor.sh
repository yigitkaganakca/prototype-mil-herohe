#!/usr/bin/env bash
# Fetch the external repositories used as MIL baselines and for feature
# extraction, pinned to the exact commits used for the reported results.
# Run once from the repository root:  bash setup_vendor.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/herohe/gp2/vendor"

clone_at() {
  # clone_at <git-url> <dest-dir> <commit-sha>
  local url="$1" dest="$2" sha="$3"
  if [ -d "$dest/.git" ]; then
    echo "[skip] $dest already present"
  else
    echo "[clone] $url -> $dest @ $sha"
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch --depth 1 origin "$sha" 2>/dev/null || git -C "$dest" fetch origin
  git -C "$dest" checkout -q "$sha"
}

# ---- MIL baselines (pinned) ----
clone_at https://github.com/AMLab-Amsterdam/AttentionDeepMIL "$VENDOR/AttentionDeepMIL" eb0434ba2795711a45d693d60120ae53532b1b93
clone_at https://github.com/szc19990412/TransMIL          "$VENDOR/TransMIL"          9d6aee57c7c72375fb9132dc58cd8c9b0f0a949c
clone_at https://github.com/uta-smile/DeepAttnMISL        "$VENDOR/DeepAttnMISL"       d7099ed88c452aec37f16fc8e93cc0c068794c2a
# CLAM is resolved at <repo-root>/CLAM-master by herohe/gp2/vendor/paths.py
clone_at https://github.com/mahmoodlab/CLAM               "$ROOT/CLAM-master"          53e2409d4a8189c682c173382964a85f114f923c

# ---- Feature extraction (TRIDENT) ----
# Used for Virchow2 / ResNet-50 patch feature extraction. Pin to the commit
# you used if you need bit-exact features; latest main is otherwise fine.
clone_at https://github.com/mahmoodlab/TRIDENT            "$ROOT/TRIDENT"              HEAD || true

echo "Done. Baselines under herohe/gp2/vendor/, CLAM at ./CLAM-master, TRIDENT at ./TRIDENT"
