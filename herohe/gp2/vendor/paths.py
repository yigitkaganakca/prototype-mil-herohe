"""Resolve pinned upstream baseline repos and extend ``sys.path`` for imports."""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent
GP2_ROOT = VENDOR_ROOT.parent
HEROHE_ROOT = GP2_ROOT.parent
GRADCODE_ROOT = HEROHE_ROOT.parent

CLAM_ROOT = GRADCODE_ROOT / "CLAM-master"
ATTENTION_DEEP_MIL_ROOT = VENDOR_ROOT / "AttentionDeepMIL"
TRANSMIL_ROOT = VENDOR_ROOT / "TransMIL"
DEEP_ATTNMISL_ROOT = VENDOR_ROOT / "DeepAttnMISL"


def _prepend(path: Path) -> None:
    s = str(path)
    if path.is_dir() and s not in sys.path:
        sys.path.insert(0, s)


def ensure_clam_path() -> Path:
    if not CLAM_ROOT.is_dir():
        raise FileNotFoundError(
            f"CLAM repo not found at {CLAM_ROOT}. "
            "Clone mahmoodlab/CLAM to gradCode/CLAM-master."
        )
    _prepend(CLAM_ROOT)
    return CLAM_ROOT


def ensure_transmil_path() -> Path:
    if not TRANSMIL_ROOT.is_dir():
        raise FileNotFoundError(
            f"TransMIL repo not found at {TRANSMIL_ROOT}. "
            "Clone szc19990412/TransMIL under herohe/gp2/vendor/TransMIL."
        )
    _prepend(TRANSMIL_ROOT)
    return TRANSMIL_ROOT


def ensure_deepattnmisl_path() -> Path:
    if not DEEP_ATTNMISL_ROOT.is_dir():
        raise FileNotFoundError(
            f"DeepAttnMISL repo not found at {DEEP_ATTNMISL_ROOT}. "
            "Clone uta-smile/DeepAttnMISL under herohe/gp2/vendor/DeepAttnMISL."
        )
    _prepend(DEEP_ATTNMISL_ROOT)
    return DEEP_ATTNMISL_ROOT


def attention_deep_mil_root() -> Path:
    if not ATTENTION_DEEP_MIL_ROOT.is_dir():
        raise FileNotFoundError(
            f"AttentionDeepMIL repo not found at {ATTENTION_DEEP_MIL_ROOT}."
        )
    return ATTENTION_DEEP_MIL_ROOT
