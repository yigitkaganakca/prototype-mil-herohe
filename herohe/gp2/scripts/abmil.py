"""Thin re-export of AB-MIL for scripts that add this directory to ``sys.path``.

The canonical implementation lives in ``herohe.gp2.models.abmil`` (Ilse et al. 2018).
``smoke_abmil_forward.py`` imports ``ABMIL``, ``ABMILConfig``, and ``count_parameters`` from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from herohe.gp2.models.abmil import ABMIL, ABMILConfig, GatedAttention, count_parameters

__all__ = ["ABMIL", "ABMILConfig", "GatedAttention", "count_parameters"]
