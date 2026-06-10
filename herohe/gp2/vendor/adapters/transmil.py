"""TransMIL adapter — subclass official ``TransMIL`` from szc19990412/TransMIL.

Inherits ``pos_layer``, ``layer1``, ``layer2``, ``norm``, ``_fc2``, ``cls_token`` from
upstream. Only replaces ``_fc1`` (configurable ``in_dim``) and fixes the hard-coded
``.cuda()`` call in the official ``forward``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from herohe.gp2.vendor.paths import TRANSMIL_ROOT, ensure_transmil_path


def _load_upstream_transmil_class():
    ensure_transmil_path()
    path = TRANSMIL_ROOT / "models" / "TransMIL.py"
    spec = importlib.util.spec_from_file_location("transmil_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream TransMIL from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TransMIL


_UpstreamTransMIL = _load_upstream_transmil_class()


@dataclass
class TransMILConfig:
    in_dim: int = 2560
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 4


class TransMIL(_UpstreamTransMIL):
    """Official TransMIL with configurable input dim and device-safe forward."""

    def __init__(self, cfg: TransMILConfig | None = None):
        cfg = cfg or TransMILConfig()
        if cfg.n_layers != 2:
            raise ValueError(
                "Official TransMIL uses exactly 2 TransLayers with PPEG between them; "
                f"got n_layers={cfg.n_layers}"
            )
        super().__init__(n_classes=cfg.num_classes)
        self.cfg = cfg
        self._fc1 = nn.Sequential(nn.Linear(cfg.in_dim, cfg.d_model), nn.ReLU())

    def forward(self, x: torch.Tensor | None = None, coords=None, **kwargs) -> dict:
        del coords
        data = x if x is not None else kwargs.get("data")
        if data is None:
            raise ValueError("TransMIL expects bag tensor as x or data=...")

        h = self._fc1(data.float())
        seq_len = h.shape[1]
        grid_h = int(np.ceil(np.sqrt(seq_len)))
        grid_w = grid_h
        pad = grid_h * grid_w - seq_len
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)

        b = h.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1).to(device=h.device, dtype=h.dtype)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, grid_h, grid_w)
        h = self.layer2(h)
        cls_out = self.norm(h)[:, 0]
        logits = self._fc2(cls_out)
        return {
            "logits": logits,
            "cls": cls_out,
            "Y_prob": F.softmax(logits, dim=-1),
            "Y_hat": torch.argmax(logits, dim=1),
        }
