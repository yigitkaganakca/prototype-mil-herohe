"""AB-MIL adapter — upstream ``GatedAttention`` attention modules from AttentionDeepMIL.

Official repo ships MNIST CNN + gated pooling in one class. We subclass ``GatedAttention``,
drop the unused CNN/classifier heads, and apply the **upstream** ``attention_V`` /
``attention_U`` / ``attention_w`` modules on projected precomputed patch features.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from herohe.gp2.vendor.paths import ATTENTION_DEEP_MIL_ROOT


def _load_upstream_model_module():
    path = ATTENTION_DEEP_MIL_ROOT / "model.py"
    spec = importlib.util.spec_from_file_location("attention_deep_mil_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream AttentionDeepMIL from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_upstream_gated_pooling() -> tuple[nn.Module, int]:
    mod = _load_upstream_model_module()
    gate = mod.GatedAttention()
    for part in (gate.feature_extractor_part1, gate.feature_extractor_part2, gate.classifier):
        for p in part.parameters():
            p.requires_grad = False
    return gate, int(gate.M)


@dataclass
class ABMILConfig:
    in_dim: int = 2560
    hidden_dim: int = 512
    attn_dim: int = 256
    num_classes: int = 4
    dropout: float = 0.25


class ABMIL(nn.Module):
    """Project patches to upstream M=500, pool with official gated-attention modules."""

    def __init__(self, cfg: ABMILConfig | None = None):
        super().__init__()
        cfg = cfg or ABMILConfig()
        self.cfg = cfg
        self.gate, upstream_m = _build_upstream_gated_pooling()
        self.proj = nn.Sequential(
            nn.Linear(cfg.in_dim, upstream_m),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(upstream_m, cfg.num_classes)

    def _pool(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (N, M) — uses upstream attention_V/U/w from ``GatedAttention``."""
        a_v = self.gate.attention_V(h)
        a_u = self.gate.attention_U(h)
        scores = self.gate.attention_w(a_v * a_u).squeeze(-1)
        attn = F.softmax(scores, dim=0)
        bag = (attn.unsqueeze(-1) * h).sum(dim=0)
        return bag, attn

    def forward(self, x: torch.Tensor) -> dict:
        batched = x.ndim == 3
        if batched:
            if x.shape[0] != 1:
                raise ValueError(f"ABMIL expects batch size 1 for variable N; got B={x.shape[0]}")
            x = x.squeeze(0)
        h = self.proj(x)
        bag, attn = self._pool(h)
        logits = self.head(bag)
        if batched:
            logits = logits.unsqueeze(0)
        return {"logits": logits, "attn": attn, "bag": bag}


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    attn = attn.clamp(min=1e-8)
    return -(attn * attn.log()).sum()
