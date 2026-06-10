"""TransMIL-style bag classifier (legacy local reimplementation).

DEPRECATED: use ``herohe.gp2.vendor.adapters.transmil`` (official TransLayer + PPEG).
Kept for reference only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransMILConfig:
    in_dim: int = 2560
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_classes: int = 4
    use_coord_pe: bool = True


class TransMIL(nn.Module):
    def __init__(self, cfg: TransMILConfig | None = None):
        super().__init__()
        cfg = cfg or TransMILConfig()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.in_dim, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.use_coord_pe = cfg.use_coord_pe
        if cfg.use_coord_pe:
            self.coord_pe = nn.Sequential(
                nn.Linear(2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, 4096 + 1, cfg.d_model))
            nn.init.normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> dict:
        """x: (B, N, D); coords: (B, N, 2) optional, same N as x."""
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, D); got {tuple(x.shape)}")
        B, N, _ = x.shape
        tok = self.patch_embed(x)
        if self.use_coord_pe:
            if coords is None:
                pe = torch.zeros(B, N, tok.shape[-1], device=tok.device, dtype=tok.dtype)
            else:
                c = coords.float()
                c_min = c.amin(dim=1, keepdim=True)
                c_max = c.amax(dim=1, keepdim=True)
                c = (c - c_min) / (c_max - c_min + 1e-6)
                pe = self.coord_pe(c)
            tok = tok + pe
        else:
            if N + 1 > self.pos_embed.shape[1]:
                raise ValueError(
                    f"N={N} exceeds baked pos_embed length {self.pos_embed.shape[1] - 1}; "
                    "enable use_coord_pe or increase pos_embed."
                )
            tok = tok + self.pos_embed[:, :N, :]

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        out = self.encoder(seq)
        cls_out = self.norm(out[:, 0])
        logits = self.head(cls_out)
        return {"logits": logits, "cls": cls_out}
