"""CLAM-MB adapter — imports ``CLAM_MB`` from mahmoodlab/CLAM."""

from __future__ import annotations

import torch.nn as nn

from herohe.gp2.vendor.paths import ensure_clam_path


def make_clam_mb(
    *,
    n_classes: int,
    embed_dim: int,
    dropout: float = 0.25,
    k_sample: int = 8,
    subtyping: bool | None = None,
    gate: bool = True,
    size_arg: str = "small",
) -> nn.Module:
    ensure_clam_path()
    from models.model_clam import CLAM_MB  # noqa: WPS433 — upstream vendor import

    if subtyping is None:
        subtyping = n_classes > 2
    return CLAM_MB(
        gate=gate,
        size_arg=size_arg,
        dropout=dropout,
        k_sample=k_sample,
        n_classes=n_classes,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=subtyping,
        embed_dim=embed_dim,
    )
