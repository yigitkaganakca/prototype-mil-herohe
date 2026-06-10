"""Build MIL baseline models from pinned upstream vendor adapters."""

from __future__ import annotations

import torch.nn as nn

from herohe.gp2.vendor.adapters.abmil import ABMIL, ABMILConfig
from herohe.gp2.vendor.adapters.attnmisl import AttnMISLClassifier, AttnMISLConfig
from herohe.gp2.vendor.adapters.clam import make_clam_mb
from herohe.gp2.vendor.adapters.transmil import TransMIL, TransMILConfig


def build_baseline_model(
    aggregator: str,
    num_classes: int,
    feature_dim: int,
    *,
    clam_dropout: float = 0.4,
    clam_k_sample: int = 8,
    trans_d_model: int = 512,
    trans_layers: int = 2,
    trans_heads: int = 8,
    trans_dropout: float = 0.25,
    abmil_hidden: int = 512,
    abmil_attn: int = 256,
    abmil_dropout: float = 0.4,
    attnmisl_cluster_num: int = 8,
    attnmisl_dropout: float = 0.5,
    prototype_centers=None,
) -> nn.Module:
    agg = aggregator.lower()
    if agg == "abmil":
        return ABMIL(
            ABMILConfig(
                in_dim=feature_dim,
                hidden_dim=abmil_hidden,
                attn_dim=abmil_attn,
                num_classes=num_classes,
                dropout=abmil_dropout,
            )
        )
    if agg == "clam":
        return make_clam_mb(
            n_classes=num_classes,
            embed_dim=feature_dim,
            dropout=clam_dropout,
            k_sample=clam_k_sample,
            subtyping=(num_classes > 2),
        )
    if agg == "transmil":
        return TransMIL(
            TransMILConfig(
                in_dim=feature_dim,
                d_model=trans_d_model,
                n_layers=trans_layers,
                n_heads=trans_heads,
                dropout=trans_dropout,
                num_classes=num_classes,
            )
        )
    if agg == "attnmisl":
        model = AttnMISLClassifier(
            AttnMISLConfig(
                in_dim=feature_dim,
                cluster_num=attnmisl_cluster_num,
                num_classes=num_classes,
                dropout=attnmisl_dropout,
            )
        )
        if prototype_centers is not None:
            model.set_prototype_centers(prototype_centers)
        return model
    raise ValueError(f"Unknown aggregator {aggregator!r}")
