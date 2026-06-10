#!/usr/bin/env python3
"""Smoke test: instantiate each vendor baseline and run one forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.vendor.adapters.abmil import ABMIL  # noqa: E402
from herohe.gp2.vendor.adapters.attnmisl import (  # noqa: E402
    AttnMISLClassifier,
    assign_patches_to_clusters,
    format_cluster_mifcn,
)
from herohe.gp2.vendor.adapters.clam import make_clam_mb  # noqa: E402
from herohe.gp2.vendor.adapters.transmil import TransMIL, TransMILConfig  # noqa: E402


def _check(name: str, logits: torch.Tensor) -> None:
    assert logits.ndim == 2, f"{name}: expected (B, C) logits, got {tuple(logits.shape)}"
    print(f"  OK  {name}  logits={tuple(logits.shape)}")


def _upstream_lineage(name: str) -> None:
    mod = {
        "abmil": "AttentionDeepMIL.model.GatedAttention (attention_V/U/w)",
        "clam": "CLAM-master.models.model_clam.CLAM_MB",
        "transmil": "TransMIL.models.TransMIL.TransMIL (subclass)",
        "attnmisl": "DeepAttnMISL_model.DeepAttnMIL_Surv (subclass)",
    }[name]
    print(f"       upstream: {mod}")


def main() -> None:
    device = torch.device("cpu")
    d = 2560
    n = 128
    x = torch.randn(1, n, d)
    h = x.squeeze(0)
    centers = torch.randn(8, d)

    print("=== vendor baseline smoke test (strong upstream fidelity) ===")

    abmil = ABMIL().to(device)
    out = abmil(x)
    _check("abmil", out["logits"])
    _upstream_lineage("abmil")
    assert hasattr(abmil, "gate") and hasattr(abmil.gate, "attention_V")

    clam = make_clam_mb(n_classes=3, embed_dim=d).to(device)
    logits, *_ = clam(h, label=None, instance_eval=False)
    _check("clam", logits)
    _upstream_lineage("clam")

    transmil = TransMIL(TransMILConfig(in_dim=d, num_classes=2)).to(device)
    out = transmil(x)
    _check("transmil", out["logits"])
    _upstream_lineage("transmil")
    assert hasattr(transmil, "layer1") and hasattr(transmil, "pos_layer")

    attnmisl = AttnMISLClassifier()
    attnmisl.set_prototype_centers(centers)
    attnmisl.to(device)
    out = attnmisl(x)
    _check("attnmisl", out["logits"])
    _upstream_lineage("attnmisl")
    assert isinstance(attnmisl.core, torch.nn.Module)
    assert hasattr(attnmisl.core, "masked_softmax")
    # Paper-faithful: MI-FCN pools each cluster to one vector; attention is over C phenotypes.
    assert isinstance(attnmisl.core.embedding_net[0], torch.nn.Conv2d)
    with torch.no_grad():
        clusters, mask = assign_patches_to_clusters(h, centers)
        graph = [format_cluster_mifcn(clusters[i]) for i in range(8)]
        pooled = []
        for g in graph:
            emb = attnmisl.core.embedding_net(g)
            pooled.append(emb.view(emb.size(0), -1))
        phen = torch.cat(pooled, dim=0)
        assert phen.shape == (8, 64), f"expected 8 phenotype tokens, got {tuple(phen.shape)}"
        assert mask.shape == (8,)

    print("=== all vendor baselines passed ===")


if __name__ == "__main__":
    main()
