"""AttnMISL adapter — paper-faithful ``DeepAttnMIL_Surv`` from uta-smile/DeepAttnMISL.

Yao et al., MedIA 2020 (§2.1.2–2.1.3, Table 1):

1. **MI-FCN (Siamese)** per phenotype cluster: 1×1 conv → ReLU → global avg pool → r_j ∈ R^64
2. **Attention MIL pooling** over C phenotype vectors (Eq. 1–2) → z ∈ R^64
3. **FC head** 64 → 32 → num_classes (survival in upstream; CE here for HEROHE)

We keep upstream ``attention``, ``masked_softmax``, and ``forward`` aggregation unchanged.
Only the embedding input width (Virchow2 2560 vs VGG 4096) and ``fc6`` (CE classes) differ.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from herohe.gp2.vendor.paths import DEEP_ATTNMISL_ROOT


def _load_upstream_class():
    path = DEEP_ATTNMISL_ROOT / "DeepAttnMISL_model.py"
    spec = importlib.util.spec_from_file_location("deepattnmisl_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream DeepAttnMISL from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DeepAttnMIL_Surv


_UpstreamDeepAttnMISL = _load_upstream_class()


@dataclass
class AttnMISLConfig:
    in_dim: int = 2560
    cluster_num: int = 8
    embed_dim: int = 64
    num_classes: int = 2
    dropout: float = 0.5


def assign_patches_to_clusters(
    x: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Nearest-prototype assignment → C cluster patch sets + presence mask.

    Empty clusters receive a single zero patch and ``mask[c]=0``, matching upstream
    ``MIL_dataloader`` behaviour for ``masked_softmax`` over phenotypes.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (N, D); got {tuple(x.shape)}")
    k = int(centers.shape[0])
    x_n = F.normalize(x.float(), dim=1)
    c_n = F.normalize(centers.float().to(x.device), dim=1)
    assign = (x_n @ c_n.T).argmax(dim=1)
    clusters: list[torch.Tensor] = []
    mask = x.new_ones(k)
    for idx in range(k):
        sel = x[assign == idx]
        if sel.numel() == 0:
            mask[idx] = 0
            clusters.append(x.new_zeros(1, x.shape[1]))
        else:
            clusters.append(sel)
    return clusters, mask


def format_cluster_mifcn(patches: torch.Tensor) -> torch.Tensor:
    """Patch matrix → upstream MI-FCN layout ``(1, D, 1, n_patches)``."""
    if patches.ndim != 2:
        raise ValueError(f"patches must be (n, D); got {tuple(patches.shape)}")
    # Match DeepAttnMISL MIL_dataloader: swapaxes(D, n) then channel-first conv input.
    feat = patches.float().transpose(0, 1).contiguous()
    return feat.unsqueeze(0).unsqueeze(2)


class DeepAttnMISLClassifier(_UpstreamDeepAttnMISL):
    """Paper MI-FCN + inherited phenotype-level attention; CE classification head."""

    def __init__(self, cfg: AttnMISLConfig):
        super().__init__(cluster_num=cfg.cluster_num)
        self.cfg = cfg
        self.embedding_net = nn.Sequential(
            nn.Conv2d(cfg.in_dim, cfg.embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(cfg.embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, cfg.num_classes),
        )


class AttnMISLClassifier(nn.Module):
    """Slide bag → prototype assignment → paper-faithful DeepAttnMISL forward."""

    def __init__(self, cfg: AttnMISLConfig | None = None):
        super().__init__()
        self.cfg = cfg or AttnMISLConfig()
        self.core = DeepAttnMISLClassifier(self.cfg)
        self.register_buffer("_centers", torch.zeros(self.cfg.cluster_num, self.cfg.in_dim), persistent=False)

    def set_prototype_centers(self, centers: torch.Tensor) -> None:
        c = centers.detach().float().cpu()
        if c.shape[0] != self.cfg.cluster_num:
            raise ValueError(f"Expected {self.cfg.cluster_num} prototypes, got {c.shape[0]}")
        if c.shape[1] != self.cfg.in_dim:
            raise ValueError(f"Prototype dim {c.shape[1]} != in_dim {self.cfg.in_dim}")
        self._centers = c

    def forward(
        self,
        x: torch.Tensor,
        centers: torch.Tensor | None = None,
    ) -> dict:
        if x.ndim == 3:
            if x.shape[0] != 1:
                raise ValueError(f"AttnMISL expects batch size 1; got B={x.shape[0]}")
            x = x.squeeze(0)
        proto = centers if centers is not None else self._centers.to(x.device)
        clusters, mask = assign_patches_to_clusters(x, proto)
        graph = [format_cluster_mifcn(clusters[i]) for i in range(self.cfg.cluster_num)]
        # Upstream expects mask shaped for softmax over C phenotype instances.
        mask_b = mask.unsqueeze(0)
        logits = self.core(graph, mask=mask_b)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        return {
            "logits": logits,
            "cluster_mask": mask,
            "n_phenotypes": self.cfg.cluster_num,
        }
