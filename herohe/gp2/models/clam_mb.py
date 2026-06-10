# Copyright notice: derived from CLAM (https://github.com/mahmoodlab/CLAM)
# Lu, M.Y., et al. "Data-efficient and weakly supervised computational pathology
# on whole-slide images." Nature Biomedical Engineering (2021).
#
# DEPRECATED: training/eval now imports CLAM_MB from CLAM-master via
# herohe.gp2.vendor.adapters.clam. Kept for reference only.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net(nn.Module):
    def __init__(self, L: int = 1024, D: int = 256, dropout: bool = False, n_classes: int = 1):
        super().__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L: int = 1024, D: int = 256, dropout: bool = False, n_classes: int = 1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class CLAM_MB(nn.Module):
    """Multi-branch CLAM for n_classes > 2 (Valieris-style bag MIL)."""

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.25,
        k_sample: int = 8,
        n_classes: int = 4,
        instance_loss_fn: nn.Module | None = None,
        subtyping: bool = True,
        embed_dim: int = 2560,
    ):
        super().__init__()
        if instance_loss_fn is None:
            instance_loss_fn = nn.CrossEntropyLoss()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=bool(dropout), n_classes=n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=bool(dropout), n_classes=n_classes
            )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(size[1], 2) for _ in range(n_classes)]
        )
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def _k_eff(self, n_patches: int) -> int:
        return max(1, min(self.k_sample, n_patches // 2))

    @staticmethod
    def create_positive_targets(length: int, device: torch.device) -> torch.Tensor:
        return torch.full((length,), 1, device=device, dtype=torch.long)

    @staticmethod
    def create_negative_targets(length: int, device: torch.device) -> torch.Tensor:
        return torch.full((length,), 0, device=device, dtype=torch.long)

    def inst_eval(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module, k_eff: int):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, k_eff, dim=1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_eff, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_eff, device)
        n_targets = self.create_negative_targets(k_eff, device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss

    def inst_eval_out(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module, k_eff: int):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, k_eff, dim=1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_eff, device)
        logits = classifier(top_p)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss

    def forward(
        self,
        h: torch.Tensor,
        label: torch.Tensor | None = None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        """h: (N, D) patch features; label: (1,) long."""
        A, h_proj = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        k_eff = self._k_eff(h.shape[0])

        total_inst_loss: torch.Tensor | float = 0.0
        if instance_eval and label is not None:
            total_inst_loss = 0.0
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze(0)
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    il = self.inst_eval(A[i], h_proj, classifier, k_eff)
                else:
                    if self.subtyping:
                        il = self.inst_eval_out(A[i], h_proj, classifier, k_eff)
                    else:
                        continue
                total_inst_loss = total_inst_loss + il
            if self.subtyping:
                total_inst_loss = total_inst_loss / len(self.instance_classifiers)

        M = torch.mm(A, h_proj)
        logits = torch.empty(1, self.n_classes, device=M.device, dtype=torch.float32)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict: dict = {}
        if instance_eval:
            results_dict["instance_loss"] = total_inst_loss
        if return_features:
            results_dict["features"] = M
        return logits, Y_prob, Y_hat, A_raw, results_dict
