"""PhenoHER2 v2 -- Phenotype-aware Ordinal MIL with dual-stream gating.

STATUS
------
The ``PhenoHER2`` class in this file is the SUPERSEDED v2 4-class design and is NOT used
for the reported results — those come from ``PhenoHER2Binary`` in
``phenotype_mil_binary.py``. This module is retained because the reported binary model
imports its reusable building blocks ``GatedAttention`` and ``SimpleSpatialBlock``.
Everything below describing the dual-stream / EMD ordinal / Sinkhorn design refers to the
deprecated v2 ``PhenoHER2`` class only.

Changes vs v1:
    1. Dual-stream architecture:
        - Detail stream: vanilla AB-MIL attention over patches -> s_detail
        - Phenotype stream: prototype-aware aggregation              -> s_phen
        - Slide-level gate g = sigmoid(MLP([s_phen, s_detail])) fuses logits.
       This gives the project's Risk-3.5 Plan B for free: if phenotypes fail
       the gate goes towards the detail stream (~ AB-MIL fallback). Both
       streams' logits, the gate, and both attention maps are returned.

    2. CORN ordinal head removed; ordinal pull is now handled in the loss
       (squared EMD^2 on the fused 4-class softmax against soft labels).

    3. Train-time patch dropout for robustness to bag-size variance and ROI
       inference (CAVEATS #6, #16).

    4. Test-time augmentation helper: predict_with_tta() returns mean+std
       of probabilities across n random patch sub-bags.

    5. Post-training temperature scaling: fit_temperature() learns a single
       scalar T on a validation loader; predict_calibrated() applies it.

    6. Sinkhorn-Knopp balance loss is computed in losses.py from the
       returned `assign_sim` and `soft_assign`; this module just exposes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class PhenoHER2Config:
    feature_dim: int = 2560
    hidden_dim: int = 384
    num_prototypes: int = 16
    num_classes: int = 4
    attn_hidden_dim: int = 256
    cross_proto_layers: int = 2
    cross_proto_heads: int = 4
    dropout: float = 0.1
    init_temperature: float = 1.0
    use_spatial_block: bool = False
    spatial_block_heads: int = 4
    use_cls_pool: bool = False
    # v2 additions
    patch_dropout: float = 0.0       # train-time random patch drop probability
    use_dual_stream: bool = True
    detail_attn_hidden: int = 256


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


class GatedAttention(nn.Module):
    """Per-prototype gated attention over patches (Ilse 2018, multi-head).

    For num_prototypes == 1 this collapses to vanilla AB-MIL.

    [REUSED BY THE REPORTED MODEL] ``PhenoHER2Binary`` imports this block; it is an
    active component of the reported pipeline (unlike the ``PhenoHER2`` class below).
    """

    def __init__(self, hidden_dim: int, attn_hidden_dim: int, num_prototypes: int, dropout: float):
        super().__init__()
        self.tanh = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.sigm = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(attn_hidden_dim, num_prototypes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.tanh(h) * self.sigm(h))


class SimpleSpatialBlock(nn.Module):
    """Optional local-context block over the N patch tokens.

    [REUSED BY THE REPORTED MODEL] Importable building block; only instantiated when
    ``use_spatial_block=True`` (off in the reported runs, but the class itself is shared).
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.layer = encoder_layer
        self.coord_proj = nn.Linear(2, hidden_dim, bias=False)

    def forward(self, h: torch.Tensor, coords: Optional[torch.Tensor]) -> torch.Tensor:
        if coords is not None:
            c = coords.float()
            c_min = c.amin(dim=1, keepdim=True)
            c_max = c.amax(dim=1, keepdim=True)
            c = (c - c_min) / (c_max - c_min + 1e-6)
            h = h + self.coord_proj(c)
        return self.layer(h)


# --------------------------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------------------------


class PhenoHER2(nn.Module):
    """[DEPRECATED — superseded v2 4-class model; NOT used for the reported results.]

    The reported binary / 3-class results come from ``PhenoHER2Binary``
    (``phenotype_mil_binary.py``). This class is retained for reference only.

    Phenotype-aware Ordinal MIL for HER2 scoring (4-class), v2.
    """

    def __init__(self, cfg: PhenoHER2Config):
        super().__init__()
        self.cfg = cfg

        # 1. Project Virchow2 -> working dim
        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

        # 2. Optional spatial block
        self.spatial_block = (
            SimpleSpatialBlock(cfg.hidden_dim, cfg.spatial_block_heads, cfg.dropout)
            if cfg.use_spatial_block
            else None
        )

        # 3. Phenotype prototypes (learnable)
        self.prototypes = nn.Parameter(torch.randn(cfg.num_prototypes, cfg.hidden_dim) * 0.02)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(cfg.init_temperature))))

        # 4. Phenotype-stream gated attention (per prototype)
        self.gated_attn_phen = GatedAttention(
            hidden_dim=cfg.hidden_dim,
            attn_hidden_dim=cfg.attn_hidden_dim,
            num_prototypes=cfg.num_prototypes,
            dropout=cfg.dropout,
        )

        # 5. Cross-prototype transformer
        cross_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.cross_proto_heads,
            dim_feedforward=cfg.hidden_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.cross_proto = nn.TransformerEncoder(cross_layer, num_layers=cfg.cross_proto_layers)
        if cfg.use_cls_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # 6. Detail-stream attention (vanilla AB-MIL): K=1
        if cfg.use_dual_stream:
            self.gated_attn_detail = GatedAttention(
                hidden_dim=cfg.hidden_dim,
                attn_hidden_dim=cfg.detail_attn_hidden,
                num_prototypes=1,
                dropout=cfg.dropout,
            )
            self.head_4cls_detail = nn.Linear(cfg.hidden_dim, cfg.num_classes)
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.gated_attn_detail = None
            self.head_4cls_detail = None
            self.fusion_gate = None

        # 7. Heads
        self.head_4cls_phen = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.head_aux_01 = nn.Linear(cfg.hidden_dim, 1)
        self.head_3p = nn.Linear(cfg.hidden_dim, 1)

        # 8. Per-prototype class contribution (interpretability only)
        self.proto_class_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)

        # 9. Post-training temperature scaling (fit via fit_temperature)
        # Stored as a buffer so it follows .to(device) but isn't optimised.
        self.register_buffer("calibration_temperature", torch.ones(1))

    # ------------------------------------------------------------------
    # Prototype initialisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_prototypes_from_kmeans(self, centers: torch.Tensor) -> None:
        """Load offline discovery centers (k-means or hierarchical AP) into ``prototypes``."""
        self._load_prototype_centers(centers)

    def set_prototypes_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze prototype vectors (PhiHER2 Cluster-PT uses trainable=False)."""
        self.prototypes.requires_grad = bool(trainable)

    @torch.no_grad()
    def _load_prototype_centers(self, centers: torch.Tensor) -> None:
        if centers.ndim != 2:
            raise ValueError(f"centers must be 2D, got {tuple(centers.shape)}")
        K, dim = centers.shape
        if K != self.cfg.num_prototypes:
            raise ValueError(f"centers K={K}, expected {self.cfg.num_prototypes}")
        if dim == self.cfg.feature_dim:
            projected = self.proj(centers.to(self.proj[0].weight))
            self.prototypes.data.copy_(projected)
        elif dim == self.cfg.hidden_dim:
            self.prototypes.data.copy_(centers.to(self.prototypes))
        else:
            raise ValueError(
                f"centers dim={dim}; expected {self.cfg.feature_dim} or {self.cfg.hidden_dim}"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, D); got {tuple(x.shape)}")
        B, N, _ = x.shape

        # ---- train-time patch dropout ----
        if self.training and self.cfg.patch_dropout > 0.0 and N > 64:
            keep_prob = 1.0 - self.cfg.patch_dropout
            keep = torch.rand(B, N, device=x.device) < keep_prob
            # Make sure each bag keeps at least 64 patches to avoid degenerate cases
            for b in range(B):
                if keep[b].sum() < 64:
                    keep[b] = True  # disable drop for this bag
            if mask is None:
                mask = keep
            else:
                mask = mask & keep

        # 1. Project + LN
        h = self.proj(x)                                  # (B, N, D)

        # 2. Optional spatial block (operates on full sequence; mask not applied
        # to the transformer because it's expensive; patch dropout is enough).
        if self.spatial_block is not None:
            h = self.spatial_block(h, coords)

        # 3. Soft phenotype assignment
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        tau = torch.exp(self.log_temp).clamp(min=1e-3)
        sim = torch.einsum("bnd,kd->bnk", h_norm, p_norm) / tau
        sim_for_loss = sim.clone()
        if mask is not None:
            sim_masked = sim.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        else:
            sim_masked = sim
        soft_assign = F.softmax(sim_masked, dim=-1)

        # 4. Phenotype-stream per-prototype gated attention
        attn_logits_phen = self.gated_attn_phen(h)        # (B, N, K)
        log_gate = torch.log(soft_assign.clamp_min(1e-8))
        attn_combined = attn_logits_phen + log_gate
        if mask is not None:
            attn_combined = attn_combined.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        patch_attn_phen = F.softmax(attn_combined, dim=1)
        Z = torch.einsum("bnk,bnd->bkd", patch_attn_phen, h)   # (B, K, D)

        # 5. Cross-prototype transformer
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            Z_in = torch.cat([cls, Z], dim=1)
            Z_out = self.cross_proto(Z_in)
            s_phen = Z_out[:, 0]
            phen_tokens = Z_out[:, 1:]
        else:
            phen_tokens = self.cross_proto(Z)
            s_phen = phen_tokens.mean(dim=1)              # (B, D)

        logits_phen = self.head_4cls_phen(s_phen)         # (B, 4)

        # 6. Detail stream (vanilla AB-MIL) + gated fusion
        if self.gated_attn_detail is not None:
            attn_logits_detail = self.gated_attn_detail(h).squeeze(-1)  # (B, N)
            if mask is not None:
                attn_logits_detail = attn_logits_detail.masked_fill(~mask, float("-inf"))
            patch_attn_detail = F.softmax(attn_logits_detail, dim=1)    # (B, N)
            s_detail = torch.einsum("bn,bnd->bd", patch_attn_detail, h)
            logits_detail = self.head_4cls_detail(s_detail)
            gate_in = torch.cat([s_phen, s_detail], dim=-1)
            gate = self.fusion_gate(gate_in)              # (B, 1) in (0, 1)
            logits_4cls = gate * logits_phen + (1.0 - gate) * logits_detail
            s_combined = gate * s_phen + (1.0 - gate) * s_detail
        else:
            patch_attn_detail = None
            logits_detail = None
            gate = None
            logits_4cls = logits_phen
            s_combined = s_phen

        # 7. Aux heads on the gated-combined slide repr
        logit_aux_01 = self.head_aux_01(s_combined).squeeze(-1)
        logit_3p = self.head_3p(s_combined).squeeze(-1)

        # 8. Per-prototype class contribution (FR9 explainability)
        proto_class_logits = self.proto_class_head(phen_tokens)

        return {
            "logits_4cls": logits_4cls,
            "logits_phen": logits_phen,
            "logits_detail": logits_detail,
            "logit_aux_01": logit_aux_01,
            "logit_3p": logit_3p,
            "patch_attn": patch_attn_phen,        # (B, N, K) per-prototype
            "patch_attn_detail": patch_attn_detail,  # (B, N) global AB-MIL
            "soft_assign": soft_assign,
            "assign_sim": sim_for_loss,
            "phen_tokens": phen_tokens,
            "slide_repr_phen": s_phen,
            "slide_repr": s_combined,
            "proto_class_logits": proto_class_logits,
            "fusion_gate": gate,
            "patch_mask": mask,
            "tau": tau.detach(),
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        apply_calibration: bool = True,
    ) -> dict:
        """Single forward pass with softmax probs (and optional temp scaling)."""
        out = self.forward(x, coords=coords)
        logits = out["logits_4cls"]
        if apply_calibration:
            logits = logits / self.calibration_temperature.clamp_min(1e-3)
        probs = F.softmax(logits, dim=-1)
        out["probs"] = probs
        out["pred_class"] = probs.argmax(dim=-1)
        out["prob_3p_high_conf"] = torch.sigmoid(out["logit_3p"])
        out["prob_aux_low_vs_zero"] = torch.sigmoid(out["logit_aux_01"])
        return out

    @torch.no_grad()
    def predict_with_tta(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        n_samples: int = 8,
        subsample_frac: float = 0.7,
        min_patches: int = 64,
        apply_calibration: bool = True,
    ) -> dict:
        """Test-time augmentation: average predictions across n random patch
        sub-bags. Also returns the std of probs as a cheap uncertainty signal.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, D); got {tuple(x.shape)}")
        B, N, _ = x.shape
        if N <= min_patches:
            return self.predict(x, coords=coords, apply_calibration=apply_calibration)

        keep = max(int(N * subsample_frac), min_patches)
        all_probs = []
        all_pred = []
        for _ in range(n_samples):
            idx = torch.randperm(N, device=x.device)[:keep]
            sub_x = x[:, idx]
            sub_c = coords[:, idx] if coords is not None else None
            single = self.predict(sub_x, coords=sub_c, apply_calibration=apply_calibration)
            all_probs.append(single["probs"])
            all_pred.append(single["pred_class"])
        probs_stack = torch.stack(all_probs, dim=0)              # (n, B, K)
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        return {
            "probs": mean_probs,
            "probs_std": std_probs,
            "pred_class": mean_probs.argmax(dim=-1),
            "n_tta_samples": n_samples,
        }

    # ------------------------------------------------------------------
    # Temperature scaling (post-training calibration)
    # ------------------------------------------------------------------
    def fit_temperature(self, val_loader, device: torch.device, max_iter: int = 50, lr: float = 0.01) -> float:
        """Fit a single scalar temperature on the validation set via L-BFGS on NLL.

        Updates self.calibration_temperature in-place. Returns the fitted T.
        """
        self.eval()
        all_logits = []
        all_y = []
        for batch in val_loader:
            x = batch["features"].to(device)
            c = batch["coords"].to(device) if batch["coords"] is not None else None
            y = batch["label"].to(device)
            with torch.no_grad():
                out = self.forward(x, coords=c)
            all_logits.append(out["logits_4cls"])
            all_y.append(y)
        if not all_logits:
            return float(self.calibration_temperature.item())
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        # L-BFGS on a single learnable scalar
        T = nn.Parameter(torch.ones(1, device=device))
        optim = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optim.zero_grad()
            loss = F.cross_entropy(logits / T.clamp_min(1e-3), y)
            loss.backward()
            return loss

        optim.step(closure)
        T_val = float(T.detach().clamp_min(1e-3).item())
        self.calibration_temperature.data = torch.tensor(
            [T_val], device=self.calibration_temperature.device
        )
        return T_val
