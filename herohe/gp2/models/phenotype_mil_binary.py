"""PhenoHER2-Binary: prototype MIL with configurable readout, 2-class head.

Targets the **original HEROHE deployment label**: ISH-aligned Negative (0) vs Positive (1),
not the 4-class IHC score.

Readout modes (``PhenoHER2BinaryConfig.readout``):
    ``full`` — soft-assign log-gate + K proto tokens + cross-proto Transformer + mean pool.
    ``khead`` — K independent gated-attention pools → readout → CE.
    ``khead_pool``: ``mean`` | ``concat`` | ``token_abmil`` (ABMIL over K phenotype tokens).
    ``khead`` routing: ``independent`` (default), ``log_gate`` (log soft-assign gate),
    or ``hard_partition`` (AttnMISL-style argmax routing; gated pool within each cluster only).
    ``khead_abmil`` — (legacy) K pools + ABMIL branch; not recommended for prototype MIL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phenotype_mil import GatedAttention, SimpleSpatialBlock

from herohe.gp2.vendor.adapters.abmil import attention_entropy

try:
    from .abmil import stochastic_topk_attn_mask
except ImportError:
    stochastic_topk_attn_mask = None  # type: ignore[assignment,misc]

ReadoutMode = Literal["full", "khead", "khead_abmil"]
KheadPoolMode = Literal["concat", "mean", "token_abmil"]
KheadRoutingMode = Literal["independent", "log_gate", "hard_partition"]


class InstSelector(nn.Module):
    """PhiHER2-style linear patch scorer: top-k by P(positive) on raw features."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.scorer = nn.Linear(in_dim, 2)

    def topk_indices(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """``x`` (N, D) → index tensor of length min(k, N)."""
        n = x.shape[0]
        if k >= n:
            return torch.arange(n, device=x.device, dtype=torch.long)
        probs = F.softmax(self.scorer(x), dim=-1)[:, 1]
        return torch.topk(probs, k, dim=0).indices


@dataclass
class PhenoHER2BinaryConfig:
    feature_dim: int = 2560
    hidden_dim: int = 384
    num_prototypes: int = 16
    attn_hidden_dim: int = 256
    cross_proto_layers: int = 2
    cross_proto_heads: int = 4
    dropout: float = 0.1
    init_temperature: float = 1.0
    use_spatial_block: bool = False
    spatial_block_heads: int = 4
    use_cls_pool: bool = False
    patch_dropout: float = 0.0
    use_dual_stream: bool = True
    detail_attn_hidden: int = 256
    num_classes: int = 2
    readout: ReadoutMode = "full"
    proto_attn_bias: bool = True
    khead_pool: KheadPoolMode = "concat"
    khead_routing: KheadRoutingMode = "independent"
    patch_attn_temperature: float = 1.0
    stkim_p: float = 0.0
    stkim_k: int = 10
    stkim_frac: float = 0.0
    mine_patches: int = 0
    mine_on_val: bool = False


class PhenoHER2Binary(nn.Module):
    """Prototype MIL with fused **2-logit** bag classifier."""

    def __init__(self, cfg: PhenoHER2BinaryConfig):
        super().__init__()
        self.cfg = cfg
        readout = cfg.readout

        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

        self.spatial_block = (
            SimpleSpatialBlock(cfg.hidden_dim, cfg.spatial_block_heads, cfg.dropout)
            if cfg.use_spatial_block
            else None
        )

        self.prototypes = nn.Parameter(torch.randn(cfg.num_prototypes, cfg.hidden_dim) * 0.02)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(cfg.init_temperature))))

        self.gated_attn_phen = GatedAttention(
            hidden_dim=cfg.hidden_dim,
            attn_hidden_dim=cfg.attn_hidden_dim,
            num_prototypes=cfg.num_prototypes,
            dropout=cfg.dropout,
        )

        if readout == "full":
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
            self.head_slide = None
            self.head_abmil = None
            self.head_proto = None
        else:
            self.cross_proto = None
            self.cls_token = None
            if readout == "khead" and cfg.khead_pool in ("mean", "token_abmil"):
                proto_in = cfg.hidden_dim
            else:
                proto_in = cfg.hidden_dim * cfg.num_prototypes
            self.head_proto = nn.Linear(proto_in, cfg.num_classes)
            if readout == "khead" and cfg.khead_pool == "token_abmil":
                self.token_attn_pool = GatedAttention(
                    hidden_dim=cfg.hidden_dim,
                    attn_hidden_dim=cfg.attn_hidden_dim,
                    num_prototypes=1,
                    dropout=cfg.dropout,
                )
            else:
                self.token_attn_pool = None
            if readout == "khead_abmil":
                self.gated_attn_detail = GatedAttention(
                    hidden_dim=cfg.hidden_dim,
                    attn_hidden_dim=cfg.detail_attn_hidden,
                    num_prototypes=1,
                    dropout=cfg.dropout,
                )
                self.head_abmil = nn.Linear(cfg.hidden_dim, cfg.num_classes)
            else:
                self.gated_attn_detail = None
                self.head_abmil = None
            self.head_slide = None

        if readout == "full" and cfg.use_dual_stream:
            self.gated_attn_detail = GatedAttention(
                hidden_dim=cfg.hidden_dim,
                attn_hidden_dim=cfg.detail_attn_hidden,
                num_prototypes=1,
                dropout=cfg.dropout,
            )
            self.head_bin_detail = nn.Linear(cfg.hidden_dim, cfg.num_classes)
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        elif readout == "full":
            self.gated_attn_detail = None
            self.head_bin_detail = None
            self.fusion_gate = None
        elif readout == "khead":
            self.head_bin_detail = None
            self.fusion_gate = None

        if cfg.mine_patches > 0:
            self.inst_selector = InstSelector(cfg.feature_dim)
        else:
            self.inst_selector = None

        if readout == "full":
            self.head_bin_phen = nn.Linear(cfg.hidden_dim, cfg.num_classes)
            self.proto_class_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        else:
            self.head_bin_phen = None
            self.proto_class_head = None

        self.register_buffer("calibration_temperature", torch.ones(1))

    @property
    def readout(self) -> ReadoutMode:
        return self.cfg.readout

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

    def _proto_similarity(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        tau = torch.exp(self.log_temp).clamp(min=1e-3)
        sim = torch.einsum("bnd,kd->bnk", h_norm, p_norm) / tau
        return sim, tau

    def _mine_instances(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """PhiHER2-style: keep top-k patches by linear scorer P(positive) on raw features."""
        k = self.cfg.mine_patches
        if k <= 0 or self.inst_selector is None:
            return x, coords, mask
        if not (self.training or self.cfg.mine_on_val):
            return x, coords, mask

        B, N, _ = x.shape
        out_x: list[torch.Tensor] = []
        out_c: list[torch.Tensor] = []
        out_m: list[torch.Tensor] = []
        for b in range(B):
            xb = x[b]
            if mask is not None:
                valid = torch.nonzero(mask[b], as_tuple=True)[0]
                if valid.numel() == 0:
                    valid = torch.arange(N, device=x.device)
            else:
                valid = torch.arange(N, device=x.device)
            xb = xb[valid]
            if xb.shape[0] <= k:
                sel = valid
            else:
                local_idx = self.inst_selector.topk_indices(xb, k)
                sel = valid[local_idx]
            out_x.append(x[b, sel])
            if coords is not None:
                out_c.append(coords[b, sel])
            if mask is not None:
                out_m.append(torch.ones(sel.shape[0], dtype=torch.bool, device=x.device))

        x_out = torch.stack(out_x, dim=0)
        c_out = torch.stack(out_c, dim=0) if coords is not None else None
        m_out = torch.stack(out_m, dim=0) if mask is not None else None
        return x_out, c_out, m_out

    def _khead_pool(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """K parallel gated-attention pools → (B, K, D) tokens + active-phenotype mask."""
        sim, _ = self._proto_similarity(h)
        sim_for_loss = sim.clone()
        if mask is not None:
            sim_masked = sim.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        else:
            sim_masked = sim
        soft_assign = F.softmax(sim_masked, dim=-1)

        if self.cfg.khead_routing == "hard_partition":
            return self._khead_pool_hard_partition(h, mask, soft_assign, sim_for_loss, sim_masked)

        attn_logits = self.gated_attn_phen(h)
        routing = self.cfg.khead_routing
        use_proto = self.cfg.proto_attn_bias or routing == "log_gate"
        if use_proto:
            if routing == "log_gate":
                attn_logits = attn_logits + torch.log(soft_assign.clamp_min(1e-8))
            elif self.cfg.proto_attn_bias:
                attn_logits = attn_logits + sim

        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        if (
            self.training
            and self.cfg.stkim_p > 0.0
            and stochastic_topk_attn_mask is not None
        ):
            n = attn_logits.shape[1]
            if mask is not None:
                n_valid = int(mask[0].sum().item()) if mask.ndim == 2 else n
            else:
                n_valid = n
            k = int(self.cfg.stkim_k)
            if self.cfg.stkim_frac > 0.0:
                k = max(k, int(self.cfg.stkim_frac * n_valid))
            attn_logits = stochastic_topk_attn_mask(
                attn_logits,
                k=k,
                p=float(self.cfg.stkim_p),
                mask=mask,
            )
        temp = max(float(self.cfg.patch_attn_temperature), 1e-3)
        patch_attn = F.softmax(attn_logits / temp, dim=1)
        phen_tokens = torch.einsum("bnk,bnd->bkd", patch_attn, h)
        phen_active = h.new_ones(h.shape[0], self.cfg.num_prototypes, dtype=torch.bool)
        return phen_tokens, patch_attn, soft_assign, sim_for_loss, phen_active

    def _khead_pool_hard_partition(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
        soft_assign: torch.Tensor,
        sim_for_loss: torch.Tensor,
        sim_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hard argmax prototype routing; head *k* pools only patches assigned to *k*."""
        B, _N, _D = h.shape
        K = self.cfg.num_prototypes
        hard = sim_masked.argmax(dim=-1)
        temp = max(float(self.cfg.patch_attn_temperature), 1e-3)

        patch_attn = h.new_zeros(B, h.shape[1], K)
        phen_tokens = h.new_zeros(B, K, h.shape[2])
        phen_active = h.new_zeros(B, K, dtype=torch.bool)

        for b in range(B):
            valid = mask[b] if mask is not None else torch.ones(h.shape[1], dtype=torch.bool, device=h.device)
            for k in range(K):
                in_k = (hard[b] == k) & valid
                if not bool(in_k.any()):
                    continue
                phen_active[b, k] = True
                h_k = h[b, in_k].unsqueeze(0)
                attn_logits = self.gated_attn_phen(h_k)[:, :, k]
                if (
                    self.training
                    and self.cfg.stkim_p > 0.0
                    and stochastic_topk_attn_mask is not None
                ):
                    n_k = int(in_k.sum())
                    k_top = int(self.cfg.stkim_k)
                    if self.cfg.stkim_frac > 0.0:
                        k_top = max(k_top, int(self.cfg.stkim_frac * n_k))
                    attn_logits = stochastic_topk_attn_mask(
                        attn_logits.unsqueeze(-1),
                        k=k_top,
                        p=float(self.cfg.stkim_p),
                        mask=None,
                    ).squeeze(-1)
                patch_attn_k = F.softmax(attn_logits / temp, dim=1)
                phen_tokens[b, k] = torch.einsum("bn,bnd->bd", patch_attn_k, h_k)[0]
                patch_attn[b, in_k, k] = patch_attn_k[0]

        return phen_tokens, patch_attn, soft_assign, sim_for_loss, phen_active

    def _khead_slide_repr(
        self,
        phen_tokens: torch.Tensor,
        phen_active: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pool K phenotype tokens → slide vector; optional phenotype-level attention weights."""
        pool = self.cfg.khead_pool
        if pool == "mean":
            if phen_active is not None:
                w = phen_active.float().unsqueeze(-1)
                denom = w.sum(dim=1).clamp(min=1.0)
                return (phen_tokens * w).sum(dim=1) / denom, None
            return phen_tokens.mean(dim=1), None
        if pool == "token_abmil":
            if self.token_attn_pool is None:
                raise RuntimeError("token_abmil readout requires token_attn_pool module")
            attn_logits = self.token_attn_pool(phen_tokens).squeeze(-1)
            if phen_active is not None:
                attn_logits = attn_logits.masked_fill(~phen_active, float("-inf"))
            phen_w = F.softmax(attn_logits, dim=-1)
            slide = torch.einsum("bk,bkd->bd", phen_w, phen_tokens)
            return slide, phen_w
        return phen_tokens.reshape(phen_tokens.shape[0], -1), None

    def _forward_khead(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> dict:
        B = h.shape[0]
        phen_tokens, patch_attn, soft_assign, sim_for_loss, phen_active = self._khead_pool(h, mask)
        s_proto, phen_token_attn = self._khead_slide_repr(phen_tokens, phen_active)
        logits_proto = self.head_proto(s_proto)
        s_phen = phen_tokens.mean(dim=1)

        if self.cfg.readout == "khead_abmil":
            attn_logits_detail = self.gated_attn_detail(h).squeeze(-1)
            if mask is not None:
                attn_logits_detail = attn_logits_detail.masked_fill(~mask, float("-inf"))
            patch_attn_detail = F.softmax(attn_logits_detail, dim=1)
            s_detail = torch.einsum("bn,bnd->bd", patch_attn_detail, h)
            logits_detail = self.head_abmil(s_detail)
            logits_bin = logits_detail + logits_proto
            s_combined = torch.cat([s_proto, s_detail], dim=-1)
            gate = None
        else:
            patch_attn_detail = None
            logits_detail = None
            gate = None
            logits_bin = logits_proto
            s_combined = s_proto
            s_detail = None

        return {
            "logits_bin": logits_bin,
            "logits_phen": logits_proto,
            "logits_detail": logits_detail,
            "patch_attn": patch_attn,
            "patch_attn_detail": patch_attn_detail,
            "soft_assign": soft_assign,
            "assign_sim": sim_for_loss,
            "phen_tokens": phen_tokens,
            "phen_token_attn": phen_token_attn,
            "phenotype_active": phen_active,
            "slide_repr_phen": s_phen,
            "slide_repr": s_combined,
            "proto_class_logits": None,
            "fusion_gate": gate,
            "patch_mask": mask,
            "tau": torch.exp(self.log_temp).clamp(min=1e-3).detach(),
        }

    @staticmethod
    def mean_patch_attention_entropy(patch_attn: torch.Tensor) -> torch.Tensor:
        """Mean Shannon entropy over K khead patch-attention maps (softmax over N)."""
        if patch_attn.ndim == 2:
            patch_attn = patch_attn.unsqueeze(0)
        k = patch_attn.shape[-1]
        ent = patch_attn.new_zeros(())
        for i in range(k):
            ent = ent + attention_entropy(patch_attn[0, :, i])
        return ent / k

    def _forward_full(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> dict:
        sim, tau = self._proto_similarity(h)
        sim_for_loss = sim.clone()
        if mask is not None:
            sim_masked = sim.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        else:
            sim_masked = sim
        soft_assign = F.softmax(sim_masked, dim=-1)

        attn_logits_phen = self.gated_attn_phen(h)
        log_gate = torch.log(soft_assign.clamp_min(1e-8))
        attn_combined = attn_logits_phen + log_gate
        if mask is not None:
            attn_combined = attn_combined.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        patch_attn_phen = F.softmax(attn_combined, dim=1)
        Z = torch.einsum("bnk,bnd->bkd", patch_attn_phen, h)

        if self.cls_token is not None:
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            Z_in = torch.cat([cls, Z], dim=1)
            Z_out = self.cross_proto(Z_in)
            s_phen = Z_out[:, 0]
            phen_tokens = Z_out[:, 1:]
        else:
            phen_tokens = self.cross_proto(Z)
            s_phen = phen_tokens.mean(dim=1)

        logits_phen = self.head_bin_phen(s_phen)

        if self.gated_attn_detail is not None:
            attn_logits_detail = self.gated_attn_detail(h).squeeze(-1)
            if mask is not None:
                attn_logits_detail = attn_logits_detail.masked_fill(~mask, float("-inf"))
            patch_attn_detail = F.softmax(attn_logits_detail, dim=1)
            s_detail = torch.einsum("bn,bnd->bd", patch_attn_detail, h)
            logits_detail = self.head_bin_detail(s_detail)
            gate_in = torch.cat([s_phen, s_detail], dim=-1)
            gate = self.fusion_gate(gate_in)
            logits_bin = gate * logits_phen + (1.0 - gate) * logits_detail
            s_combined = gate * s_phen + (1.0 - gate) * s_detail
        else:
            patch_attn_detail = None
            logits_detail = None
            gate = None
            logits_bin = logits_phen
            s_combined = s_phen

        proto_class_logits = self.proto_class_head(phen_tokens)

        return {
            "logits_bin": logits_bin,
            "logits_phen": logits_phen,
            "logits_detail": logits_detail,
            "patch_attn": patch_attn_phen,
            "patch_attn_detail": patch_attn_detail,
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

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, D); got {tuple(x.shape)}")
        B, N, _ = x.shape

        x, coords, mask = self._mine_instances(x, coords, mask)
        B, N, _ = x.shape

        if self.training and self.cfg.patch_dropout > 0.0 and N > 64:
            keep_prob = 1.0 - self.cfg.patch_dropout
            keep = torch.rand(B, N, device=x.device) < keep_prob
            for b in range(B):
                if keep[b].sum() < 64:
                    keep[b] = True
            if mask is None:
                mask = keep
            else:
                mask = mask & keep

        h = self.proj(x)
        if self.spatial_block is not None:
            h = self.spatial_block(h, coords)

        if self.cfg.readout in ("khead", "khead_abmil"):
            return self._forward_khead(h, mask)
        return self._forward_full(h, mask)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        apply_calibration: bool = True,
    ) -> dict:
        out = self.forward(x, coords=coords)
        logits = out["logits_bin"]
        if apply_calibration:
            logits = logits / self.calibration_temperature.clamp_min(1e-3)
        probs = F.softmax(logits, dim=-1)
        out["probs"] = probs
        out["pred_class"] = probs.argmax(dim=-1)
        out["prob_positive"] = probs[:, 1]
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
        if x.ndim != 3:
            raise ValueError(f"x must be (B, N, D); got {tuple(x.shape)}")
        B, N, _ = x.shape
        if N <= min_patches:
            return self.predict(x, coords=coords, apply_calibration=apply_calibration)

        keep = max(int(N * subsample_frac), min_patches)
        all_probs = []
        for _ in range(n_samples):
            idx = torch.randperm(N, device=x.device)[:keep]
            sub_x = x[:, idx]
            sub_c = coords[:, idx] if coords is not None else None
            single = self.predict(sub_x, coords=sub_c, apply_calibration=apply_calibration)
            all_probs.append(single["probs"])
        probs_stack = torch.stack(all_probs, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        return {
            "probs": mean_probs,
            "probs_std": std_probs,
            "pred_class": mean_probs.argmax(dim=-1),
            "prob_positive": mean_probs[:, 1],
            "n_tta_samples": n_samples,
        }

    def fit_temperature(self, val_loader, device: torch.device, max_iter: int = 50, lr: float = 0.01) -> float:
        self.eval()
        all_logits = []
        all_y = []
        for batch in val_loader:
            x = batch["features"].to(device)
            c = batch["coords"].to(device) if batch["coords"] is not None else None
            y = batch["label"].to(device)
            with torch.no_grad():
                out = self.forward(x, coords=c)
            all_logits.append(out["logits_bin"])
            all_y.append(y)
        if not all_logits:
            return float(self.calibration_temperature.item())
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        T = nn.Parameter(torch.ones(1, device=device))
        optim = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optim.zero_grad()
            loss = F.cross_entropy(logits / T.clamp_min(1e-3), y.view(-1))
            loss.backward()
            return loss

        optim.step(closure)
        T_val = float(T.detach().clamp_min(1e-3).item())
        self.calibration_temperature.data = torch.tensor(
            [T_val], device=self.calibration_temperature.device
        )
        return T_val
