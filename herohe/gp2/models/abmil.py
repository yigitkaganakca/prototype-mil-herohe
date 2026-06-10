"""AB-MIL — re-exports vendor adapter (Ilse 2018 / AttentionDeepMIL).

Prefer ``herohe.gp2.vendor.adapters.abmil`` or ``herohe.gp2.vendor.factory``.
"""

import torch

from herohe.gp2.vendor.adapters.abmil import (  # noqa: F401
    ABMIL,
    ABMILConfig,
    attention_entropy,
)


def stochastic_topk_attn_mask(
    attn_logits: torch.Tensor,
    k: int,
    p: float,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """ACMIL STKIM: zero top-k pre-softmax logits with prob p (train-only caller)."""
    if p <= 0.0 or k <= 0:
        return attn_logits
    out = attn_logits.clone()
    batched_heads = out.ndim == 3
    if not batched_heads:
        out = out.unsqueeze(-1)
    b, n, h = out.shape
    for bi in range(b):
        valid = torch.ones(n, dtype=torch.bool, device=out.device)
        if mask is not None:
            m = mask[bi] if mask.ndim == 2 else mask
            valid = m.bool()
        n_valid = int(valid.sum().item())
        if n_valid <= 0:
            continue
        k_eff = min(k, n_valid)
        for hi in range(h):
            logits_h = out[bi, :, hi].masked_fill(~valid, float("-inf"))
            top_idx = torch.topk(logits_h, k_eff, dim=0).indices
            if p >= 1.0:
                drop = top_idx
            else:
                drop = top_idx[torch.rand(k_eff, device=out.device) < p]
            if drop.numel() > 0:
                out[bi, drop, hi] = float("-inf")
    return out if batched_heads else out.squeeze(-1)
