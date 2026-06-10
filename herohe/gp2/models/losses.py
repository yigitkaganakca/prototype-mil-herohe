"""Loss functions for the prototype MIL models.

WHAT THE REPORTED RESULTS ACTUALLY USE
--------------------------------------
The reported binary / 3-class model is trained with ``PhenoHER2BinaryLoss`` and, in the
released run scripts, ONLY the class-weighted cross-entropy term is active: the run
scripts pass ``--w_balance 0`` and ``--w_orth 0`` (and ``--w_attn_entropy 0``), so the
Sinkhorn prototype-balance and prototype-orthogonality terms are switched off. In other
words, the effective training objective is class-weighted CE (effective-number weights,
Cui et al. 2019) with label smoothing.

DEPRECATED / EXPERIMENTAL (kept for reference, NOT used for the reported numbers)
--------------------------------------------------------------------------------
* ``PhenoHER2Loss`` + ``LossWeights`` — the full v2 4-class multi-task objective
  (EMD ordinal + 0-vs-1+ and 3+ aux heads + Sinkhorn balance + orthogonality). The v2
  4-class model/trainer that consumed this was retired; this whole class is unused.
* ``squared_emd_loss`` / ``soft_ordinal_target`` — ordinal (EMD²) trial. ``soft_ordinal_target``
  is still imported by the binary trainer to build bag-mixup soft targets, but mixup is
  disabled in the reported runs (``--mixup_alpha 0 --mixup_p 0``).
* ``sinkhorn`` / ``sinkhorn_balance_loss`` — SwAV-style prototype-balance trial
  (``w_balance`` weight); set to 0 in the reported runs.
* ``prototype_orthogonality_loss`` — prototype-diversity trial (``w_orth`` weight);
  set to 0 in the reported runs.

Original v2 design notes follow.

Components:
    1. Class-weighted Cross-Entropy on the 4-class head (FR6).
       Weights via "Effective Number of Samples" (Cui et al., CVPR 2019).

    2. Squared Earth Mover's Distance (EMD^2) ordinal loss on the same 4-class
       softmax, against neighbour-smoothed soft targets. Replaces the v1 CORN
       loss. EMD^2 (Hou, Yu & Samaras 2016) gives smoother gradients and
       penalises far-off predictions quadratically -- exactly what the HER2
       clinical risk function asks for (predicting 0 when truth is 3+ is
       far worse than mistaking 1+ for 2+).

    3. Binary Cross-Entropy for HER2-0 vs HER2-Low (1+) head (NFR3 sub-goal).

    4. Binary Cross-Entropy for the HER2-3+ confidence head (FR7 high-confidence flag).

    5. Sinkhorn-Knopp prototype balance loss (replaces v1 entropy regulariser).
       Computes a doubly-stochastic target assignment Q via SwAV-style Sinkhorn
       iterations (no_grad), then trains the predicted soft assignment A to
       match Q via cross-entropy. This balances prototype usage *per slide*
       rather than only on the marginal, and prevents collapse much more
       reliably than the entropy regulariser.

    6. Orthogonality regulariser on the prototype matrix (unchanged from v1).

The weights of the six components are exposed as constructor arguments so they
can be tuned in WP4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Class-weight utilities
# --------------------------------------------------------------------------------------


def effective_number_class_weights(class_counts, beta: float = 0.999) -> torch.Tensor:
    """Effective Number of Samples class weighting (Cui et al., CVPR 2019).

    w_c = (1 - beta) / (1 - beta^{n_c}); then normalised so the mean weight is 1.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if torch.any(counts <= 0):
        counts = torch.where(counts <= 0, torch.ones_like(counts), counts)
    eff_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / eff_num
    weights = weights / weights.mean()
    return weights.float()


# --------------------------------------------------------------------------------------
# Soft ordinal target + Squared EMD loss
# --------------------------------------------------------------------------------------


def soft_ordinal_target(
    targets: torch.Tensor, num_classes: int, smoothing: float = 0.1
) -> torch.Tensor:
    """Build a neighbour-smoothed soft target for ordinal classification.

    For each integer label y in [0, num_classes-1] the resulting distribution puts
    the bulk of the mass on y, splits `smoothing` mass between y-1 and y+1 (where
    they exist), and keeps non-adjacent classes at exactly zero.

    Args:
        targets: (B,) long tensor in [0, num_classes-1].
        num_classes: K (4 for HER2: 0,1,2,3).
        smoothing: total mass moved away from the true class to its neighbours.
            0.0 reproduces standard one-hot; values in [0.05, 0.2] are typical.

    Returns:
        (B, K) float tensor whose rows each sum to 1.
    """
    if smoothing < 0.0 or smoothing > 1.0:
        raise ValueError(f"smoothing must be in [0, 1]; got {smoothing}")
    B = targets.shape[0]
    out = targets.new_zeros(B, num_classes, dtype=torch.float32)
    for c in range(num_classes):
        mask = targets == c
        if not torch.any(mask):
            continue
        has_left = c > 0
        has_right = c < num_classes - 1
        n_neighbours = int(has_left) + int(has_right)
        # Mass spread across the existing neighbours
        side = smoothing / max(n_neighbours, 1) if n_neighbours > 0 else 0.0
        center = 1.0 - smoothing if n_neighbours > 0 else 1.0
        out[mask, c] = center
        if has_left:
            out[mask, c - 1] = side
        if has_right:
            out[mask, c + 1] = side
    return out


def squared_emd_loss(probs: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    """[DEPRECATED — v2 ordinal trial; not used in the reported model.]

    Squared Earth Mover's Distance loss for ordinal classification.

    L(p, q) = (1/(K-1)) * sum_{k=1..K-1} ( CDF_p(k) - CDF_q(k) )^2

    where CDF_p(k) = sum_{j<=k-1} p_j  (and CDF_p(K) is always 1).

    Args:
        probs: (B, K) softmax probabilities.
        target_distribution: (B, K) soft target distribution (rows sum to 1).

    Returns:
        Scalar tensor: mean over the batch.
    """
    if probs.shape != target_distribution.shape:
        raise ValueError(
            f"probs {tuple(probs.shape)} != target {tuple(target_distribution.shape)}"
        )
    if probs.dim() != 2:
        raise ValueError(f"probs must be (B, K); got {tuple(probs.shape)}")
    cdf_p = torch.cumsum(probs, dim=1)
    cdf_q = torch.cumsum(target_distribution, dim=1)
    # The last bin is always 1.0 for both; drop it.
    diff = cdf_p[:, :-1] - cdf_q[:, :-1]
    K = probs.shape[1]
    return (diff.pow(2).sum(dim=1) / max(K - 1, 1)).mean()


# --------------------------------------------------------------------------------------
# Sinkhorn-Knopp prototype balance loss
# --------------------------------------------------------------------------------------


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, n_iter: int = 3, epsilon: float = 0.05) -> torch.Tensor:
    """[DEPRECATED — prototype-balance trial; ``w_balance=0`` in the reported runs.]

    SwAV-style Sinkhorn-Knopp normalisation.

    Args:
        scores: (N, K) raw assignment logits (e.g. cosine sim / tau).
        n_iter: number of Sinkhorn iterations (3 is standard).
        epsilon: temperature for the implicit softmax; smaller -> harder.

    Returns:
        (N, K) soft assignment with rows summing to 1 and column-mass
        approximately uniform across the K prototypes.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be (N, K); got {tuple(scores.shape)}")
    Q = torch.exp(scores / epsilon).t()                 # (K, N)
    Q = Q / (Q.sum() + 1e-8)
    K, N = Q.shape
    K_t = torch.tensor(float(K), device=Q.device)
    N_t = torch.tensor(float(N), device=Q.device)
    for _ in range(n_iter):
        # row-normalise (over N for each K) -> column-mass uniform
        sum_rows = Q.sum(dim=1, keepdim=True).clamp_min(1e-8)
        Q = Q / sum_rows / K_t
        # column-normalise (over K for each N) -> rows sum to 1/N
        sum_cols = Q.sum(dim=0, keepdim=True).clamp_min(1e-8)
        Q = Q / sum_cols / N_t
    Q = Q * N_t                                        # rows now sum to 1
    return Q.t()                                       # (N, K)


def sinkhorn_balance_loss(
    sim: torch.Tensor,
    soft_assign: torch.Tensor,
    mask: torch.Tensor | None = None,
    n_iter: int = 3,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """[DEPRECATED — prototype-balance trial; ``w_balance=0`` in the reported runs.]

    SwAV-style cross-entropy between predicted assignment and the
    Sinkhorn-balanced target.

    Args:
        sim: (B, N, K) raw similarity logits used to compute the soft_assign
             (we re-use them, not the post-softmax probabilities, for
             numerical-stable Sinkhorn).
        soft_assign: (B, N, K) predicted soft assignment (post-softmax).
        mask: (B, N) bool, True = valid patch. Optional.
        n_iter: Sinkhorn iterations.
        epsilon: Sinkhorn temperature.

    Returns:
        Scalar mean-CE loss across (B, N) of valid patches.
    """
    if sim.shape != soft_assign.shape:
        raise ValueError(
            f"sim {tuple(sim.shape)} != soft_assign {tuple(soft_assign.shape)}"
        )
    B = sim.shape[0]
    losses = []
    for b in range(B):
        s_b = sim[b]                           # (N, K)
        a_b = soft_assign[b]                   # (N, K)
        if mask is not None:
            valid = mask[b]
            if valid.sum() == 0:
                continue
            s_b = s_b[valid]
            a_b = a_b[valid]
        with torch.no_grad():
            q = sinkhorn(s_b, n_iter=n_iter, epsilon=epsilon)   # (N, K) target
        # Cross-entropy per patch: -sum_k q * log(a)
        log_a = torch.log(a_b.clamp_min(1e-8))
        ce = -(q * log_a).sum(dim=-1).mean()
        losses.append(ce)
    if not losses:
        return sim.new_zeros(())
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------------------
# Prototype orthogonality
# --------------------------------------------------------------------------------------


def prototype_orthogonality_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """[DEPRECATED — prototype-diversity trial; ``w_orth=0`` in the reported runs.]

    Penalise the off-diagonal cosine similarity of the K prototypes."""
    P = F.normalize(prototypes, dim=-1)
    G = P @ P.t()
    K = G.shape[0]
    off = G - torch.eye(K, device=G.device, dtype=G.dtype)
    return (off.pow(2).sum()) / (K * (K - 1) + 1e-6)


# --------------------------------------------------------------------------------------
# Multi-task loss container
# --------------------------------------------------------------------------------------


@dataclass
class LossWeights:
    """[DEPRECATED — weights for the unused v2 4-class ``PhenoHER2Loss``.]

    Loss weights for PhenoHER2 v2.
    `emd` replaces `ordinal` (CORN) from v1.
    `balance` replaces `diversity` (entropy reg) from v1.
    """

    ce: float = 1.0       # 4-class CE (one-hot)
    emd: float = 0.5      # EMD^2 ordinal (soft labels)
    aux01: float = 0.5    # 0 vs 1+ binary
    high3p: float = 0.25  # 3+ confidence
    balance: float = 0.05  # Sinkhorn balance (prototype load)
    orthogonality: float = 0.01


class PhenoHER2Loss(nn.Module):
    """[DEPRECATED — v2 4-class multi-task loss; NOT used for the reported results.]

    The reported binary / 3-class model uses ``PhenoHER2BinaryLoss`` below. This class
    (EMD ordinal + aux heads + Sinkhorn balance + orthogonality) was the v2 4-class
    objective and is retained only for reference.

    Multi-task loss for PhenoHER2 v2.

    Args:
        class_counts: per-class counts in the training fold for the CE weights.
        prototype_param: reference to model.prototypes (for orthogonality reg).
        weights: LossWeights dataclass.
        beta: effective-number-of-samples beta (default 0.999).
        num_classes: number of HER2 classes (default 4).
        soft_label_smoothing: mass moved to ordinal neighbours in EMD target.
        sinkhorn_iter: number of Sinkhorn iterations.
        sinkhorn_epsilon: Sinkhorn temperature.

    Forward:
        loss, parts = loss_fn(model_output, targets)
    """

    def __init__(
        self,
        class_counts,
        prototype_param: torch.nn.Parameter,
        weights: LossWeights = LossWeights(),
        beta: float = 0.999,
        num_classes: int = 4,
        soft_label_smoothing: float = 0.1,
        sinkhorn_iter: int = 3,
        sinkhorn_epsilon: float = 0.05,
    ):
        super().__init__()
        self.weights = weights
        self.num_classes = num_classes
        self.soft_label_smoothing = soft_label_smoothing
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_epsilon = sinkhorn_epsilon
        ce_w = effective_number_class_weights(class_counts, beta=beta)
        self.register_buffer("ce_weights", ce_w)
        self._prototype_param = prototype_param

    def forward(self, out: dict, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if targets.dtype != torch.long:
            targets = targets.long()
        parts: dict[str, torch.Tensor] = {}

        # 1. 4-class weighted CE on the fused logits
        ce = F.cross_entropy(out["logits_4cls"], targets, weight=self.ce_weights)
        parts["ce"] = ce.detach()

        # 2. EMD^2 against soft ordinal target (computed from the fused softmax)
        probs = F.softmax(out["logits_4cls"], dim=-1)
        soft_q = soft_ordinal_target(targets, self.num_classes, self.soft_label_smoothing)
        emd = squared_emd_loss(probs, soft_q.to(probs.device))
        parts["emd"] = emd.detach()

        # 3. 0 vs 1+ aux head: only on samples with label in {0, 1}
        mask01 = targets <= 1
        if torch.any(mask01):
            aux01 = F.binary_cross_entropy_with_logits(
                out["logit_aux_01"][mask01],
                targets[mask01].float(),
                reduction="mean",
            )
        else:
            aux01 = out["logit_aux_01"].sum() * 0.0
        parts["aux01"] = aux01.detach()

        # 4. 3+ confidence head
        target_3p = (targets == (self.num_classes - 1)).float()
        high3p = F.binary_cross_entropy_with_logits(out["logit_3p"], target_3p)
        parts["high3p"] = high3p.detach()

        # 5. Sinkhorn balance loss on prototype assignment
        sim = out.get("assign_sim")          # (B, N, K) raw cosine logits
        soft_assign = out.get("soft_assign")  # (B, N, K) softmax of sim
        if sim is None or soft_assign is None:
            balance = out["logits_4cls"].sum() * 0.0
        else:
            balance = sinkhorn_balance_loss(
                sim,
                soft_assign,
                mask=out.get("patch_mask"),
                n_iter=self.sinkhorn_iter,
                epsilon=self.sinkhorn_epsilon,
            )
        parts["balance"] = balance.detach()

        # 6. Prototype orthogonality
        orth = prototype_orthogonality_loss(self._prototype_param)
        parts["orthogonality"] = orth.detach()

        w = self.weights
        total = (
            w.ce * ce
            + w.emd * emd
            + w.aux01 * aux01
            + w.high3p * high3p
            + w.balance * balance
            + w.orthogonality * orth
        )
        parts["total"] = total.detach()
        return total, parts


# --------------------------------------------------------------------------------------
# PhenoHER2-Binary (2-class CE + prototype balance + orthogonality)
# --------------------------------------------------------------------------------------


@dataclass
class BinaryLossWeights:
    """Loss weights for the reported binary / 3-class model.

    NOTE: the dataclass defaults below enable balance/orthogonality, but the released
    run scripts override them to 0 (``--w_balance 0 --w_orth 0``). For the reported
    results only ``ce`` (class-weighted cross-entropy) is active; ``balance`` (Sinkhorn)
    and ``orthogonality`` are deprecated trials left switched off.
    """

    ce: float = 1.0
    balance: float = 0.05  # deprecated trial — set to 0 in the reported runs
    orthogonality: float = 0.01  # deprecated trial — set to 0 in the reported runs


class PhenoHER2BinaryLoss(nn.Module):
    """Class-weighted CE on fused logits + Sinkhorn balance + prototype orth."""

    def __init__(
        self,
        class_counts,
        prototype_param: torch.nn.Parameter,
        weights: BinaryLossWeights = BinaryLossWeights(),
        beta: float = 0.999,
        sinkhorn_iter: int = 3,
        sinkhorn_epsilon: float = 0.05,
        label_smoothing: float = 0.0,
        use_class_weights: bool = True,
    ):
        super().__init__()
        self.weights = weights
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.label_smoothing = float(label_smoothing)
        if use_class_weights:
            ce_w = effective_number_class_weights(class_counts, beta=beta)
        else:
            ce_w = None
        if ce_w is not None:
            self.register_buffer("ce_weights", ce_w)
        else:
            self.ce_weights = None
        self._prototype_param = prototype_param

    def forward(self, out: dict, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if targets.dtype != torch.long:
            targets = targets.long()
        parts: dict[str, torch.Tensor] = {}

        ce = F.cross_entropy(
            out["logits_bin"],
            targets.view(-1),
            weight=self.ce_weights,
            label_smoothing=self.label_smoothing,
        )
        parts["ce"] = ce.detach()

        w = self.weights
        balance = ce.new_zeros(())
        if w.balance > 0.0:
            sim = out.get("assign_sim")
            soft_assign = out.get("soft_assign")
            if sim is not None and soft_assign is not None:
                balance = sinkhorn_balance_loss(
                    sim,
                    soft_assign,
                    mask=out.get("patch_mask"),
                    n_iter=self.sinkhorn_iter,
                    epsilon=self.sinkhorn_epsilon,
                )
            parts["balance"] = balance.detach()

        orth = ce.new_zeros(())
        if w.orthogonality > 0.0:
            orth = prototype_orthogonality_loss(self._prototype_param)
            parts["orthogonality"] = orth.detach()

        total = w.ce * ce + w.balance * balance + w.orthogonality * orth
        parts["total"] = total.detach()
        return total, parts
