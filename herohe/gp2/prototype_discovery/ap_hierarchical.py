"""PhiHER2-style hierarchical Affinity Propagation for phenotype prototype discovery.

Matches the official PhiHER2 implementation (``utils/cluster_utils.py``):

* Stage 1: AP on each slide's patch embeddings → slide-level exemplar indices/features.
* Stage 2: AP on the **union** of all stage-1 exemplars → data-driven L global prototypes.
* Similarity: ``exp(-λ · d / max(d))`` on normalized Euclidean distances (PMIL-style).
* AP: ``affinity='precomputed'``, ``preference=0``, ``damping=0.5`` (HEROHE.yaml defaults).
* Exemplars are **actual input feature vectors** at ``cluster_centers_indices_``, not sklearn
  ``cluster_centers_``.

Reference: Yan et al., PhiHER2 (Bioinformatics 2024, btae236).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances

# Benign sklearn AP message-passing warnings on precomputed affinities (see btae236 / PhiHER2).
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"sklearn\.utils\.extmath",
)


@dataclass
class Stage1Result:
    slide_id: str
    n_patches: int
    n_exemplars: int
    exemplars: np.ndarray  # (c_i, D)


@dataclass
class HierarchicalAPResult:
    centers: np.ndarray  # (L, D)
    stage1: list[Stage1Result] = field(default_factory=list)
    n_stage2_input: int = 0
    preference: float = 0.0
    damping: float = 0.5
    lamb: float = 0.25


def pmil_euclidean_similarity(X: np.ndarray, lamb: float) -> np.ndarray:
    """PhiHER2 / PMIL similarity: exp(-λ · normalized Euclidean distance)."""
    X64 = np.asarray(X, dtype=np.float64)
    dists = pairwise_distances(X64, metric="euclidean")
    max_d = float(np.max(dists))
    if max_d > 0.0:
        dists = dists / max_d
    return np.exp(-dists * float(lamb))


def _ap_exemplar_features_from_similarity(
    X: np.ndarray,
    similarity: np.ndarray,
    *,
    preference: float,
    damping: float,
    random_state: int,
) -> np.ndarray:
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"X must be non-empty 2D, got {X.shape}")
    if X.shape[0] == 1:
        return X.astype(np.float32, copy=True)
    ap = AffinityPropagation(
        affinity="precomputed",
        preference=float(preference),
        damping=float(damping),
        max_iter=500,
        convergence_iter=30,
        random_state=int(random_state),
        verbose=False,
    )
    ap.fit(similarity)
    idx = ap.cluster_centers_indices_
    if idx is None or len(idx) == 0:
        return X.mean(axis=0, keepdims=True).astype(np.float32)
    return X[np.asarray(idx, dtype=np.int64)].astype(np.float32, copy=False)


def _ap_exemplar_features(
    X: np.ndarray,
    *,
    preference: float,
    damping: float,
    lamb: float,
    random_state: int,
) -> np.ndarray:
    """Run AP on X (n, D) with precomputed PMIL similarity; return exemplar rows (L, D)."""
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"X must be non-empty 2D, got {X.shape}")
    if X.shape[0] == 1:
        return X.astype(np.float32, copy=True)

    similarity = pmil_euclidean_similarity(X, lamb)
    return _ap_exemplar_features_from_similarity(
        X,
        similarity,
        preference=preference,
        damping=damping,
        random_state=random_state,
    )


def count_ap_exemplars(
    X: np.ndarray,
    *,
    preference: float,
    damping: float,
    lamb: float,
    seed: int,
    similarity: np.ndarray | None = None,
) -> int:
    """Return L from a single AP fit (for preference search)."""
    if similarity is None:
        return int(
            _ap_exemplar_features(
                X,
                preference=preference,
                damping=damping,
                lamb=lamb,
                random_state=seed,
            ).shape[0]
        )
    return int(
        _ap_exemplar_features_from_similarity(
            X,
            similarity,
            preference=preference,
            damping=damping,
            random_state=seed,
        ).shape[0]
    )


def tune_stage2_preference(
    X: np.ndarray,
    target_L: int,
    *,
    damping: float = 0.5,
    lamb: float = 0.25,
    seed: int = 0,
    preference_min: float = -0.5,
    preference_max: float = 0.5,
    preference_step: float = 0.005,
) -> tuple[float, int]:
    """Find stage-2 AP preference so L is close to ``target_L``.

    Uses bracketing + binary search on preference (higher pref → more exemplars
    on PMIL similarities, lower pref → fewer). Falls back to a coarse grid if
    bracketing fails.
    """
    if target_L < 1:
        raise ValueError(f"target_L must be >= 1, got {target_L}")
    print(
        f"[hierarchical_ap] tuning stage2 preference for target_L={target_L}",
        flush=True,
    )
    similarity = pmil_euclidean_similarity(X, lamb)

    def _L(pref: float) -> int:
        return count_ap_exemplars(
            X,
            preference=pref,
            damping=damping,
            lamb=lamb,
            seed=seed,
            similarity=similarity,
        )

    hi = float(preference_max)
    L_hi = _L(hi)
    lo = float(preference_min)
    L_lo = _L(lo)

    # Expand lower bound until L drops below target (or floor hit).
    for _ in range(12):
        if L_lo <= target_L or lo <= -2.0:
            break
        lo -= 0.15
        L_lo = _L(lo)

    # Expand upper bound until L rises above target.
    for _ in range(12):
        if L_hi >= target_L or hi >= 1.0:
            break
        hi += 0.05
        L_hi = _L(hi)

    best_pref, best_L, best_gap = hi, L_hi, abs(L_hi - target_L)

    if L_lo <= target_L <= L_hi or L_hi <= target_L <= L_lo:
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            L_mid = _L(mid)
            gap = abs(L_mid - target_L)
            if gap < best_gap:
                best_gap, best_pref, best_L = gap, mid, L_mid
            if L_mid > target_L:
                hi = mid
            else:
                lo = mid

    # Coarse grid fallback / refinement around best
    for delta in np.linspace(-0.05, 0.05, 21):
        pref = best_pref + float(delta)
        L = _L(pref)
        gap = abs(L - target_L)
        if gap < best_gap:
            best_gap, best_pref, best_L = gap, pref, L

    print(
        f"[hierarchical_ap] tune done: preference={best_pref:.4f} L={best_L} "
        f"(target={target_L}, gap={best_gap})",
        flush=True,
    )
    return best_pref, best_L


def hierarchical_ap_prototypes(
    slide_patches: dict[str, np.ndarray],
    *,
    preference: float = 0.0,
    stage2_preference: float | None = None,
    target_L: int | None = None,
    stage2_method: str = "ap",
    damping: float = 0.5,
    lamb: float = 0.25,
    seed: int = 0,
    min_patches_per_slide: int = 8,
) -> HierarchicalAPResult:
    """Discover global prototype centers from per-slide patch bags (two-stage AP).

    Args:
        slide_patches: mapping slide_id → (n, D) patch feature matrix (already subsampled).
        preference: AP preference for stage 1 (PhiHER2 HEROHE.yaml: 0).
        stage2_preference: optional override for stage 2; defaults to ``preference``.
        target_L: if set, grid-search stage-2 preference to approximate this L.
        damping: AP damping in (0.5, 1] (PhiHER2: 0.5).
        lamb: PMIL similarity scale (PhiHER2 HEROHE.yaml: 0.25).
        min_patches_per_slide: skip slides with fewer patches.
    """
    stage1_results: list[Stage1Result] = []
    pooled: list[np.ndarray] = []

    for i, (slide_id, X) in enumerate(sorted(slide_patches.items())):
        if X.shape[0] < min_patches_per_slide:
            continue
        exemplars = _ap_exemplar_features(
            X,
            preference=preference,
            damping=damping,
            lamb=lamb,
            random_state=seed + i * 17,
        )
        if (i + 1) % 25 == 0 or i == 0:
            print(
                f"[hierarchical_ap] stage1 {i + 1}/{len(slide_patches)} slides  "
                f"last={slide_id} c_i={exemplars.shape[0]}",
                flush=True,
            )
        stage1_results.append(
            Stage1Result(
                slide_id=slide_id,
                n_patches=int(X.shape[0]),
                n_exemplars=int(exemplars.shape[0]),
                exemplars=exemplars,
            )
        )
        pooled.append(exemplars)

    if not pooled:
        raise RuntimeError("hierarchical AP: no slide-level exemplars produced")

    stage2_in = np.concatenate(pooled, axis=0)
    n_in = int(stage2_in.shape[0])
    if stage2_method == "kmeans":
        if target_L is None:
            raise ValueError("stage2_method=kmeans requires target_L")
        global_centers = stage2_kmeans_from_pool(stage2_in, int(target_L), seed=seed)
        s2_pref = float("nan")
    else:
        s2_pref = float(preference if stage2_preference is None else stage2_preference)
        if target_L is not None:
            s2_pref, est_L = tune_stage2_preference(
                stage2_in,
                int(target_L),
                damping=damping,
                lamb=lamb,
                seed=seed + 99991,
            )
            print(
                f"[hierarchical_ap] target_L={target_L} → stage2_preference={s2_pref:.4f} "
                f"(estimated L={est_L})",
                flush=True,
            )
        print(
            f"[hierarchical_ap] stage2 AP on n={n_in} pooled exemplars  preference={s2_pref}",
            flush=True,
        )
        global_centers = _ap_exemplar_features(
            stage2_in,
            preference=s2_pref,
            damping=damping,
            lamb=lamb,
            random_state=seed + 99991,
        )

    return HierarchicalAPResult(
        centers=global_centers,
        stage1=stage1_results,
        n_stage2_input=n_in,
        preference=float(s2_pref),
        damping=float(damping),
        lamb=float(lamb),
    )


def stage2_kmeans_from_pool(
    stage2_in: np.ndarray,
    K: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Condense pooled stage-1 exemplars to exactly K centers (MiniBatchKMeans)."""
    from sklearn.cluster import MiniBatchKMeans

    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if stage2_in.shape[0] <= K:
        return stage2_in.astype(np.float32, copy=False)
    km = MiniBatchKMeans(
        n_clusters=int(K),
        random_state=int(seed),
        n_init=10,
        batch_size=4096,
        max_iter=200,
    )
    km.fit(stage2_in)
    print(f"[hierarchical_ap] stage2 k-means on n={stage2_in.shape[0]} → K={K}", flush=True)
    return km.cluster_centers_.astype(np.float32, copy=False)


def stage2_ap_from_pool(
    stage2_in: np.ndarray,
    *,
    preference: float | None = None,
    target_L: int | None = None,
    damping: float = 0.5,
    lamb: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Run stage-2 AP on a precomputed pool of stage-1 exemplars."""
    s2_pref = 0.0 if preference is None else float(preference)
    if target_L is not None:
        s2_pref, est_L = tune_stage2_preference(
            stage2_in,
            int(target_L),
            damping=damping,
            lamb=lamb,
            seed=seed + 99991,
        )
        print(
            f"[hierarchical_ap] target_L={target_L} → stage2_preference={s2_pref:.4f} "
            f"(estimated L={est_L})",
            flush=True,
        )
    print(
        f"[hierarchical_ap] stage2 AP on n={stage2_in.shape[0]} pooled exemplars  "
        f"preference={s2_pref}",
        flush=True,
    )
    centers = _ap_exemplar_features(
        stage2_in,
        preference=s2_pref,
        damping=damping,
        lamb=lamb,
        random_state=seed + 99991,
    )
    return centers, s2_pref
