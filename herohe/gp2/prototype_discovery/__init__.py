"""Offline phenotype prototype discovery (PhiHER2-style AP, k-means legacy).

Used before MIL training to build interpretable prototype vectors from frozen
patch embeddings. Does not change the PhenoBIN forward graph — only the initial
(or frozen) ``model.prototypes`` values.
"""

from .ap_hierarchical import (
    HierarchicalAPResult,
    hierarchical_ap_prototypes,
    pmil_euclidean_similarity,
    stage2_ap_from_pool,
    stage2_kmeans_from_pool,
)
from .io import load_prototype_checkpoint, save_hierarchical_ap_checkpoint, save_prototype_checkpoint
from .patch_sampling import (
    collect_patches_per_slide,
    slide_ids_from_csv,
    train_slide_ids_from_folds,
)

__all__ = [
    "HierarchicalAPResult",
    "hierarchical_ap_prototypes",
    "stage2_ap_from_pool",
    "stage2_kmeans_from_pool",
    "load_prototype_checkpoint",
    "save_hierarchical_ap_checkpoint",
    "save_prototype_checkpoint",
    "collect_patches_per_slide",
    "slide_ids_from_csv",
    "train_slide_ids_from_folds",
]
