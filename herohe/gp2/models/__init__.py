"""PhenoHER2 model package.

Phenotype-aware ordinal MIL for granular HER2 scoring on H&E WSIs.

Public surface:
    PhenoHER2Config, PhenoHER2  -- 4-class IHC model
    PhenoHER2BinaryConfig, PhenoHER2Binary  -- ISH binary (Negative/Positive)
    PhenoHER2Loss / PhenoHER2BinaryLoss
    HerohePatchBagDataset       -- feature bags + label (IHC or gt_binary)
"""

from .phenotype_mil import PhenoHER2, PhenoHER2Config
from .phenotype_mil_binary import PhenoHER2Binary, PhenoHER2BinaryConfig
from .losses import (
    PhenoHER2Loss,
    PhenoHER2BinaryLoss,
    LossWeights,
    BinaryLossWeights,
    effective_number_class_weights,
    soft_ordinal_target,
    squared_emd_loss,
)
from .dataset import (
    HerohePatchBagDataset,
    load_herohe_labels,
    load_herohe_binary_labels,
    load_herohe_valieris_3_labels,
    collate_single_bag,
)
from .abmil import ABMIL, ABMILConfig
from .transmil import TransMIL, TransMILConfig
from .clam_mb import CLAM_MB

__all__ = [
    "PhenoHER2",
    "PhenoHER2Config",
    "PhenoHER2Binary",
    "PhenoHER2BinaryConfig",
    "PhenoHER2Loss",
    "PhenoHER2BinaryLoss",
    "LossWeights",
    "BinaryLossWeights",
    "effective_number_class_weights",
    "soft_ordinal_target",
    "squared_emd_loss",
    "HerohePatchBagDataset",
    "load_herohe_labels",
    "load_herohe_binary_labels",
    "load_herohe_valieris_3_labels",
    "collate_single_bag",
    "ABMIL",
    "ABMILConfig",
    "TransMIL",
    "TransMILConfig",
    "CLAM_MB",
]
