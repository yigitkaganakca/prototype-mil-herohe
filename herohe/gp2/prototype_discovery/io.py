"""Save/load prototype discovery checkpoints for PhenoBIN training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .ap_hierarchical import HierarchicalAPResult


def save_prototype_checkpoint(
    path: str | Path,
    centers: torch.Tensor,
    *,
    method: str,
    feature_dim: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a .pt file consumable by ``train_phenobin_mil.py --prototypes``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "centers": centers.detach().cpu().float(),
        "K": int(centers.shape[0]),
        "feature_dim": int(feature_dim),
        "method": method,
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, str(path))


def save_hierarchical_ap_checkpoint(
    path: str | Path,
    result: HierarchicalAPResult,
    *,
    slide_ids: list[str],
    patches_per_slide: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> None:
    centers = torch.from_numpy(result.centers)
    meta: dict[str, Any] = {
        "n_slides_used": len(result.stage1),
        "patches_per_slide": patches_per_slide,
        "seed": seed,
        "n_stage2_input": result.n_stage2_input,
        "preference": result.preference,
        "stage2_preference": result.preference,
        "damping": result.damping,
        "lamb": result.lamb,
        "stage1_exemplar_counts": {s.slide_id: s.n_exemplars for s in result.stage1},
        "train_slide_ids": slide_ids,
    }
    if extra:
        meta.update(extra)
    save_prototype_checkpoint(
        path,
        centers,
        method="hierarchical_ap",
        feature_dim=int(result.centers.shape[1]),
        metadata=meta,
    )


def load_prototype_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a prototype .pt file (k-means or hierarchical AP)."""
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "centers" in blob:
        return blob
    if torch.is_tensor(blob):
        return {"centers": blob.float(), "K": int(blob.shape[0]), "method": "legacy_tensor"}
    raise ValueError(f"Unrecognized prototype checkpoint format: {path}")
