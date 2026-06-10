"""HEROHE whole-slide image path helpers.

Training WSIs live under ``herohe/{slide_id}.mrxs`` (repo root relative to ``herohe/``).
Some training slides may be absent locally (deleted to save space).

Test WSIs live under ``herohe/wsi_test/{slide_id}.mrxs`` with a sibling data folder
``herohe/wsi_test/{slide_id}/``.

IMPORTANT: train and test share the same numeric slide IDs but are **different specimens**.
Always pass ``split="test"`` or ``split="train"`` explicitly when resolving paths for
figures, features, or qualitative analysis.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
HEROHE = _REPO / "herohe"
WSI_TEST_DIR = HEROHE / "wsi_test"
FEAT_TRAIN = HEROHE / "gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
FEAT_TEST = HEROHE / "gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2"
TRIDENT_TRAIN = HEROHE / "gp2/results_trident_mac_full"
TRIDENT_TEST = HEROHE / "gp2/results_trident_test"


def resolve_wsi(slide_id: str, split: str = "test") -> Path:
    """Return OpenSlide-readable path for a HEROHE slide."""
    sid = str(slide_id)
    if split == "test":
        p = WSI_TEST_DIR / f"{sid}.mrxs"
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"Test WSI not found: {p}. Test slides live under herohe/wsi_test/."
        )
    if split == "train":
        for root in (HEROHE, _REPO):
            p = root / f"{sid}.mrxs"
            if p.is_file():
                return p
        raise FileNotFoundError(
            f"Training WSI {sid}.mrxs not found under {HEROHE}. "
            "It may have been deleted locally; re-download if needed."
        )
    raise ValueError(f"split must be 'test' or 'train', got {split!r}")


def features_dir(split: str = "test") -> Path:
    return FEAT_TEST if split == "test" else FEAT_TRAIN


def trident_root(split: str = "test") -> Path:
    return TRIDENT_TEST if split == "test" else TRIDENT_TRAIN
