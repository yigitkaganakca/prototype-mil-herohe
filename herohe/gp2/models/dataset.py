"""Per-slide Virchow2 feature-bag dataset for HEROHE.

Loads patch embeddings from `<features_dir>/<slide_id>.h5` (TRIDENT layout) and
pairs them with the HER2 IHC label from the HEROHE ground truth CSV.

The HEROHE CSV is semicolon-delimited and looks like:

    Case;Immunohistochemistry;ISH Group;Final Result (Ground truth);Age;...
    1;2;5;Negative;24;8;1.60;2.40;F;1.50
    ...

We parse the `Case` column as the slide_id (string). With ``label_mode="ihc"`` (default),
`Immunohistochemistry` gives the 4-class IHC label in {0, 1, 2, 3}. With
``label_mode="gt_binary"``, `Final Result (Ground truth)` gives the original HEROHE
binary task: Negative=0, Positive=1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal, Optional
import zlib

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


HEROHE_LABEL_COL = "Immunohistochemistry"
HEROHE_CASE_COL = "Case"
# ISH-aligned deployment label (HEROHE challenge): Negative vs Positive
HEROHE_FINAL_GT_COL = "Final Result (Ground truth)"


def load_herohe_labels(labels_csv: str) -> pd.DataFrame:
    """Load and validate the HEROHE ground truth CSV.

    Returns a DataFrame with two columns:
        slide_id (str) -- e.g. "1", "12", ...
        label (int)    -- HER2 IHC score in {0, 1, 2, 3}
    """
    df = pd.read_csv(labels_csv, sep=";")
    if HEROHE_CASE_COL not in df.columns or HEROHE_LABEL_COL not in df.columns:
        raise ValueError(
            f"Labels CSV must contain '{HEROHE_CASE_COL}' and '{HEROHE_LABEL_COL}' columns; "
            f"found {list(df.columns)}"
        )
    df = df[[HEROHE_CASE_COL, HEROHE_LABEL_COL]].copy()
    df = df.rename(columns={HEROHE_CASE_COL: "slide_id", HEROHE_LABEL_COL: "label"})
    df["slide_id"] = df["slide_id"].astype(str)
    df["label"] = df["label"].astype(int)
    if not df["label"].between(0, 3).all():
        bad = df[~df["label"].between(0, 3)]
        raise ValueError(f"Labels outside [0, 3]: {bad.head().to_dict('records')}")
    return df.reset_index(drop=True)


def load_herohe_binary_labels(labels_csv: str) -> pd.DataFrame:
    """Load HEROHE slide-level binary labels (ISH ground truth).

    Maps ``Final Result (Ground truth)`` to integers: Negative -> 0, Positive -> 1,
    matching ``make_folds.py`` and ``folds_v1.csv`` ``gt_binary``.

    Returns:
        DataFrame with columns ``slide_id`` (str), ``label`` (int in {0, 1}).
    """
    df = pd.read_csv(labels_csv, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if HEROHE_CASE_COL not in df.columns or HEROHE_FINAL_GT_COL not in df.columns:
        raise ValueError(
            f"Binary labels CSV must contain '{HEROHE_CASE_COL}' and '{HEROHE_FINAL_GT_COL}'; "
            f"found {list(df.columns)}"
        )
    df = df[[HEROHE_CASE_COL, HEROHE_FINAL_GT_COL]].copy()
    df = df.rename(
        columns={HEROHE_CASE_COL: "slide_id", HEROHE_FINAL_GT_COL: "_gt_raw"}
    )
    df["slide_id"] = df["slide_id"].astype(str).str.strip()
    gt_raw = df["_gt_raw"].astype(str).str.strip().str.lower()
    df["label"] = gt_raw.map({"negative": 0, "positive": 1}).astype("Int64")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    if not df["label"].isin([0, 1]).all():
        bad = df[~df["label"].isin([0, 1])]
        raise ValueError(f"Unexpected Final Result values: {bad.head().to_dict('records')}")
    return df[["slide_id", "label"]].reset_index(drop=True)


def load_herohe_valieris_3_labels(labels_csv: str) -> pd.DataFrame:
    """Load ASCO/CAP-aligned 3-class labels (Valieris et al. 2024 mapping).

    Class 0 = HER2-negative (IHC 0)
    Class 1 = HER2-low (IHC 1+, or IHC 2+ with ISH-negative Final Result)
    Class 2 = HER2-high (IHC 3+, or IHC 2+ with ISH-positive Final Result)
    """
    df = pd.read_csv(labels_csv, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if HEROHE_CASE_COL not in df.columns or HEROHE_LABEL_COL not in df.columns:
        raise ValueError(
            f"Valieris 3-class CSV must contain '{HEROHE_CASE_COL}' and '{HEROHE_LABEL_COL}'; "
            f"found {list(df.columns)}"
        )
    if HEROHE_FINAL_GT_COL not in df.columns:
        raise ValueError(
            f"Valieris 3-class CSV must contain '{HEROHE_FINAL_GT_COL}' for IHC 2+ split"
        )
    df = df[[HEROHE_CASE_COL, HEROHE_LABEL_COL, HEROHE_FINAL_GT_COL]].copy()
    df = df.rename(
        columns={
            HEROHE_CASE_COL: "slide_id",
            HEROHE_LABEL_COL: "_ihc",
            HEROHE_FINAL_GT_COL: "_final",
        }
    )
    df["slide_id"] = df["slide_id"].astype(str).str.strip()
    ihc = pd.to_numeric(df["_ihc"], errors="coerce")
    final = df["_final"].astype(str).str.strip().str.lower()
    labels = []
    for ihc_val, fin in zip(ihc, final):
        if pd.isna(ihc_val):
            labels.append(np.nan)
            continue
        ihc_int = int(ihc_val)
        if ihc_int == 0:
            labels.append(0)
        elif ihc_int == 1:
            labels.append(1)
        elif ihc_int == 2:
            if fin == "negative":
                labels.append(1)
            elif fin == "positive":
                labels.append(2)
            else:
                labels.append(np.nan)
        elif ihc_int == 3:
            labels.append(2)
        else:
            labels.append(np.nan)
    df["label"] = labels
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    if not df["label"].between(0, 2).all():
        bad = df[~df["label"].between(0, 2)]
        raise ValueError(f"Valieris labels outside [0, 2]: {bad.head().to_dict('records')}")
    return df[["slide_id", "label"]].reset_index(drop=True)


@dataclass
class BagSample:
    slide_id: str
    label: int
    features: torch.Tensor   # (N, D_in)
    coords: Optional[torch.Tensor]   # (N, 2) or None
    n_patches: int


class HerohePatchBagDataset(Dataset):
    """Dataset of (Virchow2 feature bag, HER2 label) pairs.

    Args:
        features_dir: directory containing `<slide_id>.h5` files.
                      Expected datasets in each h5: 'features' (N, D_in) float32
                      and optionally 'coords' (N, 2) int64.
        labels_csv: path to HEROHE ground truth CSV.
        label_mode: ``"ihc"`` (default) uses 4-class IHC score {0,1,2,3};
                    ``"gt_binary"`` uses ISH Final Result Negative/Positive -> {0,1};
                    ``"valieris_3"`` uses ASCO/CAP neg / low / high -> {0,1,2}.
        slide_ids: optional iterable of slide_ids to include (otherwise everything
                   in `labels_csv` that has a matching .h5 file is used).
        max_patches: optional cap on bag size; if a slide has more patches we
                     uniformly subsample without replacement (training-time augmentation).
        subsample_mode: ``"random"`` draws a new subset on every ``__getitem__`` (train).
                        ``"deterministic"`` uses a fixed per-slide subset keyed by ``slide_id``
                        + ``seed`` (stable validation / inference).
        seed: reproducibility for subsampling.
        return_coords: if True, also return the coords tensor.
    """

    def __init__(
        self,
        features_dir: str,
        labels_csv: str,
        label_mode: str = "ihc",
        slide_ids: Optional[Iterable[str]] = None,
        max_patches: Optional[int] = None,
        subsample_mode: Literal["random", "deterministic"] = "random",
        seed: Optional[int] = None,
        return_coords: bool = True,
    ):
        super().__init__()
        self.features_dir = features_dir
        self.label_mode = label_mode
        self.return_coords = return_coords
        self.max_patches = max_patches
        if subsample_mode not in ("random", "deterministic"):
            raise ValueError(
                f"subsample_mode must be 'random' or 'deterministic', got {subsample_mode!r}"
            )
        self.subsample_mode = subsample_mode
        self._seed = 0 if seed is None else int(seed)
        self._rng = np.random.default_rng(self._seed)

        if label_mode == "ihc":
            labels_df = load_herohe_labels(labels_csv)
        elif label_mode == "gt_binary":
            labels_df = load_herohe_binary_labels(labels_csv)
        elif label_mode == "valieris_3":
            labels_df = load_herohe_valieris_3_labels(labels_csv)
        else:
            raise ValueError(
                f"label_mode must be 'ihc', 'gt_binary', or 'valieris_3', got {label_mode!r}"
            )
        if slide_ids is not None:
            slide_ids = set(str(s) for s in slide_ids)
            labels_df = labels_df[labels_df["slide_id"].isin(slide_ids)]

        # Drop slides without a corresponding .h5 file
        rows = []
        for _, row in labels_df.iterrows():
            sid = row["slide_id"]
            fp = os.path.join(features_dir, f"{sid}.h5")
            if os.path.isfile(fp):
                rows.append({"slide_id": sid, "label": int(row["label"]), "path": fp})
        if not rows:
            raise RuntimeError(
                f"No (slide_id, .h5) pairs found under {features_dir} for labels {labels_csv}"
            )
        self.entries = rows

    def __len__(self) -> int:
        return len(self.entries)

    def class_counts(self, num_classes: Optional[int] = None) -> np.ndarray:
        if num_classes is None:
            if self.label_mode == "ihc":
                num_classes = 4
            elif self.label_mode == "valieris_3":
                num_classes = 3
            else:
                num_classes = 2
        counts = np.zeros(num_classes, dtype=np.int64)
        for e in self.entries:
            counts[e["label"]] += 1
        return counts

    def slide_ids(self) -> list[str]:
        return [e["slide_id"] for e in self.entries]

    def labels(self) -> np.ndarray:
        return np.array([e["label"] for e in self.entries], dtype=np.int64)

    def _subsample_indices(self, n: int, slide_id: str) -> np.ndarray:
        if self.max_patches is None or n <= self.max_patches:
            return np.arange(n, dtype=np.int64)
        if self.subsample_mode == "deterministic":
            key = f"{self._seed}:{slide_id}".encode()
            slide_seed = zlib.adler32(key) & 0xFFFFFFFF
            rng = np.random.default_rng(slide_seed)
        else:
            rng = self._rng
        return np.sort(rng.choice(n, size=self.max_patches, replace=False))

    def __getitem__(self, idx: int) -> BagSample:
        entry = self.entries[idx]
        with h5py.File(entry["path"], "r") as f:
            feats = np.asarray(f["features"][:])
            coords = np.asarray(f["coords"][:]) if "coords" in f else None
        n = feats.shape[0]
        sel = self._subsample_indices(n, entry["slide_id"])
        if sel.shape[0] < n:
            feats = feats[sel]
            if coords is not None:
                coords = coords[sel]
            n = feats.shape[0]
        feats_t = torch.from_numpy(feats).float()
        coords_t = (
            torch.from_numpy(coords).float()
            if (coords is not None and self.return_coords)
            else None
        )
        return BagSample(
            slide_id=entry["slide_id"],
            label=entry["label"],
            features=feats_t,
            coords=coords_t,
            n_patches=n,
        )


def collate_single_bag(batch):
    """Collate function for batch_size=1: just unwrap the singleton list."""
    if len(batch) != 1:
        raise ValueError(
            f"collate_single_bag expects batch_size=1, got {len(batch)}. "
            "Multi-slide batching requires padding+masking; not implemented yet."
        )
    s = batch[0]
    feats = s.features.unsqueeze(0)                    # (1, N, D_in)
    coords = s.coords.unsqueeze(0) if s.coords is not None else None
    label = torch.tensor([s.label], dtype=torch.long)  # (1,)
    return {
        "features": feats,
        "coords": coords,
        "label": label,
        "slide_id": s.slide_id,
        "n_patches": s.n_patches,
    }
