"""Patch feature sampling from TRIDENT-style slide .h5 bags."""

from __future__ import annotations

import os
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def slide_ids_from_csv(csv_path: str | os.PathLike) -> list[str]:
    """Read slide IDs from a CSV with ``slide_id`` or TRIDENT ``wsi`` column."""
    df = pd.read_csv(csv_path)
    if "slide_id" in df.columns:
        return df["slide_id"].astype(str).tolist()
    if "wsi" in df.columns:
        return [Path(s).stem for s in df["wsi"].astype(str)]
    raise ValueError(f"{csv_path} must have a 'slide_id' or 'wsi' column")


def train_slide_ids_from_folds(folds_csv: str | os.PathLike, val_fold: int) -> list[str]:
    """Return slide IDs in all folds except ``val_fold`` (training pool for one split)."""
    df = pd.read_csv(folds_csv)
    if "slide_id" not in df.columns or "fold" not in df.columns:
        raise ValueError(f"{folds_csv} must have slide_id and fold columns")
    df["slide_id"] = df["slide_id"].astype(str)
    df["fold"] = df["fold"].astype(int)
    return sorted(df.loc[df["fold"] != int(val_fold), "slide_id"].tolist())


def collect_slide_patch_sample(
    features_dir: str | os.PathLike,
    slide_id: str,
    patches_per_slide: int,
    seed: int,
) -> np.ndarray | None:
    """Return (n, D) float32 patch features for one slide, or None if missing/empty."""
    fp = os.path.join(str(features_dir), f"{slide_id}.h5")
    if not os.path.isfile(fp):
        return None
    rng = np.random.default_rng(_slide_seed(seed, slide_id))
    with h5py.File(fp, "r") as f:
        feats = f["features"]
        n = int(feats.shape[0])
        if n == 0:
            return None
        if n <= patches_per_slide:
            arr = np.asarray(feats[:], dtype=np.float32)
        else:
            idx = np.sort(rng.choice(n, size=patches_per_slide, replace=False))
            arr = np.asarray(feats[idx], dtype=np.float32)
    return arr


def collect_patches_per_slide(
    features_dir: str | os.PathLike,
    slide_ids: list[str],
    patches_per_slide: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load subsampled patch matrices keyed by slide_id. Returns (data, missing_ids)."""
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for sid in slide_ids:
        arr = collect_slide_patch_sample(features_dir, sid, patches_per_slide, seed)
        if arr is None:
            missing.append(sid)
            continue
        out[sid] = arr
    return out, missing


def _slide_seed(seed: int, slide_id: str) -> int:
    """Deterministic per-slide RNG seed (matches HerohePatchBagDataset deterministic subsample)."""
    key = f"{seed}:{slide_id}".encode()
    return zlib.adler32(key) & 0xFFFFFFFF
