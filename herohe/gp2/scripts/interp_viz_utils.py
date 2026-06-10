"""Shared helpers for hard-partition interpretability figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def is_valid_patch(arr: np.ndarray, min_mean: float = 18.0, max_mean: float = 250.0, min_std: float = 10.0) -> bool:
    """Reject empty/black/saturated/uniform tiles from failed OpenSlide reads."""
    if arr is None or arr.size == 0:
        return False
    flat = np.asarray(arr, dtype=np.float32)
    mean = float(flat.mean())
    std = float(flat.std())
    return min_mean <= mean <= max_mean and std >= min_std


def pick_prototype_exemplar(
    coords: np.ndarray,
    feats: np.ndarray,
    centers: np.ndarray,
    proto_k: int,
    wsi_path: Path,
    attrs: dict,
    out_px: int = 150,
    top_n: int = 40,
) -> np.ndarray | None:
    """Best RGB exemplar for prototype k: highest cosine among assigned patches."""
    x_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    c_n = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    sim = x_n @ c_n.T
    assign = sim.argmax(axis=1)
    sel = np.where(assign == proto_k)[0]
    if len(sel) == 0:
        return None
    order = sel[np.argsort(-sim[sel, proto_k])[:top_n]]
    for pi in order:
        cx, cy = int(coords[pi, 0]), int(coords[pi, 1])
        tile = read_wsi_patch(wsi_path, cx, cy, attrs, out_px=out_px)
        if tile is not None:
            return tile
    return None


def read_wsi_patch(
    wsi_path: Path,
    x: int,
    y: int,
    attrs: dict,
    out_px: int = 128,
    min_mean: float = 18.0,
) -> np.ndarray | None:
    """Read one RGB patch; return None if out-of-bounds or invalid (black) tile."""
    from openslide import OpenSlide
    from PIL import Image

    ps0 = int(float(attrs.get("patch_size_level0", attrs.get("patch_size", 256))))
    with OpenSlide(str(wsi_path)) as slide:
        w, h = slide.dimensions
        if x < 0 or y < 0 or x + ps0 > w or y + ps0 > h:
            return None
        im = slide.read_region((int(x), int(y)), 0, (ps0, ps0)).convert("RGB")
    if im.size != (out_px, out_px):
        im = im.resize((out_px, out_px), Image.Resampling.LANCZOS)
    arr = np.asarray(im)
    return arr if is_valid_patch(arr, min_mean=min_mean) else None


def pick_valid_patch_indices(
    coords: np.ndarray,
    wsi_path: Path,
    attrs: dict,
    n_want: int,
    seed: int,
    out_px: int = 160,
) -> list[int]:
    """Return up to n_want patch indices with valid RGB crops (skip black tiles)."""
    n = coords.shape[0]
    rng = np.random.default_rng(seed)
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    spaced = np.linspace(0, len(order) - 1, min(n, len(order)), dtype=int)
    candidates = list(order[spaced])
    rng.shuffle(candidates)
    picked: list[int] = []
    for j in candidates:
        cx, cy = int(coords[j, 0]), int(coords[j, 1])
        tile = read_wsi_patch(wsi_path, cx, cy, attrs, out_px=out_px)
        if tile is not None:
            picked.append(int(j))
        if len(picked) >= n_want:
            break
    if len(picked) < n_want:
        for j in order:
            if j in picked:
                continue
            cx, cy = int(coords[j, 0]), int(coords[j, 1])
            tile = read_wsi_patch(wsi_path, cx, cy, attrs, out_px=out_px)
            if tile is not None:
                picked.append(int(j))
            if len(picked) >= n_want:
                break
    return picked


def hard_assign(soft_assign: np.ndarray) -> np.ndarray:
    """Per-patch prototype index from soft assignment (N, K)."""
    return soft_assign.argmax(axis=1)


def rank_patches_per_head(
    hard: np.ndarray,
    patch_attn: np.ndarray,
    k: int,
    top_m: int,
) -> np.ndarray:
    """Return up to top_m patch indices assigned to head k, ranked by within-cluster attn."""
    in_k = np.where(hard == k)[0]
    if len(in_k) == 0:
        return np.array([], dtype=np.int64)
    scores = patch_attn[in_k, k]
    order = np.argsort(-scores)[: min(top_m, len(in_k))]
    return in_k[order]


def active_heads(phenotype_active, hard_frac, phen_token_attn, min_hard: float = 0.02, min_w: float = 0.05):
    """Head indices with material hard-assign mass and/or token weight."""
    K = len(hard_frac)
    active = []
    for k in range(K):
        if phenotype_active is not None and not phenotype_active[k]:
            continue
        w = phen_token_attn[k] if phen_token_attn is not None else 0.0
        if hard_frac[k] >= min_hard or w >= min_w:
            active.append(k)
    return active


def tissue_bbox_from_coords(
    coords: np.ndarray,
    attrs: dict,
    thumb_shape: tuple[int, int],
    margin_frac: float = 0.04,
) -> tuple[int, int, int, int]:
    """Return x0, y0, x1, y1 pixel bbox on thumbnail covering patch coords."""
    th, tw = thumb_shape[0], thumb_shape[1]
    w0 = float(attrs["level0_width"])
    h0 = float(attrs["level0_height"])
    ps0 = float(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
    sx, sy = tw / w0, th / h0
    xs = coords[:, 0] * sx
    ys = coords[:, 1] * sy
    xe = (coords[:, 0] + ps0) * sx
    ye = (coords[:, 1] + ps0) * sy
    mx = margin_frac * tw
    my = margin_frac * th
    x0 = int(max(0, np.floor(xs.min() - mx)))
    y0 = int(max(0, np.floor(ys.min() - my)))
    x1 = int(min(tw, np.ceil(xe.max() + mx)))
    y1 = int(min(th, np.ceil(ye.max() + my)))
    return x0, y0, x1, y1


def attention_heatmap_cropped(
    coords: np.ndarray,
    attn: np.ndarray,
    attrs: dict,
    thumb: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    min_thumb_px: int = 6,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Build normalized attention heatmap aligned to (optionally cropped) thumbnail."""
    th, tw = thumb.shape[:2]
    if bbox is None:
        bbox = tissue_bbox_from_coords(coords, attrs, (th, tw))
    x0, y0, x1, y1 = bbox
    crop = thumb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    w0 = float(attrs["level0_width"])
    h0 = float(attrs["level0_height"])
    patch = float(attrs["patch_size_level0"])
    sx_full, sy_full = tw / w0, th / h0

    heat = np.zeros((ch, cw), dtype=np.float64)
    pw = max(min_thumb_px, int(round(patch * sx_full)))
    ph = max(min_thumb_px, int(round(patch * sy_full)))
    for (cx, cy), v in zip(coords, attn):
        px = int(cx * sx_full) - x0
        py = int(cy * sy_full) - y0
        if px + pw < 0 or py + ph < 0 or px >= cw or py >= ch:
            continue
        xa, xb = max(0, px), min(cw, px + pw)
        ya, yb = max(0, py), min(ch, py + ph)
        heat[ya:yb, xa:xb] = np.maximum(heat[ya:yb, xa:xb], float(v))
    if heat.max() > 0:
        heat /= heat.max()
    return crop, heat, bbox


def patch_attention_scatter(
    coords: np.ndarray,
    attn: np.ndarray,
    attrs: dict,
    bbox: tuple[int, int, int, int],
    thumb_shape: tuple[int, int],
    percentile: float = 75.0,
    min_radius: float = 3.5,
    max_radius: float = 16.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, sizes, attn_norm) for high-attention patch centers on cropped thumbnail."""
    th, tw = thumb_shape[0], thumb_shape[1]
    x0, y0, _, _ = bbox
    w0 = float(attrs["level0_width"])
    h0 = float(attrs["level0_height"])
    patch = float(attrs["patch_size_level0"])
    sx, sy = tw / w0, th / h0
    px = coords[:, 0] * sx - x0 + (patch * sx) / 2
    py = coords[:, 1] * sy - y0 + (patch * sy) / 2
    a = attn / (attn.max() + 1e-8)
    thr = np.percentile(a, percentile) if len(a) else 1.0
    keep = a >= thr
    if not keep.any():
        keep = a >= np.median(a)
    sizes = min_radius + (max_radius - min_radius) * a[keep]
    return px[keep], py[keep], sizes, a[keep]
