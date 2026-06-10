"""Render top-attended RGB patches per prototype for qualitative review.

Loads a PhenoHER2 / PhenoHER2-Binary checkpoint, subsamples a slide bag the same way as
training (``max_patches`` + ``seed``), scores patches by ``soft_assign * patch_attn`` per
prototype, reads tiles from the WSI with OpenSlide using TRIDENT ``coords`` attributes
(``patch_size_level0`` at level 0), and writes one PNG montage (K rows × top_m columns).

Requires:
  - openslide-python, pillow, h5py, torch

HEROHE layout in this workspace: MIRAX slides sit at the **gradCode repo root** as
``<slide_id>.mrxs`` (e.g. ``.../gradCode/47.mrxs``). Pass that file with ``--wsi_path``, or
use a manifest whose ``slide_path`` is absolute or relative to ``--slides_root``.

Example (``47.mrxs`` at repo root, PhenoHER2-Binary fold 0):

    python herohe/gp2/scripts/prototype_patch_montage.py \\
        --checkpoint herohe/gp2/runs/phenobin_5fold_parallel/fold_0/best.pt \\
        --features_dir herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2 \\
        --wsi_path /ABS/path/to/gradCode/47.mrxs \\
        --slide_id 47 \\
        --top_m 8 \\
        --out_dir herohe/gp2/runs/phenobin_5fold_parallel/montages_fold0 \\
        --device cpu

``--seed`` defaults to ``args.seed`` inside the checkpoint when omitted. Match
``--max_patches`` to training (default 4096). OpenSlide reads tiles on CPU; use
``--device mps`` only for the torch model if desired.

If the manifest uses relative paths like ``47.mrxs``, set ``--slides_root`` to the directory
that contains those files (e.g. the gradCode root).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import PhenoHER2, PhenoHER2Binary, PhenoHER2BinaryConfig, PhenoHER2Config


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_binary_ckpt(blob: dict, sd: dict) -> bool:
    if blob.get("model_type") == "PhenoHER2Binary":
        return True
    return any(k.startswith("head_bin_phen") for k in sd.keys())


def _cfg_from_blob(blob: dict, binary: bool):
    raw = blob["config"]
    if binary:
        names = {f.name for f in fields(PhenoHER2BinaryConfig)}
        return PhenoHER2BinaryConfig(**{k: raw[k] for k in names if k in raw})
    names = {f.name for f in fields(PhenoHER2Config)}
    return PhenoHER2Config(**{k: raw[k] for k in names if k in raw})


def load_manifest(path: Path, slides_root: Path | None) -> dict[str, Path]:
    import csv

    out: dict[str, Path] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row["slide_id"]).strip()
            sp = Path(row["slide_path"].strip())
            if not sp.is_absolute():
                if slides_root is None:
                    raise ValueError("Manifest has relative slide_path; pass --slides_root")
                sp = slides_root / sp
            out[sid] = sp
    return out


def load_bag_from_h5(
    features_dir: str,
    slide_id: str,
    max_patches: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None, dict, np.ndarray]:
    """Returns features (1,N,D), coords (1,N,2) or None, coord attrs, index mapping orig->subset (subset indices)."""
    fp = os.path.join(features_dir, f"{slide_id}.h5")
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    with h5py.File(fp, "r") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)
        coords = np.asarray(f["coords"][:], dtype=np.int64) if "coords" in f else None
        attrs = dict(f["coords"].attrs) if coords is not None else {}
    n = feats.shape[0]
    idx = np.arange(n, dtype=np.int64)
    if max_patches is not None and n > max_patches:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_patches, replace=False))
        feats = feats[idx]
        if coords is not None:
            coords = coords[idx]
    x = torch.from_numpy(feats).unsqueeze(0)
    c = torch.from_numpy(coords.astype(np.float32)).unsqueeze(0) if coords is not None else None
    return x, c, attrs, idx


def read_patch_pil(slide_path: Path, x: int, y: int, attrs: dict, thumb: int) -> Image.Image:
    from openslide import OpenSlide

    ps0 = int(float(attrs.get("patch_size_level0", attrs.get("patch_size", 256))))
    with OpenSlide(str(slide_path)) as slide:
        im = slide.read_region((int(x), int(y)), 0, (ps0, ps0)).convert("RGB")
    if im.size != (thumb, thumb):
        im = im.resize((thumb, thumb), Image.Resampling.BILINEAR)
    return im


def resolve_slide_path(
    slide_id: str,
    manifest: dict[str, Path] | None,
    wsi_path: Path | None,
    slides_root: Path | None,
) -> Path:
    if wsi_path is not None:
        return wsi_path
    if manifest is None or slide_id not in manifest:
        raise ValueError(f"No slide path for {slide_id}; pass --wsi_path or fix manifest")
    p = manifest[slide_id]
    return p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--slide_id", required=True)
    ap.add_argument("--manifest_csv", type=Path, default=None, help="CSV with slide_id, slide_path")
    ap.add_argument("--slides_root", type=Path, default=None, help="Prefix if manifest paths are relative")
    ap.add_argument("--wsi_path", type=Path, default=None, help="Override WSI path for this slide")
    ap.add_argument("--top_m", type=int, default=8, help="Top patches per prototype (columns)")
    ap.add_argument("--thumb", type=int, default=256, help="Each tile size after resize (px)")
    ap.add_argument("--max_patches", type=int, default=4096, help="Match train_phenobin_mil")
    ap.add_argument("--seed", type=int, default=None, help="Subsampling seed (default: from checkpoint args['seed'] if present, else 0)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    try:
        import openslide  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install openslide-python (and system OpenSlide) to read WSIs.") from e

    blob = torch.load(args.checkpoint, map_location="cpu")
    train_seed = 0
    if args.seed is not None:
        train_seed = int(args.seed)
    else:
        ta = blob.get("args") or {}
        if isinstance(ta, dict) and "seed" in ta:
            train_seed = int(ta["seed"])

    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest_csv, args.slides_root) if args.manifest_csv else None
    wsi = resolve_slide_path(args.slide_id, manifest, args.wsi_path, args.slides_root)
    if not wsi.is_file():
        raise FileNotFoundError(f"WSI not found: {wsi}")

    sd = blob["model_state"]
    binary = _is_binary_ckpt(blob, sd)
    cfg = _cfg_from_blob(blob, binary)
    if binary:
        model = PhenoHER2Binary(cfg)
    else:
        model = PhenoHER2(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)

    x, coords, attrs, subset_idx = load_bag_from_h5(
        args.features_dir, args.slide_id, args.max_patches, train_seed
    )
    x = x.to(device)
    coords_t = coords.to(device) if coords is not None else None

    out = model(x, coords=coords_t)
    sa = out["soft_assign"][0].float().cpu().numpy()
    pa = out["patch_attn"][0].float().cpu().numpy()
    score = sa * pa
    N, K = score.shape
    top_m = min(args.top_m, N)

    coords_np = coords[0].numpy().astype(np.int64) if coords is not None else None
    if coords_np is None:
        raise RuntimeError("This slide's .h5 has no coords dataset; cannot render RGB patches.")

    meta: dict = {
        "slide_id": args.slide_id,
        "wsi_path": str(wsi),
        "checkpoint": str(args.checkpoint),
        "binary": binary,
        "K": int(K),
        "top_m": int(top_m),
        "N_bag": int(N),
        "max_patches": args.max_patches,
        "seed_used": train_seed,
        "coord_attrs": {k: (float(v) if hasattr(v, "item") else str(v)) for k, v in attrs.items()},
        "per_proto": [],
    }

    thumb = args.thumb
    rows: list[Image.Image] = []
    for k in range(K):
        order = np.argsort(-score[:, k])[:top_m]
        tiles: list[Image.Image] = []
        entry = {"prototype": k, "patches": []}
        for rank, j in enumerate(order):
            j = int(j)
            cx, cy = int(coords_np[j, 0]), int(coords_np[j, 1])
            try:
                tile = read_patch_pil(wsi, cx, cy, attrs, thumb)
            except Exception as exc:
                tile = Image.new("RGB", (thumb, thumb), color=(40, 40, 40))
                d = ImageDraw.Draw(tile)
                d.text((8, thumb // 2), f"read err\n{exc}", fill=(255, 80, 80))
            tiles.append(tile)
            orig_idx = int(subset_idx[j]) if subset_idx is not None else j
            entry["patches"].append(
                {
                    "rank": rank,
                    "bag_index": j,
                    "h5_patch_index": orig_idx,
                    "x": cx,
                    "y": cy,
                    "score": float(score[j, k]),
                }
            )
        row = Image.new("RGB", (thumb * top_m + (top_m - 1) * 2, thumb + 24), (20, 20, 20))
        dr = ImageDraw.Draw(row)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        dr.text((4, 2), f"k={k}", fill=(220, 220, 220), font=font)
        x0 = 0
        y0 = 22
        for t in tiles:
            row.paste(t, (x0, y0))
            x0 += thumb + 2
        rows.append(row)
        meta["per_proto"].append(entry)

    gap = 4
    H = sum(r.size[1] for r in rows) + gap * (len(rows) - 1)
    W = max(r.size[0] for r in rows)
    canvas = Image.new("RGB", (W, H), (10, 10, 10))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.size[1] + gap

    out_png = args.out_dir / f"montage_slide{args.slide_id}_foldckpt.png"
    canvas.save(out_png, quality=95)
    out_json = args.out_dir / f"montage_slide{args.slide_id}_meta.json"
    with out_json.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
