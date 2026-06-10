"""Build architecture-section preprocessing figure from TRIDENT Otsu outputs.

Uses existing TRIDENT artefacts only (no re-segmentation):
  - thumbnails/{slide_id}.jpg          — downsampled H&E thumbnail
  - contours/{slide_id}.jpg            — Otsu tissue contours overlaid (segmenter=otsu)
  - 20x_256px_0px_overlap/visualization/{slide_id}.jpg — patch grid on thumbnail
  - features_virchow2/{slide_id}.h5      — coords for RGB patch crops via OpenSlide

Example:

    python herohe/gp2/scripts/render_architecture_pipeline_figure.py \\
        --slide_id 304 \\
        --trident_job_dir herohe/gp2/results_trident_test \\
        --split test \\
        --out_dir herohe/report/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.scripts.herohe_wsi_paths import resolve_wsi as herohe_resolve_wsi
from herohe.gp2.scripts.interp_viz_utils import (
    pick_valid_patch_indices,
    read_wsi_patch,
    tissue_bbox_from_coords,
)


def trident_paths(job_dir: Path, slide_id: str) -> dict[str, Path]:
    job = job_dir.resolve()
    job_preset = job / "20x_256px_0px_overlap"
    paths = {
        "thumbnail": job / "thumbnails" / f"{slide_id}.jpg",
        "contours": job / "contours" / f"{slide_id}.jpg",
        "patch_viz": job_preset / "visualization" / f"{slide_id}.jpg",
        "features_h5": job_preset / "features_virchow2" / f"{slide_id}.h5",
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing TRIDENT artefacts for slide {slide_id}: {missing}. "
            "Run TRIDENT seg (otsu) → coords → feat first."
        )
    return paths


def load_h5_meta(h5_path: Path) -> tuple[np.ndarray, dict, int, int]:
    with h5py.File(h5_path, "r") as f:
        n_patches = int(f["features"].shape[0])
        dim = int(f["features"].shape[1])
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        attrs = dict(f["coords"].attrs)
    return coords, attrs, n_patches, dim


def build_patch_mosaic(
    wsi_path: Path,
    coords: np.ndarray,
    attrs: dict,
    grid_size: int,
    seed: int,
    tile_px: int = 180,
) -> np.ndarray:
    n_show = grid_size * grid_size
    picked = pick_valid_patch_indices(coords, wsi_path, attrs, n_show, seed, out_px=tile_px)
    fallback = np.full((tile_px, tile_px, 3), 240, dtype=np.uint8)
    patch_tiles = []
    for j in picked:
        cx, cy = int(coords[j, 0]), int(coords[j, 1])
        tile = read_wsi_patch(wsi_path, cx, cy, attrs, out_px=tile_px)
        patch_tiles.append(tile if tile is not None else fallback)
    while len(patch_tiles) < n_show:
        patch_tiles.append(fallback)
    mosaic = np.zeros((tile_px * grid_size, tile_px * grid_size, 3), dtype=np.uint8)
    for i, tile in enumerate(patch_tiles[:n_show]):
        r, c = divmod(i, grid_size)
        mosaic[r * tile_px : (r + 1) * tile_px, c * tile_px : (c + 1) * tile_px] = tile
    return mosaic


def render_panel(
    slide_id: str,
    paths: dict[str, Path],
    wsi_path: Path,
    out_dir: Path,
    grid_size: int,
    seed: int,
    dpi: int,
) -> dict:
    coords, attrs, n_patches, feat_dim = load_h5_meta(paths["features_h5"])

    thumb = np.array(Image.open(paths["thumbnail"]).convert("RGB"))
    contours = np.array(Image.open(paths["contours"]).convert("RGB"))
    patch_viz = np.array(Image.open(paths["patch_viz"]).convert("RGB"))

    bbox = tissue_bbox_from_coords(coords, attrs, thumb.shape[:2], margin_frac=0.03)
    x0, y0, x1, y1 = bbox
    thumb = thumb[y0:y1, x0:x1]
    contours = contours[y0:y1, x0:x1]
    patch_viz = patch_viz[y0:y1, x0:x1]

    patch_mosaic = build_patch_mosaic(wsi_path, coords, attrs, grid_size, seed, tile_px=180)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"arch_pipeline_slide{slide_id}"

    fig = plt.figure(figsize=(14, 10.5), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, wspace=0.06, hspace=0.14)

    panels = [
        (gs[0, 0], thumb, "(a) H&E thumbnail (tissue crop)"),
        (gs[0, 1], contours, "(b) Otsu tissue segmentation"),
        (gs[1, 0], patch_viz, "(c) Patch coordinates ($256\\times256$, 20$\\times$, 0 overlap)"),
        (gs[1, 1], patch_mosaic, f"(d) Example RGB patches (Virchow2 ${feat_dim}$-D, $N={n_patches}$)"),
    ]
    for spec, img, title in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(
        f"HEROHE slide {slide_id} — preprocessing pipeline (Otsu segmenter, tissue crop)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    combined_path = out_dir / f"{stem}.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig2, axes = plt.subplots(1, 4, figsize=(16.5, 4.4), facecolor="white")
    strip_imgs = [thumb, contours, patch_viz, patch_mosaic]
    strip_titles = [
        "(a) H&E thumbnail",
        "(b) Otsu segmentation",
        "(c) Patch grid",
        "(d) RGB patches",
    ]
    for ax, img, t in zip(axes, strip_imgs, strip_titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=10, fontweight="bold")
        ax.axis("off")
    fig2.suptitle(f"Slide {slide_id} · Otsu · tissue crop · 20× / 256 px", fontsize=12, y=1.02)
    strip_path = out_dir / f"{stem}_strip.png"
    fig2.savefig(strip_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    singles: dict[str, str] = {}
    single_specs = [
        ("a_thumbnail", thumb),
        ("b_otsu_contours", contours),
        ("c_patch_grid", patch_viz),
        ("d_patch_mosaic", patch_mosaic),
    ]
    for name, img in single_specs:
        p = out_dir / f"{stem}_{name}.png"
        Image.fromarray(img).save(p, quality=95)
        singles[name] = str(p)

    meta = {
        "slide_id": slide_id,
        "segmenter": "otsu",
        "trident_job_dir": str(paths["thumbnail"].parents[1]),
        "wsi_path": str(wsi_path),
        "n_patches": n_patches,
        "feature_dim": feat_dim,
        "patch_size_px": int(attrs.get("patch_size", 256)),
        "target_magnification": float(attrs.get("target_magnification", 20)),
        "overlap_px": int(attrs.get("overlap", 0)),
        "outputs": {
            "combined": str(combined_path),
            "strip": str(strip_path),
            **singles,
        },
    }
    meta_path = out_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slide_id", required=True)
    ap.add_argument(
        "--trident_job_dir",
        type=Path,
        default=_REPO / "herohe/gp2/results_trident_mac_full",
    )
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--wsi_path", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=_REPO / "herohe/report/figures")
    ap.add_argument("--grid_size", type=int, default=3, help="Patch mosaic is grid_size²")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    try:
        import openslide  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install openslide-python to crop RGB patches from WSI.") from e

    paths = trident_paths(args.trident_job_dir, args.slide_id)
    wsi = args.wsi_path.resolve() if args.wsi_path else herohe_resolve_wsi(args.slide_id, split=args.split)
    meta = render_panel(
        args.slide_id,
        paths,
        wsi,
        args.out_dir,
        grid_size=args.grid_size,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
