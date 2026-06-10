"""Rescue tissue segmentation for slides whose Otsu run produced an empty
GeoDataFrame because no individual contour reached TRIDENT's hard-coded
min_contour_area=1000 µm² threshold.

Strategy:
  1. Monkey-patch trident.IO.mask_to_gdf so its default min_contour_area is
     much smaller (50 µm²). The Otsu mask itself is unchanged; only the
     post-processing area filter is loosened.
  2. Re-run segment_tissue() on the listed cases with the same Otsu segmenter
     and seg_mag (1.25x) used in the full pipeline, overwriting the empty
     geojsons + thumbnails + contour overlays in the existing job_dir.

This script does NOT run coords/features extraction; do that with TRIDENT's
batch script afterwards on the same slides only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Make TRIDENT importable from its sibling working tree.
TRIDENT_DIR = str(_REPO_ROOT / "TRIDENT")
if TRIDENT_DIR not in sys.path:
    sys.path.insert(0, TRIDENT_DIR)


def patch_min_contour_area(new_default: float) -> None:
    """Lower the default min_contour_area used by mask_to_gdf.

    Note: WSI._segment_tissue passes min_contour_area=1000 explicitly, so we
    also wrap mask_to_gdf to clamp any caller-provided value to new_default.
    """
    import trident.IO as IO

    original = IO.mask_to_gdf

    def patched_mask_to_gdf(*args, **kwargs):
        kwargs["min_contour_area"] = new_default
        return original(*args, **kwargs)

    IO.mask_to_gdf = patched_mask_to_gdf

    import trident.wsi_objects.WSI as WSI_mod
    if hasattr(WSI_mod, "mask_to_gdf"):
        WSI_mod.mask_to_gdf = patched_mask_to_gdf


def archive_old(job_dir: str, case_ids: list[str]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(job_dir, f"_rescue_backup_{ts}")
    os.makedirs(backup, exist_ok=True)
    for sub in ("contours_geojson", "contours", "thumbnails"):
        src_dir = os.path.join(job_dir, sub)
        dst_dir = os.path.join(backup, sub)
        os.makedirs(dst_dir, exist_ok=True)
        for cid in case_ids:
            for ext in (".geojson", ".jpg"):
                src = os.path.join(src_dir, cid + ext)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dst_dir, cid + ext))
    return backup


def run_rescue(
    case_ids: list[str],
    wsi_dir: str,
    job_dir: str,
    min_contour_area: float,
    seg_mag: float,
    batch_size: int,
    device: str,
) -> None:
    patch_min_contour_area(min_contour_area)

    from trident.segmentation_models.load import segmentation_model_factory
    from trident.wsi_objects.WSIFactory import load_wsi

    backup = archive_old(job_dir, case_ids)
    print(f"Backed up old artefacts -> {backup}")

    seg_model = segmentation_model_factory("otsu")

    for cid in case_ids:
        wsi_path = os.path.join(wsi_dir, f"{cid}.mrxs")
        if not os.path.exists(wsi_path):
            print(f"[SKIP] {wsi_path} missing")
            continue
        print(f"\n=== Rescuing case {cid} ===")
        slide = load_wsi(slide_path=wsi_path, lazy_init=False)
        gdf_path = slide.segment_tissue(
            segmentation_model=seg_model,
            target_mag=seg_mag,
            holes_are_tissue=True,
            batch_size=batch_size,
            device=device,
            verbose=False,
            job_dir=job_dir,
        )
        # mask_to_gdf may still return empty if Otsu found literally nothing;
        # report so caller can react.
        try:
            import geopandas as gpd
            gdf = gpd.read_file(gdf_path)
            print(f"  -> wrote {gdf_path}  contours={len(gdf)}  total_area={float(gdf.area.sum()):.0f}")
        except Exception as e:  # pragma: no cover - just diagnostic
            print(f"  -> wrote {gdf_path}  (could not read for stats: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["146", "159", "356"])
    ap.add_argument("--wsi_dir", default=str(_REPO_ROOT))
    ap.add_argument(
        "--job_dir",
        default=str(_REPO_ROOT / "herohe/gp2/results_trident_mac_full"),
    )
    ap.add_argument("--min_contour_area", type=float, default=50.0,
                    help="Loosened minimum contour area in WSI µm² (default 50, vs TRIDENT's 1000).")
    ap.add_argument("--seg_mag", type=float, default=1.25,
                    help="Magnification at which Otsu runs (matches the full pipeline).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu",
                    help="Otsu has no NN; CPU is fine and avoids MPS overhead.")
    args = ap.parse_args()

    run_rescue(
        case_ids=args.cases,
        wsi_dir=args.wsi_dir,
        job_dir=args.job_dir,
        min_contour_area=args.min_contour_area,
        seg_mag=args.seg_mag,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
