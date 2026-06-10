import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from openslide import OpenSlide


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate academic-style preprocessing QC panels and summary CSV."
    )
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--integrity_csv", type=Path, required=True)
    parser.add_argument("--process_csv", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True, help="Preprocess output directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="QC output directory.")
    parser.add_argument("--clam_dir", type=Path, required=True, help="Path to CLAM repo root.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file used in preprocessing for seg/filter parameters.",
    )
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--thumb_size", type=int, default=1024)
    return parser.parse_args()


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_open_img(path: Path):
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def resolve_slide_path(p: str):
    return Path(p)


def read_patch_count(h5_path: Path):
    if not h5_path.exists():
        return 0
    with h5py.File(h5_path, "r") as f:
        if "coords" in f:
            return int(f["coords"].shape[0])
    return 0


def default_config():
    return {
        "seg_params": {
            "seg_level": -1,
            "sthresh": 8,
            "mthresh": 7,
            "close": 4,
            "use_otsu": False,
            "keep_ids": "none",
            "exclude_ids": "none",
        },
        "filter_params": {"a_t": 100, "a_h": 16, "max_n_holes": 8},
    }


def load_config(config_path: Path):
    cfg = default_config()
    if config_path is None:
        return cfg

    with config_path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    for section in ["seg_params", "filter_params"]:
        if section in user_cfg and isinstance(user_cfg[section], dict):
            cfg[section].update(user_cfg[section])
    return cfg


def parse_ids(raw_value):
    raw = str(raw_value)
    if raw != "none" and len(raw) > 0:
        return np.array(raw.split(",")).astype(int)
    return []


def coverage_ratio(n_patches: int, patch_size: int, level0_dims):
    if not level0_dims:
        return np.nan
    w, h = level0_dims
    total = float(w) * float(h)
    if total <= 0:
        return np.nan
    approx = (n_patches * (patch_size**2)) / total
    return min(1.0, float(approx))


def exact_tissue_ratio_from_segmentation(slide_path: Path, seg_params, filter_params, whole_slide_image_cls):
    wsi = whole_slide_image_cls(str(slide_path))
    current_seg = seg_params.copy()
    current_filter = filter_params.copy()

    if current_seg["seg_level"] < 0:
        if len(wsi.level_dim) == 1:
            current_seg["seg_level"] = 0
        else:
            current_seg["seg_level"] = wsi.getOpenSlide().get_best_level_for_downsample(64)

    current_seg["keep_ids"] = parse_ids(current_seg["keep_ids"])
    current_seg["exclude_ids"] = parse_ids(current_seg["exclude_ids"])
    wsi.segmentTissue(**current_seg, filter_params=current_filter)

    tissue_area = float(sum(cv2.contourArea(c) for c in (wsi.contours_tissue or [])))
    hole_area = float(
        sum(cv2.contourArea(h) for holes in (wsi.holes_tissue or []) for h in holes)
    )
    level0_w, level0_h = wsi.level_dim[0]
    denom = float(level0_w) * float(level0_h)
    if denom <= 0:
        return np.nan
    ratio = max(0.0, tissue_area - hole_area) / denom
    return min(1.0, ratio)


def to_display(img: Image.Image, size: int):
    out = img.copy()
    out.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    x = (size - out.width) // 2
    y = (size - out.height) // 2
    canvas.paste(out, (x, y))
    return canvas


def blend_raw_mask(raw_thumb: Image.Image, mask_img: Image.Image):
    mask_resized = mask_img.resize(raw_thumb.size)
    return Image.blend(raw_thumb, mask_resized, alpha=0.45)


def draw_title(img: Image.Image, text: str):
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 18), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return canvas


def panel_2x2(raw_thumb, mask_img, stitch_img, overlay_img, panel_size=768):
    tl = draw_title(to_display(raw_thumb, panel_size), "Raw Thumbnail")
    tr = draw_title(to_display(mask_img, panel_size), "Segmentation Mask View")
    bl = draw_title(to_display(stitch_img, panel_size), "Patch Stitch Coverage")
    br = draw_title(to_display(overlay_img, panel_size), "Raw + Mask Overlay")
    panel = Image.new("RGB", (panel_size * 2, panel_size * 2), (240, 240, 240))
    panel.paste(tl, (0, 0))
    panel.paste(tr, (panel_size, 0))
    panel.paste(bl, (0, panel_size))
    panel.paste(br, (panel_size, panel_size))
    return panel


def main():
    args = parse_args()
    sys.path.insert(0, str(args.clam_dir))
    from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    manifest = read_manifest(args.manifest_csv)
    integrity_df = pd.read_csv(args.integrity_csv)
    process_df = pd.read_csv(args.process_csv)

    rows = []
    for row in manifest:
        slide_id = str(row["slide_id"])
        slide_path = resolve_slide_path(row["slide_path"])

        proc_row = process_df[process_df["slide_id"].astype(str) == slide_id]
        int_row = integrity_df[integrity_df["slide_id"].astype(str) == slide_id]

        status = proc_row["status"].iloc[0] if len(proc_row) else "missing_in_process"
        seg_time = float(proc_row["seg_time_sec"].iloc[0]) if len(proc_row) else np.nan
        patch_time = float(proc_row["patch_time_sec"].iloc[0]) if len(proc_row) else np.nan
        integrity_status = int_row["status"].iloc[0] if len(int_row) else "missing_in_integrity"
        missing_files = int_row["missing_files"].iloc[0] if len(int_row) else ""

        patch_h5 = args.results_dir / "patches" / f"{slide_id}.h5"
        mask_path = args.results_dir / "masks" / f"{slide_id}.jpg"
        stitch_path = args.results_dir / "stitches" / f"{slide_id}.jpg"

        n_patches = read_patch_count(patch_h5)

        level0_w = np.nan
        level0_h = np.nan
        raw_thumb = None
        try:
            wsi = OpenSlide(str(slide_path))
            level0_w, level0_h = wsi.level_dimensions[0]
            raw_thumb = wsi.get_thumbnail((args.thumb_size, args.thumb_size)).convert("RGB")
            wsi.close()
        except Exception:  # noqa: BLE001
            pass

        exact_cov = np.nan
        try:
            exact_cov = exact_tissue_ratio_from_segmentation(
                slide_path, cfg["seg_params"], cfg["filter_params"], WholeSlideImage
            )
        except Exception:  # noqa: BLE001
            pass

        approx_cov = coverage_ratio(
            n_patches, args.patch_size, None if np.isnan(level0_w) else (level0_w, level0_h)
        )

        panel_path = ""
        mask_img = safe_open_img(mask_path)
        stitch_img = safe_open_img(stitch_path)
        if raw_thumb is not None and mask_img is not None and stitch_img is not None:
            overlay = blend_raw_mask(raw_thumb, mask_img)
            panel = panel_2x2(raw_thumb, mask_img, stitch_img, overlay, panel_size=700)
            panel_path_obj = panel_dir / f"{slide_id}_qc_panel.jpg"
            panel.save(panel_path_obj, quality=95)
            panel_path = str(panel_path_obj)

        rows.append(
            {
                "slide_id": slide_id,
                "slide_path": str(slide_path),
                "process_status": status,
                "integrity_status": integrity_status,
                "missing_files": missing_files,
                "seg_time_sec": seg_time,
                "patch_time_sec": patch_time,
                "patch_count": n_patches,
                "level0_width": level0_w,
                "level0_height": level0_h,
                "exact_tissue_ratio": exact_cov,
                "approx_coverage_ratio": approx_cov,
                "mask_exists": int(mask_path.exists()),
                "stitch_exists": int(stitch_path.exists()),
                "patch_h5_exists": int(patch_h5.exists()),
                "qc_panel_path": panel_path,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("slide_id", key=lambda s: s.astype(int))
    out_csv = args.output_dir / "preprocessing_qc_summary.csv"
    out_df.to_csv(out_csv, index=False)

    report_md = args.output_dir / "preprocessing_qc_report.md"
    n_total = len(out_df)
    n_proc = int((out_df["process_status"] == "processed").sum())
    n_ok = int((out_df["integrity_status"] == "ok").sum())
    n_panels = int((out_df["qc_panel_path"].astype(str) != "").sum())
    mean_patches = float(out_df["patch_count"].mean()) if n_total else 0.0
    mean_seg = float(out_df["seg_time_sec"].mean()) if n_total else 0.0
    mean_patch_time = float(out_df["patch_time_sec"].mean()) if n_total else 0.0
    mean_exact_cov = float(out_df["exact_tissue_ratio"].mean()) if n_total else 0.0
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Preprocessing QC Report\n\n")
        f.write("## Summary\n")
        f.write(f"- Slides in manifest: {n_total}\n")
        f.write(f"- Processed slides: {n_proc}\n")
        f.write(f"- Integrity OK slides: {n_ok}\n")
        f.write(f"- QC panels generated: {n_panels}\n")
        f.write(f"- Mean patch count per slide: {mean_patches:.2f}\n")
        f.write(f"- Mean segmentation time (s): {mean_seg:.3f}\n")
        f.write(f"- Mean patch extraction time (s): {mean_patch_time:.3f}\n\n")
        f.write(f"- Mean exact tissue ratio: {mean_exact_cov:.4f}\n\n")
        f.write("## Files\n")
        f.write(f"- QC table: `{out_csv}`\n")
        f.write(f"- QC panels: `{panel_dir}`\n")

    print(f"Wrote QC summary: {out_csv}")
    print(f"Wrote QC report: {report_md}")
    print(f"Panels: {n_panels}/{n_total}")


if __name__ == "__main__":
    main()
