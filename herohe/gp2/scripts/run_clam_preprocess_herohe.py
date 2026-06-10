import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CLAM segmentation + patch coordinate extraction for HEROHE manifest."
    )
    parser.add_argument("--clam_dir", type=Path, required=True, help="Path to CLAM repo root.")
    parser.add_argument("--manifest_csv", type=Path, required=True, help="Manifest CSV path.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--patch_size", type=int, default=None, help="Optional override for config run.patch_size.")
    parser.add_argument("--step_size", type=int, default=None, help="Optional override for config run.step_size.")
    parser.add_argument("--patch_level", type=int, default=None, help="Optional override for config run.patch_level.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file for seg/filter/vis/patch/run parameters.",
    )
    parser.add_argument("--seg", action="store_true", default=True, help="Run tissue segmentation.")
    parser.add_argument("--patch", action="store_true", default=True, help="Run patch coordinate extraction.")
    parser.add_argument("--stitch", action="store_true", default=False, help="Generate stitched previews.")
    parser.add_argument("--max_slides", type=int, default=None, help="Optional cap for fast debugging.")
    parser.add_argument(
        "--skip_incomplete",
        action="store_true",
        default=True,
        help="Skip missing slide files instead of failing fast.",
    )
    parser.add_argument(
        "--strict_wsi_format",
        action="store_true",
        default=True,
        help="Require OpenSlide.detect_format() to recognize the input path.",
    )
    return parser.parse_args()


def load_rows(manifest_csv: Path):
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def default_config():
    return {
        "run": {"patch_size": 256, "step_size": 256, "patch_level": 0},
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
        "vis_params": {"vis_level": -1, "line_thickness": 250},
        "patch_params": {"use_padding": True, "contour_fn": "four_pt"},
    }


def load_config(config_path: Path):
    cfg = default_config()
    if config_path is None:
        return cfg

    with config_path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    # Structured config format
    has_structured_sections = any(
        section in user_cfg for section in ["run", "seg_params", "filter_params", "vis_params", "patch_params"]
    )
    if has_structured_sections:
        for section in ["run", "seg_params", "filter_params", "vis_params", "patch_params"]:
            if section in user_cfg and isinstance(user_cfg[section], dict):
                cfg[section].update(user_cfg[section])
        return cfg

    # Flat CLAM preset-like config format (mirrors tcga.csv keys)
    seg_keys = ["seg_level", "sthresh", "mthresh", "close", "use_otsu", "keep_ids", "exclude_ids"]
    filter_keys = ["a_t", "a_h", "max_n_holes"]
    vis_keys = ["vis_level", "line_thickness"]
    patch_keys = ["use_padding", "contour_fn"]
    unsupported_keys = ["white_thresh", "black_thresh"]

    for key in seg_keys:
        if key in user_cfg:
            cfg["seg_params"][key] = user_cfg[key]
    for key in filter_keys:
        if key in user_cfg:
            cfg["filter_params"][key] = user_cfg[key]
    for key in vis_keys:
        if key in user_cfg:
            cfg["vis_params"][key] = user_cfg[key]
    for key in patch_keys:
        if key in user_cfg:
            cfg["patch_params"][key] = user_cfg[key]

    present_unsupported = [k for k in unsupported_keys if k in user_cfg]
    if present_unsupported:
        print(
            "Note: flat config includes keys not used by CLAM fast patching path: "
            f"{present_unsupported}. They are recorded but ignored in runtime."
        )
    return cfg


def parse_ids(raw_value):
    raw = str(raw_value)
    if raw != "none" and len(raw) > 0:
        return np.array(raw.split(",")).astype(int)
    return []


def patch_runtime_kwargs(current_patch, patch_level, patch_size, step_size, patch_save_dir):
    # CLAM fast pipeline (process_contours) only consumes these patch keys.
    runtime = {
        "patch_level": patch_level,
        "patch_size": patch_size,
        "step_size": step_size,
        "save_path": str(patch_save_dir),
    }
    for key in ["use_padding", "contour_fn"]:
        if key in current_patch:
            runtime[key] = current_patch[key]
    return runtime


def main():
    args = parse_args()

    sys.path.insert(0, str(args.clam_dir))
    from create_patches_fp import patching, segment, stitching  # noqa: E402
    from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402
    from openslide import OpenSlide  # noqa: E402

    patch_save_dir = args.save_dir / "patches"
    mask_save_dir = args.save_dir / "masks"
    stitch_save_dir = args.save_dir / "stitches"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    patch_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    stitch_save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    seg_params = cfg["seg_params"]
    filter_params = cfg["filter_params"]
    vis_params = cfg["vis_params"]
    patch_params = cfg["patch_params"]

    patch_size = args.patch_size if args.patch_size is not None else int(cfg["run"]["patch_size"])
    step_size = args.step_size if args.step_size is not None else int(cfg["run"]["step_size"])
    patch_level = args.patch_level if args.patch_level is not None else int(cfg["run"]["patch_level"])
    rows = load_rows(args.manifest_csv)
    if args.max_slides is not None:
        rows = rows[: args.max_slides]

    report_rows = []
    for i, row in enumerate(rows, start=1):
        slide_id = row["slide_id"]
        slide_path = Path(row["slide_path"])
        print(f"[{i}/{len(rows)}] Processing {slide_id} -> {slide_path}")

        if not slide_path.exists():
            msg = "missing_slide_path"
            if args.skip_incomplete:
                report_rows.append(
                    {
                        "slide_id": slide_id,
                        "slide_path": str(slide_path),
                        "status": msg,
                        "seg_time_sec": -1,
                        "patch_time_sec": -1,
                        "stitch_time_sec": -1,
                        "patch_h5_path": "",
                        "mask_path": "",
                        "stitch_path": "",
                    }
                )
                continue
            raise FileNotFoundError(f"Slide path does not exist: {slide_path}")

        try:
            detected_fmt = OpenSlide.detect_format(str(slide_path))
            if args.strict_wsi_format and detected_fmt is None:
                raise RuntimeError(
                    f"Unsupported WSI format for path: {slide_path}. "
                    "This may be a preview image renamed as .mrxs."
                )

            wsi = WholeSlideImage(str(slide_path))
            # Avoid filename collision because all MIRAX entries are named Slidedat.ini.
            wsi.name = str(slide_id)

            current_seg = seg_params.copy()
            current_filter = filter_params.copy()
            current_vis = vis_params.copy()
            current_patch = patch_params.copy()

            if current_vis["vis_level"] < 0:
                if len(wsi.level_dim) == 1:
                    current_vis["vis_level"] = 0
                else:
                    current_vis["vis_level"] = wsi.getOpenSlide().get_best_level_for_downsample(64)

            if current_seg["seg_level"] < 0:
                if len(wsi.level_dim) == 1:
                    current_seg["seg_level"] = 0
                else:
                    current_seg["seg_level"] = wsi.getOpenSlide().get_best_level_for_downsample(64)

            current_seg["keep_ids"] = parse_ids(current_seg["keep_ids"])
            current_seg["exclude_ids"] = parse_ids(current_seg["exclude_ids"])

            w, h = wsi.level_dim[current_seg["seg_level"]]
            if w * h > 1e8:
                raise RuntimeError(f"Segmentation level too large: {w} x {h}")

            seg_time = -1.0
            if args.seg:
                wsi, seg_time = segment(
                    wsi, seg_params=current_seg, filter_params=current_filter, mask_file=None
                )

            mask_path = mask_save_dir / f"{slide_id}.jpg"
            mask = wsi.visWSI(**current_vis)
            mask.save(mask_path)

            patch_time = -1.0
            patch_h5_path = patch_save_dir / f"{slide_id}.h5"
            if args.patch:
                patch_kwargs = patch_runtime_kwargs(
                    current_patch=current_patch,
                    patch_level=patch_level,
                    patch_size=patch_size,
                    step_size=step_size,
                    patch_save_dir=patch_save_dir,
                )
                _unused, patch_time = patching(WSI_object=wsi, **patch_kwargs)

            stitch_time = -1.0
            stitch_path = stitch_save_dir / f"{slide_id}.jpg"
            if args.stitch and patch_h5_path.exists():
                heatmap, stitch_time = stitching(str(patch_h5_path), wsi, downscale=64)
                heatmap.save(stitch_path)

            report_rows.append(
                {
                    "slide_id": slide_id,
                    "slide_path": str(slide_path),
                    "status": "processed",
                    "seg_time_sec": seg_time,
                    "patch_time_sec": patch_time,
                    "stitch_time_sec": stitch_time,
                    "patch_h5_path": str(patch_h5_path) if patch_h5_path.exists() else "",
                    "mask_path": str(mask_path),
                    "stitch_path": str(stitch_path) if stitch_path.exists() else "",
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            report_rows.append(
                {
                    "slide_id": slide_id,
                    "slide_path": str(slide_path),
                    "status": f"failed:{type(exc).__name__}",
                    "seg_time_sec": -1,
                    "patch_time_sec": -1,
                    "stitch_time_sec": -1,
                    "patch_h5_path": "",
                    "mask_path": "",
                    "stitch_path": "",
                }
            )
            print(f"Failed {slide_id}: {exc}")

    report_path = args.save_dir / "process_list_autogen.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"Done. Wrote report: {report_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total elapsed: {time.time() - t0:.2f}s")
