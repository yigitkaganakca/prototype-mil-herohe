import argparse
import csv
import time
from pathlib import Path

import h5py
import numpy as np
from openslide import OpenSlide
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-filter CLAM patch coordinates by patch intensity std-dev."
    )
    parser.add_argument("--manifest_csv", type=Path, required=True, help="Slide manifest CSV.")
    parser.add_argument("--input_patch_dir", type=Path, required=True, help="Directory with CLAM .h5 coord files.")
    parser.add_argument("--output_patch_dir", type=Path, required=True, help="Output directory for filtered .h5 files.")
    parser.add_argument("--report_csv", type=Path, required=True, help="Filtering summary CSV.")
    parser.add_argument("--std_threshold", type=float, default=12.0, help="Keep patches with grayscale std >= threshold.")
    parser.add_argument("--limit_slides", type=int, default=None, help="Optional slide cap for smoke tests.")
    return parser.parse_args()


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def open_coords(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    return coords, attrs


def save_coords(h5_path: Path, coords: np.ndarray, attrs: dict):
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("coords", data=coords, dtype=coords.dtype)
        for k, v in attrs.items():
            ds.attrs[k] = v


def patch_stddev(slide: OpenSlide, x: int, y: int, patch_level: int, patch_size: int) -> float:
    patch = slide.read_region((int(x), int(y)), int(patch_level), (int(patch_size), int(patch_size))).convert("RGB")
    arr = np.asarray(patch, dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return float(np.std(gray))


def main():
    args = parse_args()
    args.output_patch_dir.mkdir(parents=True, exist_ok=True)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest_csv)
    if args.limit_slides is not None:
        rows = rows[: args.limit_slides]

    report = []
    for i, row in enumerate(rows, start=1):
        slide_id = str(row["slide_id"])
        slide_path = Path(row["slide_path"])
        in_h5 = args.input_patch_dir / f"{slide_id}.h5"
        out_h5 = args.output_patch_dir / f"{slide_id}.h5"

        start = time.time()
        if not in_h5.exists():
            report.append(
                {
                    "slide_id": slide_id,
                    "slide_path": str(slide_path),
                    "status": "missing_input_h5",
                    "n_in": 0,
                    "n_out": 0,
                    "keep_ratio": 0.0,
                    "std_threshold": args.std_threshold,
                    "elapsed_sec": 0.0,
                }
            )
            continue

        try:
            coords, attrs = open_coords(in_h5)
            patch_size = int(attrs.get("patch_size", 256))
            patch_level = int(attrs.get("patch_level", 0))
            slide = OpenSlide(str(slide_path))

            keep = []
            for x, y in coords:
                if patch_stddev(slide, x, y, patch_level, patch_size) >= args.std_threshold:
                    keep.append((x, y))
            slide.close()

            keep_arr = np.array(keep, dtype=coords.dtype) if keep else np.empty((0, 2), dtype=coords.dtype)
            save_coords(out_h5, keep_arr, attrs)

            n_in = int(coords.shape[0])
            n_out = int(keep_arr.shape[0])
            report.append(
                {
                    "slide_id": slide_id,
                    "slide_path": str(slide_path),
                    "status": "filtered",
                    "n_in": n_in,
                    "n_out": n_out,
                    "keep_ratio": (n_out / n_in) if n_in > 0 else 0.0,
                    "std_threshold": args.std_threshold,
                    "elapsed_sec": time.time() - start,
                }
            )
            print(f"[{i}/{len(rows)}] {slide_id}: kept {n_out}/{n_in}")
        except Exception as exc:  # noqa: BLE001
            report.append(
                {
                    "slide_id": slide_id,
                    "slide_path": str(slide_path),
                    "status": f"failed:{type(exc).__name__}",
                    "n_in": 0,
                    "n_out": 0,
                    "keep_ratio": 0.0,
                    "std_threshold": args.std_threshold,
                    "elapsed_sec": time.time() - start,
                }
            )
            print(f"[{i}/{len(rows)}] {slide_id}: failed - {exc}")

    with args.report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slide_id",
                "slide_path",
                "status",
                "n_in",
                "n_out",
                "keep_ratio",
                "std_threshold",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(report)

    print(f"Wrote stddev filter report: {args.report_csv}")


if __name__ == "__main__":
    main()
