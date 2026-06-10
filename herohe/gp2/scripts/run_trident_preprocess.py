import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TRIDENT segmentation + coordinate extraction on a WSI directory."
    )
    parser.add_argument("--trident_dir", type=Path, required=True, help="Path to local TRIDENT repo.")
    parser.add_argument("--wsi_dir", type=Path, required=True, help="Directory containing WSI files.")
    parser.add_argument("--job_dir", type=Path, required=True, help="TRIDENT output job directory.")
    parser.add_argument("--mag", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--segmenter", type=str, default="hest", help="hest | grandqc | otsu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python_exe", type=Path, default=None, help="Python executable to use.")
    return parser.parse_args()


def run_cmd(cmd, cwd: Path):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    args = parse_args()
    py = str(args.python_exe) if args.python_exe else sys.executable

    args.job_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: segmentation
    seg_cmd = [
        py,
        "run_batch_of_slides.py",
        "--task",
        "seg",
        "--wsi_dir",
        str(args.wsi_dir),
        "--job_dir",
        str(args.job_dir),
        "--gpu",
        str(args.gpu),
        "--segmenter",
        str(args.segmenter),
    ]
    run_cmd(seg_cmd, args.trident_dir)

    # Step 2: coordinates
    coord_cmd = [
        py,
        "run_batch_of_slides.py",
        "--task",
        "coords",
        "--wsi_dir",
        str(args.wsi_dir),
        "--job_dir",
        str(args.job_dir),
        "--mag",
        str(args.mag),
        "--patch_size",
        str(args.patch_size),
        "--overlap",
        str(args.overlap),
    ]
    run_cmd(coord_cmd, args.trident_dir)

    print("TRIDENT preprocessing completed.")


if __name__ == "__main__":
    main()
