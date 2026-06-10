#!/usr/bin/env python3
"""Verify HEROHE test MIRAX slides: .mrxs + paired .dat tree + OpenSlide."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import openslide


def dat_count(slide_dir: Path) -> int:
    if not slide_dir.is_dir():
        return -1
    return len(list(slide_dir.glob("*.dat")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wsi_dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "herohe" / "wsi_test",
    )
    ap.add_argument(
        "--ref_dir",
        type=Path,
        default=None,
        help="Optional Google Drive Test/ folder to compare .dat counts",
    )
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    wsi_dir = args.wsi_dir.expanduser().resolve()
    slide_ids = sorted(int(p.stem) for p in wsi_dir.glob("*.mrxs"))
    if not slide_ids:
        raise SystemExit(f"No .mrxs in {wsi_dir}")

    ref_dir = args.ref_dir.expanduser().resolve() if args.ref_dir else None
    bad_dat: list[dict] = []
    openslide_fail: list[dict] = []

    for sid in slide_ids:
        n = dat_count(wsi_dir / str(sid))
        if n <= 0:
            bad_dat.append({"slide_id": sid, "local_dat": n, "reason": "missing_or_empty_dat_dir"})
            continue
        if ref_dir is not None:
            ref_n = dat_count(ref_dir / str(sid))
            if n != ref_n:
                bad_dat.append({"slide_id": sid, "local_dat": n, "ref_dat": ref_n})
        mrxs = wsi_dir / f"{sid}.mrxs"
        try:
            with openslide.OpenSlide(str(mrxs)) as slide:
                _ = slide.dimensions
        except Exception as e:
            openslide_fail.append({"slide_id": sid, "error": str(e)})

    report = {
        "wsi_dir": str(wsi_dir),
        "n_slides": len(slide_ids),
        "n_dat_files": len(list(wsi_dir.rglob("*.dat"))),
        "dat_mismatch_or_missing": len(bad_dat),
        "openslide_fail": len(openslide_fail),
        "ok": len(bad_dat) == 0 and len(openslide_fail) == 0,
        "bad_dat": bad_dat[:50],
        "openslide_failures": openslide_fail[:50],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json:
        args.out_json.write_text(text)

    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
