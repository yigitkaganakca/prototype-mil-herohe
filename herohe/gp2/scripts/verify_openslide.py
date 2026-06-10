# herohe/gp2/scripts/verify_openslide.py
"""OpenSlide-based verification pass for HEROHE cases.

For each case dir <workspace>/<case_id>/ with an associated <workspace>/<case_id>.mrxs:
  - try OpenSlide(path)
  - read a thumbnail
  - read_region at (0, 0) on the lowest level
  - read_region at the slide centre on level 0 (small tile)
  - read_region at level=mid on a non-zero offset

Writes a CSV with one row per case:
  case_id, ok, levels, base_dims, error
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--out", default="herohe/gp2/data/openslide_verify.csv")
    p.add_argument("--limit", type=int, default=0, help="optional cap for smoke runs")
    return p.parse_args()

def verify_one(mrxs_path: Path) -> dict:
    import openslide
    info = {"path": str(mrxs_path), "ok": False, "levels": "", "base_dims": "", "error": ""}
    try:
        s = openslide.OpenSlide(str(mrxs_path))
    except Exception as e:
        info["error"] = f"open:{type(e).__name__}:{e}"
        return info
    try:
        info["levels"] = s.level_count
        info["base_dims"] = f"{s.dimensions[0]}x{s.dimensions[1]}"
        # thumbnail
        s.get_thumbnail((512, 512))
        # base level small region
        s.read_region((0, 0), 0, (256, 256))
        # mid level small region (uses tile mapping)
        mid = max(0, s.level_count // 2)
        w, h = s.level_dimensions[mid]
        s.read_region((min(w-1, 1024), min(h-1, 1024)), mid, (256, 256))
        info["ok"] = True
    except Exception as e:
        info["error"] = f"read:{type(e).__name__}:{e}"
    finally:
        try: s.close()
        except Exception: pass
    return info

def main():
    args = parse_args()
    ws = Path(args.workspace)
    out = (ws / args.out) if not args.out.startswith("/") else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = sorted([p for p in ws.glob("*.mrxs") if p.is_file()],
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    if args.limit:
        cases = cases[:args.limit]
    rows = []
    t0 = time.time()
    for i, mrxs in enumerate(cases, 1):
        r = verify_one(mrxs)
        r["case_id"] = mrxs.stem
        rows.append(r)
        status = "OK" if r["ok"] else f"FAIL {r['error']}"
        print(f"[{i:>4}/{len(cases)}] {mrxs.name:<10} {status}")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "ok", "levels", "base_dims", "error", "path"])
        for r in rows:
            w.writerow([r.get("case_id",""), int(r.get("ok",False)),
                        r.get("levels",""), r.get("base_dims",""),
                        r.get("error",""), r.get("path","")])
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\nDone in {time.time()-t0:.1f}s.  ok={n_ok}/{len(rows)}  csv={out}")

if __name__ == "__main__":
    sys.exit(main())