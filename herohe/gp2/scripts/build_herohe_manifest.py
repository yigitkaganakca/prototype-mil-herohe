import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build HEROHE manifest from training ground truth CSV."
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        required=True,
        help="Path to HEROHE labels CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--slides_root",
        type=Path,
        required=True,
        help="Root containing case folders like 1/, 2/, ...",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--case_ids",
        type=str,
        default=None,
        help="Optional comma-separated case IDs to include (example: 1,2,3,10).",
    )
    parser.add_argument(
        "--slide_ini_name",
        type=str,
        default="Slidedat.ini",
        help="Slide entry filename inside each case folder.",
    )
    parser.add_argument(
        "--prefer_mrxs",
        action="store_true",
        default=True,
        help="If case_id.mrxs exists in case folder, use it as slide path.",
    )
    return parser.parse_args()


def parse_case_filter(case_ids_arg):
    if not case_ids_arg:
        return None
    return {x.strip() for x in case_ids_arg.split(",") if x.strip()}


def main():
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    include_cases = parse_case_filter(args.case_ids)
    rows_out = []

    with args.labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        required = {"Case", "Immunohistochemistry"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in labels CSV: {sorted(missing)}")

        for row in reader:
            case_raw = (row.get("Case") or "").strip()
            ihc_raw = (row.get("Immunohistochemistry") or "").strip()

            if not case_raw or ihc_raw not in {"0", "1", "2", "3"}:
                continue

            if include_cases and case_raw not in include_cases:
                continue

            slide_id = case_raw
            case_dir = args.slides_root / case_raw
            # Support both common MIRAX layouts:
            # 1) side-by-side: <root>/<case_id>.mrxs + <root>/<case_id>/ This is how it should be to openslide to work
            # 2) nested:      <root>/<case_id>/<case_id>.mrxs + <root>/<case_id>/
            root_mrxs_path = args.slides_root / f"{case_raw}.mrxs"

            if args.prefer_mrxs and root_mrxs_path.exists():
                slide_path = root_mrxs_path
            else:
                slide_path = case_dir / args.slide_ini_name

            rows_out.append(
                {
                    "case_id": case_raw,
                    "slide_id": slide_id,
                    "slide_path": str(slide_path),
                    "ihc_label": int(ihc_raw),
                }
            )

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "slide_id", "slide_path", "ihc_label"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
