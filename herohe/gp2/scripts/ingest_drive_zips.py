#!/usr/bin/env python3
#manual control results wout real data count
#break at 36 till 82, 83 94 fine in between? check needed
# okay till 150
# 

"""
Ingest HEROHE WSI data from Google Drive `drive-download-*.zip` archives.

What this does
--------------
1. Scans a source directory (default ~/Downloads) for `drive-download-*.zip*` entries.
   Classifies each as: complete (real zip), partial (.zip.part chunk),
   empty (0-byte placeholder), or broken (non-zip data with .zip suffix).
2. For each complete zip:
     - Lists members, groups by case id (top-level folder named with digits).
     - Extracts each member into <workspace>/<case_id>/, streaming.
       (zipfile.ZipFile.open verifies CRC32 of every file as it's read, so
       extraction success implies bit-identical content with the source.)
     - On collision: if size + CRC32 match the zip-stored CRC -> skip. Otherwise
       saves the incoming as `<file>.conflict_<zipname>` and logs the conflict.
     - **After a successful extract, deletes the source zip** to reclaim disk
       (override with --keep-zips).
3. Persists per-zip state in a JSON file so re-runs skip already-done zips.
4. Verifies every <workspace>/<case_id>/ directory:
     - Slidedat.ini, Index.dat, sibling <id>.mrxs presence
     - Contiguous Data*.dat sequence
     - Cross-checks Data*.dat count against [DATAFILE] FILE_COUNT in Slidedat.ini
5. With --cleanup: retroactively deletes any source zip whose state is `done`
   but whose file still exists on disk (useful after upgrading from an older
   conservative version that kept zips around).

Idempotent. Safe to interrupt and resume. Writes a human-readable status CSV
to <log_dir>/ingest_status.csv and a verbose log to ingest_log_<ts>.log.

Run
---
    python herohe/gp2/scripts/ingest_drive_zips.py
    python herohe/gp2/scripts/ingest_drive_zips.py --keep-zips
    python herohe/gp2/scripts/ingest_drive_zips.py --cleanup --no-extract  # retroactive zip purge
    python herohe/gp2/scripts/ingest_drive_zips.py --no-extract            # verify only
    python herohe/gp2/scripts/ingest_drive_zips.py --dry-run               # plan only
"""

from __future__ import annotations

import argparse
import configparser
import csv
import filecmp
import json
import logging
import re
import shutil
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

CASE_ID_RE = re.compile(r"^\d+$")
MRXS_FILE_RE = re.compile(r"^(\d+)\.mrxs$")
DATA_FILE_RE = re.compile(r"^Data(\d+)\.dat$")
ALLOWED_OTHER_FILES = {"Slidedat.ini", "Index.dat", "desktop.ini"}
ZIP_GLOB = "*.zip*"  # broad: any naming convention; validated by content below
DRIVE_PARENT_GLOB = "drive-download-*"
IGNORED_DIR_NAMES = {"__MACOSX", ".DS_Store"}


# ---------------------------------------------------------------- logging --
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ingest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ------------------------------------------------------------- discovery ---
def zip_contains_mirax_data(zip_path: Path) -> bool:
    """Relevant iff the zip contains either:
       - a top-level <digits>/<Data*.dat | Slidedat.ini | Index.dat>, OR
       - a top-level <digits>.mrxs entry file.""" # not the case for many times but just in case 
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for m in zf.infolist():
                if m.is_dir():
                    continue
                parts = Path(m.filename).parts
                if any(p == "__MACOSX" for p in parts):
                    continue
                fname = parts[-1]
                if len(parts) == 1 and MRXS_FILE_RE.match(fname):
                    return True
                if len(parts) >= 2:
                    case_id = parts[0]
                    if CASE_ID_RE.match(case_id) and (
                        DATA_FILE_RE.match(fname) or fname in {"Slidedat.ini", "Index.dat"}
                    ):
                        return True
    except (zipfile.BadZipFile, OSError):
        return False
    return False


def classify_entries(src: Path) -> Dict[str, List[Path]]:
    """Bucket all *.zip* entries by type. Any valid zip whose contents don't
    look like MIRAX case data is silently ignored (not put into any bucket)."""
    buckets: Dict[str, List[Path]] = {
        "complete": [],
        "partial": [],
        "empty": [],
        "broken": [],
    }
    for p in sorted(src.glob(ZIP_GLOB)):
        if not p.is_file():
            continue
        size = p.stat().st_size
        if p.name.endswith(".zip.part") or ".zip.part" in p.name:
            buckets["partial"].append(p) # requires manual handling for these for sure
            continue
        if size == 0:
            buckets["empty"].append(p)
            continue
        if not p.name.endswith(".zip"):
            buckets["partial"].append(p)
            continue
        if not zipfile.is_zipfile(p):
            buckets["broken"].append(p)
            continue
        if zip_contains_mirax_data(p):
            buckets["complete"].append(p)
        # else: valid zip but unrelated content (e.g. installer); silently ignore
    return buckets


# ----------------------------------------------------------------- state ---
def zip_signature(zip_path: Path) -> str:
    st = zip_path.stat()
    return f"{st.st_size}_{int(st.st_mtime)}"


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(state_path)


# ----------------------------------------------------------- file checks ---
def compute_crc32(p: Path) -> int:
    crc = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def is_member_relevant(zinfo: zipfile.ZipInfo) -> Tuple[bool, Optional[str], Optional[str], bool]:
    """Returns (keep, case_id, filename, is_mrxs_root).

    is_mrxs_root=True means the file should be written to workspace/<filename>
    (i.e. the .mrxs entry file), NOT into workspace/<case_id>/<filename>.
    """
    if zinfo.is_dir():
        return False, None, None, False
    parts = Path(zinfo.filename).parts
    if any(part == "__MACOSX" for part in parts):
        return False, None, None, False
    fname = parts[-1]
    if fname.startswith("._") or fname == ".DS_Store":
        return False, None, None, False
    # top-level <digits>.mrxs -> workspace root entry file
    if len(parts) == 1:
        m = MRXS_FILE_RE.match(fname)
        if m:
            return True, m.group(1), fname, True
        return False, None, None, False
    case_id = parts[0]
    if not CASE_ID_RE.match(case_id):
        return False, None, None, False
    if not (DATA_FILE_RE.match(fname) or fname in ALLOWED_OTHER_FILES):
        return False, None, None, False
    return True, case_id, fname, False


# ------------------------------------------------------------ extraction ---
def extract_zip(
    zip_path: Path,
    workspace: Path,
    logger: logging.Logger,
    dry_run: bool = False,
) -> Tuple[Set[str], int, int]:
    """Extract one zip. Returns (case_ids_touched, files_written, conflicts)."""
    cases_touched: Set[str] = set()
    files_written = 0
    conflicts = 0
    ignored = 0

    with zipfile.ZipFile(zip_path) as zf:
        by_case: Dict[str, List[Tuple[zipfile.ZipInfo, bool]]] = defaultdict(list)
        for m in zf.infolist():
            keep, case_id, fname, is_mrxs_root = is_member_relevant(m)
            if not keep:
                if not m.is_dir():
                    ignored += 1
                continue
            by_case[case_id].append((m, is_mrxs_root))

        for case_id, members in sorted(by_case.items(), key=lambda kv: int(kv[0])):
            cases_touched.add(case_id)
            tgt_dir = workspace / case_id
            need_dir = any(not is_mrxs for _, is_mrxs in members)
            if need_dir and not dry_run:
                tgt_dir.mkdir(parents=True, exist_ok=True)
            for m, is_mrxs_root in members:
                fname = Path(m.filename).name
                dest = (workspace / fname) if is_mrxs_root else (tgt_dir / fname)
                if dest.exists():
                    same_size = dest.stat().st_size == m.file_size
                    if same_size and compute_crc32(dest) == (m.CRC & 0xFFFFFFFF):
                        continue
                    conflict_dest = dest.parent / f"{fname}.conflict_{zip_path.stem}"
                    logger.warning(
                        "  CONFLICT %s differs; keeping existing, saving incoming as %s",
                        dest.relative_to(workspace), conflict_dest.name,
                    )
                    if not dry_run:
                        with zf.open(m) as src, open(conflict_dest, "wb") as out:
                            shutil.copyfileobj(src, out, length=1 << 20)
                    conflicts += 1
                else:
                    if not dry_run:
                        with zf.open(m) as src, open(dest, "wb") as out:
                            shutil.copyfileobj(src, out, length=1 << 20)
                    files_written += 1

    if ignored:
        logger.info("  (ignored %d irrelevant entries)", ignored)
    return cases_touched, files_written, conflicts


# ----------------------------------------------------- loose-dir ingest ---
def looks_like_mirax_case_dir(d: Path) -> bool:
    """Heuristic: a real case dir contains at least one MIRAX-relevant file
    (Data*.dat, Slidedat.ini, or Index.dat). Fragment dirs that have only
    a few Data*.dat files (from interrupted downloads) still qualify and
    can usefully fill gaps in incomplete cases."""
    if not d.is_dir():
        return False
    for f in d.iterdir():
        if not f.is_file():
            continue
        if DATA_FILE_RE.match(f.name) or f.name in {"Slidedat.ini", "Index.dat"}:
            return True
    return False


def discover_loose_case_dirs(src: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    Returns list of (case_dir, parent_dir_or_None).
    parent_dir is set when the case_dir lives inside a drive-download-*/ folder
    (so we know which parent to consider deleting after).
    """
    found: List[Tuple[Path, Optional[Path]]] = []
    for child in sorted(src.iterdir()):
        name = child.name
        if name in IGNORED_DIR_NAMES or name.startswith("."):
            continue
        if not child.is_dir():
            continue
        if CASE_ID_RE.match(name) and looks_like_mirax_case_dir(child):
            found.append((child, None))
            continue
        if child.match(DRIVE_PARENT_GLOB):
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and CASE_ID_RE.match(sub.name) and looks_like_mirax_case_dir(sub):
                    found.append((sub, child))
    return found


def ingest_case_dir(
    case_dir: Path,
    workspace: Path,
    logger: logging.Logger,
    dry_run: bool = False,
) -> Tuple[str, int, int, int]:
    """Merge a single loose case dir into workspace/<case_id>/.

    Returns (case_id, files_copied, conflicts, ignored).
    Conflict detection is byte-for-byte (filecmp.cmp shallow=False).
    """
    case_id = case_dir.name
    target = workspace / case_id
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    conflicts = 0
    ignored = 0

    for src_file in sorted(case_dir.iterdir()):
        if not src_file.is_file():
            continue
        fname = src_file.name
        if fname.startswith("."):
            ignored += 1
            continue
        if not (DATA_FILE_RE.match(fname) or fname in ALLOWED_OTHER_FILES):
            ignored += 1
            continue

        dest = target / fname
        if dest.exists():
            same_size = dest.stat().st_size == src_file.stat().st_size
            if same_size and filecmp.cmp(dest, src_file, shallow=False):
                continue
            conflict_dest = target / f"{fname}.conflict_{case_dir.parent.name}_{case_id}"
            logger.warning(
                "  CONFLICT %s/%s differs; keeping existing, saving incoming as %s",
                case_id, fname, conflict_dest.name,
            )
            if not dry_run:
                shutil.copy2(src_file, conflict_dest)
            conflicts += 1
        else:
            if not dry_run:
                shutil.copy2(src_file, dest)
            files_copied += 1

    return case_id, files_copied, conflicts, ignored


def remove_dir_if_empty(d: Path, logger: logging.Logger) -> bool:
    """Delete d if it contains no relevant content. Returns True if deleted."""
    if not d.exists():
        return False
    leftovers = [
        f for f in d.iterdir()
        if f.name not in IGNORED_DIR_NAMES and not f.name.startswith(".")
    ]
    if leftovers:
        logger.info("  KEPT parent %s (still has %d non-empty children)", d.name, len(leftovers))
        return False
    try:
        shutil.rmtree(d)
        logger.info("  DELETED parent dir %s", d.name)
        return True
    except OSError as e:
        logger.warning("  could not delete %s: %s", d, e)
        return False


# ---------------------------------------------------------- verification ---
@dataclass
class CaseStatus:
    case_id: str
    has_slidedat: bool
    has_index: bool
    has_mrxs: bool
    data_count: int
    expected_data_count: Optional[int]
    missing_indices: List[int] = field(default_factory=list)
    extra_files: List[str] = field(default_factory=list)
    is_complete: bool = False


def parse_expected_datafile_count(slidedat: Path) -> Optional[int]:
    cfg = configparser.ConfigParser(strict=False)
    cfg.optionxform = str
    try:
        with slidedat.open("r", encoding="utf-8-sig", errors="ignore") as f:
            cfg.read_file(f)
        if "DATAFILE" in cfg and "FILE_COUNT" in cfg["DATAFILE"]:
            return int(cfg["DATAFILE"]["FILE_COUNT"])
    except Exception:
        return None
    return None


def verify_case(case_dir: Path, mrxs_root: Path) -> CaseStatus:
    fnames = {f.name for f in case_dir.iterdir() if f.is_file()}
    has_slidedat = "Slidedat.ini" in fnames
    has_index = "Index.dat" in fnames
    has_mrxs = (mrxs_root / f"{case_dir.name}.mrxs").exists()

    data_indices: List[int] = []
    extras: List[str] = []
    for fn in fnames:
        m = DATA_FILE_RE.match(fn)
        if m:
            data_indices.append(int(m.group(1)))
        elif fn not in ALLOWED_OTHER_FILES and not fn.startswith(".") and ".conflict_" not in fn:
            extras.append(fn)
    data_indices.sort()

    expected = None
    if has_slidedat:
        expected = parse_expected_datafile_count(case_dir / "Slidedat.ini")

    if data_indices:
        full = set(range(0, max(data_indices) + 1))
        missing = sorted(full - set(data_indices))
    else:
        missing = []

    is_complete = (
        has_slidedat
        and has_index
        and has_mrxs
        and not missing
        and (expected is None or len(data_indices) == expected)
    )
    return CaseStatus(
        case_id=case_dir.name,
        has_slidedat=has_slidedat,
        has_index=has_index,
        has_mrxs=has_mrxs,
        data_count=len(data_indices),
        expected_data_count=expected,
        missing_indices=missing,
        extra_files=sorted(extras),
        is_complete=is_complete,
    )


def write_status_csv(statuses: List[CaseStatus], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case_id", "is_complete", "has_mrxs", "has_slidedat", "has_index",
            "data_count", "expected_data_count", "missing_indices", "extra_files",
        ])
        for s in sorted(statuses, key=lambda x: int(x.case_id)):
            w.writerow([
                s.case_id, s.is_complete, s.has_mrxs, s.has_slidedat, s.has_index,
                s.data_count, s.expected_data_count if s.expected_data_count is not None else "",
                ",".join(str(i) for i in s.missing_indices),
                ",".join(s.extra_files),
            ])


# ------------------------------------------------------------------- cli ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="~/Downloads", help="Source dir with drive-download-*.zip files")
    p.add_argument("--workspace", default=str(Path(__file__).resolve().parents[3]),
                   help="Target workspace where <case_id>/ directories live")
    p.add_argument("--state-file", default=None,
                   help="State JSON path (default: <src>/_herohe_extract_state.json)")
    p.add_argument("--log-dir", default=None,
                   help="Where to write logs and status CSV (default: <workspace>/herohe/gp2/data)")
    p.add_argument("--keep-zips", action="store_true",
                   help="Do NOT delete source zips OR loose case dirs after successful ingest "
                        "(default: delete)")
    p.add_argument("--cleanup", action="store_true",
                   help="Retroactively delete any zip whose state is 'done' but whose file still exists")
    p.add_argument("--no-extract", action="store_true",
                   help="Skip extraction step; only run verification")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only, no writes anywhere")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    state_file = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file else src / "_herohe_extract_state.json"
    )
    log_dir = (
        Path(args.log_dir).expanduser().resolve()
        if args.log_dir else workspace / "herohe" / "gp2" / "data"
    )
    log_path = log_dir / f"ingest_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
    csv_path = log_dir / "ingest_status.csv"
    logger = setup_logger(log_path)

    if not src.exists():
        logger.error("Source dir does not exist: %s", src)
        return 1
    if not workspace.exists():
        logger.error("Workspace does not exist: %s", workspace)
        return 1

    logger.info("src=%s", src)
    logger.info("workspace=%s", workspace)
    logger.info("state=%s", state_file)
    logger.info("log=%s", log_path)
    logger.info("dry_run=%s no_extract=%s keep_zips=%s cleanup=%s",
                args.dry_run, args.no_extract, args.keep_zips, args.cleanup)

    buckets = classify_entries(src)
    logger.info(
        "Classified: complete=%d partial=%d empty=%d broken=%d",
        len(buckets["complete"]), len(buckets["partial"]),
        len(buckets["empty"]), len(buckets["broken"]),
    )

    if buckets["partial"]:
        logger.info("=== INCOMPLETE DOWNLOADS (re-download these from Drive) ===")
        for p in buckets["partial"]:
            logger.info("  PARTIAL %s (%.1f MB)", p.name, p.stat().st_size / 1e6)
    if buckets["empty"]:
        logger.info("=== EMPTY 0-BYTE PLACEHOLDERS ===")
        for p in buckets["empty"]:
            logger.info("  EMPTY   %s", p.name)
    if buckets["broken"]:
        logger.info("=== BROKEN ZIPS (corrupt) ===")
        for p in buckets["broken"]:
            logger.info("  BROKEN  %s", p.name)

    state = load_state(state_file)

    if not args.no_extract:
        logger.info("=== EXTRACTION ===")
        for zip_path in buckets["complete"]:
            sig = zip_signature(zip_path)
            existing = state.get(zip_path.name)
            if existing and existing.get("signature") == sig and existing.get("status") == "done":
                logger.info("SKIP (already done): %s", zip_path.name)
                continue
            logger.info("EXTRACT %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)
            t0 = time.time()
            try:
                cases, files_written, conflicts = extract_zip(
                    zip_path, workspace, logger, dry_run=args.dry_run
                )
                elapsed = time.time() - t0
                logger.info(
                    "  done in %.1fs: cases=%s files_written=%d conflicts=%d",
                    elapsed, sorted(cases, key=int), files_written, conflicts,
                )
                if not args.dry_run:
                    state[zip_path.name] = {
                        "signature": sig,
                        "status": "done",
                        "cases_touched": sorted(cases),
                        "files_written": files_written,
                        "conflicts": conflicts,
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    save_state(state_file, state)
                    if not args.keep_zips:
                        try:
                            size_mb = zip_path.stat().st_size / 1e6
                            zip_path.unlink()
                            logger.info("  DELETED %s (%.1f MB reclaimed)", zip_path.name, size_mb)
                        except OSError as e:
                            logger.warning("  could not delete %s: %s", zip_path.name, e)
            except Exception as e:
                logger.error("  FAILED %s: %s", zip_path.name, e)
                if not args.dry_run:
                    state[zip_path.name] = {
                        "signature": sig,
                        "status": "failed",
                        "error": str(e),
                        "attempted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    save_state(state_file, state)

    if not args.no_extract:
        loose = discover_loose_case_dirs(src)
        if loose:
            logger.info("=== LOOSE CASE DIRECTORIES (%d found) ===", len(loose))
            parents_seen: Dict[Path, List[str]] = defaultdict(list)
            for case_dir, parent in loose:
                key = f"dir:{case_dir.resolve()}"
                if state.get(key, {}).get("status") == "done" and not case_dir.exists():
                    continue
                logger.info(
                    "INGEST DIR %s%s",
                    case_dir.relative_to(src) if case_dir.is_relative_to(src) else case_dir,
                    f" (parent: {parent.name})" if parent else "",
                )
                t0 = time.time()
                try:
                    case_id, copied, conflicts, ignored = ingest_case_dir(
                        case_dir, workspace, logger, dry_run=args.dry_run,
                    )
                    elapsed = time.time() - t0
                    logger.info(
                        "  done in %.1fs: case=%s files_copied=%d conflicts=%d ignored=%d",
                        elapsed, case_id, copied, conflicts, ignored,
                    )
                    if not args.dry_run:
                        state[key] = {
                            "kind": "loose_dir",
                            "status": "done",
                            "case_id": case_id,
                            "files_copied": copied,
                            "conflicts": conflicts,
                            "parent": str(parent) if parent else None,
                            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        save_state(state_file, state)
                        if not args.keep_zips:
                            try:
                                shutil.rmtree(case_dir)
                                logger.info("  DELETED source dir %s", case_dir.name)
                                if parent is not None:
                                    parents_seen[parent].append(case_id)
                            except OSError as e:
                                logger.warning("  could not delete %s: %s", case_dir, e)
                except Exception as e:
                    logger.error("  FAILED %s: %s", case_dir, e)
                    if not args.dry_run:
                        state[key] = {
                            "kind": "loose_dir",
                            "status": "failed",
                            "error": str(e),
                            "attempted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        save_state(state_file, state)

            if parents_seen and not args.keep_zips and not args.dry_run:
                logger.info("=== PARENT DIR CLEANUP ===")
                for parent in parents_seen:
                    remove_dir_if_empty(parent, logger)

    logger.info("=== VERIFICATION ===")
    case_dirs = [
        d for d in workspace.iterdir()
        if d.is_dir() and CASE_ID_RE.match(d.name)
    ]
    statuses = [verify_case(d, workspace) for d in case_dirs]
    n_ok = sum(1 for s in statuses if s.is_complete)
    n_total = len(statuses)
    for s in sorted(statuses, key=lambda x: int(x.case_id)):
        marker = "OK        " if s.is_complete else "INCOMPLETE"
        logger.info(
            "  %s case=%-4s data=%d/%s missing=%s mrxs=%s",
            marker, s.case_id, s.data_count,
            s.expected_data_count if s.expected_data_count is not None else "?",
            s.missing_indices if s.missing_indices else "-",
            s.has_mrxs,
        )
    write_status_csv(statuses, csv_path)
    logger.info("Wrote status CSV: %s", csv_path)

    if args.cleanup and not args.dry_run:
        logger.info("=== RETROACTIVE CLEANUP ===")
        deleted = 0
        bytes_reclaimed = 0
        for zip_name, info in list(state.items()):
            if info.get("status") != "done":
                continue
            zip_path = src / zip_name
            if not zip_path.exists():
                continue
            try:
                size = zip_path.stat().st_size
                zip_path.unlink()
                deleted += 1
                bytes_reclaimed += size
                logger.info("  DELETED %s (%.1f MB)", zip_name, size / 1e6)
            except OSError as e:
                logger.warning("  could not delete %s: %s", zip_name, e)
        logger.info("Retroactive cleanup: deleted=%d, reclaimed=%.1f MB",
                    deleted, bytes_reclaimed / 1e6)

    logger.info("=== DONE: %d/%d cases fully verified ===", n_ok, n_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
