"""
scripts/prepare_all_releases.py
--------------------------------
Download → window → delete raw EEG, one release at a time.

Strategy:
  For each release:
    1. Run prepare_data() to download raw EEG and save windowed .npy files
    2. Delete the raw EEG cache for that release to free disk space
    3. Move on to the next release

This keeps peak disk usage to:
    ~3-4 GB raw EEG (one release at a time)
  + cumulative windowed .npy files (~0.5-1 GB per release)

Skips releases that are already fully windowed (resumable).

Usage:
    python scripts/prepare_all_releases.py
    python scripts/prepare_all_releases.py --task contrastChangeDetection
    python scripts/prepare_all_releases.py --releases R6,R7,R8   # subset
    python scripts/prepare_all_releases.py --skip R1,R2          # skip done ones
"""

import gc
import sys
import shutil
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import DATA_DIR, WINDOWS_DIR, TASK
from scripts.prepare_ccd_data import prepare_data


# ── Release → OpenNeuro dataset ID map ───────────────────────────────────────
RELEASE_MAP = {
    "R1":  "ds005505",
    "R2":  "ds005506",
    "R3":  "ds005507",
    "R4":  "ds005508",
    "R5":  "ds005509",
    "R6":  "ds005510",
    "R7":  "ds005511",
    "R8":  "ds005512",
    "R9":  "ds005514",
    "R10": "ds005515",
    "R11": "ds005516",
}


def get_disk_usage(path: Path) -> str:
    """Return human-readable disk usage of a directory."""
    if not path.exists():
        return "0B"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def get_free_disk(path: Path) -> str:
    """Return human-readable free disk space."""
    import shutil as sh
    usage = sh.disk_usage(path)
    free = usage.free
    for unit in ["B", "KB", "MB", "GB"]:
        if free < 1024:
            return f"{free:.1f} {unit}"
        free /= 1024
    return f"{free:.1f} TB"


def is_release_windowed(release: str, task: str) -> bool:
    """Check if a release has already been fully windowed."""
    out_dir = WINDOWS_DIR / f"{task}_{release}"
    subjects_file = out_dir / "subjects.txt"
    if not subjects_file.exists():
        return False
    subjects = subjects_file.read_text().strip().splitlines()
    return len(subjects) > 0


def delete_raw_cache(dataset_id: str) -> None:
    """
    Delete the raw EEG cache for a dataset to free disk space.
    The windowed .npy files in data/windows/ are NOT touched.
    """
    # EEGDash caches raw data under data/{dataset_id}/
    raw_dir = DATA_DIR / dataset_id
    if raw_dir.exists():
        size = get_disk_usage(raw_dir)
        print(f"  [cleanup] Deleting raw cache {raw_dir}  ({size})")
        shutil.rmtree(raw_dir)
        print(f"  [cleanup] ✓ Deleted")
    else:
        print(f"  [cleanup] No raw cache found at {raw_dir} — skipping")


def main(
    task: str = TASK,
    releases: list[str] = None,
    skip: list[str] = None,
    overwrite: bool = False,
    subject_batch_size: int = 5,
    delete_raw: bool = True,
) -> None:

    if releases is None:
        releases = list(RELEASE_MAP.keys())
    if skip:
        releases = [r for r in releases if r not in skip]

    print("=" * 65)
    print("PREPARE ALL RELEASES — download → window → delete raw")
    print("=" * 65)
    print(f"  Task     : {task}")
    print(f"  Releases : {releases}")
    print(f"  Delete raw after windowing : {delete_raw}")
    print(f"  Free disk: {get_free_disk(DATA_DIR)}")
    print()

    results = {}

    for i, release in enumerate(releases, 1):
        dataset_id = RELEASE_MAP.get(release)
        if dataset_id is None:
            print(f"\n[{i}/{len(releases)}] {release} — unknown release, skipping")
            continue

        print(f"\n{'='*65}")
        print(f"[{i}/{len(releases)}] Release {release}  ({dataset_id})")
        print(f"  Free disk : {get_free_disk(DATA_DIR)}")
        print("=" * 65)

        # ── Check if already done ─────────────────────────────────────────────
        if not overwrite and is_release_windowed(release, task):
            windows_dir = WINDOWS_DIR / f"{task}_{release}"
            n_subjects = len((windows_dir / "subjects.txt").read_text().strip().splitlines())
            print(f"  [skip] Already windowed ({n_subjects} subjects). Use --overwrite to redo.")
            # Still delete raw cache if it exists and wasn't cleaned up before
            if delete_raw:
                delete_raw_cache(dataset_id)
            results[release] = "skipped"
            continue

        # ── Download + window ─────────────────────────────────────────────────
        try:
            prepare_data(
                dataset_id=dataset_id,
                task=task,
                release=release,
                overwrite=overwrite,
                subject_batch_size=subject_batch_size,
            )
            results[release] = "done"
        except Exception:
            print(f"\n  [ERROR] prepare_data failed for {release}")
            traceback.print_exc()
            results[release] = "failed"
            gc.collect()
            # Don't delete raw on failure — may need it to retry
            continue

        # ── Delete raw EEG cache ──────────────────────────────────────────────
        if delete_raw:
            delete_raw_cache(dataset_id)

        gc.collect()
        print(f"\n  Free disk after cleanup: {get_free_disk(DATA_DIR)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print("=" * 65)
    for release, status in results.items():
        icon = "✓" if status == "done" else ("↷" if status == "skipped" else "✗")
        windows_dir = WINDOWS_DIR / f"{task}_{release}"
        if windows_dir.exists():
            subjects_file = windows_dir / "subjects.txt"
            n = len(subjects_file.read_text().strip().splitlines()) if subjects_file.exists() else 0
            size = get_disk_usage(windows_dir)
            print(f"  {icon} {release:>3} : {status:<8}  {n:>4} subjects  {size}")
        else:
            print(f"  {icon} {release:>3} : {status}")

    done_releases  = [r for r, s in results.items() if s in ("done", "skipped")]
    total_subjects = 0
    for release in done_releases:
        f = WINDOWS_DIR / f"{task}_{release}" / "subjects.txt"
        if f.exists():
            total_subjects += len(f.read_text().strip().splitlines())

    print(f"\n  Total windowed subjects across all releases : {total_subjects}")
    print(f"  Estimated total windows (~58/subject)       : {total_subjects * 58:,}")
    print(f"  Final free disk                             : {get_free_disk(DATA_DIR)}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, window, and delete raw EEG for all releases sequentially."
    )
    parser.add_argument("--task",     type=str, default=TASK,
                        help="Task name (default: from config.py)")
    parser.add_argument("--releases", type=str, default=None,
                        help="Comma-separated releases to process, e.g. R1,R2,R3 (default: all)")
    parser.add_argument("--skip",     type=str, default=None,
                        help="Comma-separated releases to skip, e.g. R1,R5")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Re-process releases that are already windowed")
    parser.add_argument("--keep-raw",    action="store_true",
                        help="Do NOT delete raw EEG cache after windowing")
    parser.add_argument("--batch-size",  type=int, default=5,
                        help="Subjects per GC cycle in prepare_data (default: 5)")
    args = parser.parse_args()

    releases = args.releases.split(",") if args.releases else None
    skip     = args.skip.split(",")     if args.skip     else None

    main(
        task=args.task,
        releases=releases,
        skip=skip,
        overwrite=args.overwrite,
        delete_raw=not args.keep_raw,
        subject_batch_size=args.batch_size,
    )