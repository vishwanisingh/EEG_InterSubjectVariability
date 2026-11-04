"""
prepare_ssl_data.py
-------------------
Download all CCD releases and create SLIDING WINDOWS for SSL pretraining.

Unlike prepare_ccd_data.py (which cuts event-locked windows around stimulus
onsets), this script slides a fixed-length window across the entire raw EEG
recording with a fixed stride. No event structure, no RT labels — just raw
EEG epochs for unsupervised pretraining.

Why sliding windows for SSL:
  - Event-locked: ~60 windows per subject (only trial periods)
  - Sliding:      ~300 windows per subject (entire recording)
  - SSL benefits from MORE unlabeled data, so sliding wins even though
    many windows won't contain a stimulus response.

On-disk layout (compatible with existing EEGWindowDataset):
    data/windows/ssl_ccd_{release}/
        subjects.txt
        sub-{subject_id}/
            windows.npy     float32  (n_windows, n_chans, n_times)
            metadata.npz    subject, window_start_s, recording_id

Strategy (same as prepare_all_releases.py):
  For each release: download raw → sliding window → delete raw → next release
  Peak disk: ~3-4 GB raw + cumulative windowed .npy files (~1-2 GB per release)

Usage:
    # All releases (recommended — runs sequentially, resumable)
    python prepare_ssl_data.py

    # Specific releases only
    python prepare_ssl_data.py --releases R5,R6,R7

    # Skip already-done releases
    python prepare_ssl_data.py --skip R1,R2,R3

    # Check disk usage without downloading
    python prepare_ssl_data.py --dry_run
"""

import gc
import sys
import shutil
import argparse
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import DATA_DIR, WINDOWS_DIR, SFREQ
from src.data.loader import load_eegdash_dataset


# ── Constants ─────────────────────────────────────────────────────────────────
TASK = "contrastChangeDetection"

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
# R5 = held-out test release (must never be seen during pretraining)
# R7 = no CCD task in this release
TEST_RELEASE     = "R5"
EXCLUDED_RELEASES = {"R5", "R7"}
ALL_RELEASES     = [r for r in RELEASE_MAP if r not in EXCLUDED_RELEASES]

WINDOW_LEN_S  = 2.0                      # seconds — same as supervised pipeline
STRIDE_S      = 1.0                      # seconds — 50% overlap
WINDOW_SAMPS  = int(WINDOW_LEN_S * SFREQ)  # 200 samples at 100 Hz
STRIDE_SAMPS  = int(STRIDE_S * SFREQ)      # 100 samples

# Minimum recording length to bother windowing (skip very short recordings)
MIN_RECORDING_S = 10.0
MIN_RECORDING_SAMPS = int(MIN_RECORDING_S * SFREQ)


# ─────────────────────────────────────────────────────────────────────────────
# Disk helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_disk_usage(path: Path) -> str:
    if not path.exists():
        return "0 B"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _get_free_disk(path: Path) -> str:
    import shutil as sh
    free = sh.disk_usage(path).free
    for unit in ["B", "KB", "MB", "GB"]:
        if free < 1024:
            return f"{free:.1f} {unit}"
        free /= 1024
    return f"{free:.1f} TB"


def _delete_raw_cache(dataset_id: str) -> None:
    raw_dir = DATA_DIR / dataset_id
    if raw_dir.exists():
        size = _get_disk_usage(raw_dir)
        print(f"  [cleanup] Deleting {raw_dir}  ({size})")
        shutil.rmtree(raw_dir)
        print(f"  [cleanup] ✓ Done")
    else:
        print(f"  [cleanup] No raw cache at {raw_dir} — skipping")


def _already_done(subject_id: str, out_dir: Path) -> bool:
    d = out_dir / f"sub-{subject_id}"
    return (d / "windows.npy").exists() and (d / "metadata.npz").exists()


def _is_release_windowed(release: str) -> bool:
    out_dir = WINDOWS_DIR / f"{TASK}_sliding_{release}"
    subjects_file = out_dir / "subjects.txt"
    if not subjects_file.exists():
        return False
    return len(subjects_file.read_text().strip().splitlines()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Core: sliding windows for one subject
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_windows_one_subject(
    single_sub,
    subject_id: str,
) -> tuple | None:
    """
    Extract sliding windows from all recordings of one subject.

    No event structure needed — just slide across the raw signal.

    Returns (windows_array, meta_dict) or None if no valid windows.
    windows_array : float32 (n_windows, n_chans, n_times)
    meta_dict     : subject, window_start_s, recording_id
    """
    all_windows   = []
    all_starts    = []
    all_rec_ids   = []

    for rec_idx, ds in enumerate(single_sub.datasets):
        try:
            raw = ds.raw
            # Load raw data into memory for this recording
            raw.load_data(verbose=False)

            n_samples = raw.n_times
            n_chans   = len(raw.ch_names)

            if n_samples < MIN_RECORDING_SAMPS:
                print(f"    [skip rec {rec_idx}] too short ({n_samples} samples)")
                raw._data = None
                continue

            # Slide window across the recording
            starts = range(0, n_samples - WINDOW_SAMPS + 1, STRIDE_SAMPS)
            rec_windows = []

            for start in starts:
                stop   = start + WINDOW_SAMPS
                window = raw.get_data(
                    start=start, stop=stop
                ).astype(np.float32)   # (n_chans, n_times)

                # Skip windows with flat/dead channels (artifact)
                if np.all(window == 0) or np.isnan(window).any():
                    continue

                rec_windows.append(window)
                all_starts.append(start / SFREQ)         # in seconds
                all_rec_ids.append(rec_idx)

            # Free raw data immediately after windowing
            raw._data = None
            del raw

            all_windows.extend(rec_windows)
            print(f"    rec {rec_idx}: {len(rec_windows)} windows "
                  f"from {n_samples} samples")

        except Exception as e:
            print(f"    [warn] rec {rec_idx}: {e}")
            continue

    gc.collect()

    if not all_windows:
        print(f"    [skip] {subject_id}: 0 windows extracted")
        return None

    data = np.stack(all_windows, axis=0)   # (N, C, T)
    meta = {
        "subject":        [subject_id] * len(all_windows),
        "window_start_s": all_starts,
        "recording_id":   all_rec_ids,
        # Dummy target field so EEGWindowDataset doesn't complain
        # SSL pretraining ignores this — it's only needed for dataset compat
        "target":         [0.0] * len(all_windows),
    }

    print(f"    ✓ {subject_id}: {len(all_windows)} windows  shape={data.shape}")
    return data, meta


def _save_subject(
    subject_id: str,
    data: np.ndarray,
    meta: dict,
    out_dir: Path,
) -> None:
    sub_dir = out_dir / f"sub-{subject_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.save(sub_dir / "windows.npy", data)
    np.savez(
        sub_dir / "metadata.npz",
        **{k: np.array(v, dtype=object) for k, v in meta.items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-release pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ssl_release(
    release: str,
    overwrite: bool = False,
    subject_batch_size: int = 5,
    delete_raw: bool = True,
) -> dict:
    """
    Download CCD for one release, extract sliding windows, save, delete raw.

    Returns summary dict with done/skipped/failed counts.
    """
    dataset_id = RELEASE_MAP[release]
    out_dir    = WINDOWS_DIR / f"{TASK}_sliding_{release}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Release : {release}  ({dataset_id})")
    print(f"Output  : {out_dir}")
    print(f"Window  : {WINDOW_LEN_S}s  stride {STRIDE_S}s")
    print(f"Free disk: {_get_free_disk(DATA_DIR)}")
    print('='*60)

    # ── [1/3] Get subject list ────────────────────────────────────────────────
    print(f"\n[1/3] Fetching subject list ...")
    try:
        full_ds  = load_eegdash_dataset(
            dataset_id=dataset_id,
            cache_dir=DATA_DIR,
            task=TASK,
        )
        subjects = full_ds.description["subject"].dropna().unique().tolist()
        n_total  = len(subjects)
        print(f"  → {n_total} subjects")
        del full_ds
        gc.collect()
    except Exception as e:
        print(f"  [ERROR] Could not fetch subject list: {e}")
        return {"done": [], "skipped": [], "failed": [], "error": str(e)}

    # ── [2/3] Process subjects one at a time ──────────────────────────────────
    print(f"\n[2/3] Extracting sliding windows ...")
    done, skipped, failed = [], [], []

    for i, subject_id in enumerate(subjects):
        print(f"\n  [{i+1}/{n_total}] {subject_id}")

        if not overwrite and _already_done(subject_id, out_dir):
            print(f"    [skip] already on disk")
            skipped.append(subject_id)
            continue

        try:
            single_sub = load_eegdash_dataset(
                dataset_id=dataset_id,
                cache_dir=DATA_DIR,
                task=TASK,
                subject=subject_id,
            )

            result = _sliding_windows_one_subject(single_sub, subject_id)

            if result is None:
                skipped.append(subject_id)
            else:
                data, meta = result
                _save_subject(subject_id, data, meta, out_dir)
                done.append(subject_id)

            del single_sub, result
            gc.collect()

        except Exception:
            print(f"    [ERROR] {subject_id}")
            traceback.print_exc()
            failed.append(subject_id)
            gc.collect()

        if (i + 1) % subject_batch_size == 0:
            gc.collect()
            print(f"  [gc] after {i+1} subjects")

    # Write subject index
    all_processed = sorted(set(done) | set(skipped))
    (out_dir / "subjects.txt").write_text("\n".join(all_processed))

    if failed:
        (out_dir / "failed_subjects.txt").write_text("\n".join(sorted(failed)))

    # ── [3/3] Delete raw cache ────────────────────────────────────────────────
    if delete_raw:
        print(f"\n[3/3] Cleaning up raw cache ...")
        _delete_raw_cache(dataset_id)
    else:
        print(f"\n[3/3] Keeping raw cache (--keep_raw)")

    print(f"\n  ✓ {len(done)} windowed  ↷ {len(skipped)} skipped  ✗ {len(failed)} failed")
    print(f"  Windowed data: {_get_disk_usage(out_dir)}")
    print(f"  Free disk:     {_get_free_disk(DATA_DIR)}")

    return {"done": done, "skipped": skipped, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# Main — all releases sequentially
# ─────────────────────────────────────────────────────────────────────────────

def main(
    releases: list = None,
    skip: list = None,
    overwrite: bool = False,
    delete_raw: bool = True,
    subject_batch_size: int = 5,
    dry_run: bool = False,
) -> None:

    if releases is None:
        releases = ALL_RELEASES
    if skip:
        releases = [r for r in releases if r not in skip]

    # Hard guard — never window the test release or R7 for SSL
    blocked = [r for r in releases if r in EXCLUDED_RELEASES]
    if blocked:
        print(f"[ERROR] The following releases are excluded from SSL pretraining: {blocked}")
        print(f"  R5 = held-out test release. R7 = no CCD task.")
        print(f"  Remove them from --releases or they will be skipped.")
        releases = [r for r in releases if r not in EXCLUDED_RELEASES]
        if not releases:
            return

    print("=" * 60)
    print("SSL DATA PREPARATION — sliding windows, CCD pretrain releases")
    print("=" * 60)
    print(f"  Pretrain releases : {releases}")
    print(f"  Excluded          : {sorted(EXCLUDED_RELEASES)}  (R5=test, R7=no CCD)")
    print(f"  Window / stride   : {WINDOW_LEN_S}s / {STRIDE_S}s")
    print(f"  Delete raw        : {delete_raw}")
    print(f"  Free disk         : {_get_free_disk(DATA_DIR)}")

    if dry_run:
        print("\n[DRY RUN] Checking which releases are already done:")
        for r in releases:
            done = _is_release_windowed(r)
            out  = WINDOWS_DIR / f"{TASK}_sliding_{r}"
            size = _get_disk_usage(out) if out.exists() else "—"
            print(f"  {r:>3} : {'✓ done' if done else '○ pending'}  ({size})")
        return

    all_results = {}
    for i, release in enumerate(releases, 1):
        if release not in RELEASE_MAP:
            print(f"\n[warn] Unknown release '{release}', skipping")
            continue

        print(f"\n\n[{i}/{len(releases)}] Processing release {release}")

        if not overwrite and _is_release_windowed(release):
            out    = WINDOWS_DIR / f"{TASK}_sliding_{release}"
            n_subj = len((out / "subjects.txt").read_text().strip().splitlines())
            size   = _get_disk_usage(out)
            print(f"  [skip] Already windowed ({n_subj} subjects, {size})")
            # Still clean up raw if it somehow exists
            if delete_raw:
                _delete_raw_cache(RELEASE_MAP[release])
            all_results[release] = "skipped"
            continue

        try:
            result = prepare_ssl_release(
                release=release,
                overwrite=overwrite,
                subject_batch_size=subject_batch_size,
                delete_raw=delete_raw,
            )
            all_results[release] = "done" if not result.get("error") else "failed"
        except Exception:
            print(f"\n[ERROR] Release {release} failed")
            traceback.print_exc()
            all_results[release] = "failed"

        gc.collect()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    total_windows = 0
    total_subjects = 0
    for release, status in all_results.items():
        icon = "✓" if status == "done" else ("↷" if status == "skipped" else "✗")
        out  = WINDOWS_DIR / f"{TASK}_sliding_{release}"
        if out.exists():
            subj_file = out / "subjects.txt"
            n = len(subj_file.read_text().strip().splitlines()) if subj_file.exists() else 0
            size = _get_disk_usage(out)
            total_subjects += n
            print(f"  {icon} {release:>3} : {status:<8}  {n:>4} subjects  {size}")
        else:
            print(f"  {icon} {release:>3} : {status}")

    est_windows = total_subjects * int(
        (5 * 60 - WINDOW_LEN_S) / STRIDE_S
    )
    print(f"\n  Total subjects (across releases) : {total_subjects}")
    print(f"  Estimated total windows (~300/s) : {est_windows:,}")
    print(f"  Final free disk                  : {_get_free_disk(DATA_DIR)}")
    print(f"\n  Next step:")
    print(f"    python ssl_pretrain.py")
    print('='*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all CCD releases and create sliding windows for SSL."
    )
    parser.add_argument(
        "--releases", type=str, default=None,
        help="Comma-separated releases, e.g. R1,R2,R3. Default: all R1-R11."
    )
    parser.add_argument(
        "--skip", type=str, default=None,
        help="Comma-separated releases to skip, e.g. R1,R5."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process releases already on disk."
    )
    parser.add_argument(
        "--keep_raw", action="store_true",
        help="Do NOT delete raw EEG cache after windowing."
    )
    parser.add_argument(
        "--batch_size", type=int, default=5,
        help="GC every N subjects (default: 5). Lower = safer memory."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Check which releases are done without downloading anything."
    )
    args = parser.parse_args()

    releases = args.releases.split(",") if args.releases else None
    skip     = args.skip.split(",")     if args.skip     else None

    main(
        releases=releases,
        skip=skip,
        overwrite=args.overwrite,
        delete_raw=not args.keep_raw,
        subject_batch_size=args.batch_size,
        dry_run=args.dry_run,
    )