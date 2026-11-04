"""
prepare_ccd_data.py
-------------------
Download, preprocess, window, and save CCD dataset to disk.

Two windowing modes
-------------------
event (default)
    Stimulus-locked 2s windows with RT labels.
    Output: data/windows/contrastChangeDetection_{release}/
    ~60 windows/subject. Used for supervised training and evaluation.

sliding (for SSL pretraining)
    2s windows every 1s across the full recording. No labels.
    Output: data/windows/contrastChangeDetection_sliding_{release}/
    ~300+ windows/subject. Used for SSL pretraining only.
    R5 (test) and R7 (no CCD task) are excluded from sliding.

Memory strategy: one subject at a time — peak RAM = one subject's raw EEG.

Usage
-----
    # Event-locked (supervised) — default, same as before
    python prepare_ccd_data.py
    python prepare_ccd_data.py --mode event --release R5

    # Sliding windows (SSL) — all non-test releases
    python prepare_ccd_data.py --mode sliding --all_releases --delete_raw

    # Both modes for one release
    python prepare_ccd_data.py --mode all --release R1 --delete_raw

    # All releases, both modes
    python prepare_ccd_data.py --mode all --all_releases --delete_raw
"""

import gc
import sys
import shutil
import argparse
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

from configs import get_task_config
from configs.config import (
    TASK, DATASET_ID, DATA_DIR, WINDOWS_DIR, RESULTS_DIR, SFREQ, RELEASE
)
from src.data.loader import load_eegdash_dataset
from src.utils.plotting import plot_response_time_distribution


# ── Constants ─────────────────────────────────────────────────────────────────
RELEASE_MAP = {
    "R1":  "ds005505", "R2":  "ds005506", "R3":  "ds005507",
    "R4":  "ds005508", "R5":  "ds005509", "R6":  "ds005510",
    "R7":  "ds005511", "R8":  "ds005512", "R9":  "ds005514",
    "R10": "ds005515", "R11": "ds005516",
}

# R5 = held-out test release — event OK, sliding EXCLUDED
# R7 = no CCD task in this release
SSL_EXCLUDED = {"R5", "R7"}
ALL_RELEASES = list(RELEASE_MAP.keys())
SSL_RELEASES = [r for r in ALL_RELEASES if r not in SSL_EXCLUDED]

WINDOW_LEN_S = 2.0
STRIDE_S     = 1.0
WINDOW_SAMPS = int(WINDOW_LEN_S * SFREQ)   # 200 samples
STRIDE_SAMPS = int(STRIDE_S * SFREQ)        # 100 samples
MIN_SAMPS    = int(10.0 * SFREQ)            # skip recordings < 10s

METADATA_KEYS = (
    "target",
    "rt_from_stimulus",
    "rt_from_trialstart",
    "stimulus_onset",
    "response_onset",
    "correct",
    "response_type",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _already_done(subject_id: str, out_dir: Path) -> bool:
    d = out_dir / f"sub-{subject_id}"
    return (d / "windows.npy").exists() and (d / "metadata.npz").exists()


def _save_subject(subject_id: str, data: np.ndarray, meta: dict, out_dir: Path) -> None:
    sub_dir = out_dir / f"sub-{subject_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.save(sub_dir / "windows.npy", data)
    np.savez(sub_dir / "metadata.npz",
             **{k: np.array(v, dtype=object) for k, v in meta.items()})


def _disk_usage(path: Path) -> str:
    if not path.exists():
        return "0 B"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _free_disk(path: Path) -> str:
    free = shutil.disk_usage(path).free
    for unit in ["B", "KB", "MB", "GB"]:
        if free < 1024:
            return f"{free:.1f} {unit}"
        free /= 1024
    return f"{free:.1f} TB"


def _delete_raw(release: str) -> None:
    raw_dir = DATA_DIR / RELEASE_MAP[release]
    if raw_dir.exists():
        size = _disk_usage(raw_dir)
        print(f"\n  [cleanup] Deleting {raw_dir}  ({size}) ...")
        shutil.rmtree(raw_dir)
        print(f"  [cleanup] Done")
    else:
        print(f"\n  [cleanup] {raw_dir} not found — already deleted?")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: event-locked windows (supervised)
# ─────────────────────────────────────────────────────────────────────────────

def _event_windows_one_subject(
    single_rec: BaseConcatDataset,
    cfg: dict,
    subject_id: str,
) -> tuple | None:
    """
    Stimulus-locked 2s windows with RT labels.
    Original windowing logic — unchanged from previous version.
    """
    anchor     = cfg["anchor"]
    window_len = cfg["window_len"]
    shift      = cfg["shift_after_stim"]

    transforms = [
        Preprocessor(
            annotate_trials_with_target,
            target_field=cfg["target_field"],
            epoch_length=window_len,
            require_stimulus=cfg["require_stimulus"],
            require_response=cfg["require_response"],
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(single_rec, transforms, n_jobs=1)

    filtered = keep_only_recordings_with(anchor, single_rec)
    if len(filtered.datasets) == 0:
        print(f"    [skip] {subject_id}: no recordings with anchor '{anchor}'")
        return None

    start_s = int(shift * SFREQ)
    stop_s  = int((shift + window_len) * SFREQ)
    size_s  = int(window_len * SFREQ)

    windows = create_windows_from_events(
        filtered,
        mapping={anchor: 0},
        trial_start_offset_samples=start_s,
        trial_stop_offset_samples=stop_s,
        window_size_samples=size_s,
        window_stride_samples=SFREQ,
        preload=False,
    )
    windows = add_extras_columns(windows, filtered, desc=anchor, keys=METADATA_KEYS)

    n = len(windows)
    if n == 0:
        print(f"    [skip] {subject_id}: 0 windows after windowing")
        return None

    n_chans = windows[0][0].shape[0]
    data    = np.empty((n, n_chans, size_s), dtype=np.float32)
    for i in range(n):
        x, _, _ = windows[i]
        data[i] = x.numpy().astype(np.float32) if hasattr(x, 'numpy') else x.astype(np.float32)

    meta_df   = windows.get_metadata()
    meta_dict = {}
    for key in METADATA_KEYS:
        meta_dict[key] = meta_df[key].tolist() if key in meta_df.columns else [None] * n
    meta_dict["subject"] = [subject_id] * n

    print(f"    ✓ {subject_id}: {n} event windows  shape={data.shape}")
    return data, meta_dict


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: sliding windows (SSL pretraining)
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_windows_one_subject(
    single_rec: BaseConcatDataset,
    subject_id: str,
) -> tuple | None:
    """
    2s/1s sliding windows across the full CCD recording.

    CCD inter-trial intervals contain useful baseline EEG (fixation,
    anticipatory attention) so we window the entire recording — no clipping.
    No RT labels — dummy target=0.0 for dataset compatibility.
    """
    all_windows, all_starts, all_rec_ids = [], [], []

    for rec_idx, ds in enumerate(single_rec.datasets):
        try:
            raw = ds.raw
            raw.load_data(verbose=False)

            n_samples = raw.n_times
            if n_samples < MIN_SAMPS:
                print(f"    [skip rec {rec_idx}] too short ({n_samples} samples)")
                raw._data = None
                continue

            print(f"    rec {rec_idx}: {n_samples/SFREQ:.0f}s recording")

            rec_windows = []
            for start in range(0, n_samples - WINDOW_SAMPS + 1, STRIDE_SAMPS):
                window = raw.get_data(
                    start=start, stop=start + WINDOW_SAMPS
                ).astype(np.float32)
                if np.isnan(window).any() or np.all(window == 0):
                    continue
                rec_windows.append(window)
                all_starts.append(start / SFREQ)
                all_rec_ids.append(rec_idx)

            raw._data = None
            del raw

            all_windows.extend(rec_windows)
            print(f"    rec {rec_idx}: {len(rec_windows)} sliding windows")

        except Exception as e:
            print(f"    [warn] rec {rec_idx}: {e}")
            continue

    gc.collect()

    if not all_windows:
        print(f"    [skip] {subject_id}: 0 windows extracted")
        return None

    data = np.stack(all_windows, axis=0)
    meta = {
        "subject":        [subject_id] * len(all_windows),
        "window_start_s": all_starts,
        "recording_id":   all_rec_ids,
        "target":         [0.0] * len(all_windows),  # dummy for dataset compat
    }
    print(f"    ✓ {subject_id}: {len(all_windows)} sliding windows  shape={data.shape}")
    return data, meta


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    dataset_id: str = DATASET_ID,
    task: str = TASK,
    release: str = RELEASE,
    mode: str = "event",
    overwrite: bool = False,
    subject_batch_size: int = 5,
    delete_raw: bool = False,
) -> None:
    """
    Download -> window -> save, one subject at a time.

    Parameters
    ----------
    dataset_id : str
        OpenNeuro dataset ID (e.g. "ds005509").
    task : str
        Task name — must exist in configs/task_configs.py.
    release : str
        Release label used as folder suffix.
    mode : str
        "event"   — stimulus-locked windows with RT labels (supervised).
        "sliding" — full-recording sliding windows, no labels (SSL).
    overwrite : bool
        Re-process subjects already on disk.
    subject_batch_size : int
        Run gc.collect() every N subjects.
    delete_raw : bool
        Delete raw EEG cache after windowing.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix  = "_sliding" if mode == "sliding" else ""
    out_dir = WINDOWS_DIR / f"{task}{suffix}_{release}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_task_config(task)

    print(f"\n{'='*60}")
    print(f"Task    : {task}")
    print(f"Release : {release}  ({dataset_id})")
    print(f"Mode    : {mode}")
    print(f"Output  : {out_dir}")
    print(f"Free    : {_free_disk(DATA_DIR)}")
    print('='*60)

    # ── [1/3] Get subject list ────────────────────────────────────────────────
    print(f"\n[1/3] Fetching subject list ...")
    full_dataset = load_eegdash_dataset(
        dataset_id=dataset_id,
        cache_dir=DATA_DIR,
        task=task,
    )
    print(f"    Total recordings: {len(full_dataset.datasets)}")
    subjects = full_dataset.description["subject"].dropna().unique().tolist()
    n_total  = len(subjects)
    print(f"  -> {n_total} subjects found")
    del full_dataset
    gc.collect()

    # ── [2/3] Process subjects ────────────────────────────────────────────────
    print(f"\n[2/3] Processing subjects  (mode={mode}, batch={subject_batch_size})")
    done, skipped, failed = [], [], []
    all_meta = []

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
                task=task,
                subject=subject_id,
            )

            if mode == "event":
                result = _event_windows_one_subject(single_sub, cfg, subject_id)
            else:
                result = _sliding_windows_one_subject(single_sub, subject_id)

            if result is None:
                skipped.append(subject_id)
            else:
                data, meta = result
                _save_subject(subject_id, data, meta, out_dir)
                if mode == "event":
                    all_meta.append(meta)
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

    # ── Write subject index ───────────────────────────────────────────────────
    all_processed = sorted(set(done) | set(skipped))
    (out_dir / "subjects.txt").write_text("\n".join(all_processed))

    # ── [3/3] Summary ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Done.")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  ✓ Processed : {len(done)}")
    print(f"  ↷ Skipped   : {len(skipped)}")
    print(f"  ✗ Failed    : {len(failed)}")
    if failed:
        print(f"    -> {failed}")
    print(f"  Saved to    : {out_dir}  ({_disk_usage(out_dir)})")
    print('='*60)

    if mode == "event" and cfg["task_type"] == "active" and all_meta:
        import pandas as pd
        combined = pd.DataFrame({
            k: sum([m[k] for m in all_meta], []) for k in METADATA_KEYS
        })
        plot_response_time_distribution(
            combined,
            RESULTS_DIR / f"{task}_{release}_rt_distribution.png",
        )

    if delete_raw:
        _delete_raw(release)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: download only (no windowing)
# ─────────────────────────────────────────────────────────────────────────────

def download_only(
    dataset_id: str,
    task: str = TASK,
    subject_batch_size: int = 5,
) -> None:
    """
    Download raw EEG files for all subjects without windowing.

    Useful when you want to download overnight and window separately,
    or when you want to inspect the raw files before committing to windowing.

    For CCD, eegdash downloads the .set file on first load_data() call.
    We trigger that by calling load_data() then immediately freeing RAM.
    """
    print(f"\n[1/2] Fetching subject list  dataset={dataset_id}  task={task}")
    full_ds  = load_eegdash_dataset(dataset_id=dataset_id, cache_dir=DATA_DIR, task=task)
    subjects = full_ds.description["subject"].dropna().unique().tolist()
    n_total  = len(subjects)
    print(f"  -> {n_total} subjects")
    del full_ds
    gc.collect()

    print(f"\n[2/2] Downloading raw files ...")
    done, failed = [], []

    for i, subject_id in enumerate(subjects):
        print(f"\n  [{i+1}/{n_total}] {subject_id}", flush=True)
        try:
            single = load_eegdash_dataset(
                dataset_id=dataset_id,
                cache_dir=DATA_DIR,
                task=task,
                subject=subject_id,
            )
            for ds in single.datasets:
                ds.raw.load_data(verbose=False)
                ds.raw._data = None   # free RAM immediately after download
            del single
            gc.collect()
            done.append(subject_id)
            print(f"    ✓ downloaded", flush=True)
        except Exception as e:
            failed.append(subject_id)
            print(f"    ✗ {type(e).__name__}: {e}", flush=True)
            gc.collect()

        if (i + 1) % subject_batch_size == 0:
            gc.collect()

    print(f"\n  ✓ {len(done)} downloaded  ✗ {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: inspect (print event/annotation structure)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_release(dataset_id: str, task: str = TASK, n_subjects: int = 3) -> None:
    """
    Print MNE annotation structure for the first N subjects of a release.

    CCD uses MNE annotations (added by eegdash) rather than a separate
    events.tsv — this shows you what anchors exist and the task period
    timing before you commit to windowing.
    """
    print(f"\n[1/2] Fetching subject list  dataset={dataset_id}  task={task}")
    full_ds  = load_eegdash_dataset(dataset_id=dataset_id, cache_dir=DATA_DIR, task=task)
    subjects = full_ds.description["subject"].dropna().unique().tolist()
    print(f"  -> {len(subjects)} subjects total. Showing first {n_subjects}.")
    del full_ds
    gc.collect()

    print(f"\n[2/2] Inspecting annotations ...")

    for subject_id in subjects[:n_subjects]:
        print(f"\n  {'='*50}")
        print(f"  Subject: {subject_id}")
        try:
            single = load_eegdash_dataset(
                dataset_id=dataset_id,
                cache_dir=DATA_DIR,
                task=task,
                subject=subject_id,
            )
            for rec_idx, ds in enumerate(single.datasets):
                raw = ds.raw
                raw.load_data(verbose=False)

                print(f"\n  Recording {rec_idx}:")
                print(f"    Duration  : {raw.n_times / raw.info['sfreq']:.1f}s  "
                      f"({raw.n_times} samples @ {raw.info['sfreq']} Hz)")
                print(f"    Channels  : {len(raw.ch_names)}")

                if raw.annotations and len(raw.annotations) > 0:
                    ann = raw.annotations
                    print(f"    Annotations: {len(ann)} total")
                    print(f"    Unique descriptions:")
                    from collections import Counter
                    counts = Counter(ann.description)
                    for desc, cnt in sorted(counts.items()):
                        print(f"      {cnt:>4}x  {desc}")
                    print(f"    Time range: {ann.onset.min():.2f}s → "
                          f"{(ann.onset + ann.duration).max():.2f}s")
                    print(f"    First 5 annotations:")
                    for j in range(min(5, len(ann))):
                        print(f"      t={ann.onset[j]:.3f}s  "
                              f"dur={ann.duration[j]:.3f}s  "
                              f"desc='{ann.description[j]}'")
                else:
                    print(f"    Annotations: none found")

                raw._data = None
                del raw

            del single
            gc.collect()

        except Exception as e:
            print(f"  [ERROR] {subject_id}: {e}")
            traceback.print_exc()
            gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and window CCD EEG data."
    )
    parser.add_argument(
        "--mode", type=str, default="event",
        choices=["event", "sliding", "all", "download", "inspect"],
        help=(
            "event    : stimulus-locked windows with RT labels (supervised). "
            "sliding  : full-recording sliding windows, no labels (SSL). "
            "all      : both event and sliding. "
            "download : download raw files only, no windowing. "
            "inspect  : print annotation structure for first N subjects."
        )
    )
    parser.add_argument(
        "--release", type=str, default=RELEASE,
        help=f"Single release e.g. R5 (default: {RELEASE})."
    )
    parser.add_argument(
        "--all_releases", action="store_true",
        help="Process all releases sequentially."
    )
    parser.add_argument(
        "--delete_raw", action="store_true",
        help="Delete raw EEG cache after windowing each release."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process subjects already on disk."
    )
    parser.add_argument(
        "--batch_size", type=int, default=5,
        help="GC every N subjects (default: 5)."
    )
    parser.add_argument(
        "--n_inspect", type=int, default=3,
        help="Number of subjects to inspect (inspect mode only, default: 3)."
    )
    args = parser.parse_args()

    releases = ALL_RELEASES if args.all_releases else [args.release]

    print(f"Mode     : {args.mode}")
    print(f"Releases : {releases}")
    if args.mode not in ("download", "inspect"):
        print(f"Excluded from sliding: {sorted(SSL_EXCLUDED)}  (R5=test, R7=no CCD)")

    # ── Inspect: just show annotation structure, no downloading ──────────────
    if args.mode == "inspect":
        for release in releases:
            if release not in RELEASE_MAP:
                print(f"[warn] Unknown release '{release}', skipping")
                continue
            print(f"\n{'#'*60}")
            print(f"# Inspect: {release}  ({RELEASE_MAP[release]})")
            print(f"{'#'*60}")
            inspect_release(
                dataset_id=RELEASE_MAP[release],
                task=TASK,
                n_subjects=args.n_inspect,
            )

    # ── Download only: fetch raw files, no windowing ─────────────────────────
    elif args.mode == "download":
        for release in releases:
            if release not in RELEASE_MAP:
                print(f"[warn] Unknown release '{release}', skipping")
                continue
            print(f"\n{'#'*60}")
            print(f"# Download: {release}  ({RELEASE_MAP[release]})")
            print(f"{'#'*60}")
            download_only(
                dataset_id=RELEASE_MAP[release],
                task=TASK,
                subject_batch_size=args.batch_size,
            )

    # ── Event / sliding / all: download + window ──────────────────────────────
    else:
        modes = ["event", "sliding"] if args.mode == "all" else [args.mode]

        for release in releases:
            if release not in RELEASE_MAP:
                print(f"\n[warn] Unknown release '{release}', skipping")
                continue

            dataset_id = RELEASE_MAP[release]

            print(f"\n{'#'*60}")
            print(f"# Release: {release}  ({dataset_id})")
            print(f"{'#'*60}")

            for mode in modes:
                if mode == "sliding" and release in SSL_EXCLUDED:
                    print(f"\n[skip] {release} excluded from SSL sliding windows")
                    continue

                try:
                    prepare_data(
                        dataset_id=dataset_id,
                        task=TASK,
                        release=release,
                        mode=mode,
                        overwrite=args.overwrite,
                        subject_batch_size=args.batch_size,
                        delete_raw=False,  # delete after ALL modes done
                    )
                except Exception:
                    print(f"\n[ERROR] {TASK} {release} {mode}")
                    traceback.print_exc()
                gc.collect()

            # Delete raw once after all modes are done for this release
            if args.delete_raw:
                _delete_raw(release)