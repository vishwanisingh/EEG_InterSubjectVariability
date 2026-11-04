"""
prepare_sl_data.py
------------------
Window SeqLearning (8target and 6target) EEG from already-downloaded raw files.

Event structure (confirmed from inspect):
    dot_noX_ON   — stimulus onset: dot X lights up (~590ms duration)
    dot_noX_OFF  — stimulus offset
    learningBlock_N — block boundary markers
    seqLearning_start / seqLearning_stop — session markers
    No response events — user_answer/correct_answer are always NaN.
    → No RT prediction possible from events alone.

Two windowing modes
-------------------
sliding (for SSL pretraining)
    Slide a 2s window every 1s across the task period only:
    seqLearning_start → seqLearning_stop (avoids long pre-task break ~90s).
    Yields ~150-200 windows/subject.
    Output: data/windows/ssl_sl_{task}_{release}/

event (for transfer evaluation)
    Lock 2s window to each dot_noX_ON event.
    Yields ~50 windows/subject (5 blocks x ~10 dots).
    No RT target (dummy 0.0).
    Output: data/windows/sl_{task}_{release}/

Usage
-----
    # Sliding windows for SSL — all releases, both tasks
    python prepare_sl_data.py --mode sliding --all_releases

    # Event-locked for transfer eval — specific release
    python prepare_sl_data.py --mode event --release R5

    # Both modes
    python prepare_sl_data.py --mode all --release R5

    # Inspect event codes
    python prepare_sl_data.py --mode inspect --release R5
"""

import gc
import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import DATA_DIR, WINDOWS_DIR, SFREQ, RELEASE


# ── Constants ─────────────────────────────────────────────────────────────────
TASKS = ["seqLearning8target", "seqLearning6target"]

RELEASE_MAP = {
    "R1":  "ds005505", "R2":  "ds005506", "R3":  "ds005507",
    "R4":  "ds005508", "R5":  "ds005509", "R6":  "ds005510",
    "R7":  "ds005511", "R8":  "ds005512", "R9":  "ds005514",
    "R10": "ds005515", "R11": "ds005516",
}

# R5  = held-out test release — event windows OK, sliding EXCLUDED
# R7  = no SL task confirmed — skip entirely
SSL_EXCLUDED  = {"R5", "R7"}   # never in sliding (pretraining)
ALL_RELEASES  = list(RELEASE_MAP.keys())
SSL_RELEASES  = [r for r in ALL_RELEASES if r not in SSL_EXCLUDED]

# Windowing params — same as CCD SSL for consistency
WINDOW_LEN_S  = 2.0
STRIDE_S      = 1.0
WINDOW_SAMPS  = int(WINDOW_LEN_S * SFREQ)   # 200 samples at 100 Hz
STRIDE_SAMPS  = int(STRIDE_S * SFREQ)        # 100 samples

# Confirmed event codes from inspect
DOT_ON_EVENTS = {f"dot_no{i}_ON" for i in range(1, 9)}  # dot_no1_ON ... dot_no8_ON
SESSION_START = "seqLearning_start"
SESSION_STOP  = "seqLearning_stop"


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
    np.savez(
        sub_dir / "metadata.npz",
        **{k: np.array(v, dtype=object) for k, v in meta.items()},
    )


def _load_raw_and_events(eeg_path: Path, events_path: Path, subject_id: str):
    """Load raw EEG + events. Returns (raw, events_df) or None."""
    try:
        events_df = pd.read_csv(events_path, sep="\t")
        events_df["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
        events_df = (events_df
                     .dropna(subset=["onset"])
                     .sort_values("onset")
                     .reset_index(drop=True))
    except Exception as e:
        print(f"    [skip] cannot read events: {e}")
        return None

    try:
        raw = mne.io.read_raw(str(eeg_path), preload=True, verbose=False)
    except Exception as e:
        print(f"    [skip] cannot load EEG: {e}")
        return None

    if raw.info["sfreq"] != SFREQ:
        print(f"    [resample] {raw.info['sfreq']} Hz -> {SFREQ} Hz")
        raw.resample(SFREQ, verbose=False)

    return raw, events_df


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: sliding windows (SSL pretraining)
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_windows(
    eeg_path: Path,
    events_path: Path,
    subject_id: str,
) -> tuple | None:
    """
    Slide 2s/1s windows within seqLearning_start -> seqLearning_stop only.
    Avoids the long pre-task break (~90s) that adds noise to SSL pretraining.
    Falls back to full recording if session markers are missing.
    """
    result = _load_raw_and_events(eeg_path, events_path, subject_id)
    if result is None:
        return None
    raw, events_df = result

    n_samples = raw.n_times

    # Find task boundaries
    start_rows = events_df[events_df["value"] == SESSION_START]
    stop_rows  = events_df[events_df["value"] == SESSION_STOP]

    if not start_rows.empty and not stop_rows.empty:
        task_start = int(start_rows.iloc[0]["onset"] * SFREQ)
        task_stop  = int(stop_rows.iloc[-1]["onset"] * SFREQ)
        print(f"    [task] {start_rows.iloc[0]['onset']:.1f}s -> "
              f"{stop_rows.iloc[-1]['onset']:.1f}s")
    else:
        task_start = 0
        task_stop  = n_samples
        print(f"    [warn] No session markers — windowing full recording")

    task_start = max(0, task_start)
    task_stop  = min(n_samples, task_stop)

    if task_stop - task_start < WINDOW_SAMPS:
        print(f"    [skip] task period too short ({task_stop - task_start} samples)")
        del raw
        return None

    all_windows, all_starts = [], []
    for start in range(task_start, task_stop - WINDOW_SAMPS + 1, STRIDE_SAMPS):
        window = raw.get_data(start=start, stop=start + WINDOW_SAMPS).astype(np.float32)
        if np.isnan(window).any() or np.all(window == 0):
            continue
        all_windows.append(window)
        all_starts.append(start / SFREQ)

    del raw
    gc.collect()

    if not all_windows:
        print(f"    [skip] 0 windows extracted")
        return None

    data = np.stack(all_windows)
    meta = {
        "subject":        [subject_id] * len(all_windows),
        "window_start_s": all_starts,
        "target":         [0.0] * len(all_windows),  # dummy for dataset compat
    }
    print(f"    -> {len(all_windows)} sliding windows  shape={data.shape}")
    return data, meta


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: event-locked windows (transfer evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def _event_windows(
    eeg_path: Path,
    events_path: Path,
    subject_id: str,
) -> tuple | None:
    """
    Lock 2s window to each dot_noX_ON event.
    Metadata includes dot_number and block_number for analysis.
    No RT target — dummy 0.0.
    """
    result = _load_raw_and_events(eeg_path, events_path, subject_id)
    if result is None:
        return None
    raw, events_df = result

    n_samples    = raw.n_times
    dot_events   = events_df[events_df["value"].isin(DOT_ON_EVENTS)].copy()
    block_labels = events_df[
        events_df["value"].str.startswith("learningBlock_", na=False)
    ].copy()

    if dot_events.empty:
        print(f"    [skip] no dot_ON events found")
        del raw
        return None

    def _get_block(onset):
        prior = block_labels[block_labels["onset"] <= onset]
        if prior.empty:
            return 0
        try:
            return int(prior.iloc[-1]["value"].split("_")[-1])
        except ValueError:
            return 0

    all_windows, all_dot_nums, all_blocks, all_onsets = [], [], [], []

    for _, row in dot_events.iterrows():
        onset_s   = float(row["onset"])
        start_s   = int(onset_s * SFREQ)
        stop_s    = start_s + WINDOW_SAMPS

        if stop_s > n_samples:
            continue

        window = raw.get_data(start=start_s, stop=stop_s).astype(np.float32)
        if np.isnan(window).any() or np.all(window == 0):
            continue

        try:
            dot_num = int(row["value"].split("_no")[1].split("_")[0])
        except (IndexError, ValueError):
            dot_num = -1

        all_windows.append(window)
        all_dot_nums.append(dot_num)
        all_blocks.append(_get_block(onset_s))
        all_onsets.append(onset_s)

    del raw
    gc.collect()

    if not all_windows:
        print(f"    [skip] 0 event windows extracted")
        return None

    data = np.stack(all_windows)
    meta = {
        "subject":          [subject_id] * len(all_windows),
        "stimulus_onset":   all_onsets,
        "dot_number":       all_dot_nums,
        "block_number":     all_blocks,
        "target":           [0.0] * len(all_windows),
        "rt_from_stimulus": [float("nan")] * len(all_windows),
    }
    print(f"    -> {len(all_windows)} event windows  shape={data.shape}")
    return data, meta


# ─────────────────────────────────────────────────────────────────────────────
# Per-release runner
# ─────────────────────────────────────────────────────────────────────────────

def window_release(
    task: str,
    release: str,
    mode: str,
    overwrite: bool = False,
    subject_batch_size: int = 5,
) -> None:
    dataset_id = RELEASE_MAP[release]
    raw_dir    = DATA_DIR / dataset_id
    # sliding → {task}_sliding_{release}
    # event   → {task}_{release}  (matches supervised pipeline convention)
    suffix  = "_sliding" if mode == "sliding" else ""
    out_dir = WINDOWS_DIR / f"{task}{suffix}_{release}"

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task    : {task}")
    print(f"Release : {release}  ({dataset_id})")
    print(f"Mode    : {mode}")
    print(f"Output  : {out_dir}")
    print('='*60)

    if not raw_dir.exists():
        print(f"  [ERROR] Raw data not found at {raw_dir}")
        return

    eeg_files = sorted(raw_dir.glob(f"sub-*/eeg/sub-*_task-{task}_eeg.set"))
    if not eeg_files:
        print(f"  [warn] No .set files found for {task} in {raw_dir}")
        return

    n_total   = len(eeg_files)
    window_fn = _sliding_windows if mode == "sliding" else _event_windows
    print(f"\n  Found {n_total} EEG files")

    done, skipped, failed = [], [], []

    for i, eeg_path in enumerate(eeg_files):
        subject_id  = eeg_path.parent.parent.name.replace("sub-", "")
        events_path = eeg_path.parent / eeg_path.name.replace("_eeg.set", "_events.tsv")

        print(f"\n  [{i+1}/{n_total}] {subject_id}")

        if not overwrite and _already_done(subject_id, out_dir):
            print(f"    [skip] already on disk")
            skipped.append(subject_id)
            continue

        if not events_path.exists():
            print(f"    [skip] events file missing")
            skipped.append(subject_id)
            continue

        try:
            result = window_fn(eeg_path, events_path, subject_id)
            if result is None:
                skipped.append(subject_id)
            else:
                data, meta = result
                _save_subject(subject_id, data, meta, out_dir)
                done.append(subject_id)
            del result
            gc.collect()
        except Exception:
            print(f"    [ERROR]")
            traceback.print_exc()
            failed.append(subject_id)
            gc.collect()

        if (i + 1) % subject_batch_size == 0:
            gc.collect()

    all_processed = sorted(set(done) | set(skipped))
    (out_dir / "subjects.txt").write_text("\n".join(all_processed))
    if failed:
        (out_dir / "failed_subjects.txt").write_text("\n".join(sorted(failed)))

    print(f"\n[Done] {task} {release} ({mode})")
    print(f"  -> {len(done)} windowed  {len(skipped)} skipped  {len(failed)} failed")
    print(f"  Output: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Inspect
# ─────────────────────────────────────────────────────────────────────────────

def delete_raw_cache(release: str) -> None:
    """Delete the raw EEG cache for a release to free disk space."""
    import shutil
    dataset_id = RELEASE_MAP[release]
    raw_dir    = DATA_DIR / dataset_id
    if raw_dir.exists():
        size = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file())
        print(f"  [cleanup] Deleting {raw_dir}  ({size/1024**3:.2f} GB) ...")
        shutil.rmtree(raw_dir)
        print(f"  [cleanup] Done")
    else:
        print(f"  [cleanup] {raw_dir} not found — already deleted?")


def inspect_events(task: str, release: str) -> None:
    dataset_id   = RELEASE_MAP[release]
    raw_dir      = DATA_DIR / dataset_id
    events_files = sorted(raw_dir.glob(f"sub-*/eeg/sub-*_task-{task}_events.tsv"))

    if not events_files:
        print(f"No events files found for {task} {release} in {raw_dir}")
        return

    print(f"\nInspecting {task} {release} — {len(events_files)} subjects")
    print(f"Showing first 3 subjects:\n")

    for ef in events_files[:3]:
        subject_id = ef.parent.parent.name.replace("sub-", "")
        print(f"  Subject: {subject_id}")
        try:
            df = pd.read_csv(ef, sep="\t")
            print(f"  Columns : {list(df.columns)}")
            print(f"  Rows    : {len(df)}")
            print(f"  Event codes (value column):")
            for val, cnt in df["value"].value_counts().items():
                print(f"    {cnt:>4}x  {val}")
            print(f"\n  First 10 rows:")
            print(df.head(10).to_string(index=False))
        except Exception as e:
            print(f"  Error: {e}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def delete_raw_cache(release: str) -> None:
    """Delete the raw EEG cache for one release to free disk space."""
    import shutil
    dataset_id = RELEASE_MAP[release]
    raw_dir    = DATA_DIR / dataset_id
    if raw_dir.exists():
        total = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file())
        size  = f"{total/1024**3:.2f} GB" if total > 1024**2 else f"{total/1024:.0f} KB"
        print(f"\n  [cleanup] Deleting {raw_dir}  ({size}) ...", flush=True)
        shutil.rmtree(raw_dir)
        print(f"  [cleanup] Done", flush=True)
    else:
        print(f"\n  [cleanup] {raw_dir} not found — already deleted?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="sliding",
        choices=["sliding", "event", "all", "inspect"],
        help=(
            "sliding: 2s/1s windows within task period (SSL). "
            "event: 2s windows locked to dot_ON events (transfer eval). "
            "all: both modes. inspect: print event codes."
        )
    )
    parser.add_argument("--task", type=str, default=None, choices=TASKS)
    parser.add_argument("--release", type=str, default=None,
                        help="Single release e.g. R5. Default: RELEASE from config.")
    parser.add_argument("--all_releases", action="store_true",
                        help="Process all releases R1-R11 sequentially.")
    parser.add_argument("--delete_raw", action="store_true",
                        help="Delete raw EEG cache after windowing each release.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subject_batch_size", type=int, default=5)
    args = parser.parse_args()

    tasks    = TASKS if args.task is None else [args.task]
    releases = (ALL_RELEASES if args.all_releases
                else ([args.release] if args.release else [RELEASE]))

    print(f"Mode       : {args.mode}")
    print(f"Tasks      : {tasks}")
    print(f"Releases   : {releases}")
    print(f"SSL excl.  : {sorted(SSL_EXCLUDED)}  (R5=test, R7=no SL task)")
    print(f"Delete raw : {args.delete_raw}")

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "inspect":
        for task in tasks:
            for release in releases:
                inspect_events(task, release)

    else:
        modes = ["sliding", "event"] if args.mode == "all" else [args.mode]

        for release in releases:
            if release not in RELEASE_MAP:
                print(f"[warn] Unknown release '{release}', skipping")
                continue

            print(f"\n\n{'#'*60}")
            print(f"# Release: {release}  ({RELEASE_MAP[release]})")
            print(f"{'#'*60}")

            for mode in modes:
                # Sliding windows: skip test/excluded releases
                if mode == "sliding" and release in SSL_EXCLUDED:
                    print(f"\n  [skip] {release} excluded from SSL sliding windows")
                    continue

                for task in tasks:
                    try:
                        window_release(
                            task=task,
                            release=release,
                            mode=mode,
                            overwrite=args.overwrite,
                            subject_batch_size=args.subject_batch_size,
                        )
                    except Exception:
                        print(f"\n[ERROR] {task} {release} {mode}")
                        traceback.print_exc()
                    gc.collect()

            # Delete raw for this release before moving to next
            if args.delete_raw:
                delete_raw_cache(release)

            gc.collect()