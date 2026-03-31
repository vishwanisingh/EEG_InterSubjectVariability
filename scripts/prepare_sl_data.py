"""
scripts/prepare_sl_data.py
--------------------------
Download, window, and save SeqLearning (8target and 6target) data to disk.

Two modes:
  --mode download  : download raw EEG files only (no windowing)
  --mode window    : window already-downloaded files (no downloading)
  --mode all       : download then window (default)

SeqLearning cannot use eegdash's built-in annotate_trials_with_target
because that function is hardcoded for contrastChangeDetection event codes.
This script uses custom windowing based on the actual SeqLearning event
structure discovered from the raw events.tsv files.

On-disk output (identical layout to prepare_ccd_data.py):
    data/windows/{task}_{release}/
        subjects.txt
        sub-{subject_id}/
            windows.npy       float32  (n_windows, n_chans, n_times)
            metadata.npz      target, rt_from_stimulus, correct,
                              stimulus_onset, response_onset, subject

Usage
-----
    # Download all releases for both tasks
    python scripts/prepare_sl_data.py --mode download

    # Download specific task and release
    python scripts/prepare_sl_data.py --mode download --task seqLearning8target --release R5

    # Window after download (inspect events first to confirm structure)
    python scripts/prepare_sl_data.py --mode window --release R5

    # Download + window in one shot
    python scripts/prepare_sl_data.py --mode all --release R5

    # All releases, both tasks, download only
    python scripts/prepare_sl_data.py --mode download --all_releases
"""

import gc
import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import DATA_DIR, WINDOWS_DIR, RESULTS_DIR, SFREQ, RELEASE
from src.data.loader import load_eegdash_dataset


# ── Constants ─────────────────────────────────────────────────────────────────
TASKS = ["seqLearning8target", "seqLearning6target"]

RELEASE_MAP = {
    "R1":  "ds005505", "R2":  "ds005506", "R3":  "ds005507",
    "R4":  "ds005508", "R5":  "ds005509", "R6":  "ds005510",
    "R7":  "ds005511", "R8":  "ds005512", "R9":  "ds005514",
    "R10": "ds005515", "R11": "ds005516",
}
ALL_RELEASES = list(RELEASE_MAP.keys())

WINDOW_LEN  = 2.0                        # seconds (same as CCD)
MIN_RT      = 0.05                       # minimum plausible RT in seconds
MAX_RT      = 10.0                       # maximum plausible RT in seconds

METADATA_KEYS = (
    "target",           # RT in seconds — regression target
    "rt_from_stimulus", # same as target, explicit name
    "correct",          # bool: correct response
    "stimulus_onset",   # absolute stimulus onset in seconds
    "response_onset",   # absolute response onset in seconds
)


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_task_release(
    task: str,
    release: str,
    dataset_id: str,
) -> bool:
    """
    Download all subjects for one task/release combination.
    Downloads subject-by-subject to trigger actual EEG file download.
    Returns True if successful, False if task not available in this release.
    """
    print(f"\n  Downloading {task} {release} ({dataset_id}) ...", flush=True)

    # Step 1: get subject list
    try:
        full_ds  = load_eegdash_dataset(
            dataset_id=dataset_id,
            cache_dir=DATA_DIR,
            task=task,
        )
        subjects = full_ds.description["subject"].dropna().unique().tolist()
        n_total  = len(subjects)
        print(f"  Found {n_total} subjects — downloading EEG files ...",
              flush=True)
        del full_ds
        gc.collect()
    except Exception as e:
        print(f"  ✗ {release}: task not available — {type(e).__name__}: {e}",
              flush=True)
        return False

    # Step 2: download each subject individually by accessing raw EEG data
    # Simply calling load_eegdash_dataset only queries metadata.
    # We must access ds.raw.load_data() to trigger actual file download.
    done = 0
    failed = 0
    for i, subject_id in enumerate(subjects):
        print(f"  [{i+1}/{n_total}] {subject_id}", flush=True)
        try:
            single = load_eegdash_dataset(
                dataset_id=dataset_id,
                cache_dir=DATA_DIR,
                task=task,
                subject=subject_id,
            )
            # Access raw data to trigger download of .set file
            for ds in single.datasets:
                ds.raw.load_data(verbose=False)
                ds.raw._data = None   # free RAM immediately after download
            del single
            gc.collect()
            done += 1
            print(f"    ✓ downloaded", flush=True)
        except Exception as e:
            failed += 1
            print(f"    ✗ {type(e).__name__}: {e}", flush=True)
            gc.collect()

        if (i + 1) % 10 == 0:
            gc.collect()

    print(f"\n  {release} done: {done}/{n_total} downloaded "
          f"({failed} failed)", flush=True)
    return True


def download_all(
    tasks: list,
    releases: list,
) -> dict:
    """
    Download all task/release combinations.
    Returns summary dict: {task: {release: success_bool}}
    """
    summary = {task: {} for task in tasks}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Downloading: {task}")
        print('='*60)

        for release in releases:
            dataset_id = RELEASE_MAP[release]
            success = download_task_release(task, release, dataset_id)
            summary[task][release] = success

    # Print download summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    for task in tasks:
        succeeded = [r for r, ok in summary[task].items() if ok]
        failed    = [r for r, ok in summary[task].items() if not ok]
        print(f"\n  {task}:")
        print(f"    ✓ {len(succeeded)} releases: {succeeded}")
        if failed:
            print(f"    ✗ {len(failed)} not available: {failed}")
    print('='*60)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Trial table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_trial_table_sl(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a trial table from a SeqLearning events file.

    SeqLearning trial structure (determined from events.tsv inspection):
    - Stimulus event: marks when a circle flashes on screen
    - Response event: marks when subject presses button
    - RT = response onset - stimulus onset

    NOTE: This function will be updated once the actual event codes are
    confirmed from the downloaded files. The current implementation uses
    common SeqLearning event code patterns from the HBN dataset.
    Run --mode inspect after downloading to verify event codes.

    Parameters
    ----------
    events_df : pd.DataFrame
        Contents of _events.tsv file.

    Returns
    -------
    pd.DataFrame with columns:
        stimulus_onset, response_onset, rt_from_stimulus, correct, target
    Returns empty DataFrame if no valid trials found.
    """
    df = events_df.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

    # Print unique event codes for inspection on first call
    if hasattr(build_trial_table_sl, "_printed_codes"):
        pass
    else:
        build_trial_table_sl._printed_codes = True
        unique_vals = df["value"].value_counts()
        print(f"    [event codes] {dict(unique_vals)}", flush=True)

    # ── Detect event structure automatically ──────────────────────────────────
    # Common SeqLearning patterns in HBN dataset:
    #   stimulus: 'target', 'stimulus', 'circle_on', or numeric codes
    #   response: 'response', 'button_press', 'keypress'
    # We try multiple patterns and use whichever matches.

    all_values = set(df["value"].dropna().unique())

    # Pattern 1: explicit stimulus/response labels
    stimulus_events = df[df["value"].isin([
        "target", "stimulus", "circle_on", "seq_target",
        "seqLearning_target", "target_on",
    ])]
    response_events = df[df["value"].isin([
        "response", "button_press", "keypress", "correct_response",
        "incorrect_response", "seq_response",
    ])]

    # Pattern 2: if no named events, look for trial_start + first response
    if stimulus_events.empty or response_events.empty:
        trial_starts = df[df["value"].isin([
            "trial_start", "seq_trial_start", "seqLearning_start",
            "new_trial",
        ])]
        all_responses = df[df["value"].isin([
            "response", "button_press", "keypress",
            "correct_response", "incorrect_response",
        ])]

        if not trial_starts.empty and not all_responses.empty:
            stimulus_events = trial_starts
            response_events = all_responses

    if stimulus_events.empty:
        print(f"    [warn] No stimulus events found. "
              f"Available values: {list(all_values)[:10]}", flush=True)
        return pd.DataFrame()

    if response_events.empty:
        print(f"    [warn] No response events found. "
              f"Available values: {list(all_values)[:10]}", flush=True)
        return pd.DataFrame()

    # ── Build trial table ─────────────────────────────────────────────────────
    rows = []
    stim_list = stimulus_events.reset_index(drop=True)
    resp_list = response_events.copy()

    for i, stim in stim_list.iterrows():
        stim_onset = float(stim["onset"])

        # Next stimulus onset defines boundary
        next_stim = (
            float(stim_list.iloc[i + 1]["onset"])
            if i + 1 < len(stim_list)
            else stim_onset + MAX_RT
        )

        # First response after this stimulus
        resp_window = resp_list[
            (resp_list["onset"] > stim_onset) &
            (resp_list["onset"] < next_stim)
        ]

        if resp_window.empty:
            continue

        first_resp   = resp_window.iloc[0]
        resp_onset   = float(first_resp["onset"])
        rt           = resp_onset - stim_onset

        if rt < MIN_RT or rt > MAX_RT:
            continue

        # Correctness if available
        correct = None
        if "correct" in first_resp.index:
            try:
                correct = bool(int(first_resp["correct"]))
            except (ValueError, TypeError):
                pass
        elif "feedback" in first_resp.index:
            feedback = str(first_resp.get("feedback", ""))
            if feedback == "correct":
                correct = True
            elif feedback == "incorrect":
                correct = False

        rows.append({
            "stimulus_onset":   stim_onset,
            "response_onset":   resp_onset,
            "rt_from_stimulus": rt,
            "correct":          correct,
            "target":           rt,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject windowing
# ─────────────────────────────────────────────────────────────────────────────

def _window_one_subject(
    eeg_path: Path,
    events_path: Path,
    subject_id: str,
) -> tuple | None:
    """
    Extract windows for one subject from local files.
    Resamples to SFREQ (100 Hz) if needed.

    Returns (windows_array, meta_dict) or None.
    """
    import mne

    # ── Load events ───────────────────────────────────────────────────────────
    try:
        events_df = pd.read_csv(events_path, sep="\t")
    except Exception as e:
        print(f"    [skip] {subject_id}: cannot read events — {e}", flush=True)
        return None

    trials = build_trial_table_sl(events_df)
    if trials.empty:
        print(f"    [skip] {subject_id}: no valid trials", flush=True)
        return None

    print(f"    [info] {len(trials)} valid trials", flush=True)

    # ── Load EEG ──────────────────────────────────────────────────────────────
    try:
        raw = mne.io.read_raw(str(eeg_path), preload=True, verbose=False)
    except Exception as e:
        print(f"    [skip] {subject_id}: cannot load EEG — {e}", flush=True)
        return None

    # Resample to 100 Hz to match CCD
    if raw.info["sfreq"] != SFREQ:
        print(f"    [info] Resampling {raw.info['sfreq']} Hz → {SFREQ} Hz",
              flush=True)
        raw.resample(SFREQ, verbose=False)

    sfreq       = raw.info["sfreq"]
    n_samples   = raw.n_times
    win_samples = int(WINDOW_LEN * sfreq)
    n_chans     = len(raw.ch_names)

    print(f"    [info] EEG: {n_chans} ch × {n_samples} samples @ {sfreq} Hz",
          flush=True)

    # ── Extract windows ───────────────────────────────────────────────────────
    all_windows = []
    all_meta    = {k: [] for k in METADATA_KEYS}
    all_meta["subject"] = []

    for _, trial in trials.iterrows():
        start_samp = int(trial["stimulus_onset"] * sfreq)
        stop_samp  = start_samp + win_samples

        if stop_samp > n_samples:
            continue

        window = raw.get_data(
            start=start_samp, stop=stop_samp
        ).astype(np.float32)

        all_windows.append(window)
        all_meta["target"].append(float(trial["target"]))
        all_meta["rt_from_stimulus"].append(float(trial["rt_from_stimulus"]))
        all_meta["correct"].append(trial["correct"])
        all_meta["stimulus_onset"].append(float(trial["stimulus_onset"]))
        all_meta["response_onset"].append(float(trial["response_onset"]))
        all_meta["subject"].append(subject_id)

    del raw
    gc.collect()

    if not all_windows:
        print(f"    [skip] {subject_id}: 0 windows extracted", flush=True)
        return None

    data = np.stack(all_windows, axis=0)
    print(f"    ✓ {subject_id}: {len(all_windows)} windows  shape={data.shape}",
          flush=True)
    return data, all_meta


# ─────────────────────────────────────────────────────────────────────────────
# Save / check helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _already_done(subject_id: str, out_dir: Path) -> bool:
    d = out_dir / f"sub-{subject_id}"
    return (d / "windows.npy").exists() and (d / "metadata.npz").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Inspect events (helper to verify event codes before windowing)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_events(task: str, release: str) -> None:
    """
    Print event code summary for one task/release to verify structure
    before committing to full windowing run.
    """
    dataset_id = RELEASE_MAP[release]
    raw_dir    = DATA_DIR / dataset_id

    events_files = sorted(raw_dir.glob(
        f"sub-*/eeg/sub-*_task-{task}_events.tsv"
    ))

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
            counts = df["value"].value_counts()
            for val, cnt in counts.items():
                print(f"    {cnt:>4}x  {val}")
            print(f"\n  First 10 rows:")
            print(df.head(10).to_string(index=False))
        except Exception as e:
            print(f"  Error reading: {e}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Per-release windowing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def window_task_release(
    task: str,
    release: str,
    overwrite: bool = False,
    subject_batch_size: int = 5,
) -> None:
    """
    Window all subjects for one task/release from local files.
    """
    dataset_id = RELEASE_MAP[release]
    raw_dir    = DATA_DIR / dataset_id
    out_dir    = WINDOWS_DIR / f"{task}_{release}"

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task    : {task}")
    print(f"Release : {release}  ({dataset_id})")
    print(f"Source  : {raw_dir}")
    print(f"Output  : {out_dir}")
    print('='*60)

    if not raw_dir.exists():
        print(f"  [ERROR] Raw data directory not found: {raw_dir}")
        print(f"  Run: python scripts/prepare_sl_data.py --mode download "
              f"--task {task} --release {release}")
        return

    # Discover EEG files
    eeg_files = sorted(raw_dir.glob(f"sub-*/eeg/sub-*_task-{task}_eeg.set"))

    if not eeg_files:
        print(f"  [warn] No .set files found for {task} in {raw_dir}")
        print(f"  The task may not be available in this release, "
              f"or data was not downloaded.")
        return

    n_total = len(eeg_files)
    print(f"\n  Found {n_total} EEG files")

    # Reset event code print flag per release
    if hasattr(build_trial_table_sl, "_printed_codes"):
        del build_trial_table_sl._printed_codes

    done, skipped, failed = [], [], []

    for i, eeg_path in enumerate(eeg_files):
        subject_id  = eeg_path.parent.parent.name.replace("sub-", "")
        events_path = eeg_path.parent / eeg_path.name.replace(
            "_eeg.set", "_events.tsv"
        )

        print(f"\n  [{i+1}/{n_total}] {subject_id}", flush=True)

        if not overwrite and _already_done(subject_id, out_dir):
            print(f"    [skip] already on disk", flush=True)
            skipped.append(subject_id)
            continue

        if not events_path.exists():
            print(f"    [skip] events file missing", flush=True)
            skipped.append(subject_id)
            continue

        try:
            result = _window_one_subject(eeg_path, events_path, subject_id)

            if result is None:
                skipped.append(subject_id)
            else:
                data, meta = result
                _save_subject(subject_id, data, meta, out_dir)
                done.append(subject_id)

            del result
            gc.collect()

        except Exception:
            print(f"    [ERROR] {subject_id}", flush=True)
            traceback.print_exc()
            failed.append(subject_id)
            gc.collect()

        if (i + 1) % subject_batch_size == 0:
            gc.collect()
            print(f"  [gc] after {i+1} subjects", flush=True)

    # Write subject index
    all_processed = sorted(set(done) | set(skipped))
    (out_dir / "subjects.txt").write_text("\n".join(all_processed))

    if failed:
        (out_dir / "failed_subjects.txt").write_text(
            "\n".join(sorted(failed))
        )

    print(f"\n[Done] {task} {release}")
    print(f"  ✓ Windowed : {len(done)}")
    print(f"  ↷ Skipped  : {len(skipped)}")
    print(f"  ✗ Failed   : {len(failed)}")
    print(f"  Output     : {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and/or window SeqLearning EEG data."
    )
    parser.add_argument(
        "--mode", type=str, default="download",
        choices=["download", "window", "all", "inspect"],
        help=(
            "'download' = download raw files only. "
            "'window'   = window already-downloaded files. "
            "'all'      = download then window. "
            "'inspect'  = print event codes to verify structure before windowing."
        )
    )
    parser.add_argument(
        "--task", type=str, default=None,
        choices=TASKS + [None],
        help=f"Which task to process. Default: both {TASKS}."
    )
    parser.add_argument(
        "--release", type=str, default=None,
        help=f"Single release, e.g. R5. Ignored if --all_releases is set."
    )
    parser.add_argument(
        "--all_releases", action="store_true",
        help="Process all releases R1-R11."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process subjects already saved to disk (window mode only)."
    )
    parser.add_argument(
        "--subject_batch_size", type=int, default=5,
        help="Run gc.collect() every N subjects (default: 5)."
    )
    args = parser.parse_args()

    # Resolve tasks
    tasks    = TASKS if args.task is None else [args.task]

    # Resolve releases
    if args.all_releases:
        releases = ALL_RELEASES
    elif args.release:
        if args.release not in RELEASE_MAP:
            print(f"[ERROR] Unknown release '{args.release}'. "
                  f"Available: {ALL_RELEASES}")
        releases = [args.release]
    else:
        releases = [RELEASE]   # default from config

    print(f"\nMode     : {args.mode}")
    print(f"Tasks    : {tasks}")
    print(f"Releases : {releases}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "inspect":
        for task in tasks:
            for release in releases:
                inspect_events(task, release)

    elif args.mode == "download":
        download_all(tasks, releases)

    elif args.mode == "window":
        for task in tasks:
            for release in releases:
                try:
                    window_task_release(
                        task=task,
                        release=release,
                        overwrite=args.overwrite,
                        subject_batch_size=args.subject_batch_size,
                    )
                except Exception:
                    print(f"\n[ERROR] {task} {release}")
                    traceback.print_exc()

    elif args.mode == "all":
        download_all(tasks, releases)
        for task in tasks:
            for release in releases:
                try:
                    window_task_release(
                        task=task,
                        release=release,
                        overwrite=args.overwrite,
                        subject_batch_size=args.subject_batch_size,
                    )
                except Exception:
                    print(f"\n[ERROR] {task} {release}")
                    traceback.print_exc()