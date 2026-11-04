"""
scripts/prepare_ss_data.py
--------------------------
Create windowed dataset for SymbolSearch from already-downloaded raw files.

NO downloading — assumes raw EEG files are already on disk at:
    data/{dataset_id}/sub-{subject_id}/eeg/sub-{subject_id}_task-symbolSearch_eeg.set
    data/{dataset_id}/sub-{subject_id}/eeg/sub-{subject_id}_task-symbolSearch_events.tsv

SymbolSearch event structure:
    newPage       → stimulus onset (new page of symbols displayed)
    trialResponse → subject response per symbol row
                    user_answer=1: subject pressed "found target"
                    correct_answer=1: target was actually present

Trial definition:
    One trial  = one newPage event
    RT (target) = time from newPage → first trialResponse
    Correct     = user_answer == correct_answer on first response

Window:
    2 seconds locked to newPage onset (same length as CCD)

On-disk output (identical layout to prepare_ccd_data.py):
    data/windows/symbolSearch_{release}/
        subjects.txt
        sub-{subject_id}/
            windows.npy       float32  (n_windows, n_chans, n_times)
            metadata.npz      target, rt_from_page, correct, n_responses,
                              page_onset, subject

Usage
-----
    # Download only
    python scripts/prepare_ss_data.py --mode download --release R5
    python scripts/prepare_ss_data.py --mode download --all_releases

    # Window only (assumes raw files already downloaded)
    python scripts/prepare_ss_data.py --mode window --release R5
    python scripts/prepare_ss_data.py --mode window --all_releases

    # Download then window
    python scripts/prepare_ss_data.py --mode all --release R5
    python scripts/prepare_ss_data.py --mode all --all_releases

    # Overwrite existing windows
    python scripts/prepare_ss_data.py --mode window --release R5 --overwrite
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

from configs.config import DATA_DIR, WINDOWS_DIR, RESULTS_DIR, SFREQ, RELEASE


# ── Constants ─────────────────────────────────────────────────────────────────
TASK = "symbolSearch"

RELEASE_MAP = {
    "R1":  "ds005505", "R2":  "ds005506", "R3":  "ds005507",
    "R4":  "ds005508", "R5":  "ds005509", "R6":  "ds005510",
    "R7":  "ds005511", "R8":  "ds005512", "R9":  "ds005514",
    "R10": "ds005515", "R11": "ds005516",
}
ALL_RELEASES = list(RELEASE_MAP.keys())

WINDOW_LEN  = 2.0                        # seconds (same as CCD)
MIN_RT      = 0.1                        # minimum plausible RT in seconds
MAX_RT      = 20.0                       # maximum plausible RT in seconds

METADATA_KEYS = (
    "target",        # RT in seconds (same as rt_from_page, kept for compat)
    "rt_from_page",  # RT from newPage onset to first response
    "correct",       # bool: user_answer == correct_answer on first response
    "n_responses",   # number of responses on this page
    "page_onset",    # absolute newPage onset in seconds
)


# ─────────────────────────────────────────────────────────────────────────────
# Trial table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_trial_table_ss(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-page trial table from a SymbolSearch events file.

    Parameters
    ----------
    events_df : pd.DataFrame
        Contents of _events.tsv. Must have columns: onset, value,
        user_answer, correct_answer.

    Returns
    -------
    pd.DataFrame with columns:
        page_onset, rt_from_page, correct, n_responses, target
    Returns empty DataFrame if no valid trials found.
    """
    df = events_df.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

    pages     = df[df["value"] == "newPage"].reset_index(drop=True)
    responses = df[df["value"] == "trialResponse"].copy()

    if pages.empty:
        return pd.DataFrame()

    rows = []
    for i, page in pages.iterrows():
        page_onset = float(page["onset"])

        # Next page onset defines the boundary of this page's responses
        next_onset = (
            float(pages.iloc[i + 1]["onset"])
            if i + 1 < len(pages)
            else float("inf")
        )

        page_resps = responses[
            (responses["onset"] > page_onset) &
            (responses["onset"] < next_onset)
        ].reset_index(drop=True)

        if page_resps.empty:
            continue

        first = page_resps.iloc[0]
        rt    = float(first["onset"]) - page_onset

        if rt < MIN_RT or rt > MAX_RT:
            continue

        try:
            correct = bool(int(first["user_answer"]) == int(first["correct_answer"]))
        except (ValueError, TypeError):
            correct = None

        rows.append({
            "page_onset":   page_onset,
            "rt_from_page": rt,
            "correct":      correct,
            "n_responses":  len(page_resps),
            "target":       rt,
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

    Parameters
    ----------
    eeg_path    : Path to .set file
    events_path : Path to _events.tsv file
    subject_id  : subject ID string

    Returns (windows_array, meta_dict) or None.
    """
    # ── Load events ───────────────────────────────────────────────────────────
    try:
        events_df = pd.read_csv(events_path, sep="\t")
    except Exception as e:
        print(f"    [skip] {subject_id}: cannot read events — {e}")
        return None

    trials = build_trial_table_ss(events_df)
    if trials.empty:
        print(f"    [skip] {subject_id}: no valid trials in events file")
        return None

    print(f"    [info] {subject_id}: {len(trials)} valid trials from events")

    # ── Load EEG and resample to 100 Hz to match CCD ────────────────────────
    try:
        raw = mne.io.read_raw(str(eeg_path), preload=True, verbose=False)
    except Exception as e:
        print(f"    [skip] {subject_id}: cannot load EEG — {e}")
        return None

    orig_sfreq = raw.info["sfreq"]
    if orig_sfreq != SFREQ:
        print(f"    [info] Resampling {orig_sfreq} Hz → {SFREQ} Hz")
        raw.resample(SFREQ, verbose=False)

    sfreq       = raw.info["sfreq"]   # now SFREQ (100 Hz)
    n_samples   = raw.n_times
    win_samples = int(WINDOW_LEN * sfreq)   # 200 samples
    n_chans     = len(raw.ch_names)

    print(f"    [info] EEG: {n_chans} ch × {n_samples} samples @ {sfreq} Hz")

    # ── Extract windows ───────────────────────────────────────────────────────
    all_windows = []
    all_meta    = {k: [] for k in METADATA_KEYS}
    all_meta["subject"] = []

    for _, trial in trials.iterrows():
        start_samp = int(trial["page_onset"] * sfreq)
        stop_samp  = start_samp + win_samples

        if stop_samp > n_samples:
            continue

        window = raw.get_data(
            start=start_samp, stop=stop_samp
        ).astype(np.float32)  # (n_chans, n_times)

        all_windows.append(window)
        all_meta["target"].append(float(trial["target"]))
        all_meta["rt_from_page"].append(float(trial["rt_from_page"]))
        all_meta["correct"].append(trial["correct"])
        all_meta["n_responses"].append(int(trial["n_responses"]))
        all_meta["page_onset"].append(float(trial["page_onset"]))
        all_meta["subject"].append(subject_id)

    del raw
    gc.collect()

    if not all_windows:
        print(f"    [skip] {subject_id}: 0 windows extracted "
              f"(all trials beyond recording end)")
        return None

    data = np.stack(all_windows, axis=0)  # (n_windows, n_chans, n_times)
    print(f"    ✓ {subject_id}: {len(all_windows)} windows  shape={data.shape}")
    return data, all_meta


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
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
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_ss_data(release: str) -> bool:
    """
    Download SymbolSearch raw EEG files for one release.
    Returns True if successful, False if task not available.
    """
    import gc
    from src.data.loader import load_eegdash_dataset

    dataset_id = RELEASE_MAP[release]
    print(f"\n  Downloading {TASK} {release} ({dataset_id}) ...", flush=True)

    try:
        full_ds  = load_eegdash_dataset(
            dataset_id=dataset_id,
            cache_dir=DATA_DIR,
            task=TASK,
        )
        subjects = full_ds.description["subject"].dropna().unique().tolist()
        n_total  = len(subjects)
        print(f"  Found {n_total} subjects — downloading EEG files ...", flush=True)
        del full_ds
        gc.collect()
    except Exception as e:
        print(f"  ✗ {release}: task not available — {type(e).__name__}: {e}", flush=True)
        return False

    done = 0
    failed = 0
    for i, subject_id in enumerate(subjects):
        print(f"  [{i+1}/{n_total}] {subject_id}", flush=True)
        try:
            single = load_eegdash_dataset(
                dataset_id=dataset_id,
                cache_dir=DATA_DIR,
                task=TASK,
                subject=subject_id,
            )
            for ds in single.datasets:
                ds.raw.load_data(verbose=False)
                ds.raw._data = None
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

    print(f"\n  {release} done: {done}/{n_total} downloaded ({failed} failed)", flush=True)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-release pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ss_data(
    release: str,
    overwrite: bool = False,
    subject_batch_size: int = 5,
) -> None:
    """
    Window all SymbolSearch subjects for one release from local files.
    No downloading — assumes raw files already exist in DATA_DIR.
    """
    dataset_id  = RELEASE_MAP[release]
    raw_dir     = DATA_DIR / dataset_id          # e.g. data/ds005509/
    out_dir     = WINDOWS_DIR / f"{TASK}_{release}"

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task       : {TASK}")
    print(f"Release    : {release}  ({dataset_id})")
    print(f"Source     : {raw_dir}")
    print(f"Output     : {out_dir}")
    print(f"Window     : {WINDOW_LEN}s locked to newPage onset")
    print(f"Target     : RT from newPage → first trialResponse")
    print('='*60)

    if not raw_dir.exists():
        print(f"  [ERROR] Raw data directory not found: {raw_dir}")
        print(f"  Download the data first for release {release}")
        return

    # ── Discover all symbolSearch EEG files ───────────────────────────────────
    eeg_files = sorted(raw_dir.glob(
        f"sub-*/eeg/sub-*_task-{TASK}_eeg.set"
    ))

    if not eeg_files:
        print(f"  [ERROR] No symbolSearch .set files found in {raw_dir}")
        return

    n_total = len(eeg_files)
    print(f"\n  Found {n_total} EEG files")

    # ── Process subjects ──────────────────────────────────────────────────────
    done, skipped, failed = [], [], []

    for i, eeg_path in enumerate(eeg_files):
        # Derive subject_id from filename: sub-NDARXXX_task-..._eeg.set
        subject_id   = eeg_path.parent.parent.name.replace("sub-", "")
        events_path  = eeg_path.parent / eeg_path.name.replace(
            "_eeg.set", "_events.tsv"
        )

        print(f"\n  [{i+1}/{n_total}] {subject_id}")

        if not overwrite and _already_done(subject_id, out_dir):
            print(f"    [skip] already on disk")
            skipped.append(subject_id)
            continue

        if not events_path.exists():
            print(f"    [skip] events file missing: {events_path.name}")
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

    if failed:
        (out_dir / "failed_subjects.txt").write_text(
            "\n".join(sorted(failed))
        )
        print(f"  Failed subjects → {out_dir / 'failed_subjects.txt'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[Done]  {TASK}  {release}")
    print(f"  ✓ Windowed : {len(done)}")
    print(f"  ↷ Skipped  : {len(skipped)}")
    print(f"  ✗ Failed   : {len(failed)}")
    print(f"  Output     : {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and/or window SymbolSearch EEG data."
    )
    parser.add_argument(
        "--mode", type=str, default="window",
        choices=["download", "window", "all"],
        help=(
            "'download' = download raw files only. "
            "'window'   = window already-downloaded files. "
            "'all'      = download then window."
        )
    )
    parser.add_argument(
        "--release", type=str, default=RELEASE,
        help=f"Release label, e.g. R5 (default: {RELEASE}). "
             f"Ignored if --all_releases is set."
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

    releases = ALL_RELEASES if args.all_releases else [args.release]

    if args.mode == "download":
        for rel in releases:
            if rel not in RELEASE_MAP:
                print(f"[warn] Unknown release '{rel}', skipping")
                continue
            download_ss_data(rel)

    elif args.mode == "window":
        succeeded, failed_releases = [], []
        for rel in releases:
            if rel not in RELEASE_MAP:
                print(f"[warn] Unknown release '{rel}', skipping")
                continue
            print(f"\n\n{'#'*60}")
            print(f"# Release: {rel}  ({RELEASE_MAP[rel]})")
            print(f"{'#'*60}")
            try:
                prepare_ss_data(
                    release=rel,
                    overwrite=args.overwrite,
                    subject_batch_size=args.subject_batch_size,
                )
                succeeded.append(rel)
            except Exception:
                print(f"\n[ERROR] Failed on release {rel}")
                traceback.print_exc()
                failed_releases.append(rel)

        if len(releases) > 1:
            print(f"\n\n{'='*60}")
            print(f"SUMMARY — task: {TASK}")
            print(f"  ✓ {len(succeeded)} succeeded : {succeeded}")
            print(f"  ✗ {len(failed_releases)} failed    : {failed_releases}")
            print('='*60)

    elif args.mode == "all":
        succeeded, failed_releases = [], []
        for rel in releases:
            if rel not in RELEASE_MAP:
                print(f"[warn] Unknown release '{rel}', skipping")
                continue
            print(f"\n\n{'#'*60}")
            print(f"# Release: {rel}  ({RELEASE_MAP[rel]})")
            print(f"{'#'*60}")
            ok = download_ss_data(rel)
            if not ok:
                failed_releases.append(rel)
                continue
            try:
                prepare_ss_data(
                    release=rel,
                    overwrite=args.overwrite,
                    subject_batch_size=args.subject_batch_size,
                )
                succeeded.append(rel)
            except Exception:
                print(f"\n[ERROR] Failed windowing on release {rel}")
                traceback.print_exc()
                failed_releases.append(rel)

        print(f"\n\n{'='*60}")
        print(f"SUMMARY — task: {TASK}")
        print(f"  ✓ {len(succeeded)} succeeded : {succeeded}")
        print(f"  ✗ {len(failed_releases)} failed    : {failed_releases}")
        print('='*60)