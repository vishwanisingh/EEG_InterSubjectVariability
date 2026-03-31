"""
scripts/prepare_ccd_data.py
-----------------------
Run ONCE to download, preprocess, window, and save the dataset to disk.

Memory strategy: process ONE SUBJECT AT A TIME — each subject's raw EEG is
loaded, windowed, saved as numpy arrays to disk, then freed before the next
subject loads.

Peak RAM = ~one subject's raw EEG (~100-500 MB) instead of the whole dataset.

On-disk layout:
    data/windows/{task}_{release}/
        subjects.txt              list of successfully processed subject IDs
        sub-{subject_id}/
            windows.npy           float32  (n_windows, n_chans, n_times)
            metadata.npz          mixed    target, rt, correct, subject, ...

Usage:
    python scripts/prepare_ccd_data.py
"""

import gc
import sys
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
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_one_subject(
    single_rec: BaseConcatDataset,
    cfg: dict,
    subject_id: str,
) -> tuple | None:
    """
    Preprocess and window a single-subject dataset.

    Returns (windows_array, meta_dict) or None if no valid windows.
    windows_array : float32 ndarray  (n_windows, n_chans, n_times)
    meta_dict     : dict of lists, one entry per window
    """
    anchor     = cfg["anchor"]
    window_len = cfg["window_len"]
    shift      = cfg["shift_after_stim"]

    # Annotate trials with RT labels and add windowing anchors
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

    # Drop recordings that don't have the event anchor
    filtered = keep_only_recordings_with(anchor, single_rec)
    if len(filtered.datasets) == 0:
        print(f"    [skip] {subject_id}: no recordings with anchor '{anchor}'")
        return None

    # Create windows WITHOUT preloading into RAM
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

    # Pull windows one at a time to keep memory flat
    n_chans = windows[0][0].shape[0]
    data    = np.empty((n, n_chans, size_s), dtype=np.float32)

    for i in range(n):
        x, _, _ = windows[i]
        # Handle both torch.Tensor and numpy.ndarray
        if hasattr(x, 'numpy'):
            data[i] = x.numpy().astype(np.float32)
        else:
            data[i] = x.astype(np.float32)

    meta_df   = windows.get_metadata()
    meta_dict = {}
    for key in METADATA_KEYS:
        meta_dict[key] = meta_df[key].tolist() if key in meta_df.columns else [None] * n
    meta_dict["subject"] = [subject_id] * n

    print(f"    ✓ {subject_id}: {n} windows  shape={data.shape}")
    return data, meta_dict


def _save_subject(subject_id: str, data: np.ndarray, meta: dict, out_dir: Path) -> None:
    sub_dir = out_dir / f"sub-{subject_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.save(sub_dir / "windows.npy", data)
    np.savez(sub_dir / "metadata.npz", **{k: np.array(v, dtype=object) for k, v in meta.items()})


def _already_done(subject_id: str, out_dir: Path) -> bool:
    d = out_dir / f"sub-{subject_id}"
    return (d / "windows.npy").exists() and (d / "metadata.npz").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    dataset_id: str = DATASET_ID,
    task: str = TASK,
    release: str = RELEASE,
    overwrite: bool = False,
    subject_batch_size: int = 5,
) -> None:
    """
    Download → preprocess → window → save, one subject at a time.

    Parameters
    ----------
    dataset_id : str
        OpenNeuro dataset ID (e.g. "ds005509").
    task : str
        Task name — must exist in configs/task_configs.py.
    release : str
        Release label used as part of the save folder name.
    overwrite : bool
        Re-process subjects that are already saved to disk.
    subject_batch_size : int
        Run gc.collect() every N subjects to release memory.
        Lower values = safer memory, slightly slower.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_dir = WINDOWS_DIR / f"{task}_{release}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_task_config(task)

    # ── [1/3] Query metadata to get the subject list ──────────────────────────
    print(f"\n[1/3] Fetching subject list  dataset={dataset_id}  task={task}")
    full_dataset = load_eegdash_dataset(
        dataset_id=dataset_id,
        cache_dir=DATA_DIR,
        task=task,
    )

    print(f"    Total recordings: {len(full_dataset.datasets)}")
    subjects = full_dataset.description["subject"].dropna().unique().tolist()
    n_total  = len(subjects)
    print(f"  → {n_total} subjects found")

    # Free the full dataset — we only needed the subject list
    del full_dataset
    gc.collect()

    # ── [2/3] Process subjects one at a time ──────────────────────────────────
    print(f"\n[2/3] Processing subjects  (batch_size={subject_batch_size})")
    done, skipped, failed = [], [], []
    all_meta = []

    for i, subject_id in enumerate(subjects):
        print(f"\n  [{i+1}/{n_total}] {subject_id}")

        if not overwrite and _already_done(subject_id, out_dir):
            print(f"    [skip] already on disk")
            skipped.append(subject_id)
            continue

        try:
            # Load only this one subject
            single_sub = load_eegdash_dataset(
                dataset_id=dataset_id,
                cache_dir=DATA_DIR,
                task=task,
                subject=subject_id,
            )

            result = _window_one_subject(single_sub, cfg, subject_id)

            if result is None:
                skipped.append(subject_id)
            else:
                data, meta = result
                _save_subject(subject_id, data, meta, out_dir)
                all_meta.append(meta)
                done.append(subject_id)

            # Explicitly delete subject data from memory
            del single_sub, result
            gc.collect()

        except Exception:
            print(f"    [ERROR] {subject_id}")
            traceback.print_exc()
            failed.append(subject_id)
            gc.collect()

        # Periodic GC even on successful runs
        if (i + 1) % subject_batch_size == 0:
            gc.collect()
            print(f"  [gc] Memory released after {i+1} subjects")

    # ── Write subject index so train.py knows what's available ───────────────
    all_processed = sorted(set(done) | set(skipped))
    (out_dir / "subjects.txt").write_text("\n".join(all_processed))

    # ── [3/3] Summary + RT plot ───────────────────────────────────────────────
    print(f"\n[3/3] Done.")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  ✓ Processed : {len(done)}")
    print(f"  ↷ Skipped   : {len(skipped)}  (already done or no valid windows)")
    print(f"  ✗ Failed    : {len(failed)}")
    if failed:
        print(f"    → {failed}")
    print(f"  Saved to    : {out_dir}")
    print("="*60)

    if cfg["task_type"] == "active" and all_meta:
        import pandas as pd
        combined = pd.DataFrame({
            k: sum([m[k] for m in all_meta], []) for k in METADATA_KEYS
        })
        plot_response_time_distribution(
            combined,
            RESULTS_DIR / f"{task}_{release}_rt_distribution.png",
        )


if __name__ == "__main__":
    prepare_data(
        dataset_id=DATASET_ID,
        task=TASK,
        release=RELEASE,
        overwrite=False,
        subject_batch_size=5,   # tune this lower if still crashing
    )