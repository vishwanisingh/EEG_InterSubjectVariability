"""
inspect_dataset.py
------------------
Run this locally to see exactly what's in your windowed data folders.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --task symbolSearch --release R5
    python inspect_dataset.py --task contrastChangeDetection --release R5
    python inspect_dataset.py --all_tasks
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

WINDOWS_DIR = Path("data/windows")

ALL_TASKS = [
    # Active (button press, RT target)
    "contrastChangeDetection",
    "symbolSearch",
    "seqLearning8target",
    "seqLearning6target",
    # Passive (no response events, no RT target)
    "surroundSupp",
    "RestingState",
    "DespicableMe",
    "ThePresent",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def inspect_subject(sub_dir: Path) -> dict:
    """Return a summary dict for one subject folder."""
    subject_id   = sub_dir.name.replace("sub-", "")
    windows_path = sub_dir / "windows.npy"
    meta_path    = sub_dir / "metadata.npz"

    row = {
        "subject":         subject_id,
        "has_windows":     windows_path.exists(),
        "has_metadata":    meta_path.exists(),
        "windows_size":    human_size(windows_path),
        "windows_shape":   None,
        "n_windows":       None,
        "n_chans":         None,
        "n_times":         None,
        "meta_keys":       None,
        "target_n_valid":  None,
        "target_mean":     None,
        "target_std":      None,
        "target_min":      None,
        "target_max":      None,
        "error":           None,
    }

    if not windows_path.exists() and not meta_path.exists():
        row["error"] = "EMPTY FOLDER"
        return row

    # ── Windows ───────────────────────────────────────────────────────────────
    if windows_path.exists():
        try:
            arr = np.load(windows_path, mmap_mode="r")
            row["windows_shape"] = str(arr.shape)
            if arr.ndim == 3:
                row["n_windows"] = arr.shape[0]
                row["n_chans"]   = arr.shape[1]
                row["n_times"]   = arr.shape[2]
            else:
                row["error"] = f"BAD SHAPE: {arr.shape}"
        except Exception as e:
            row["error"] = f"windows load error: {e}"
    else:
        row["error"] = "MISSING windows.npy"

    # ── Metadata ──────────────────────────────────────────────────────────────
    if meta_path.exists():
        try:
            meta = np.load(meta_path, allow_pickle=True)
            keys = list(meta.files)
            row["meta_keys"] = ", ".join(keys)

            # Try to read target field
            target_key = None
            for k in ["target", "rt_from_stimulus", "rt_from_page"]:
                if k in keys:
                    target_key = k
                    break

            if target_key:
                targets = meta[target_key].tolist()
                valid   = [float(t) for t in targets
                           if t is not None and not (isinstance(t, float) and np.isnan(t))]
                row["target_n_valid"] = len(valid)
                if valid:
                    row["target_mean"] = round(float(np.mean(valid)),  4)
                    row["target_std"]  = round(float(np.std(valid)),   4)
                    row["target_min"]  = round(float(np.min(valid)),   4)
                    row["target_max"]  = round(float(np.max(valid)),   4)
        except Exception as e:
            row["error"] = (row["error"] or "") + f" | meta load error: {e}"
    else:
        row["error"] = (row["error"] or "") + " | MISSING metadata.npz"

    return row


def inspect_task_release(task: str, release: str) -> None:
    root = WINDOWS_DIR / f"{task}_{release}"

    print(f"\n{'='*65}")
    print(f"Task    : {task}")
    print(f"Release : {release}")
    print(f"Path    : {root}")
    print("=" * 65)

    if not root.exists():
        print(f"  ✗ Directory does not exist. Data not prepared yet.")
        return

    # subjects.txt
    subjects_file = root / "subjects.txt"
    if subjects_file.exists():
        subjects = subjects_file.read_text().strip().splitlines()
        print(f"  subjects.txt : {len(subjects)} entries")
    else:
        print(f"  subjects.txt : MISSING")
        subjects = []

    # Scan all sub-* folders
    sub_dirs = sorted(root.glob("sub-*"))
    print(f"  sub-* folders: {len(sub_dirs)}")

    if not sub_dirs:
        print("  No subject folders found.")
        return

    rows = []
    for sub_dir in sub_dirs:
        rows.append(inspect_subject(sub_dir))

    df = pd.DataFrame(rows)

    # ── Status counts ─────────────────────────────────────────────────────────
    complete = df["has_windows"] & df["has_metadata"] & df["error"].isna()
    empty    = ~df["has_windows"] & ~df["has_metadata"]
    partial  = ~complete & ~empty

    print(f"\n  Status:")
    print(f"    ✓ Complete (windows + metadata) : {complete.sum()}")
    print(f"    ~ Partial (one file missing)    : {partial.sum()}")
    print(f"    ✗ Empty (no files)              : {empty.sum()}")

    if complete.sum() > 0:
        good = df[complete]
        print(f"\n  Windows shape (first subject)  : {good['windows_shape'].iloc[0]}")
        print(f"  Channels                        : {good['n_chans'].iloc[0]}")
        print(f"  Time points                     : {good['n_times'].iloc[0]}")
        print(f"  Metadata keys                   : {good['meta_keys'].iloc[0]}")
        print(f"\n  Windows per subject:")
        print(f"    Min    : {good['n_windows'].min()}")
        print(f"    Max    : {good['n_windows'].max()}")
        print(f"    Mean   : {good['n_windows'].mean():.1f}")
        print(f"    Median : {good['n_windows'].median():.1f}")
        if good["target_mean"].notna().any():
            print(f"\n  Target (RT) stats across all complete subjects:")
            print(f"    Mean   : {good['target_mean'].mean():.4f} s")
            print(f"    Std    : {good['target_std'].mean():.4f} s  (avg within-subject std)")
            print(f"    Min RT : {good['target_min'].min():.4f} s")
            print(f"    Max RT : {good['target_max'].max():.4f} s")
        else:
            print(f"\n  Target (RT): none (passive task, no labels)")

    # ── Error breakdown ───────────────────────────────────────────────────────
    errors = df[df["error"].notna()]
    if len(errors) > 0:
        print(f"\n  Error breakdown:")
        for err_type, count in errors["error"].value_counts().items():
            print(f"    {count:>4}x  {err_type}")

    # ── First 10 subjects detail ──────────────────────────────────────────────
    print(f"\n  First 10 subjects:")
    cols = ["subject", "has_windows", "has_metadata", "windows_shape",
            "n_windows", "target_mean", "target_std", "error"]
    print(df[cols].head(10).to_string(index=False))

    # ── Save full report ──────────────────────────────────────────────────────
    out_path = Path("results") / f"inspect_{task}_{release}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  Full report saved → {out_path}")


# ── Raw EEG check ─────────────────────────────────────────────────────────────

def check_raw_eeg(task: str, release: str) -> None:
    """Check if raw EEG .set files exist for this task/release."""
    RELEASE_MAP = {
        "R1": "ds005505", "R2": "ds005506", "R3": "ds005507",
        "R4": "ds005508", "R5": "ds005509", "R6": "ds005510",
        "R7": "ds005511", "R8": "ds005512", "R9": "ds005514",
        "R10": "ds005515", "R11": "ds005516",
    }
    dataset_id = RELEASE_MAP.get(release, "unknown")
    raw_dir    = Path("data") / dataset_id

    print(f"\n  Raw EEG check ({raw_dir}):")
    if not raw_dir.exists():
        print(f"    ✗ Raw data directory not found: {raw_dir}")
        print(f"      Run the download step first.")
        return

    eeg_files = sorted(raw_dir.glob(f"sub-*/eeg/sub-*_task-{task}_eeg.set"))
    tsv_files = sorted(raw_dir.glob(f"sub-*/eeg/sub-*_task-{task}_events.tsv"))

    print(f"    .set   files found : {len(eeg_files)}")
    print(f"    .tsv   files found : {len(tsv_files)}")

    if eeg_files:
        print(f"    Example: {eeg_files[0]}")
        # Peek at events for first subject
        if tsv_files:
            try:
                df = pd.read_csv(tsv_files[0], sep="\t")
                print(f"\n    Events file columns : {list(df.columns)}")
                print(f"    Event value counts  :")
                for val, cnt in df["value"].value_counts().head(10).items():
                    print(f"      {cnt:>5}x  {val}")
            except Exception as e:
                print(f"    Could not read events: {e}")
    else:
        print(f"    ✗ No .set files found — raw EEG not downloaded for task='{task}'")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect windowed EEG dataset on disk."
    )
    parser.add_argument("--task",       type=str, default="symbolSearch")
    parser.add_argument("--release",    type=str, default="R5")
    parser.add_argument("--all_tasks",  action="store_true",
                        help="Inspect all known tasks for the given release.")
    parser.add_argument("--check_raw",  action="store_true",
                        help="Also check if raw .set EEG files exist.")
    args = parser.parse_args()

    tasks = ALL_TASKS if args.all_tasks else [args.task]

    for task in tasks:
        inspect_task_release(task, args.release)
        if args.check_raw:
            check_raw_eeg(task, args.release)

    print("\nDone.")