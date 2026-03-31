"""
scripts/check_trials.py
-----------------------
Check how many trials (windows) each subject has.

This is critical — RT normalization only makes sense if each subject has
enough trials to compute a meaningful mean and std. If subjects only have
5-10 trials, the std estimate is noisy and normalization is unreliable.

Also checks RT variance per subject — if a subject has very low RT std,
normalization will amplify noise rather than signal.

Usage:
    python scripts/check_trials.py
    python scripts/check_trials.py --release R5
    python scripts/check_trials.py --release R5 --task contrastChangeDetection
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from configs.config import TASK, RELEASE, WINDOWS_DIR, EXCLUDED_SUBJECTS


def check_trials(task: str, release: str) -> None:

    root_dir = WINDOWS_DIR / f"{task}_{release}"

    if not root_dir.exists():
        print(f"ERROR: {root_dir} does not exist. Run prepare_data.py first.")
        sys.exit(1)

    # ── Load subjects ─────────────────────────────────────────────────────────
    subjects_file = root_dir / "subjects.txt"
    if not subjects_file.exists():
        print(f"ERROR: subjects.txt not found in {root_dir}")
        sys.exit(1)

    all_subjects = subjects_file.read_text().strip().splitlines()
    subjects = [s for s in all_subjects if s not in EXCLUDED_SUBJECTS]

    print(f"\n{'='*60}")
    print(f"Trial count check")
    print(f"  Task    : {task}")
    print(f"  Release : {release}")
    print(f"  Subjects: {len(subjects)} (excluded {len(all_subjects) - len(subjects)})")
    print(f"{'='*60}\n")

    # ── Per-subject stats ─────────────────────────────────────────────────────
    rows = []
    missing = []

    for sid in subjects:
        sub_dir = root_dir / f"sub-{sid}"
        windows_path  = sub_dir / "windows.npy"
        metadata_path = sub_dir / "metadata.npz"

        if not windows_path.exists() or not metadata_path.exists():
            missing.append(sid)
            continue

        # Load metadata only — don't need the actual EEG
        meta = np.load(metadata_path, allow_pickle=True)
        targets = meta["target"].tolist()

        # Filter out None/nan
        valid_rts = [float(t) for t in targets if t is not None and not (isinstance(t, float) and np.isnan(t))]
        n_windows  = len(targets)
        n_valid_rt = len(valid_rts)

        if n_valid_rt > 1:
            rt_mean = np.mean(valid_rts)
            rt_std  = np.std(valid_rts)
            rt_min  = np.min(valid_rts)
            rt_max  = np.max(valid_rts)
            rt_cv   = rt_std / rt_mean  # coefficient of variation
        else:
            rt_mean = rt_std = rt_min = rt_max = rt_cv = float("nan")

        rows.append({
            "subject":    sid,
            "n_windows":  n_windows,
            "n_valid_rt": n_valid_rt,
            "rt_mean":    round(rt_mean, 4) if not np.isnan(rt_mean) else None,
            "rt_std":     round(rt_std,  4) if not np.isnan(rt_std)  else None,
            "rt_min":     round(rt_min,  4) if not np.isnan(rt_min)  else None,
            "rt_max":     round(rt_max,  4) if not np.isnan(rt_max)  else None,
            "rt_cv":      round(rt_cv,   4) if not np.isnan(rt_cv)   else None,
        })

    df = pd.DataFrame(rows)

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"Trials per subject:")
    print(f"  Min    : {df['n_windows'].min()}")
    print(f"  Max    : {df['n_windows'].max()}")
    print(f"  Mean   : {df['n_windows'].mean():.1f}")
    print(f"  Median : {df['n_windows'].median():.1f}")
    print(f"  Std    : {df['n_windows'].std():.1f}")

    print(f"\nRT std per subject (how much within-subject variance exists):")
    print(f"  Min    : {df['rt_std'].min():.4f}s")
    print(f"  Max    : {df['rt_std'].max():.4f}s")
    print(f"  Mean   : {df['rt_std'].mean():.4f}s")
    print(f"  Median : {df['rt_std'].median():.4f}s")

    # ── Warnings ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WARNINGS")
    print(f"{'='*60}")

    # Too few trials — normalization unreliable
    few_trials = df[df["n_windows"] < 20]
    if len(few_trials) > 0:
        print(f"\n⚠ Subjects with < 20 trials ({len(few_trials)} subjects):")
        print(f"  RT normalization stats will be noisy for these subjects.")
        for _, row in few_trials.iterrows():
            print(f"    {row['subject']} : {row['n_windows']} trials")
    else:
        print(f"\n✓ All subjects have >= 20 trials — normalization stats reliable")

    # Very low RT std — normalization will amplify noise
    low_std = df[df["rt_std"] < 0.05]
    if len(low_std) > 0:
        print(f"\n⚠ Subjects with RT std < 0.05s ({len(low_std)} subjects):")
        print(f"  These subjects show very little trial-level RT variation.")
        print(f"  Normalization may amplify noise rather than signal.")
        for _, row in low_std.iterrows():
            print(f"    {row['subject']} : std={row['rt_std']}s  mean={row['rt_mean']}s")
    else:
        print(f"\n✓ All subjects have RT std >= 0.05s — enough within-subject variance")

    # Missing subjects
    if missing:
        print(f"\n⚠ {len(missing)} subjects listed in subjects.txt but missing data folders:")
        for s in missing:
            print(f"    {s}")

    # ── Trial count distribution ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRIAL COUNT DISTRIBUTION")
    print(f"{'='*60}")
    bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 99999]
    labels = ["<10", "10-20", "20-30", "30-50", "50-75", "75-100", "100-150", "150-200", "200+"]
    df["bucket"] = pd.cut(df["n_windows"], bins=bins, labels=labels)
    dist = df["bucket"].value_counts().sort_index()
    for bucket, count in dist.items():
        bar = "█" * count
        print(f"  {bucket:>8} trials : {count:>4} subjects  {bar}")

    # ── Full table ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PER-SUBJECT BREAKDOWN")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path("results") / f"trial_counts_{task}_{release}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["bucket"], errors="ignore").to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    median_trials = df["n_windows"].median()
    mean_rt_std   = df["rt_std"].mean()

    if median_trials >= 30 and mean_rt_std >= 0.1:
        print(f"✓ RT normalization is JUSTIFIED")
        print(f"  Median {median_trials:.0f} trials per subject is enough for reliable stats")
        print(f"  Mean within-subject RT std of {mean_rt_std:.3f}s means real trial-level variance exists")
    elif median_trials < 20:
        print(f"✗ RT normalization is RISKY")
        print(f"  Median only {median_trials:.0f} trials per subject — stats will be noisy")
    else:
        print(f"~ RT normalization is MARGINAL — check the warnings above carefully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    type=str, default=TASK)
    parser.add_argument("--release", type=str, default=RELEASE)
    args = parser.parse_args()

    check_trials(task=args.task, release=args.release)