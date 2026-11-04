"""
scripts/check_data_quality.py
------------------------------
Post-hoc data quality audit on windowed EEG data (already on disk).
Works directly on windows.npy — no raw files needed.

Checks per subject
------------------
  1. Amplitude threshold  — windows with peak-to-peak > AMPLITUDE_THRESH
  2. Flat channels        — channels with std < FLAT_THRESH across all windows
  3. Noisy channels       — channels with std > NOISY_FACTOR × median channel std
  4. Bad epochs           — windows with max |amplitude| > EPOCH_THRESH
  5. ICA kurtosis         — per-subject FastICA; high-kurtosis components suggest
                            residual artifacts (blinks, muscle, electrode pops)
  6. Cross-channel corr   — mean pairwise correlation; very high values suggest
                            bridged electrodes or reference leakage

Units: MNE raw.get_data() returns Volts, so all windows.npy are in Volts.
       Thresholds are in Volts; display values are converted to µV.

Output
------
  results/data_quality/{task}_{release}_quality.csv   per-subject summary
  results/data_quality/{task}_{release}_bad_windows.csv  flagged window indices

Usage
-----
    # Default task/release from configs/config.py
    python scripts/check_data_quality.py

    # Specific task and release
    python scripts/check_data_quality.py --task symbolSearch --release R5

    # Check all tasks × releases present on disk
    python scripts/check_data_quality.py --all
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import TASK, RELEASE, WINDOWS_DIR, RESULTS_DIR

# ── Thresholds ────────────────────────────────────────────────────────────────

AMPLITUDE_THRESH = 0.15     # 150 µV in mV  — peak-to-peak per window flag
EPOCH_THRESH     = 0.10     # 100 µV in mV  — max |amplitude| per window flag
FLAT_THRESH      = 0.0005   # 0.5 µV  in mV — channel std below this = flat/dead
NOISY_FACTOR     = 5.0      # channel std > NOISY_FACTOR × median = noisy
N_ICA_COMPONENTS = 20       # ICA components to fit per subject
KURTOSIS_THRESH  = 5.0      # excess kurtosis above this = likely artifact comp
CORR_THRESH      = 0.98     # mean pairwise corr above this = bridging concern

V2UV = 1000                 # mV → µV for display


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject checks
# ─────────────────────────────────────────────────────────────────────────────

def check_amplitude(windows: np.ndarray) -> dict:
    """
    windows : (N, C, T)  float32, Volts
    Returns flagging stats for peak-to-peak and max |amplitude|.
    """
    ptp   = windows.max(axis=2) - windows.min(axis=2)   # (N, C)
    maxabs = np.abs(windows).max(axis=2)                  # (N, C)

    ptp_per_window  = ptp.max(axis=1)                    # (N,)  worst channel per window
    maxabs_per_win  = maxabs.max(axis=1)                 # (N,)

    n_flagged_ptp   = int((ptp_per_window  > AMPLITUDE_THRESH).sum())
    n_flagged_epoch = int((maxabs_per_win  > EPOCH_THRESH).sum())

    return {
        "n_windows":           windows.shape[0],
        "n_flagged_ptp":       n_flagged_ptp,
        "pct_flagged_ptp":     round(100 * n_flagged_ptp  / windows.shape[0], 2),
        "n_flagged_epoch":     n_flagged_epoch,
        "pct_flagged_epoch":   round(100 * n_flagged_epoch / windows.shape[0], 2),
        "max_ptp_uv":          round(float(ptp_per_window.max())  * V2UV, 2),
        "mean_ptp_uv":         round(float(ptp_per_window.mean()) * V2UV, 2),
        "flagged_window_idx":  np.where(maxabs_per_win > EPOCH_THRESH)[0].tolist(),
    }


def check_channels(windows: np.ndarray) -> dict:
    """
    windows : (N, C, T)
    Detect flat and noisy channels.
    """
    # Per-channel std over all windows and time points
    ch_std = windows.reshape(windows.shape[0], windows.shape[1], -1).std(axis=(0, 2))  # (C,)

    flat_mask  = ch_std < FLAT_THRESH
    median_std = float(np.median(ch_std))
    noisy_mask = ch_std > (NOISY_FACTOR * median_std) if median_std > 0 else np.zeros_like(flat_mask)

    return {
        "n_flat_channels":   int(flat_mask.sum()),
        "flat_channel_idx":  np.where(flat_mask)[0].tolist(),
        "n_noisy_channels":  int(noisy_mask.sum()),
        "noisy_channel_idx": np.where(noisy_mask)[0].tolist(),
        "median_ch_std_uv":  round(median_std * V2UV, 4),
        "max_ch_std_uv":     round(float(ch_std.max()) * V2UV, 4),
    }


def check_ica(windows: np.ndarray, n_components: int = N_ICA_COMPONENTS) -> dict:
    """
    windows : (N, C, T)
    Reshape to (N*T, C), run FastICA, report kurtosis of each component.
    High excess kurtosis (> KURTOSIS_THRESH) flags likely artifact components.
    """
    N, C, T = windows.shape
    n_comp = min(n_components, C, N * T // 4)

    # (N*T, C) — treat concatenated windows as pseudo-continuous signal
    X = windows.transpose(0, 2, 1).reshape(-1, C).astype(np.float64)

    try:
        ica = FastICA(
            n_components=n_comp,
            random_state=42,
            max_iter=500,
            tol=0.01,
        )
        sources = ica.fit_transform(X)   # (N*T, n_comp)
    except Exception as e:
        return {"ica_error": str(e), "n_artifact_components": -1}

    # Excess kurtosis: kurt = E[x^4] / E[x^2]^2 - 3
    def excess_kurtosis(x):
        x = x - x.mean()
        var = (x ** 2).mean()
        if var < 1e-12:
            return 0.0
        return float((x ** 4).mean() / var ** 2) - 3.0

    kurtosis_vals = np.array([excess_kurtosis(sources[:, i]) for i in range(n_comp)])
    n_artifact    = int((np.abs(kurtosis_vals) > KURTOSIS_THRESH).sum())

    return {
        "n_ica_components":   n_comp,
        "n_artifact_components": n_artifact,
        "max_kurtosis":       round(float(np.abs(kurtosis_vals).max()), 2),
        "mean_kurtosis":      round(float(np.abs(kurtosis_vals).mean()), 2),
        "kurtosis_values":    [round(k, 2) for k in kurtosis_vals.tolist()],
    }


def check_channel_correlation(windows: np.ndarray, sample_size: int = 200) -> dict:
    """
    windows : (N, C, T)
    Compute mean pairwise channel correlation on a random subset of windows.
    Very high mean corr (> CORR_THRESH) may indicate bridged electrodes or
    bad reference.
    """
    N, C, T = windows.shape
    idx     = np.random.RandomState(42).choice(N, size=min(sample_size, N), replace=False)
    subset  = windows[idx]                          # (sample, C, T)
    flat    = subset.reshape(len(idx), C, -1)       # (sample, C, T)

    # Stack windows along time, then correlate channels
    X = flat.transpose(1, 0, 2).reshape(C, -1).T   # (sample*T, C)

    # Pearson correlation matrix
    X_c  = X - X.mean(axis=0)
    norms = np.linalg.norm(X_c, axis=0)
    norms[norms == 0] = 1.0
    X_n   = X_c / norms
    corr  = X_n.T @ X_n / X_c.shape[0]             # (C, C)

    # Only upper triangle, excluding diagonal
    upper = corr[np.triu_indices(C, k=1)]
    mean_corr = float(upper.mean())
    max_corr  = float(upper.max())

    n_bridged_pairs = int((upper > CORR_THRESH).sum())

    return {
        "mean_pairwise_corr": round(mean_corr, 4),
        "max_pairwise_corr":  round(max_corr, 4),
        "n_bridged_pairs":    n_bridged_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject entry point
# ─────────────────────────────────────────────────────────────────────────────

def audit_subject(subject_id: str, sub_dir: Path, run_ica: bool = True) -> dict:
    windows_path = sub_dir / "windows.npy"
    meta_path    = sub_dir / "metadata.npz"

    if not windows_path.exists():
        return {"subject": subject_id, "error": "windows.npy missing"}

    windows = np.load(windows_path)                 # (N, C, T)

    if windows.ndim != 3 or windows.shape[0] == 0:
        return {"subject": subject_id, "error": f"bad shape {windows.shape}"}

    N, C, T = windows.shape
    result  = {"subject": subject_id, "n_windows": N, "n_channels": C, "n_times": T}

    amp   = check_amplitude(windows)
    chans = check_channels(windows)
    corr  = check_channel_correlation(windows)

    result.update({k: v for k, v in amp.items()   if k != "flagged_window_idx"})
    result.update({k: v for k, v in chans.items() if "idx" not in k})
    result.update(corr)

    if run_ica:
        ica = check_ica(windows)
        result.update({k: v for k, v in ica.items() if k != "kurtosis_values"})

    # Overall verdict
    issues = []
    if amp["pct_flagged_epoch"] > 5:
        issues.append(f"{amp['pct_flagged_epoch']:.1f}% bad epochs")
    if chans["n_flat_channels"] > 0:
        issues.append(f"{chans['n_flat_channels']} flat ch")
    if chans["n_noisy_channels"] > 0:
        issues.append(f"{chans['n_noisy_channels']} noisy ch")
    if corr["n_bridged_pairs"] > 0:
        issues.append(f"{corr['n_bridged_pairs']} bridged pairs")
    if run_ica and result.get("n_artifact_components", 0) > 3:
        issues.append(f"{result['n_artifact_components']} ICA artifact comps")

    result["issues"]   = "; ".join(issues) if issues else "none"
    result["flag"]     = len(issues) > 0
    result["bad_window_idx"] = amp["flagged_window_idx"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task-level audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_task(task: str, release: str, run_ica: bool = True) -> None:
    root_dir = WINDOWS_DIR / f"{task}_{release}"

    if not root_dir.exists():
        print(f"  [skip] {task}_{release} not found at {root_dir}")
        return

    subjects_file = root_dir / "subjects.txt"
    if not subjects_file.exists():
        print(f"  [skip] no subjects.txt in {root_dir}")
        return

    subjects = subjects_file.read_text().strip().splitlines()
    subjects = [s for s in subjects if s]

    out_dir = RESULTS_DIR / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Task   : {task}")
    print(f"  Release: {release}")
    print(f"  Dir    : {root_dir}")
    print(f"  Subjects found: {len(subjects)}")
    print(f"  Thresholds: ptp>{AMPLITUDE_THRESH*V2UV:.0f}µV  "
          f"epoch>{EPOCH_THRESH*V2UV:.0f}µV  "
          f"flat<{FLAT_THRESH*V2UV:.1f}µV  "
          f"noisy>{NOISY_FACTOR:.0f}×median  "
          f"ICA kurt>{KURTOSIS_THRESH:.0f}")
    print('='*65)

    rows          = []
    bad_win_rows  = []
    n_flagged     = 0

    for i, sid in enumerate(subjects):
        sub_dir = root_dir / f"sub-{sid}"
        if not sub_dir.exists():
            continue

        print(f"  [{i+1:>3}/{len(subjects)}] {sid}", end="  ", flush=True)
        result = audit_subject(sid, sub_dir, run_ica=run_ica)

        flag   = result.get("flag", False)
        issues = result.get("issues", "none")
        n_win  = result.get("n_windows", "?")
        pct_ep = result.get("pct_flagged_epoch", 0)

        status = "FLAGGED" if flag else "ok"
        print(f"{n_win:>5} windows  epoch_bad={pct_ep:5.1f}%  [{status}]  {issues}")

        if flag:
            n_flagged += 1

        # Collect bad window indices separately
        bad_idx = result.pop("bad_window_idx", [])
        rows.append(result)

        for widx in bad_idx:
            bad_win_rows.append({"subject": sid, "window_idx": widx})

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(rows)
    out_csv    = out_dir / f"{task}_{release}_quality.csv"
    summary_df.to_csv(out_csv, index=False)

    if bad_win_rows:
        bad_df  = pd.DataFrame(bad_win_rows)
        bad_csv = out_dir / f"{task}_{release}_bad_windows.csv"
        bad_df.to_csv(bad_csv, index=False)
        print(f"\n  Bad window indices → {bad_csv}  ({len(bad_win_rows)} windows)")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Subjects audited : {len(rows)}")
    print(f"  Subjects flagged : {n_flagged}  ({100*n_flagged/max(len(rows),1):.1f}%)")

    if not summary_df.empty and "pct_flagged_epoch" in summary_df.columns:
        ep = summary_df["pct_flagged_epoch"]
        print(f"  Epoch rejection  : mean={ep.mean():.2f}%  max={ep.max():.2f}%  "
              f"(threshold: |amp|>{EPOCH_THRESH*V2UV:.0f}µV)")

    if not summary_df.empty and "n_flat_channels" in summary_df.columns:
        fl = summary_df["n_flat_channels"]
        print(f"  Flat channels    : {int(fl.sum())} total across {int((fl>0).sum())} subjects")

    if not summary_df.empty and "n_noisy_channels" in summary_df.columns:
        ns = summary_df["n_noisy_channels"]
        print(f"  Noisy channels   : {int(ns.sum())} total across {int((ns>0).sum())} subjects")

    if run_ica and not summary_df.empty and "n_artifact_components" in summary_df.columns:
        ac = summary_df["n_artifact_components"]
        print(f"  ICA artifact comps: mean={ac[ac>=0].mean():.1f}  "
              f"max={int(ac.max())}  (kurtosis>{KURTOSIS_THRESH:.0f})")

    flagged_df = summary_df[summary_df.get("flag", False) == True] if "flag" in summary_df.columns else pd.DataFrame()
    if not flagged_df.empty:
        print(f"\n  Flagged subjects:")
        for _, row in flagged_df.iterrows():
            print(f"    {row['subject']}  — {row.get('issues','')}")

    print(f"\n  Summary → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG data quality audit on windowed dataset."
    )
    parser.add_argument("--task",    type=str, default=TASK,    help="Task name")
    parser.add_argument("--release", type=str, default=RELEASE, help="Release label, e.g. R5")
    parser.add_argument("--all",     action="store_true",
                        help="Audit all task_release folders found in WINDOWS_DIR")
    parser.add_argument("--no_ica",  action="store_true",
                        help="Skip ICA (faster, but won't catch artifact components)")
    args = parser.parse_args()

    run_ica = not args.no_ica

    if args.all:
        folders = sorted(WINDOWS_DIR.glob("*_R*")) if WINDOWS_DIR.exists() else []
        if not folders:
            print(f"No windowed datasets found in {WINDOWS_DIR}")
            sys.exit(1)
        for folder in folders:
            parts = folder.name.rsplit("_", 1)
            if len(parts) == 2:
                audit_task(parts[0], parts[1], run_ica=run_ica)
    else:
        audit_task(args.task, args.release, run_ica=run_ica)
