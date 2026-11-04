"""
scripts/explore_challenge_data.py
---------------------------------
Exploratory look at the EEG 2025 Challenge data (EEGChallengeDataset):
what preprocessing it carries, and what it looks like amplitude- and
frequency-wise.

Challenge preprocessing (eeg2025/downsample-datasets, process_eeg_data.m):
    pop_eegfiltnew(EEG, 'locutoff', 0.5, 'hicutoff', 50)   # FIR bandpass @ 500 Hz
    pop_resample(EEG, 100)                                 # 500 Hz -> 100 Hz
    no notch, no re-reference (HBN Cz ref kept), no ICA / artifact rejection

Figures (results/data_exploration/)
-----------------------------------
  01_raw_vs_challenge_psd.png   same recordings, raw 500 Hz vs challenge 100 Hz,
                                and the empirical filter gain between them
  02_psd_subjects.png           per-subject spectra + relative band power
  03_topomaps.png               scalp maps: RMS amplitude, band power,
                                how often each channel is extreme
  04_amplitude.png              per-channel SD and per-window peak-to-peak
  05_example_traces.png         10 s of signal on landmark channels
  summary.json                  the numbers behind the figures

Usage
-----
    # Download first (mini = 20 subjects/release):
    #   EEGChallengeDataset(release="R1", cache_dir="data/challenge", mini=True, task=...)
    python scripts/explore_challenge_data.py
    python scripts/explore_challenge_data.py --task symbolSearch --release R1

    # Raw comparison (fig 1) uses any overlapping subjects found in --raw_dir
    python scripts/explore_challenge_data.py --raw_dir data/ds005505
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import welch
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

mne.set_log_level("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHALLENGE_DIR = Path("data/challenge")
OUT_DIR       = Path("results/data_exploration")

WIN_S   = 2.0        # same window length as the training pipeline
PSD_SEG = 8.0        # Welch segment (s) -> 0.125 Hz resolution at any sfreq

# EGI HydroCel-129 electrodes around the eyes (EOG proxies)
EYE_CHANS = ["E8", "E14", "E21", "E25", "E125", "E126", "E127", "E128"]

# Approximate 10-20 landmarks on the HydroCel-129 net
LANDMARKS = {
    "Fp1 (E22)": "E22", "Fp2 (E9)": "E9", "E25 (eye)": "E25", "E127 (eye)": "E127",
    "Fz (E11)": "E11", "C3 (E36)": "E36", "C4 (E104)": "E104",
    "T7 (E45)": "E45", "Pz (E62)": "E62", "O1 (E70)": "E70",
    "Oz (E75)": "E75", "O2 (E83)": "E83",
}

BANDS = {
    "delta\n1–4":    (1, 4),
    "theta\n4–8":    (4, 8),
    "alpha\n8–13":   (8, 13),
    "beta\n13–30":   (13, 30),
    "gamma\n30–45":  (30, 45),
}

AMP_THRESH_UV = [100, 150, 200]
EXTREME_FACTOR = 5.0    # channel SD > 5 x recording median = extreme (as check_data_quality.py)
FLAT_UV        = 0.5    # channel SD below this = flat
PEAK_FACTOR    = 10.0   # narrowband peak > 10 x local spectral median = interference

# Palette (dataviz reference instance, light mode)
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8f8e88"
GRID      = "#e6e5e0"
C_CHAL    = "#2a78d6"   # slot 1 blue   — challenge data
C_RAW     = "#eb6834"   # slot 2 orange — raw data
C_EYE     = "#1baf7a"   # slot 3 aqua   — periocular channels
SEQ_BLUE  = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelcolor": INK_2, "axes.edgecolor": INK_MUTED, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 2,
})


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _subject(path: Path) -> str:
    return path.name.split("_")[0]


def _load_uv(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load one recording as µV, dropping the all-zero Cz reference."""
    if path.suffix == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True)
    else:
        raw = mne.io.read_raw(path, preload=True)
    raw.pick("eeg")
    if "Cz" in raw.ch_names:
        raw.drop_channels(["Cz"])
    return raw.get_data() * 1e6, raw.ch_names, raw.info["sfreq"]


def _psd(x: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    nper = int(PSD_SEG * sfreq)
    return welch(x, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)


def _windows(x: np.ndarray, sfreq: float) -> np.ndarray:
    """(C, T) -> (W, C, n) non-overlapping windows."""
    n = int(WIN_S * sfreq)
    w = x.shape[1] // n
    return x[:, : w * n].reshape(x.shape[0], w, n).transpose(1, 0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording statistics
# ─────────────────────────────────────────────────────────────────────────────

def recording_stats(path: Path) -> dict:
    x, chs, sfreq = _load_uv(path)
    f, p = _psd(x, sfreq)
    win = _windows(x, sfreq)
    ptp = np.ptp(win, axis=2)                                 # (W, C)
    eye = [chs.index(c) for c in EYE_CHANS if c in chs]
    not_eye = [i for i in range(len(chs)) if i not in eye]
    sd = x.std(1)
    extreme = (sd > EXTREME_FACTOR * np.median(sd)) | (sd < FLAT_UV)
    return {
        "subject": _subject(path), "file": path.name, "sfreq": sfreq,
        "n_chans": len(chs) + 1, "duration_s": x.shape[1] / sfreq,
        "chs": chs, "freqs": f, "psd": p,
        "ch_std": sd, "ch_mean": x.mean(1), "extreme": extreme,
        "peaks_hz": narrowband_peaks(f, np.median(p, 0)),
        "abs_p99": np.percentile(np.abs(x), 99, axis=1),
        "ptp_any": ptp.max(1),
        "ptp_any_ok": ptp[:, ~extreme].max(1),
        "ptp_eye": ptp[:, eye].max(1),
        "ptp_brain_median": np.median(ptp[:, not_eye], axis=1),
        "x": x,
    }


def narrowband_peaks(f: np.ndarray, p: np.ndarray) -> list[float]:
    """Frequencies in 1-45 Hz standing PEAK_FACTOR x above the +-2 Hz median."""
    df = f[1] - f[0]
    half = int(2 / df)
    peaks = []
    for i in np.where((f >= 1) & (f <= 45))[0]:
        local = np.median(np.r_[p[i - half : i - 1], p[i + 2 : i + half + 1]])
        if p[i] > PEAK_FACTOR * local and p[i] == p[i - 1 : i + 2].max():
            peaks.append(round(float(f[i]), 2))
    return peaks


def band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f >= lo) & (f < hi)
    return p[..., m].sum(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — raw vs challenge on the same recordings
# ─────────────────────────────────────────────────────────────────────────────

def fig_raw_vs_challenge(pairs: list[tuple[Path, Path]], out: Path) -> dict:
    raw_psd, chal_psd, gains = [], [], []
    for raw_path, chal_path in pairs:
        xr, chr_, sr = _load_uv(raw_path)
        xc, chc, sc = _load_uv(chal_path)
        common = [c for c in chc if c in chr_]
        xr = xr[[chr_.index(c) for c in common]]
        xc = xc[[chc.index(c) for c in common]]
        fr, pr = _psd(xr, sr)
        fc, pc = _psd(xc, sc)
        pr_m, pc_m = np.median(pr, 0), np.median(pc, 0)
        raw_psd.append(pr_m)
        chal_psd.append(pc_m)
        # Both grids are 0.125 Hz, so challenge bins are the first len(fc) raw bins
        ok = pr[:, : len(fc)].min(1) > 0                      # skip channels flat in raw
        gains.append(10 * np.log10(np.median(pc[ok] / pr[ok, : len(fc)], axis=0)))

    raw_psd, chal_psd = np.median(raw_psd, 0), np.median(chal_psd, 0)
    gain = np.median(gains, 0)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.28})

    m = fr > 0.1
    a.loglog(fr[m], raw_psd[m], color=C_RAW, label="Raw HBN (500 Hz)")
    m = fc > 0.1
    a.loglog(fc[m], chal_psd[m], color=C_CHAL, label="Challenge (100 Hz)")
    for hz, txt, ha in [(0.5, "0.5 Hz", "left"), (50, "50 Hz", "right"), (60, "60 Hz line", "left")]:
        a.axvline(hz, color=INK_MUTED, lw=1, ls="--", zorder=0)
        a.text(hz * (1.06 if ha == "left" else 0.95), 1.0, txt, transform=a.get_xaxis_transform(),
               va="top", ha=ha, fontsize=8, color=INK_2)
    a.set_xlabel("Frequency (Hz)")
    a.set_ylabel("PSD, median over channels (µV²/Hz)")
    a.set_title("Same recordings: raw vs challenge", loc="left")
    a.legend(loc="lower left")

    m = fc > 0.1
    b.semilogx(fc[m], gain[m], color=C_CHAL)
    b.axhline(0, color=INK_MUTED, lw=1)
    for db in (-3, -6):
        b.axhline(db, color=INK_MUTED, lw=0.8, ls=":")
    b.set_ylim(-40, 5)
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("Gain, challenge / raw (dB)")
    b.set_title("Gain added by the challenge preprocessing", loc="left")

    def _crossing(level: float, low_side: bool) -> float | None:
        band = (fc > 0.1) & (fc < 5) if low_side else (fc > 20)
        idx = np.where(band)[0]
        g = gain[idx]
        hit = np.where(g >= level)[0] if low_side else np.where(g < level)[0]
        if not len(hit) or hit[0] == 0:
            return None
        j = hit[0]                                         # interpolate between bins
        return float(np.interp(level, sorted([g[j - 1], g[j]]),
                               [fc[idx][j - 1], fc[idx][j]][:: 1 if g[j] > g[j - 1] else -1]))

    cut = {
        "highpass_-3dB_hz": _crossing(-3, True), "highpass_-6dB_hz": _crossing(-6, True),
        "lowpass_-3dB_hz":  _crossing(-3, False), "lowpass_-6dB_hz":  _crossing(-6, False),
        "gain_db_at": {str(hz): float(np.interp(hz, fc, gain)) for hz in (0.25, 0.5, 1, 10, 30, 40, 45, 49)},
    }
    b.text(0.12, -3, "−3 dB", va="bottom", fontsize=7.5, color=INK_MUTED)
    b.text(0.12, -6, "−6 dB", va="bottom", fontsize=7.5, color=INK_MUTED)
    for key, dx, ha in [("highpass_-3dB_hz", 10, "left"), ("lowpass_-3dB_hz", -10, "right")]:
        if cut[key]:
            b.plot(cut[key], -3, "o", ms=7, color=C_CHAL, mec="white", mew=2, zorder=5)
            b.annotate(f"−3 dB at {cut[key]:.2f} Hz", (cut[key], -3), xytext=(dx, -34),
                       textcoords="offset points", ha=ha, fontsize=8, color=INK,
                       arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 0.8})

    fig.suptitle(f"Raw HBN vs EEG2025 challenge data — {len(pairs)} matched CCD recordings",
                 x=0.06, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(out)
    plt.close(fig)
    cut["n_pairs"] = len(pairs)
    cut["raw_line_peak_60hz_db"] = float(
        10 * np.log10(np.interp(60, fr, raw_psd) / np.median(raw_psd[(fr > 55) & (fr < 58)]))
    )
    return cut


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — spectra across subjects
# ─────────────────────────────────────────────────────────────────────────────

def fig_psd_subjects(recs: list[dict], out: Path) -> dict:
    f = recs[0]["freqs"]
    subjects = sorted({r["subject"] for r in recs})
    subj_psd = np.array([
        np.mean([np.median(r["psd"], 0) for r in recs if r["subject"] == s], 0) for s in subjects
    ])
    total = band_power(f, subj_psd, 1, 45)
    rel = {b: band_power(f, subj_psd, lo, hi) / total for b, (lo, hi) in BANDS.items()}

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                               gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.25})
    m = (f >= 0.5) & (f <= 50)
    for i, (lo, hi) in enumerate(BANDS.values()):
        if i % 2 == 0:
            a.axvspan(lo, hi, color="#f4f3ef", zorder=0, lw=0)
        a.text((lo + hi) / 2, 1.0, list(BANDS)[i].split("\n")[0], transform=a.get_xaxis_transform(),
               ha="center", va="top", fontsize=8, color=INK_2)
    for sp in subj_psd:
        a.semilogy(f[m], sp[m], color=C_CHAL, lw=0.8, alpha=0.3)
    a.semilogy(f[m], np.median(subj_psd, 0)[m], color=C_CHAL, lw=2.4,
               label="Median subject")
    a.plot([], [], color=C_CHAL, lw=0.8, alpha=0.4, label="Individual subjects")
    a.set_xlim(0, 50)
    a.set_ylim(bottom=1e-3)
    a.set_xlabel("Frequency (Hz)")
    a.set_ylabel("PSD, median over channels (µV²/Hz)")
    a.set_title(f"Power spectrum — {len(subjects)} subjects", loc="left")
    a.legend(loc="upper right", bbox_to_anchor=(1, 0.92))

    rng = np.random.default_rng(0)
    for i, (band, vals) in enumerate(rel.items()):
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        b.scatter(np.full(len(vals), i) + jitter, 100 * vals, s=22, color=C_CHAL,
                  edgecolor="white", linewidth=1, zorder=3)
        med = 100 * np.median(vals)
        b.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2, zorder=4)
        b.text(i + 0.32, med, f"{med:.0f}%", va="center", fontsize=8, color=INK)
        mad = np.median(np.abs(vals - np.median(vals)))
        for j in np.where(np.abs(vals - np.median(vals)) > 6 * mad)[0]:
            if band.startswith("beta"):
                b.annotate(subjects[j], (i + jitter[j], 100 * vals[j]), xytext=(-8, 0),
                           textcoords="offset points", ha="right", va="center",
                           fontsize=7.5, color=INK_2)
    b.set_xticks(range(len(rel)), list(rel))
    b.set_ylabel("Share of 1–45 Hz power (%)")
    b.set_title("Relative band power (dot = subject, bar = median)", loc="left")
    b.grid(axis="x", visible=False)

    fig.savefig(out)
    plt.close(fig)
    return {k.split("\n")[0]: float(np.median(v)) for k, v in rel.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — topographies
# ─────────────────────────────────────────────────────────────────────────────

def fig_topomaps(recs: list[dict], out: Path) -> None:
    chs = recs[0]["chs"]
    info = mne.create_info(chs, 100.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"))
    f = recs[0]["freqs"]

    rms = np.median([r["ch_std"] for r in recs], 0)
    psd = np.median([r["psd"] for r in recs], 0)               # (C, F)
    total = band_power(f, psd, 1, 45)
    maps = [
        ("RMS amplitude\n(µV)", rms, ".0f"),
        ("Delta 1–4 Hz\n(% of 1–45 Hz power)", 100 * band_power(f, psd, 1, 4) / total, ".0f"),
        ("Alpha 8–13 Hz\n(% of 1–45 Hz power)", 100 * band_power(f, psd, 8, 13) / total, ".0f"),
        ("20–45 Hz, muscle range\n(% of 1–45 Hz power)", 100 * band_power(f, psd, 20, 45) / total, ".1f"),
        ("Extreme channel\n(% of recordings)", 100 * np.mean([r["extreme"] for r in recs], 0), ".0f"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for ax, (title, vals, fmt) in zip(axes, maps):
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, cmap=SEQ_BLUE, show=False,
                                     sensors=True, contours=0,
                                     vlim=(np.percentile(vals, 2), np.percentile(vals, 98)))
        ax.set_title(title, fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, format=f"%{fmt}")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=INK_2)
    fig.suptitle("Where the signal lives — median over all recordings (nose up)",
                 x=0.06, y=0.97, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(out)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — amplitude distributions
# ─────────────────────────────────────────────────────────────────────────────

def fig_amplitude(recs: list[dict], out: Path) -> dict:
    ch_std = np.concatenate([r["ch_std"] for r in recs])
    series = [
        ("Any channel (max over 128)", np.concatenate([r["ptp_any"] for r in recs]), INK_MUTED),
        ("Any channel, extreme channels removed",
         np.concatenate([r["ptp_any_ok"] for r in recs]), INK_2),
        ("Periocular channels (max)", np.concatenate([r["ptp_eye"] for r in recs]), C_EYE),
        ("Typical channel (median)", np.concatenate([r["ptp_brain_median"] for r in recs]), C_CHAL),
    ]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                               gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.25})

    bins = np.logspace(np.log10(max(ch_std.min(), 1)), np.log10(ch_std.max()), 50)
    a.hist(ch_std, bins=bins, color=C_CHAL, edgecolor="white", linewidth=0.8)
    a.set_xscale("log")
    a.axvline(np.median(ch_std), color=INK, lw=1.5)
    a.text(np.median(ch_std) * 1.08, 0.97, f"median {np.median(ch_std):.0f} µV",
           transform=a.get_xaxis_transform(), va="top", fontsize=8)
    a.set_xlabel("Channel SD over a recording (µV, log)")
    a.set_ylabel("Channels × recordings")
    a.set_title("Per-channel amplitude", loc="left")
    a.grid(axis="x", visible=False)

    stats = {}
    x = np.logspace(0.5, 4, 300)
    for label, v, color in series:
        surv = 100 * (v[None, :] > x[:, None]).mean(1)
        b.semilogx(x, surv, color=color, label=label,
                   ls="--" if label.startswith("Any channel (") else "-")
        stats[label] = {f">{t}uV": float(100 * (v > t).mean()) for t in AMP_THRESH_UV}
        stats[label]["median_uV"] = float(np.median(v))
    for t in AMP_THRESH_UV:
        b.axvline(t, color=INK_MUTED, lw=0.8, ls=":", zorder=0)
    b.set_ylim(0, 100)
    b.set_xlabel("Peak-to-peak within a 2 s window (µV, log) — dotted: 100 / 150 / 200 µV")
    b.set_ylabel("Windows exceeding (%)")
    b.set_title(f"How often windows exceed a threshold — {len(series[0][1]):,} windows",
                loc="left")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)

    fig.savefig(out)
    plt.close(fig)
    stats["channel_sd_uV"] = {"p5": float(np.percentile(ch_std, 5)),
                              "median": float(np.median(ch_std)),
                              "p95": float(np.percentile(ch_std, 95))}
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — example traces
# ─────────────────────────────────────────────────────────────────────────────

def fig_traces(rec: dict, out: Path, seconds: float = 10.0) -> None:
    x, chs, sfreq = rec["x"], rec["chs"], rec["sfreq"]
    n = int(seconds * sfreq)
    # Pick the 10 s stretch whose periocular activity is closest to the recording median
    eye = [chs.index(c) for c in EYE_CHANS]
    starts = np.arange(0, x.shape[1] - n, n)
    act = np.array([np.ptp(x[eye, s : s + n], axis=1).max() for s in starts])
    s0 = starts[np.argsort(act)[int(0.75 * len(act))]]
    t = np.arange(n) / sfreq

    names = list(LANDMARKS)
    spacing = 120.0
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    for i, name in enumerate(names):
        y = x[chs.index(LANDMARKS[name]), s0 : s0 + n]
        off = -i * spacing
        color = C_EYE if "eye" in name or name.startswith("Fp") else C_CHAL
        ax.plot(t, y - y.mean() + off, color=color, lw=0.9)
        ax.text(-0.15, off, name, ha="right", va="center", fontsize=8.5, color=INK)
    ax.plot([seconds + 0.2] * 2, [0, -100], color=INK, lw=2, clip_on=False)
    ax.text(seconds + 0.3, -50, "100 µV", va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, seconds)
    ax.set_xlabel("Time (s)")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{rec['subject']} · {seconds:.0f} s at t = {s0 / sfreq:.0f} s "
                 f"(75th-percentile eye activity) — aqua = frontal/periocular",
                 loc="left")
    fig.savefig(out)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="contrastChangeDetection")
    p.add_argument("--release", default="R1")
    p.add_argument("--mini", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raw_dir", type=Path, default=Path("data/ds005505"),
                   help="Raw HBN BIDS dir for the raw-vs-challenge comparison (optional)")
    args = p.parse_args()

    root = CHALLENGE_DIR / f"EEG2025r{args.release[1:]}{'mini' if args.mini else ''}"
    files = sorted(root.glob(f"sub-*/eeg/*task-{args.task}_*eeg.bdf"))
    if not files:
        sys.exit(f"No challenge files under {root} for task {args.task} — download them first.")
    out_dir = OUT_DIR / f"{args.task}_{args.release}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} recordings from {root}")

    recs = [recording_stats(f) for f in files]
    summary = {
        "task": args.task, "release": args.release, "mini": args.mini,
        "n_recordings": len(recs), "n_subjects": len({r["subject"] for r in recs}),
        "sfreq_hz": sorted({r["sfreq"] for r in recs}),
        "n_channels_incl_Cz": sorted({r["n_chans"] for r in recs}),
        "duration_s": {"min": min(r["duration_s"] for r in recs),
                       "median": float(np.median([r["duration_s"] for r in recs])),
                       "max": max(r["duration_s"] for r in recs)},
        "channel_mean_abs_uV_median": float(np.median(np.abs(np.concatenate([r["ch_mean"] for r in recs])))),
        "abs_amplitude_p99_uV_median": float(np.median(np.concatenate([r["abs_p99"] for r in recs]))),
    }
    n_ext = np.array([r["extreme"].sum() for r in recs])
    freq_ext = np.mean([r["extreme"] for r in recs], 0)
    order = np.argsort(-freq_ext)[:15]
    summary["extreme_channels"] = {
        "rule": f"SD > {EXTREME_FACTOR} x recording median, or SD < {FLAT_UV} uV",
        "per_recording": {"median": float(np.median(n_ext)), "mean": float(n_ext.mean()),
                          "max": int(n_ext.max())},
        "recordings_with_any": float((n_ext > 0).mean()),
        "most_frequent": {recs[0]["chs"][i]: round(float(freq_ext[i]), 2) for i in order},
    }
    summary["narrowband_interference"] = {r["file"]: r["peaks_hz"] for r in recs if r["peaks_hz"]}

    pairs = []
    if args.raw_dir.exists():
        for f in files:
            raw = args.raw_dir / _subject(f) / "eeg" / f.name.replace(".bdf", ".set")
            if raw.exists():
                pairs.append((raw, f))
    if pairs:
        print(f"Raw vs challenge on {len(pairs)} matched recordings ...")
        summary["filter_response"] = fig_raw_vs_challenge(pairs, out_dir / "01_raw_vs_challenge_psd.png")
    else:
        print("No matching raw recordings — skipping figure 1")

    summary["relative_band_power_median"] = fig_psd_subjects(recs, out_dir / "02_psd_subjects.png")
    fig_topomaps(recs, out_dir / "03_topomaps.png")
    summary["amplitude"] = fig_amplitude(recs, out_dir / "04_amplitude.png")
    fig_traces(recs[0], out_dir / "05_example_traces.png")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFigures -> {out_dir}")


if __name__ == "__main__":
    main()
