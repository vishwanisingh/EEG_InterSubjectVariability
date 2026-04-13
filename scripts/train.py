"""
scripts/train.py
----------------
Main training script. Loads the per-subject windowed dataset from disk,
splits by subjects, applies a normalization strategy, trains a model,
and saves results.

Run prepare_data.py first if the windowed dataset doesn't exist yet.

Normalization methods (--norm)
------------------------------
  global_zscore       Per-channel mean/std across all training windows [default]
  robust              Per-channel median/IQR across all training windows
  euclidean_alignment Per-subject covariance whitening then global z-score
  temporal_baseline   Per-window baseline correction then global z-score
  frequency_band      Z-score each EEG frequency band separately then sum
  riemannian          Tangent-space projection at Riemannian mean covariance

Usage
-----
    # Single model, single norm (original behaviour)
    python scripts/train.py --model BDTCN --norm global_zscore

    # Run 5 baseline models × 1 norm method
    python scripts/train.py --norm robust --run_baselines

    # Full ablation: 5 models × all 6 norm methods (30 runs)
    python scripts/train.py --ablation

    # Specific subset
    python scripts/train.py --ablation --norm global_zscore robust ea
    python scripts/train.py --ablation --models EEGNet EEGConformer
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import (
    TASK, RELEASE, WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    BATCH_SIZE, NUM_WORKERS,
    N_CHANS, N_OUTPUTS, N_TIMES, SFREQ,
    N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, MIN_DELTA,
    VALID_FRAC, TEST_FRAC, SEED, EXCLUDED_SUBJECTS,
)
from src.data.disk_dataset import load_windowed_dataset
from src.models.registry import create_model
from src.training.trainer import train_model
from src.training.engine import validate_model
from src.utils.device import setup_device
from src.utils.io import save_results_to_csv, save_model_weights
from src.utils.plotting import plot_training_history


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_MODELS = [
    "EEGNet",
    "Deep4Net",
    "ShallowFBCSPNet",
    "EEGTCNet",
    "EEGConformer",
]

RELEASE_MAP = {
    "R1":  "ds005505", "R2":  "ds005506", "R3":  "ds005507",
    "R4":  "ds005508", "R5":  "ds005509", "R6":  "ds005510",
    "R7":  "ds005511", "R8":  "ds005512", "R9":  "ds005514",
    "R10": "ds005515", "R11": "ds005516",
}
ALL_RELEASES     = list(RELEASE_MAP.keys())
TRAIN_RELEASES   = ["R1","R2","R3","R4","R5","R6","R8","R9","R10"]
VAL_RELEASES     = ["R11"]
TEST_RELEASES    = ["R5"]

ALL_NORM_METHODS = [
    "global_zscore",
    "robust",
    "euclidean_alignment",
    "temporal_baseline",
    "frequency_band",
    "riemannian",
]

# Shorthand aliases accepted on the CLI
NORM_ALIASES = {
    "zscore": "global_zscore",
    "gz":     "global_zscore",
    "rb":     "robust",
    "ea":     "euclidean_alignment",
    "tb":     "temporal_baseline",
    "fb":     "frequency_band",
    "rm":     "riemannian",
}

# Number of leading samples used as within-window baseline.
# Windows start 500 ms after stimulus (shift_after_stim=0.5 in task_config).
# We use the first 50 ms (5 samples at 100 Hz) as an intra-window baseline —
# removes slow drifts without touching event-related dynamics.
BASELINE_SAMPLES: int = 5

# Frequency bands for frequency_band normalisation (Hz).
# Nyquist at 100 Hz = 50 Hz; gamma capped at 45 Hz.
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Normalization: parameter containers
# ─────────────────────────────────────────────────────────────────────────────

class NormParams:
    """Base class — holds fitted normalisation statistics."""
    rt_mean: float
    rt_std:  float


class GlobalZScoreParams(NormParams):
    def __init__(self, mean, std, rt_mean, rt_std):
        self.mean    = mean      # (C,)
        self.std     = std       # (C,)
        self.rt_mean = rt_mean
        self.rt_std  = rt_std


class RobustParams(NormParams):
    def __init__(self, median, iqr, rt_mean, rt_std):
        self.median  = median    # (C,)
        self.iqr     = iqr       # (C,)
        self.rt_mean = rt_mean
        self.rt_std  = rt_std


class EuclideanAlignmentParams(NormParams):
    def __init__(self, whitening, mean, std, rt_mean, rt_std):
        self.whitening = whitening   # {subject_id: ndarray (C, C)}
        self.mean      = mean        # (C,) on aligned signal
        self.std       = std         # (C,)
        self.rt_mean   = rt_mean
        self.rt_std    = rt_std


class TemporalBaselineParams(NormParams):
    def __init__(self, mean, std, rt_mean, rt_std,
                 n_baseline: int = BASELINE_SAMPLES):
        self.mean       = mean
        self.std        = std
        self.rt_mean    = rt_mean
        self.rt_std     = rt_std
        self.n_baseline = n_baseline


class FrequencyBandParams(NormParams):
    def __init__(self, band_stats, rt_mean, rt_std):
        # band_stats[name] = (mean (C,), std (C,))
        self.band_stats = band_stats
        self.rt_mean    = rt_mean
        self.rt_std     = rt_std


class RiemannianParams(NormParams):
    def __init__(self, riemann_mean, tangent_mean, tangent_std,
                 projection, rt_mean, rt_std):
        self.riemann_mean = riemann_mean   # (C, C)
        self.tangent_mean = tangent_mean   # (n_tangent,)
        self.tangent_std  = tangent_std    # (n_tangent,)
        self.projection   = projection     # (n_tangent, C, T)
        self.rt_mean      = rt_mean
        self.rt_std       = rt_std


# ─────────────────────────────────────────────────────────────────────────────
# Normalization: fitters  (compute params from training data)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_train_arrays(train_set) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all training windows into RAM once.
    Works with both EEGWindowDataset and ConcatDataset.

    Returns
    -------
    X        : float32 (N, C, T)
    y        : float32 (N,)
    subjects : list[str] length N
    """
    from torch.utils.data import ConcatDataset as _ConcatDataset
    import pandas as pd

    N = len(train_set)
    X = np.empty((N, N_CHANS, N_TIMES), dtype=np.float32)
    y = np.empty(N,                     dtype=np.float32)

    # Collect subject IDs — handle both EEGWindowDataset and ConcatDataset
    if hasattr(train_set, "get_metadata"):
        sids = train_set.get_metadata()["subject"].tolist()
    elif isinstance(train_set, _ConcatDataset):
        # Recursively gather metadata from each sub-dataset
        frames = []
        for ds in train_set.datasets:
            if hasattr(ds, "get_metadata"):
                frames.append(ds.get_metadata())
        if frames:
            sids = pd.concat(frames, ignore_index=True)["subject"].tolist()
        else:
            sids = ["unknown"] * N
    else:
        sids = ["unknown"] * N

    # Sample up to max_windows to avoid OOM on large datasets
    max_windows = 5000
    if N > max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, max_windows, replace=False)
        print(f"  [norm-fit] Sampling {max_windows}/{N} windows for norm fitting…")
    else:
        idx = np.arange(N)
        print(f"  [norm-fit] Collecting {N} training windows into RAM…")

    M = len(idx)
    X = X[:M]
    y = y[:M]
    sids = [sids[i] for i in idx] if sids[0] != "unknown" else sids[:M]

    for j, i in enumerate(idx):
        item = train_set[int(i)]
        xi, yi = item[0], item[1]
        X[j] = xi.numpy()
        y[j] = yi.item()

    return X, y, sids


def fit_norm(method: str, train_set) -> NormParams:
    """Fit normalisation parameters from the training split."""
    X, y, sids = _collect_train_arrays(train_set)
    print(f"  [norm-fit] Fitting '{method}' on {len(X)} windows…")

    if method == "global_zscore":
        mean = X.mean(axis=(0, 2))
        std  = X.std(axis=(0, 2)).clip(min=1e-8)
        return GlobalZScoreParams(
            mean=mean, std=std,
            rt_mean=float(y.mean()),
            rt_std=float(y.std().clip(min=1e-8)),
        )

    elif method == "robust":
        Xt     = X.transpose(0, 2, 1).reshape(-1, N_CHANS)   # (N*T, C)
        median = np.median(Xt, axis=0)
        iqr    = (np.percentile(Xt, 75, axis=0)
                  - np.percentile(Xt, 25, axis=0)).clip(min=1e-8)
        rt_iqr = float((np.percentile(y, 75)
                        - np.percentile(y, 25)).clip(min=1e-8))
        return RobustParams(
            median=median, iqr=iqr,
            rt_mean=float(np.median(y)), rt_std=rt_iqr,
        )

    elif method == "euclidean_alignment":
        from scipy.linalg import sqrtm as mat_sqrt
        unique_sids = list(dict.fromkeys(sids))
        whitening: Dict[str, np.ndarray] = {}
        print(f"  [EA] Whitening {len(unique_sids)} subjects…")
        for sid in unique_sids:
            mask  = np.array([s == sid for s in sids])
            X_sub = X[mask]                                      # (n, C, T)
            covs  = (np.einsum('nct,ndt->ncd', X_sub, X_sub)
                     / X_sub.shape[2])
            R     = covs.mean(axis=0) + 1e-6 * np.eye(N_CHANS)
            W     = np.real(np.linalg.inv(mat_sqrt(R)))
            whitening[sid] = W.astype(np.float32)

        # Align all training windows then compute global stats
        X_aln = np.stack([whitening[s] @ X[i] for i, s in enumerate(sids)])
        mean  = X_aln.mean(axis=(0, 2))
        std   = X_aln.std(axis=(0, 2)).clip(min=1e-8)
        return EuclideanAlignmentParams(
            whitening=whitening, mean=mean, std=std,
            rt_mean=float(y.mean()),
            rt_std=float(y.std().clip(min=1e-8)),
        )

    elif method == "temporal_baseline":
        baseline = X[:, :, :BASELINE_SAMPLES].mean(axis=2, keepdims=True)
        X_corr   = X - baseline
        mean     = X_corr.mean(axis=(0, 2))
        std      = X_corr.std(axis=(0, 2)).clip(min=1e-8)
        return TemporalBaselineParams(
            mean=mean, std=std,
            rt_mean=float(y.mean()),
            rt_std=float(y.std().clip(min=1e-8)),
        )

    elif method == "frequency_band":
        band_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, (lo, hi) in FREQ_BANDS.items():
            X_band = np.stack([_bandpass(x, lo, hi) for x in X])
            mean   = X_band.mean(axis=(0, 2)).astype(np.float32)
            std    = X_band.std(axis=(0, 2)).clip(min=1e-8).astype(np.float32)
            band_stats[name] = (mean, std)
        return FrequencyBandParams(
            band_stats=band_stats,
            rt_mean=float(y.mean()),
            rt_std=float(y.std().clip(min=1e-8)),
        )

    elif method == "riemannian":
        try:
            from pyriemann.utils.mean import mean_covariance
            from pyriemann.utils.tangentspace import tangent_space
        except ImportError:
            raise ImportError(
                "pyriemann is required for Riemannian normalisation.\n"
                "Install:  pip install pyriemann --break-system-packages"
            )
        n_tangent = N_CHANS * (N_CHANS + 1) // 2
        covs      = (np.einsum('nct,ndt->ncd', X, X) / N_TIMES
                     + 1e-6 * np.eye(N_CHANS)[None])
        print("  [Riemann] Computing Riemannian mean (may take ~30 s)…")
        R_mean    = mean_covariance(covs, metric="riemann")
        tvecs     = tangent_space(covs, R_mean)              # (N, n_tangent)
        t_mean    = tvecs.mean(axis=0).astype(np.float32)
        t_std     = tvecs.std(axis=0).clip(min=1e-8).astype(np.float32)
        rng       = np.random.default_rng(SEED)
        proj_raw  = rng.standard_normal((n_tangent, N_CHANS * N_TIMES))
        proj      = (proj_raw
                     / np.linalg.norm(proj_raw, axis=0, keepdims=True)
                     ).astype(np.float32)
        return RiemannianParams(
            riemann_mean=R_mean.astype(np.float32),
            tangent_mean=t_mean,
            tangent_std=t_std,
            projection=proj.reshape(n_tangent, N_CHANS, N_TIMES),
            rt_mean=float(y.mean()),
            rt_std=float(y.std().clip(min=1e-8)),
        )

    else:
        raise ValueError(
            f"Unknown norm method '{method}'. "
            f"Choose from: {ALL_NORM_METHODS}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Normalization: appliers  (transform one window at __getitem__ time)
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """4th-order Butterworth bandpass.  x: (C, T) → (C, T)."""
    nyq  = SFREQ / 2.0
    sos  = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def apply_norm(x: np.ndarray, params: NormParams,
               subject_id: str = "") -> np.ndarray:
    """
    Apply fitted normalisation to a single window.

    Parameters
    ----------
    x          : float32 (C, T)
    params     : fitted NormParams
    subject_id : str — only used by EA and Riemannian

    Returns
    -------
    x_norm : float32 (C, T)
    """
    if isinstance(params, GlobalZScoreParams):
        return ((x - params.mean[:, None])
                / params.std[:, None]).astype(np.float32)

    elif isinstance(params, RobustParams):
        return ((x - params.median[:, None])
                / params.iqr[:, None]).astype(np.float32)

    elif isinstance(params, EuclideanAlignmentParams):
        W = params.whitening.get(subject_id)
        x_aln = (W @ x) if W is not None else x   # unseen subject → no-op
        return ((x_aln - params.mean[:, None])
                / params.std[:, None]).astype(np.float32)

    elif isinstance(params, TemporalBaselineParams):
        baseline = x[:, :params.n_baseline].mean(axis=1, keepdims=True)
        x_corr   = x - baseline
        return ((x_corr - params.mean[:, None])
                / params.std[:, None]).astype(np.float32)

    elif isinstance(params, FrequencyBandParams):
        out = np.zeros_like(x)
        for name, (lo, hi) in FREQ_BANDS.items():
            mean, std = params.band_stats[name]
            out += (_bandpass(x, lo, hi) - mean[:, None]) / std[:, None]
        return (out / len(FREQ_BANDS)).astype(np.float32)

    elif isinstance(params, RiemannianParams):
        try:
            from pyriemann.utils.tangentspace import tangent_space
        except ImportError:
            raise ImportError("pyriemann required for Riemannian normalisation.")
        cov   = x @ x.T / N_TIMES + 1e-6 * np.eye(N_CHANS)
        tvec  = tangent_space(cov[None], params.riemann_mean)[0]
        tvec_n = (tvec - params.tangent_mean) / params.tangent_std
        return np.einsum('k,kct->ct', tvec_n,
                         params.projection).astype(np.float32)

    else:
        raise TypeError(f"Unknown NormParams type: {type(params)}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalised dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NormalisedDataset(Dataset):
    """
    Thin wrapper around EEGWindowDataset that applies normalisation
    on-the-fly in __getitem__. The base dataset is never modified.

    Parameters
    ----------
    base_dataset : EEGWindowDataset
    params       : fitted NormParams
    subject_ids  : list[str] — one per window, aligned with base_dataset
    """

    def __init__(self, base_dataset, params: NormParams,
                 subject_ids: List[str]):
        self.base    = base_dataset
        self.params  = params
        self.sids    = subject_ids

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y, _ = self.base[idx]
        x_norm = apply_norm(x.numpy(), self.params, self.sids[idx])
        rt_n   = ((y.item() - self.params.rt_mean)
                  / (self.params.rt_std + 1e-8))
        return (torch.from_numpy(x_norm),
                torch.tensor([rt_n], dtype=torch.float32))


def _sid_list(dataset) -> List[str]:
    """Return one subject-ID string per window. Handles ConcatDataset."""
    from torch.utils.data import ConcatDataset as _ConcatDataset
    import pandas as pd
    if hasattr(dataset, "get_metadata"):
        return dataset.get_metadata()["subject"].tolist()
    elif isinstance(dataset, _ConcatDataset):
        frames = []
        for ds in dataset.datasets:
            if hasattr(ds, "get_metadata"):
                frames.append(ds.get_metadata())
        if frames:
            return pd.concat(frames, ignore_index=True)["subject"].tolist()
    return ["unknown"] * len(dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Core training function  (one model × one norm method)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model_name:     str        = "BDTCN",
    norm_method:    str        = "global_zscore",
    print_stats:    bool       = True,
    results_csv:    Path       = None,
    train_releases: List[str]  = None,
    val_releases:   List[str]  = None,
    test_releases:  List[str]  = None,
) -> Dict:
    """
    Full training pipeline for one (model, norm_method) combination.

    Parameters
    ----------
    model_name     : any key in MODEL_REGISTRY
    norm_method    : one of ALL_NORM_METHODS
    print_stats    : verbose epoch logging
    results_csv    : override output CSV path (default: RESULTS_DIR/results.csv)
    train_releases : list of release labels to train on (default: TRAIN_RELEASES)
    val_releases   : list of release labels for validation (default: VAL_RELEASES)
    test_releases  : list of release labels for test (default: TEST_RELEASES)

    Returns
    -------
    dict with all training and test metrics
    """
    if train_releases is None:
        train_releases = TRAIN_RELEASES
    if val_releases is None:
        val_releases = VAL_RELEASES
    if test_releases is None:
        test_releases = TEST_RELEASES
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if results_csv is None:
        results_csv = RESULTS_DIR / "results.csv"

    # Resolve CLI shorthand aliases
    norm_method = NORM_ALIASES.get(norm_method, norm_method)
    if norm_method not in ALL_NORM_METHODS:
        raise ValueError(
            f"Unknown norm method '{norm_method}'. "
            f"Valid: {ALL_NORM_METHODS + list(NORM_ALIASES)}"
        )

    tag    = f"{norm_method}__{model_name}"
    device = setup_device()

    # ── [1/6] Load dataset ────────────────────────────────────────────────────
    print(f"\n[1/6] Loading datasets")
    print(f"  Train releases : {train_releases}")
    print(f"  Val   releases : {val_releases}")
    print(f"  Test  releases : {test_releases}")

    def _load_releases(releases, label):
        datasets = []
        for rel in releases:
            root = WINDOWS_DIR / f"{TASK}_{rel}"
            if not root.exists():
                print(f"  [skip] {TASK}_{rel} not found")
                continue
            try:
                ds = load_windowed_dataset(
                    root_dir=root,
                    target_key="target",
                    eager=False,
                    excluded_subjects=EXCLUDED_SUBJECTS,
                )
                datasets.append(ds)
                print(f"  {label} {rel}: {len(ds)} windows, {len(ds.get_subject_ids())} subjects")
            except Exception as e:
                print(f"  [skip] {rel}: {e}")
        if not datasets:
            raise RuntimeError(f"No data found for {label} releases {releases}")
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    # ── [2/6] Split ───────────────────────────────────────────────────────────
    print("\n[2/6] Building splits…")
    train_set = _load_releases(train_releases, "train")
    valid_set = _load_releases(val_releases,   "val  ")
    test_set  = _load_releases(test_releases,  "test ")
    print(f"  Train : {len(train_set)} windows")
    print(f"  Valid : {len(valid_set)} windows")
    print(f"  Test  : {len(test_set)} windows")

    # ── [3/6] Fit normalisation ───────────────────────────────────────────────
    print(f"\n[3/6] Fitting normalisation: {norm_method}")
    t0          = time.time()
    norm_params = fit_norm(norm_method, train_set)
    print(f"  Fitted in {time.time()-t0:.1f} s")

    # ── [4/6] Wrap datasets + build loaders ──────────────────────────────────
    print("\n[4/6] Wrapping datasets with normalisation…")
    train_sids = _sid_list(train_set)
    valid_sids = _sid_list(valid_set)
    test_sids  = _sid_list(test_set)

    train_ds = NormalisedDataset(train_set, norm_params, train_sids)
    valid_ds = NormalisedDataset(valid_set, norm_params, valid_sids)
    test_ds  = NormalisedDataset(test_set,  norm_params, test_sids)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # ── [5/6] Build model ─────────────────────────────────────────────────────
    print(f"\n[5/6] Building model: {model_name}")
    model = create_model(model_name, n_chans=N_CHANS, n_outputs=N_OUTPUTS,
                         n_times=N_TIMES, sfreq=SFREQ)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}  ({n_params/1e6:.3f} M)")

    # ── [6/6] Train ───────────────────────────────────────────────────────────
    print(f"\n[6/6] Training  [{tag}]…")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        device=device,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        print_stats=print_stats,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    test_metrics = validate_model(
        test_loader, model, torch.nn.MSELoss(), device,
        print_batch_stats=False,
    )
    # Denormalize RMSE and MAE back to seconds (multiply by rt_std only —
    # RMSE/MAE measure spread, not location, so rt_mean is not added back)
    rt_std  = norm_params.rt_std
    rt_mean = norm_params.rt_mean
    import math as _math
    results["test_rmse"]     = test_metrics["rmse"] * rt_std   # seconds
    results["test_mae"]      = test_metrics["mae"]  * rt_std   # seconds
    results["test_r2"]       = test_metrics["r2"]              # unitless
    results["test_nrmse"]    = _math.sqrt(max(1.0 - test_metrics["r2"], 0.0)) if not _math.isnan(test_metrics["r2"]) else float("nan")
    results["val_rmse_sec"]  = results["best_val_rmse"] * rt_std
    results["val_mae_sec"]   = results["best_val_mae"]  * rt_std
    results["val_nrmse"]     = _math.sqrt(max(1.0 - results["best_val_r2"], 0.0)) if not _math.isnan(results["best_val_r2"]) else float("nan")
    results["rt_mean_train"] = rt_mean
    results["rt_std_train"]  = rt_std
    results["norm_method"]   = norm_method
    results["model_name"]    = model_name

    print(f"\n{'='*60}")
    print(f"RESULTS  [{tag}]")
    print(f"  RT stats   : mean={rt_mean:.4f}s  std={rt_std:.4f}s")
    print(f"  Val  NRMSE : {results['val_nrmse']:.4f}  RMSE: {results['val_rmse_sec']:.4f}s  R2: {results['best_val_r2']:.4f}  (epoch {results['best_epoch']})")
    print(f"  Test NRMSE : {results['test_nrmse']:.4f}  RMSE: {results['test_rmse']:.4f}s  R2: {results['test_r2']:.4f}")
    print(f"  (competition metric: NRMSE lower is better, 1.0 = predict mean)")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results_to_csv(results, results_csv, model_name=tag, n_params=n_params)
    save_model_weights(results["best_model_state"], tag, WEIGHTS_DIR)
    plot_training_history(results["train_history"], results["val_history"],
                          tag, RESULTS_DIR)

    print(f"\n✓ Results → {results_csv}")
    print(f"✓ Weights  → {WEIGHTS_DIR / (tag + '_weights.pt')}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation loop  (N models × M norm methods)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    norm_methods:   List[str],
    model_names:    List[str],
    train_releases: List[str] = None,
    val_releases:   List[str] = None,
    test_releases:  List[str] = None,
) -> None:
    """Train every (norm_method, model) combination and print a summary."""
    norm_methods   = [NORM_ALIASES.get(m, m) for m in norm_methods]
    train_releases = train_releases or TRAIN_RELEASES
    val_releases   = val_releases   or VAL_RELEASES
    test_releases  = test_releases  or TEST_RELEASES
    split_tag      = f"tr{'_'.join(train_releases)}_val{'_'.join(val_releases)}_te{'_'.join(test_releases)}"
    results_csv    = RESULTS_DIR / f"normalization_ablation_{split_tag}.csv"

    total = len(norm_methods) * len(model_names)
    print(f"\n{'='*70}")
    print(f"NORMALIZATION ABLATION  —  {len(norm_methods)} methods × "
          f"{len(model_names)} models = {total} runs")
    print(f"{'='*70}")

    all_results = []
    idx = 0

    for norm_method in norm_methods:
        for model_name in model_names:
            idx += 1
            print(f"\n[{idx}/{total}]  norm={norm_method}  model={model_name}")
            try:
                r = train(
                    model_name=model_name,
                    norm_method=norm_method,
                    print_stats=False,
                    results_csv=results_csv,
                    train_releases=train_releases,
                    val_releases=val_releases,
                    test_releases=test_releases,
                )
                all_results.append(r)
            except Exception as e:
                import traceback
                print(f"  [ERROR] {e}")
                traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY  (sorted by Test RMSE ↑ best)")
    print(f"{'='*70}")
    header = f"{'Norm Method':<25} {'Model':<20} {'Test RMSE':>10} {'Test R²':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(all_results, key=lambda x: x.get("test_rmse", 999)):
        print(f"{r['norm_method']:<25} {r['model_name']:<20} "
              f"{r['test_rmse']:>10.4f} {r['test_r2']:>8.3f}")
    print(f"\nFull results → {results_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG regression training with normalization ablation",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Mode flags ────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--ablation", action="store_true",
        help=(
            "Run full ablation: all --norm methods × all --models.\n"
            "Default norm: all 6.  Default models: 5 baselines."
        ),
    )
    mode.add_argument(
        "--run_baselines", action="store_true",
        help="Run all 5 baseline models with a single --norm method.",
    )

    # ── Selection args ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="BDTCN",
        help="Model name for single-run mode (default: BDTCN).",
    )
    parser.add_argument(
        "--norm", nargs="+",
        default=["global_zscore"],
        metavar="METHOD",
        help=(
            "Normalisation method(s).\n"
            "Full names:  " + "  ".join(ALL_NORM_METHODS) + "\n"
            "Short aliases: zscore gz | rb | ea | tb | fb | rm\n"
            "Default: global_zscore"
        ),
    )
    parser.add_argument(
        "--models", nargs="+",
        default=BASELINE_MODELS,
        metavar="MODEL",
        help=(
            "Model names for --ablation / --run_baselines.\n"
            "Default: " + " ".join(BASELINE_MODELS)
        ),
    )

    parser.add_argument(
        "--train_releases", type=str, default=",".join(TRAIN_RELEASES),
        help=f"Comma-separated train releases (default: {','.join(TRAIN_RELEASES)})"
    )
    parser.add_argument(
        "--val_releases", type=str, default=",".join(VAL_RELEASES),
        help=f"Comma-separated val releases (default: {','.join(VAL_RELEASES)})"
    )
    parser.add_argument(
        "--test_releases", type=str, default=",".join(TEST_RELEASES),
        help=f"Comma-separated test releases (default: {','.join(TEST_RELEASES)})"
    )

    args = parser.parse_args()

    train_rels = args.train_releases.split(",")
    val_rels   = args.val_releases.split(",")
    test_rels  = args.test_releases.split(",")

    if args.ablation:
        norm_list = args.norm if args.norm != ["global_zscore"] else ALL_NORM_METHODS
        run_ablation(
            norm_methods=norm_list, model_names=args.models,
            train_releases=train_rels, val_releases=val_rels, test_releases=test_rels,
        )

    elif args.run_baselines:
        norm = args.norm[0]
        run_ablation(
            norm_methods=[norm], model_names=args.models,
            train_releases=train_rels, val_releases=val_rels, test_releases=test_rels,
        )

    else:
        norm = args.norm[0]
        train(
            model_name=args.model, norm_method=norm, print_stats=True,
            train_releases=train_rels, val_releases=val_rels, test_releases=test_rels,
        )