"""
scripts/train.py
----------------
Main training script. Loads the per-subject windowed dataset from disk,
splits by subjects, trains a model, and saves results.

Run prepare_data.py first if the windowed dataset doesn't exist yet.

Normalization methods (--norm)
------------------------------
  global_zscore       Per-channel mean/std across all training windows [default]
  robust              Per-channel median/IQR across all training windows
  euclidean_alignment Per-subject covariance whitening then global z-score
  temporal_baseline   Per-window baseline correction then global z-score
  frequency_band      Z-score each EEG frequency band separately then sum
  riemannian          Tangent-space projection at Riemannian mean covariance
  riemannian_recentering
                      Per-subject re-centering: x → M_s^{-1/2} (x − x̄), M_s =
                      Riemannian mean of the subject's window covariances
                      (val/test subjects use their own unlabeled EEG). Output is
                      already unit-scale; no global z-score

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

import os
import sys
import zlib
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

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
    "AttentionBaseNet",
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
    "riemannian_recentering",
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
    "rr":     "riemannian_recentering",
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

# Max windows per subject used to estimate its re-centering reference.
# CCD event windows are ~60/subject; the cap only bites for sliding windows.
RECENTER_MAX_WINDOWS: int = 200


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


class RiemannianRecenteringParams(NormParams):
    def __init__(self, recenter, rt_mean, rt_std):
        self.recenter = recenter     # {subject_id: ndarray (C, C)} = M_s^{-1/2}
        self.rt_mean  = rt_mean
        self.rt_std   = rt_std


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


# ── Riemannian re-centering helpers ──────────────────────────────────────────

def _window_covs(X: np.ndarray) -> torch.Tensor:
    """
    Shrunk spatial covariance per window.  X: (n, C, T) → (n, C, C) float64.

    Correlations are shrunk toward identity with OAS intensity (= OAS on
    channel-standardised data), i.e. the covariance toward its own diagonal.
    Shrinking toward trace/C·I instead lets one broken electrode (~1e6× median
    variance in some HBN subjects) swamp every other channel. Keeps matrices
    SPD at T=200 for C=129. Flat channels (Cz reference, dead electrodes at
    ~1e-30× median variance) get the median variance as a placeholder, so
    their ~0 signal is not amplified.
    """
    X  = torch.from_numpy(np.asarray(X, dtype=np.float64))
    X  = X - X.mean(dim=2, keepdim=True)
    n_c, n_t = X.shape[1], X.shape[2]
    S  = X @ X.transpose(1, 2) / n_t
    d  = S.diagonal(dim1=1, dim2=2)
    med = d.median(dim=1, keepdim=True).values
    d  = torch.where(d > 1e-6 * med, d, med)
    sd = d.sqrt()
    R  = S / (sd[:, :, None] * sd[:, None, :])
    R.diagonal(dim1=1, dim2=2).fill_(1.0)
    alpha  = (R ** 2).mean(dim=(1, 2))
    den    = (n_t + 1) * (alpha - 1.0 / n_c)
    shrink = torch.where(den > 0, (alpha + 1.0) / den,
                         torch.ones_like(den)).clamp(max=1.0)
    R = (1 - shrink)[:, None, None] * R + shrink[:, None, None] * torch.eye(n_c, dtype=R.dtype)
    return sd[:, :, None] * R * sd[:, None, :]


def _spd_fn(S: torch.Tensor, fn) -> torch.Tensor:
    """Apply `fn` to the eigenvalues of symmetric matrices (..., C, C)."""
    w, V = torch.linalg.eigh(S)
    return (V * fn(w).unsqueeze(-2)) @ V.transpose(-1, -2)


def _riemann_mean(covs: torch.Tensor, tol: float = 1e-3,
                  max_iter: int = 50) -> torch.Tensor:
    """
    Affine-invariant Riemannian (Karcher) mean of SPD matrices (n, C, C).
    Gradient descent with pyriemann's adaptive step size — a fixed step of 1
    stalls on subjects with a broken electrode. tol=1e-3 lands within ~1e-3
    Riemannian distance of the exact mean (subjects are ~30 apart) in ~10-20
    iterations; tighter tolerances often need >200.
    """
    M = covs.mean(dim=0)
    nu, tau = 1.0, float("inf")
    for _ in range(max_iter):
        M_sqrt  = _spd_fn(M, torch.sqrt)
        M_isqrt = _spd_fn(M, torch.rsqrt)
        J = _spd_fn(M_isqrt @ covs @ M_isqrt, torch.log).mean(dim=0)
        M = M_sqrt @ _spd_fn(nu * J, torch.exp) @ M_sqrt
        crit = torch.linalg.norm(J).item()
        if nu * crit < tau:
            nu, tau = 0.95 * nu, nu * crit
        else:
            nu *= 0.5
        if crit <= tol or nu <= tol:
            break
    return M


def _recentering_matrix(windows: np.ndarray, sid: str) -> np.ndarray:
    """M^{-1/2} for one subject, M = Riemannian mean of its window covariances."""
    rng = np.random.default_rng([SEED, zlib.crc32(str(sid).encode())])
    n   = len(windows)
    idx = (np.sort(rng.choice(n, RECENTER_MAX_WINDOWS, replace=False))
           if n > RECENTER_MAX_WINDOWS else np.arange(n))
    covs = _window_covs(windows[idx])
    covs = covs[covs.diagonal(dim1=1, dim2=2).sum(-1) > 0]   # drop flat windows
    return _spd_fn(_riemann_mean(covs), torch.rsqrt).numpy().astype(np.float32)


def _subject_windows(dataset):
    """Yield (subject_id, windows memmap (n, C, T)) for every subject in a split."""
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    for part in parts:
        yield from part.iter_subject_windows()


def fit_subject_references(params: NormParams, dataset) -> None:
    """
    Add a re-centering matrix for every subject in `dataset` that has none yet.

    Uses only that subject's EEG, never its labels, so val/test subjects get
    their own reference too (the standard unsupervised re-centering protocol).
    No-op for methods without per-subject references.
    """
    if not isinstance(params, RiemannianRecenteringParams):
        return
    todo = [(sid, w) for sid, w in _subject_windows(dataset)
            if sid not in params.recenter]
    # One subject per thread (torch releases the GIL); single-threaded torch
    # inside, since batched 129×129 eigh is slower multi-threaded.
    n_workers = len(os.sched_getaffinity(0))
    print(f"  [RR] Riemannian mean for {len(todo)} subjects "
          f"({n_workers} threads)…")
    t0, n_threads = time.time(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ThreadPoolExecutor(n_workers) as pool:
            Ws = pool.map(lambda t: _recentering_matrix(t[1], t[0]), todo)
            for i, ((sid, _), W) in enumerate(zip(todo, Ws), 1):
                params.recenter[sid] = W
                if i % 200 == 0:
                    print(f"  [RR] {i}/{len(todo)}  ({time.time()-t0:.0f} s)")
    finally:
        torch.set_num_threads(n_threads)


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

    elif method == "riemannian_recentering":
        # References from ALL windows of every training subject (not just the
        # sampled X). No global z-score afterwards: typical windows already
        # have unit channel std, and artifact windows (std up to ~1e4) would
        # inflate a pooled std ~30× and shrink every normal window.
        params = RiemannianRecenteringParams(
            recenter={},
            rt_mean=float(y.mean()),
            rt_std=float(y.std().clip(min=1e-8)),
        )
        fit_subject_references(params, train_set)
        return params

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
    model_name : str
        Any key from src.models.registry.MODEL_REGISTRY.
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

    elif isinstance(params, RiemannianRecenteringParams):
        W = params.recenter.get(subject_id)
        if W is None:
            raise KeyError(
                f"No re-centering reference for subject '{subject_id}'. "
                "Call fit_subject_references(params, dataset) on its split first."
            )
        # Demean like the covariances were, else DC offsets get amplified too
        return (W @ (x - x.mean(axis=1, keepdims=True))).astype(np.float32)

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
        results_csv = RESULTS_DIR / "results2.csv"

    # ── [1/5] Load dataset from disk ─────────────────────────────────────────
    print(f"\n[1/5] Loading windowed dataset from {root_dir} ...")
    dataset = load_windowed_dataset(
        root_dir=root_dir,
        target_key="target",
        eager=False,                    # lazy by default — reads from disk on demand
        excluded_subjects=EXCLUDED_SUBJECTS,
    )

    # ── [2/5] Split subjects into train / valid / test ───────────────────────
    print("\n[2/5] Splitting by subjects...")
    all_subjects = dataset.get_subject_ids()

    train_valid, test_subj = train_test_split(
        all_subjects,
        test_size=TEST_FRAC,
        random_state=check_random_state(SEED),
        shuffle=True,
    )
    adjusted_valid = VALID_FRAC / (1.0 - TEST_FRAC)
    train_subj, valid_subj = train_test_split(
        train_valid,
        test_size=adjusted_valid,
        random_state=check_random_state(SEED + 1),
        shuffle=True,
    )

    print(f"  Train : {len(train_subj)} subjects")
    print(f"  Valid : {len(valid_subj)} subjects")
    print(f"  Test  : {len(test_subj)} subjects")

    train_set, valid_set, test_set = dataset.split_by_subjects(
        train_subj, valid_subj, test_subj
    )
    print(f"  Train : {len(train_set)} windows")
    print(f"  Valid : {len(valid_set)} windows")
    print(f"  Test  : {len(test_set)} windows")

    # ── [3/6] Fit normalisation ───────────────────────────────────────────────
    print(f"\n[3/6] Fitting normalisation: {norm_method}")
    t0          = time.time()
    norm_params = fit_norm(norm_method, train_set)
    fit_subject_references(norm_params, valid_set)
    fit_subject_references(norm_params, test_set)
    print(f"  Fitted in {time.time()-t0:.1f} s")

    # ── [4/5] Build model ────────────────────────────────────────────────────
    print(f"\n[4/5] Building model: {model_name}")
    model = create_model(model_name, n_chans=N_CHANS, n_outputs=N_OUTPUTS,
                         n_times=N_TIMES, sfreq=SFREQ)
    model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── [5/5] Train ──────────────────────────────────────────────────────────
    print("\n[5/5] Training...")
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
        print_stats=True,
    )

    # ── Evaluate on test set ─────────────────────────────────────────────────
    test_loss, test_rmse = validate_model(
        test_loader, model, torch.nn.MSELoss(), device, print_batch_stats=False
    )
    results["test_rmse"] = test_rmse
    results["test_loss"] = test_loss

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"  Best Val RMSE : {results['best_val_rmse']:.6f}  (epoch {results['best_epoch']})")
    print(f"  Test RMSE     : {test_rmse:.6f}")
    print(f"  Model         : {model_name}")
    print("="*60)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_results_to_csv(results, RESULTS_DIR / "results.csv", model_name=model_name)
    save_model_weights(results["best_model_state"], model_name, WEIGHTS_DIR)
    plot_training_history(results["train_history"], results["val_history"],
                          model_name, RESULTS_DIR)

    print(f"\n✓ Results → {RESULTS_DIR}")
    print(f"✓ Weights  → {WEIGHTS_DIR}")


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
            "Short aliases: zscore gz | rb | ea | tb | fb | rm | rr\n"
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
