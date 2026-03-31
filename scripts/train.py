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
from torch.utils.data import DataLoader, Dataset
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

    Returns
    -------
    X        : float32 (N, C, T)
    y        : float32 (N,)
    subjects : list[str] length N
    """
    meta  = train_set.get_metadata()
    N     = len(train_set)
    X     = np.empty((N, N_CHANS, N_TIMES), dtype=np.float32)
    y     = np.empty(N,                     dtype=np.float32)
    sids  = meta["subject"].tolist()

    print(f"  [norm-fit] Collecting {N} training windows into RAM…")
    for i in range(N):
        xi, yi = train_set[i]
        X[i]   = xi.numpy()
        y[i]   = yi.item()

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
        x, y   = self.base[idx]
        x_norm = apply_norm(x.numpy(), self.params, self.sids[idx])
        rt_n   = ((y.item() - self.params.rt_mean)
                  / (self.params.rt_std + 1e-8))
        return (torch.from_numpy(x_norm),
                torch.tensor([rt_n], dtype=torch.float32))


def _sid_list(dataset) -> List[str]:
    """Return one subject-ID string per window."""
    return dataset.get_metadata()["subject"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Core training function  (one model × one norm method)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model_name:  str        = "BDTCN",
    norm_method: str        = "global_zscore",
    print_stats: bool       = True,
    results_csv: Path       = None,
) -> Dict:
    """
    Full training pipeline for one (model, norm_method) combination.

    Parameters
    ----------
    model_name  : any key in MODEL_REGISTRY
    norm_method : one of ALL_NORM_METHODS
    print_stats : verbose epoch logging
    results_csv : override output CSV path (default: RESULTS_DIR/results.csv)

    Returns
    -------
    dict with all training and test metrics
    """
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

    root_dir = WINDOWS_DIR / f"{TASK}_{RELEASE}"

    # ── [1/6] Load dataset ────────────────────────────────────────────────────
    print(f"\n[1/6] Loading windowed dataset from {root_dir}")
    dataset = load_windowed_dataset(
        root_dir=root_dir,
        target_key="target",
        eager=False,
        excluded_subjects=EXCLUDED_SUBJECTS,
    )

    # ── [2/6] Split by subjects ───────────────────────────────────────────────
    print("\n[2/6] Splitting by subjects…")
    all_subjects = dataset.get_subject_ids()

    train_valid, test_subj = train_test_split(
        all_subjects, test_size=TEST_FRAC,
        random_state=check_random_state(SEED), shuffle=True,
    )
    adjusted_valid = VALID_FRAC / (1.0 - TEST_FRAC)
    train_subj, valid_subj = train_test_split(
        train_valid, test_size=adjusted_valid,
        random_state=check_random_state(SEED + 1), shuffle=True,
    )
    train_set, valid_set, test_set = dataset.split_by_subjects(
        train_subj, valid_subj, test_subj
    )
    print(f"  Train : {len(train_subj)} subjects  |  {len(train_set)} windows")
    print(f"  Valid : {len(valid_subj)} subjects  |  {len(valid_set)} windows")
    print(f"  Test  : {len(test_subj)} subjects  |  {len(test_set)} windows")

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
    results["test_rmse"] = test_metrics["rmse"]
    results["test_mae"]  = test_metrics["mae"]
    results["test_r2"]   = test_metrics["r2"]
    results["norm_method"] = norm_method
    results["model_name"]  = model_name

    print(f"\n{'='*60}")
    print(f"RESULTS  [{tag}]")
    print(f"  Val  RMSE : {results['best_val_rmse']:.6f}  "
          f"MAE : {results['best_val_mae']:.6f}  "
          f"R²  : {results['best_val_r2']:.4f}  "
          f"(epoch {results['best_epoch']})")
    print(f"  Test RMSE : {results['test_rmse']:.6f}  "
          f"MAE : {results['test_mae']:.6f}  "
          f"R²  : {results['test_r2']:.4f}")
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
    norm_methods: List[str],
    model_names:  List[str],
) -> None:
    """Train every (norm_method, model) combination and print a summary."""
    norm_methods = [NORM_ALIASES.get(m, m) for m in norm_methods]
    results_csv  = RESULTS_DIR / "normalization_ablation.csv"

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
                    print_stats=False,       # keep output compact
                    results_csv=results_csv,
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

    args = parser.parse_args()

    if args.ablation:
        # Full ablation: requested norms × requested models
        norm_list = args.norm if args.norm != ["global_zscore"] else ALL_NORM_METHODS
        run_ablation(norm_methods=norm_list, model_names=args.models)

    elif args.run_baselines:
        # All baseline models with one norm method
        norm = args.norm[0]                  # take the first if multiple given
        run_ablation(norm_methods=[norm], model_names=args.models)

    else:
        # Single run — original behaviour, now with norm param
        norm = args.norm[0]
        train(model_name=args.model, norm_method=norm, print_stats=True)