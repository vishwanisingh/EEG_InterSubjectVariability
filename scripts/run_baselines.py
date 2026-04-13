"""
scripts/run_baselines.py
------------------------
Train all 5 baseline models on CCD and evaluate on:
  - CCD held-out test set        (in-domain)
  - a transfer task of your choice  (cross-task generalization)

Two split strategies are supported:

  subject  (default) — random 80:10:10 subject-level split within one release
  release            — train on earlier releases, val/test on later releases
                       e.g. train R1-R8, val R9-R10, test R11

Both EEG and RT targets are z-score normalized using training statistics only.

Usage
-----
    # Subject-wise split, transfer to SymbolSearch
    python scripts/run_baselines.py --transfer_task symbolSearch

    # Release-wise split, transfer to SeqLearning
    python scripts/run_baselines.py \\
        --split_mode release \\
        --train_releases R1,R2,R3,R4,R5,R6,R7,R8 \\
        --val_releases R9,R10 \\
        --test_releases R11 \\
        --transfer_task seqLearning8target

    # In-domain only (no transfer)
    python scripts/run_baselines.py --transfer_task none
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from configs.config import (
    WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    N_CHANS, N_OUTPUTS, N_TIMES, SFREQ,
    BATCH_SIZE, NUM_WORKERS,
    N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, MIN_DELTA,
    SEED, EXCLUDED_SUBJECTS,
)
from src.data.disk_dataset import load_windowed_dataset, EEGWindowDataset
from src.models.registry import create_model
from src.training.trainer import train_model
from src.training.engine import validate_model
from src.utils.device import setup_device
from src.utils.io import save_results_to_csv, save_model_weights
from src.utils.plotting import plot_training_history

import gc
import random


def set_seed(seed: int = SEED) -> None:
    """Seed everything for reproducible weight init and data ordering."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Baseline models (Stage 1 — five maximally different architectures) ────────
BASELINE_MODELS = [
    "ShallowFBCSPNet",
    "EEGNet",
    "EEGConformer",
    "EEGTCNet",
    "Deep4Net",
]

TRAIN_TASK = "contrastChangeDetection"

# Release → OpenNeuro dataset ID map
RELEASE_MAP = {
    "R1":  "ds005505", "R2":  "ds005506", "R3":  "ds005507",
    "R4":  "ds005508", "R5":  "ds005509", "R6":  "ds005510",
    "R7":  "ds005511", "R8":  "ds005512", "R9":  "ds005514",
    "R10": "ds005515", "R11": "ds005516",
}
ALL_RELEASES = list(RELEASE_MAP.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

class NormStats:
    """
    Per-channel EEG stats and scalar RT stats from training data only.
    Estimated from up to max_windows sampled training windows.

    When normalize_eeg=False, eeg_mean=0 and eeg_std=1 (identity transform).
    When normalize_rt=False,  rt_mean=0  and rt_std=1  (identity transform).
    """

    def __init__(
        self,
        train_dataset,
        normalize_eeg: bool = True,
        normalize_rt: bool = True,
        max_windows: int = 5000,
        eps: float = 1e-6,
    ):
        self.normalize_eeg = normalize_eeg
        self.normalize_rt  = normalize_rt

        # Identity defaults — used when normalization is disabled
        self.eeg_mean = np.zeros(N_CHANS, dtype=np.float32)
        self.eeg_std  = np.ones(N_CHANS,  dtype=np.float32)
        self.rt_mean  = 0.0
        self.rt_std   = 1.0

        if not normalize_eeg and not normalize_rt:
            print(f"  [NormStats] Both EEG and RT normalization disabled — identity transform")
            return

        print(f"  [NormStats] Estimating from up to {max_windows} windows ...")
        n   = min(max_windows, len(train_dataset))
        idx = np.random.default_rng(SEED).choice(len(train_dataset), n, replace=False)

        eeg_sum = np.zeros(N_CHANS, dtype=np.float64)
        eeg_sq  = np.zeros(N_CHANS, dtype=np.float64)
        rt_vals = []

        for i in idx:
            item = train_dataset[int(i)]
            x, y = item[0], item[1]   # safe unpack — dataset may return >2 values
            xn    = x.numpy()
            eeg_sum += xn.mean(axis=1)
            eeg_sq  += (xn ** 2).mean(axis=1)
            if not np.isnan(y.item()):
                rt_vals.append(y.item())

        if normalize_eeg:
            self.eeg_mean = (eeg_sum / n).astype(np.float32)
            self.eeg_std  = np.sqrt(
                np.maximum(eeg_sq / n - (eeg_sum / n) ** 2, 0)
            ).astype(np.float32) + eps
            print(f"  [NormStats] EEG z-score ON  — "
                  f"std range [{self.eeg_std.min():.4f}, {self.eeg_std.max():.4f}]")
        else:
            print(f"  [NormStats] EEG normalization OFF — raw signal used")

        if normalize_rt:
            rt_arr       = np.array(rt_vals, dtype=np.float64)
            self.rt_mean = float(rt_arr.mean())
            self.rt_std  = float(rt_arr.std()) + eps
            print(f"  [NormStats] RT  z-score ON  — mean={self.rt_mean:.4f}  std={self.rt_std:.4f}")
        else:
            print(f"  [NormStats] RT  normalization OFF — raw RT values used")


class NormalizedDataset(Dataset):
    """
    Wraps EEGWindowDataset with on-the-fly z-score normalization.
    Always use the NormStats from the training split.
    When normalization is disabled the NormStats identity values
    (mean=0, std=1) make this a no-op transform.
    """

    def __init__(self, dataset, stats: NormStats):
        self.dataset   = dataset
        self.stats     = stats
        self._eeg_mean = torch.from_numpy(stats.eeg_mean).unsqueeze(1)  # (C,1)
        self._eeg_std  = torch.from_numpy(stats.eeg_std).unsqueeze(1)   # (C,1)
        self._rt_mean  = stats.rt_mean
        self._rt_std   = stats.rt_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item   = self.dataset[idx]
        x, y   = item[0], item[1]   # safe unpack
        x_norm = (x - self._eeg_mean) / self._eeg_std
        y_norm = (y - self._rt_mean)  / self._rt_std
        return x_norm, y_norm

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self._rt_std + self._rt_mean


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers
# ─────────────────────────────────────────────────────────────────────────────

def subject_split(dataset: EEGWindowDataset, seed: int = SEED):
    """80:10:10 random subject-level split."""
    subjects = dataset.get_subject_ids()

    train_valid, test_subj = train_test_split(
        subjects, test_size=0.10,
        random_state=check_random_state(seed), shuffle=True,
    )
    train_subj, valid_subj = train_test_split(
        train_valid, test_size=0.10 / 0.90,
        random_state=check_random_state(seed + 1), shuffle=True,
    )
    train_set, valid_set, test_set = dataset.split_by_subjects(
        train_subj, valid_subj, test_subj
    )
    print(f"  Subject split — "
          f"train: {len(train_subj)} ({len(train_set)} windows) | "
          f"val: {len(valid_subj)} ({len(valid_set)} windows) | "
          f"test: {len(test_subj)} ({len(test_set)} windows)")
    return train_set, valid_set, test_set


def release_split(
    task: str,
    train_releases: list,
    val_releases: list,
    test_releases: list,
):
    """
    Load separate datasets per release group and concatenate.
    Train releases → training set
    Val releases   → validation set
    Test releases  → test set
    """
    def _load_releases(releases, label):
        # eager=True loads all windows into RAM and closes file handles
        # immediately — avoids FD_SETSIZE crash from thousands of open mmaps.
        # Val and test sets are always eager. Train set is eager too since
        # the mmap approach hits OS file descriptor limits with many releases.
        use_eager = True
        datasets = []
        for rel in releases:
            root = WINDOWS_DIR / f"{task}_{rel}"
            if not root.exists():
                print(f"  [skip] {task}_{rel} not found")
                continue
            try:
                ds = load_windowed_dataset(
                    root_dir=root,
                    target_key="target",
                    eager=use_eager,
                    excluded_subjects=EXCLUDED_SUBJECTS,
                )
                datasets.append(ds)
                print(f"  ✓ {task}_{rel}: {len(ds)} windows, "
                      f"{len(ds.get_subject_ids())} subjects")
            except Exception as e:
                print(f"  [skip] {task}_{rel}: {e}")

        if not datasets:
            raise RuntimeError(
                f"No data found for {label} releases {releases}. "
                f"Run prepare_data.py for each release."
            )
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    print(f"  Loading train releases : {train_releases}")
    train_set = _load_releases(train_releases, "train")
    print(f"  Loading val releases   : {val_releases}")
    valid_set = _load_releases(val_releases,   "val")
    print(f"  Loading test releases  : {test_releases}")
    test_set  = _load_releases(test_releases,  "test")

    print(f"  Release split — "
          f"train: {len(train_set)} windows | "
          f"val: {len(valid_set)} windows | "
          f"test: {len(test_set)} windows")
    return train_set, valid_set, test_set


def load_transfer_dataset(transfer_task: str, release: str) -> EEGWindowDataset:
    """Load the full transfer task dataset (all subjects, no split)."""
    root = WINDOWS_DIR / f"{transfer_task}_{release}"
    if not root.exists():
        raise FileNotFoundError(
            f"Transfer dataset not found at {root}.\n"
            f"Run: python scripts/prepare_data.py  (with TASK='{transfer_task}')"
        )
    ds = load_windowed_dataset(
        root_dir=root,
        target_key="target",
        eager=True,
        excluded_subjects=EXCLUDED_SUBJECTS,
    )
    print(f"  ✓ Transfer ({transfer_task}): {len(ds)} windows, "
          f"{len(ds.get_subject_ids())} subjects")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    stats: NormStats,
    label: str,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        x, y = batch[0].to(device).float(), batch[1].to(device).float()
        all_preds.append(model(x).cpu())
        all_targets.append(y.cpu())

    p = torch.cat(all_preds).view(-1)
    t = torch.cat(all_targets).view(-1)

    rmse_norm = float(torch.sqrt(((p - t) ** 2).mean()))
    mae_norm  = float(torch.abs(p - t).mean())
    ss_res    = float(((t - p) ** 2).sum())
    ss_tot    = float(((t - t.mean()) ** 2).sum())
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # NRMSE = RMSE / std(y_true) — scale-free, competition metric
    # = sqrt(1 - R2) when computed on normalized targets
    # A score of 1.0 = predicting the mean; lower is better
    import math
    nrmse     = math.sqrt(max(1.0 - r2, 0.0)) if not math.isnan(r2) else float("nan")

    p_sec = p * stats.rt_std + stats.rt_mean
    t_sec = t * stats.rt_std + stats.rt_mean
    rmse_sec = float(torch.sqrt(((p_sec - t_sec) ** 2).mean()))
    mae_sec  = float(torch.abs(p_sec - t_sec).mean())

    print(f"  [{label}]")
    print(f"    NRMSE: {nrmse:.4f}  R2: {r2:.4f}  (competition metric: lower is better, 1.0 = predict mean)")
    print(f"    RMSE : {rmse_sec:.4f} s  MAE: {mae_sec:.4f} s")

    return dict(
        rmse_norm=rmse_norm, mae_norm=mae_norm, r2=r2,
        nrmse=nrmse,
        rmse_sec=rmse_sec,   mae_sec=mae_sec,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_single_baseline(
    model_name: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    ccd_test_loader: DataLoader,
    transfer_loader: DataLoader | None,
    transfer_task: str,
    stats: NormStats,
    device: torch.device,
    results_csv: Path,
    split_label: str,
) -> dict | None:

    print(f"\n{'='*70}")
    print(f"Model : {model_name}  |  Split: {split_label}")
    print('='*70)

    model = None
    row   = None

    try:
        # ── Re-seed before every model so weight init is reproducible ─────────
        set_seed(SEED)

        # Build model
        try:
            model = create_model(
                model_name, n_chans=N_CHANS, n_outputs=N_OUTPUTS,
                n_times=N_TIMES, sfreq=SFREQ,
            )
            model.to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {n_params:,}  ({n_params/1e6:.3f} M)")
        except Exception as e:
            print(f"  [ERROR] Could not create {model_name}: {e}")
            return None

        # Train
        try:
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
        except Exception as e:
            print(f"  [ERROR] Training failed for {model_name}: {e}")
            import traceback; traceback.print_exc()
            return None

        # ── Evaluate — in-domain CCD test ────────────────────────────────────
        print(f"\n  Evaluating ...")
        ccd_m = evaluate(ccd_test_loader, model, device, stats,
                         label=f"{TRAIN_TASK} test (in-domain)")

        # ── Evaluate — transfer task ──────────────────────────────────────────
        transfer_m = {}
        if transfer_loader is not None:
            transfer_m = evaluate(transfer_loader, model, device, stats,
                                  label=f"{transfer_task} (zero-shot transfer)")

        # ── Degradation ───────────────────────────────────────────────────────
        delta_rmse_norm = (transfer_m.get("rmse_norm", float("nan"))
                           - ccd_m["rmse_norm"])
        delta_r2        = (ccd_m["r2"]
                           - transfer_m.get("r2", float("nan")))

        print(f"\n  {'='*60}")
        print(f"  {model_name} — Summary")
        print(f"    Val RMSE (best)   : {results['best_val_rmse']:.4f}  "
              f"(epoch {results['best_epoch']})")
        print(f"    CCD  RMSE (norm)  : {ccd_m['rmse_norm']:.4f}  "
              f"R2: {ccd_m['r2']:.4f}  "
              f"({ccd_m['rmse_sec']:.4f} s)")
        if transfer_m:
            print(f"    {transfer_task[:20]:<20} RMSE: {transfer_m['rmse_norm']:.4f}  "
                  f"R2: {transfer_m['r2']:.4f}  "
                  f"({transfer_m['rmse_sec']:.4f} s)")
            print(f"    ΔRMSE (norm)      : {delta_rmse_norm:+.4f}")
        print(f"  {'='*60}")

        # ── Save ──────────────────────────────────────────────────────────────
        # Denormalize val RMSE to seconds for consistent reporting
        val_rmse_sec = results["best_val_rmse"] * stats.rt_std + 0.0
        # Note: RMSE denormalizes by multiplying by std only (not adding mean)
        # because RMSE measures spread, not location

        row = {
            "split_mode":       split_label,
            "train_task":       TRAIN_TASK,
            "transfer_task":    transfer_task,
            # ── Primary metrics in seconds (old format compatible) ────────────
            "Test RMSE":        ccd_m["rmse_sec"],
            "Test MAE":         ccd_m["mae_sec"],
            "Test R2":          ccd_m["r2"],
            "Test NRMSE":       ccd_m["nrmse"],
            "Val RMSE":         val_rmse_sec,
            "Val MAE":          results["best_val_mae"] * stats.rt_std,
            "Val R2":           results["best_val_r2"],
            "Best Epoch":       results["best_epoch"],
            "Train Time (s)":   round(results["train_time_s"], 1),
            # ── Transfer metrics in seconds ───────────────────────────────────
            "Transfer RMSE":    transfer_m.get("rmse_sec",  float("nan")),
            "Transfer MAE":     transfer_m.get("mae_sec",   float("nan")),
            "Transfer R2":      transfer_m.get("r2",        float("nan")),
            "Transfer NRMSE":   transfer_m.get("nrmse",     float("nan")),
            "Delta RMSE (s)":   (transfer_m.get("rmse_sec", float("nan"))
                                 - ccd_m["rmse_sec"]),
            "Delta R2":         delta_r2,
            # ── Reproducibility ───────────────────────────────────────────────
            "normalize_eeg":    stats.normalize_eeg,
            "normalize_rt":     stats.normalize_rt,
            "rt_mean_train":    stats.rt_mean,
            "rt_std_train":     stats.rt_std,
        }

        # Write directly — our row schema is richer than io.py's save_results_to_csv
        import os
        row_with_model = {"Model": model_name, "Params (M)": round(n_params / 1e6, 3), **row}
        if os.path.exists(results_csv):
            existing = pd.read_csv(results_csv)
            updated  = pd.concat([existing, pd.DataFrame([row_with_model])], ignore_index=True)
        else:
            updated = pd.DataFrame([row_with_model])
        updated.to_csv(results_csv, index=False)
        print(f"  Results saved to {results_csv}")
        save_model_weights(results["best_model_state"], model_name, WEIGHTS_DIR)
        plot_training_history(results["train_history"], results["val_history"],
                              model_name, RESULTS_DIR)

    finally:
        # Always release GPU memory after each model regardless of success/failure
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Raise file descriptor limit ───────────────────────────────────────────
    # Loading many releases simultaneously opens thousands of mmap file handles.
    # Default FD_SETSIZE (1024) is too low — raise to 65536.
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f"  [fd] File descriptor limit raised to {target} (was {soft})")
    except Exception as e:
        print(f"  [fd] Could not raise file descriptor limit: {e}")
        print(f"       Run: ulimit -n 65536  before this script if crashes occur")

    device = setup_device()

    transfer_task = args.transfer_task  # e.g. "symbolSearch" or "none"
    do_transfer   = transfer_task.lower() != "none"

    # ── [1] Load / split CCD ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BASELINE TRAINING")
    print(f"  Train task     : {TRAIN_TASK}")
    print(f"  Transfer task  : {transfer_task if do_transfer else 'none'}")
    print(f"  Split mode     : {args.split_mode}")
    norm_label = (
        f"EEG={'z-score per channel' if args.normalize_eeg else 'raw (no normalization)'} | "
        f"RT={'z-score' if args.normalize_rt else 'raw (no normalization)'}"
    )
    print(f"  Normalization  : {norm_label}")
    print('='*70)

    print(f"\n[1/5] Loading and splitting {TRAIN_TASK}")

    if args.split_mode == "subject":
        # Load single release, split by subjects 80:10:10
        root = WINDOWS_DIR / f"{TRAIN_TASK}_{args.release}"
        ccd  = load_windowed_dataset(
            root_dir=root, target_key="target",
            eager=True, excluded_subjects=EXCLUDED_SUBJECTS,
        )
        print(f"  Loaded {len(ccd)} windows, "
              f"{len(ccd.get_subject_ids())} subjects from {args.release}")
        train_set, valid_set, ccd_test_set = subject_split(ccd)
        split_label = f"subject_{args.release}"

    else:  # release-wise
        train_rels = args.train_releases.split(",")
        val_rels   = args.val_releases.split(",")
        test_rels  = args.test_releases.split(",")
        train_set, valid_set, ccd_test_set = release_split(
            TRAIN_TASK, train_rels, val_rels, test_rels
        )
        split_label = f"release_tr{'_'.join(train_rels)}_te{'_'.join(test_rels)}"

    # ── [2] Normalization stats (training split only) ─────────────────────────
    print(f"\n[2/5] Computing normalization statistics")
    stats = NormStats(
        train_set,
        normalize_eeg=args.normalize_eeg,
        normalize_rt=args.normalize_rt,
        max_windows=5000,
    )

    # ── [3] Wrap with normalization ───────────────────────────────────────────
    print(f"\n[3/5] Applying normalization")
    norm_train    = NormalizedDataset(train_set,    stats)
    norm_valid    = NormalizedDataset(valid_set,    stats)
    norm_ccd_test = NormalizedDataset(ccd_test_set, stats)

    train_loader    = DataLoader(norm_train,    batch_size=BATCH_SIZE,
                                 shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
    valid_loader    = DataLoader(norm_valid,    batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS)
    ccd_test_loader = DataLoader(norm_ccd_test, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS)

    # ── [4] Load transfer task ────────────────────────────────────────────────
    transfer_loader = None
    if do_transfer:
        print(f"\n[4/5] Loading transfer task: {transfer_task}")
        # For release-wise split, use the first test release for transfer
        transfer_rel = (args.release if args.split_mode == "subject"
                        else args.test_releases.split(",")[0])
        transfer_ds     = load_transfer_dataset(transfer_task, transfer_rel)
        norm_transfer   = NormalizedDataset(transfer_ds, stats)  # CCD train stats
        transfer_loader = DataLoader(norm_transfer, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=NUM_WORKERS)
    else:
        print(f"\n[4/5] No transfer task requested")

    # ── [5] Train all baselines ───────────────────────────────────────────────
    if args.models:
        requested = [m.strip() for m in args.models.split(",")]
        unknown   = [m for m in requested if m not in BASELINE_MODELS]
        if unknown:
            print(f"[ERROR] Unknown model(s): {unknown}")
            print(f"        Available: {BASELINE_MODELS}")
            sys.exit(1)
        models_to_run = requested
    else:
        models_to_run = BASELINE_MODELS

    print(f"\n[5/5] Training {len(models_to_run)} model(s): {models_to_run}")
    results_csv = RESULTS_DIR / f"baseline_results_{split_label}.csv"
    all_results = {}

    # Global seed before the loop — individual models re-seed inside run_single_baseline
    set_seed(SEED)

    for i, model_name in enumerate(models_to_run, 1):
        print(f"\n── Baseline {i}/{len(models_to_run)}: {model_name} ──")
        row = run_single_baseline(
            model_name=model_name,
            train_loader=train_loader,
            valid_loader=valid_loader,
            ccd_test_loader=ccd_test_loader,
            transfer_loader=transfer_loader,
            transfer_task=transfer_task,
            stats=stats,
            device=device,
            results_csv=results_csv,
            split_label=split_label,
        )
        if row:
            all_results[model_name] = row

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"  Split       : {split_label}")
        print(f"  Train task  : {TRAIN_TASK}")
        print(f"  Transfer    : {transfer_task if do_transfer else 'none'}")
        norm_label = (
            f"EEG={'z-score' if args.normalize_eeg else 'raw'} | "
            f"RT={'z-score' if args.normalize_rt else 'raw'}"
        )
        print(f"  Norm        : {norm_label}")
        print()

        col = f"{'Model':<22} {'CCD RMSE':>10} {'CCD R2':>8}"
        if do_transfer:
            col += f" {'Xfer RMSE':>10} {'Xfer R2':>8} {'ΔRMSE':>8}"
        print(col)
        print("-"*70)

        for name, r in all_results.items():
            row_str = (f"{name:<22} "
                       f"{r['ccd_rmse_norm']:>10.4f} "
                       f"{r['ccd_r2']:>8.4f}")
            if do_transfer:
                row_str += (f" {r['transfer_rmse_norm']:>10.4f} "
                            f"{r['transfer_r2']:>8.4f} "
                            f"{r['delta_rmse_norm']:>+8.4f}")
            print(row_str)

        print("="*70)
        print(f"\nResults → {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline models: train on CCD, test in-domain + transfer"
    )

    # Transfer task
    parser.add_argument(
        "--transfer_task", type=str, default="symbolSearch",
        help="Transfer test task. Use 'none' for in-domain only. "
             "Options: symbolSearch, seqLearning8target, seqLearning6target, "
             "surroundSupp, none"
    )

    # Split mode
    parser.add_argument(
        "--split_mode", type=str, default="subject",
        choices=["subject", "release"],
        help="'subject' = 80:10:10 random subject split within one release. "
             "'release' = train/val/test on different releases."
    )

    # Subject-wise split: single release
    parser.add_argument(
        "--release", type=str, default="R5",
        help="Release to use for subject-wise split (e.g. R5)."
    )

    # Release-wise split: explicit release lists
    parser.add_argument(
        "--train_releases", type=str, default="R1,R2,R3,R4,R5,R6,R7,R8",
        help="Comma-separated train releases (release mode only)."
    )
    parser.add_argument(
        "--val_releases", type=str, default="R9,R10",
        help="Comma-separated val releases (release mode only)."
    )
    parser.add_argument(
        "--test_releases", type=str, default="R11",
        help="Comma-separated test releases (release mode only)."
    )

    # Model selection — single model or comma-separated list
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names to run, e.g. "
             "'EEGConformer,EEGTCNet'. Omit to run all BASELINE_MODELS. "
             f"Available: {', '.join(BASELINE_MODELS)}"
    )

    # Normalization flags
    parser.add_argument(
        "--normalize_eeg", action=argparse.BooleanOptionalAction, default=True,
        help="Z-score EEG per channel using training stats. "
             "Use --no-normalize_eeg to disable. (default: True)"
    )
    parser.add_argument(
        "--normalize_rt", action=argparse.BooleanOptionalAction, default=True,
        help="Z-score RT targets using training stats. "
             "Use --no-normalize_rt to disable. (default: True)"
    )

    args = parser.parse_args()
    main(args)