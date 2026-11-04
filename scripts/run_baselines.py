"""
scripts/run_baselines.py
------------------------
Train all baseline models sequentially on the pre-saved windowed dataset
and write a comparison CSV.

Run prepare_data.py / prepare_all_releases.py first.

Data modes:
  --data-mode single   : use only R5 (train/val/test split by subject)
  --data-mode all      : use all releases; test=R5, val=one release, train=rest

Normalization flags (both default off for backward compatibility):
  --normalize-targets  : z-score RT per subject
  --normalize-eeg      : z-score EEG per channel per subject

Usage:
    python scripts/run_baselines.py                                         # all releases, all models
    python scripts/run_baselines.py --data-mode single                      # R5 only
    python scripts/run_baselines.py --normalize-targets --normalize-eeg     # with normalization
    python scripts/run_baselines.py --model EEGConformer                    # one model
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from configs.config import (
    TASK, RELEASE, WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    BATCH_SIZE, NUM_WORKERS,
    N_CHANS, N_OUTPUTS, N_TIMES, SFREQ,
    N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, MIN_DELTA,
    VALID_FRAC, TEST_FRAC, SEED, EXCLUDED_SUBJECTS,
)
from src.data.disk_dataset import load_windowed_dataset, EEGWindowDataset
from src.models.registry import create_model, get_baseline_models
from src.training.trainer import train_model
from src.training.engine import validate_model
from src.utils.device import setup_device, set_seed, seed_worker
from src.utils.io import save_results_to_csv, save_model_weights
from src.utils.plotting import plot_training_history


# ── All available releases (R7 excluded — no CCD task) ───────────────────────
ALL_RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R8", "R9", "R10", "R11"]
TEST_RELEASE = "R5"

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
    "AttentionBaseNet",
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

# ── Shared config dicts ───────────────────────────────────────────────────────
MODEL_CONFIG = dict(
    n_chans=N_CHANS,
    n_outputs=N_OUTPUTS,
    n_times=N_TIMES,
    sfreq=SFREQ,
)

TRAINING_CONFIG = dict(
    n_epochs=N_EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    print_stats=True,
)

OUTPUT_CONFIG = dict(
    results_dir=RESULTS_DIR,
    weights_dir=WEIGHTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_release(
    release: str,
    normalize_targets: bool = False,
    normalize_eeg: bool = False,
) -> EEGWindowDataset | None:
    """Load one release from disk. Returns None if not found."""
    root_dir = WINDOWS_DIR / f"{TASK}_{release}"
    if not root_dir.exists():
        print(f"  [warn] {release} not found at {root_dir} — skipping")
        return None
    try:
        ds = load_windowed_dataset(
            root_dir=root_dir,
            target_key="target",
            eager=False,
            excluded_subjects=EXCLUDED_SUBJECTS,
            normalize_targets=normalize_targets,
            normalize_eeg=normalize_eeg,
        )
        print(f"  {release:>3} : {len(ds):>6} windows  ({len(ds.get_subject_ids())} subjects)")
        return ds
    except Exception as e:
        print(f"  [warn] {release}: {e} — skipping")
        return None


class RobustNormStats:
    """
    Per-channel EEG robust stats (median/IQR) and scalar RT stats.
    Uses median and IQR instead of mean/std — resistant to outliers.
    Estimated from up to max_windows sampled training windows.
    CCD training stats applied identically to transfer tasks.
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

        # Identity defaults
        self.eeg_mean = np.zeros(N_CHANS, dtype=np.float32)
        self.eeg_std  = np.ones(N_CHANS,  dtype=np.float32)
        self.rt_mean  = 0.0
        self.rt_std   = 1.0

        if not normalize_eeg and not normalize_rt:
            print(f"  [RobustNormStats] Both disabled — identity transform")
            return

        print(f"  [RobustNormStats] Estimating from up to {max_windows} windows ...")
        n   = min(max_windows, len(train_dataset))
        idx = np.random.default_rng(SEED).choice(len(train_dataset), n, replace=False)

        eeg_windows = []
        rt_vals     = []

        for i in idx:
            item = train_dataset[int(i)]
            x, y = item[0], item[1]
            eeg_windows.append(x.numpy())
            if not np.isnan(y.item()):
                rt_vals.append(y.item())

        if normalize_eeg:
            # Stack → (N, C, T) → reshape to (N*T, C) for per-channel stats
            X   = np.stack(eeg_windows)             # (N, C, T)
            Xt  = X.transpose(0, 2, 1).reshape(-1, N_CHANS)  # (N*T, C)
            self.eeg_mean = np.median(Xt, axis=0).astype(np.float32)
            iqr = (np.percentile(Xt, 75, axis=0)
                   - np.percentile(Xt, 25, axis=0)).clip(min=eps)
            self.eeg_std  = iqr.astype(np.float32)
            print(f"  [RobustNormStats] EEG robust ON  — "
                  f"IQR range [{self.eeg_std.min():.4f}, {self.eeg_std.max():.4f}]")
        else:
            print(f"  [RobustNormStats] EEG normalization OFF")

        if normalize_rt:
            rt_arr       = np.array(rt_vals, dtype=np.float64)
            self.rt_mean = float(np.median(rt_arr))
            rt_iqr       = float(np.percentile(rt_arr, 75) - np.percentile(rt_arr, 25))
            self.rt_std  = max(rt_iqr, eps)
            print(f"  [RobustNormStats] RT  robust ON  — median={self.rt_mean:.4f}  IQR={self.rt_std:.4f}")
        else:
            print(f"  [RobustNormStats] RT  normalization OFF")


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
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
    )
    return train_loader, valid_loader, test_loader


def build_splits_single(
    normalize_targets: bool,
    normalize_eeg: bool,
    results_tag: str,
) -> tuple:
    """Single-release mode: load R5 only, split by subjects 80/10/10."""
    print(f"\n[Data] Single-release mode (R5 only)")
    dataset = load_release(TEST_RELEASE, normalize_targets, normalize_eeg)
    if dataset is None:
        raise RuntimeError("R5 not found — run prepare_data.py first.")

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

    print(f"  Train : {len(train_subj)} subjects  {len(train_set)} windows")
    print(f"  Valid : {len(valid_subj)} subjects  {len(valid_set)} windows")
    print(f"  Test  : {len(test_subj)} subjects  {len(test_set)} windows")

    norm_tag = _norm_tag(normalize_targets, normalize_eeg)
    results_csv = RESULTS_DIR / f"baseline_results_single{norm_tag}_{results_tag}.csv"
    return train_set, valid_set, test_set, results_csv


def build_splits_all(
    val_release: str,
    normalize_targets: bool,
    normalize_eeg: bool,
    results_tag: str,
) -> tuple:
    """
    All-releases mode:
      test  = R5
      val   = val_release
      train = all remaining releases (ConcatDataset)
    """
    print(f"\n[Data] All-releases mode")
    print(f"  Test release  : {TEST_RELEASE}")
    print(f"  Val  release  : {val_release}")

    train_releases = [r for r in ALL_RELEASES if r not in (TEST_RELEASE, val_release)]
    print(f"  Train releases: {train_releases}")
    print()

    test_set  = load_release(TEST_RELEASE, normalize_targets, normalize_eeg)
    valid_set = load_release(val_release,  normalize_targets, normalize_eeg)

    if test_set is None:
        raise RuntimeError("R5 (test) not found.")
    if valid_set is None:
        raise RuntimeError(f"{val_release} (val) not found.")

    train_datasets = []
    for r in train_releases:
        ds = load_release(r, normalize_targets, normalize_eeg)
        if ds is not None:
            train_datasets.append(ds)

    if not train_datasets:
        raise RuntimeError("No training releases found.")

    train_set = ConcatDataset(train_datasets)

    print(f"\n  Train : {len(train_set):>7} windows  ({len(train_releases)} releases)")
    print(f"  Valid : {len(valid_set):>7} windows  (release {val_release})")
    print(f"  Test  : {len(test_set):>7} windows  (release {TEST_RELEASE})")

    norm_tag = _norm_tag(normalize_targets, normalize_eeg)
    results_csv = RESULTS_DIR / f"baseline_results_all_{val_release}val{norm_tag}_{results_tag}.csv"
    return train_set, valid_set, test_set, results_csv


def _norm_tag(normalize_targets: bool, normalize_eeg: bool) -> str:
    """Build a short tag for the results filename."""
    parts = []
    if normalize_targets:
        parts.append("RTnorm")
    if normalize_eeg:
        parts.append("EEGnorm")
    return ("_" + "_".join(parts)) if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_single_baseline(
    model_name: str,
    train_set,
    valid_set,
    test_set,
    results_csv: Path,
    device: torch.device,
) -> None:
    """Train, evaluate, and save one baseline model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print("=" * 70)

    bs = BATCH_SIZE_OVERRIDES.get(model_name, BATCH_SIZE)
    if bs != BATCH_SIZE:
        print(f"  [info] Reduced batch size: {bs} (memory constraint)")
    train_loader, valid_loader, test_loader = make_loaders(
        train_set, valid_set, test_set, bs
    )

    # ── Create model ──────────────────────────────────────────────────────────
    try:
        set_seed(SEED)   # re-seed before each model init for identical weight init
        model = create_model(model_name, **MODEL_CONFIG)
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{model_name} architecture:")
        print(model)
        print(f"\nTrainable parameters: {n_params:,}  ({n_params / 1e6:.3f} M)")
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            device=device,
            **TRAINING_CONFIG,
        )

        # ── Test evaluation ───────────────────────────────────────────────────
        test_metrics = validate_model(
            test_loader, model, torch.nn.MSELoss(), device, print_batch_stats=True,
        )
        results["test_rmse"] = test_metrics["rmse"]
        results["test_mae"]  = test_metrics["mae"]
        results["test_r2"]   = test_metrics["r2"]

        print(f"\n{'='*70}")
        print(f"{model_name} - Final Results:")
        print(f"  Best Val RMSE : {results['best_val_rmse']:.3f}  "
              f"MAE: {results['best_val_mae']:.3f}  R2: {results['best_val_r2']:.3f}  "
              f"(epoch {results['best_epoch']})")
        print(f"  Test RMSE     : {results['test_rmse']:.3f}  "
              f"MAE: {results['test_mae']:.3f}  R2: {results['test_r2']:.3f}")
        print(f"  Train time    : {results['train_time_s']:.1f} s")
        print(f"  Params        : {n_params / 1e6:.2f} M")
        print("=" * 70)

        save_results_to_csv(
            results, results_csv, model_name=model_name, n_params=n_params
        )
        save_model_weights(
            results["best_model_state"], model_name, OUTPUT_CONFIG["weights_dir"]
        )
        plot_training_history(
            results["train_history"], results["val_history"],
            model_name, OUTPUT_CONFIG["results_dir"],
        )

    except Exception as e:
        print(f"Error training model {model_name}: {e}")
        import traceback
        traceback.print_exc()


def main(
    data_mode: str = "all",
    val_release: str = "R11",
    normalize_targets: bool = False,
    normalize_eeg: bool = False,
    run_only: str = None,
) -> None:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EEG BASELINE MODELS TRAINING")
    print(f"  Data mode          : {data_mode.upper()}")
    print(f"  Normalize targets  : {normalize_targets}")
    print(f"  Normalize EEG      : {normalize_eeg}")
    print("=" * 70)

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
    print(f"\n[2/5] Computing normalization statistics ({args.norm_method})")
    if args.norm_method == "robust":
        stats = RobustNormStats(
            train_set,
            normalize_eeg=args.normalize_eeg,
            normalize_rt=args.normalize_rt,
            max_windows=5000,
        )
    else:
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
        transfer_rel = None
        for rel in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]:
            candidate = WINDOWS_DIR / f"{transfer_task}_{rel}"
            if candidate.exists():
                # Check it actually has valid subjects
                subj_file = candidate / "subjects.txt"
                if subj_file.exists() and len(subj_file.read_text().strip().splitlines()) > 0:
                    transfer_rel = rel
                    break
        if transfer_rel is None:
            raise FileNotFoundError(f"No valid windowed data found for {transfer_task}")
        print(f"  Using {transfer_task}_{transfer_rel} for transfer evaluation")
        transfer_ds = load_transfer_dataset(transfer_task, transfer_rel)
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
    device = setup_device()
    tag    = run_only or "all_models"

    # ── Build splits ──────────────────────────────────────────────────────────
    if data_mode == "single":
        train_set, valid_set, test_set, results_csv = build_splits_single(
            normalize_targets, normalize_eeg, tag
        )
    elif data_mode == "all":
        train_set, valid_set, test_set, results_csv = build_splits_all(
            val_release, normalize_targets, normalize_eeg, tag
        )
    else:
        raise ValueError(f"Unknown data_mode '{data_mode}'. Choose 'single' or 'all'.")

    print(f"\n  Results CSV : {results_csv}")

    # ── Determine which models to run ─────────────────────────────────────────
    baseline_models = get_baseline_models()
    if run_only:
        if run_only not in baseline_models:
            print(f"ERROR: '{run_only}' not found. Available: {baseline_models}")
            sys.exit(1)
        baseline_models = [run_only]

    print(f"\nTraining {len(baseline_models)} model(s): {baseline_models}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    for i, model_name in enumerate(baseline_models, 1):
        print(f"\n\nBaseline {i}/{len(baseline_models)}: {model_name}")
        run_single_baseline(
            model_name=model_name,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            results_csv=results_csv,
            device=device,
        )

    print("\n" + "=" * 70)
    print("ALL BASELINES COMPLETED!")
    print("=" * 70)
    print(f"\nResults : {results_csv}")
    print(f"Weights : {OUTPUT_CONFIG['weights_dir']}")
    print(f"Plots   : {OUTPUT_CONFIG['results_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-mode", type=str, default="all", choices=["single", "all"],
        help="'single' = R5 only (subject split); 'all' = all releases (release split)",
    )
    parser.add_argument(
        "--val-release", type=str, default="R11",
        help="Which release to use as validation in 'all' mode (default: R11)",
    )
    parser.add_argument(
        "--normalize-targets", action="store_true",
        help="Z-score RT per subject using that subject's own mean/std",
    )
    parser.add_argument(
        "--normalize-eeg", action="store_true",
        help="Z-score each EEG channel per subject using that subject's own stats",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run a single model by name (e.g. --model EEGConformer). Omit to run all.",
    )
    parser.add_argument(
        "--norm_method", type=str, default="global_zscore",
        choices=["global_zscore", "robust"],
        help="Normalization method: global_zscore (default) or robust (median/IQR). "
             "Both apply CCD training stats to transfer tasks."
    )

    args = parser.parse_args()
    main(
        data_mode=args.data_mode,
        val_release=args.val_release,
        normalize_targets=args.normalize_targets,
        normalize_eeg=args.normalize_eeg,
        run_only=args.model,
    )