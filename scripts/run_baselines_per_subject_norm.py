"""
scripts/run_baselines.py
------------------------
Train all baseline models sequentially on the pre-saved windowed dataset
and write a comparison CSV.

Run prepare_ccd_data.py / prepare_all_releases.py first.

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

import gc
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from configs.config import (
    TASK, WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
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

# Models that need a smaller batch size due to attention memory cost
BATCH_SIZE_OVERRIDES = {
    "EEGformer": 8,
    "DualTSST":  16,
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


def make_loaders(train_set, valid_set, test_set, batch_size: int) -> tuple:
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
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
        raise RuntimeError("R5 not found — run prepare_ccd_data.py first.")

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
        # Tag weights filename with experiment info to prevent overwrites
        weights_name = f"{model_name}_{results_csv.stem}"
        save_model_weights(
            results["best_model_state"], weights_name, OUTPUT_CONFIG["weights_dir"]
        )
        plot_training_history(
            results["train_history"], results["val_history"],
            model_name, OUTPUT_CONFIG["results_dir"],
        )

    except Exception as e:
        print(f"Error training model {model_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always release GPU memory after each model regardless of success/failure
        if "model" in dir():
            del model
        torch.cuda.empty_cache()
        gc.collect()


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
        # Release GPU memory and file descriptors between models
        torch.cuda.empty_cache()
        gc.collect()

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
    args = parser.parse_args()
    main(
        data_mode=args.data_mode,
        val_release=args.val_release,
        normalize_targets=args.normalize_targets,
        normalize_eeg=args.normalize_eeg,
        run_only=args.model,
    )