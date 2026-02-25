"""
scripts/run_baselines.py
------------------------
Train all baseline models sequentially on the pre-saved windowed dataset
and write a comparison CSV.

Run prepare_data.py first if the windowed dataset doesn't exist yet.

Usage:
    python scripts/run_baselines.py
"""

import sys
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
from src.models.registry import create_model, get_baseline_models
from src.training.trainer import train_model
from src.training.engine import validate_model
from src.utils.device import setup_device
from src.utils.io import save_results_to_csv, save_model_weights
from src.utils.plotting import plot_training_history


# ── Shared config dicts (built once, reused per model) ───────────────────────

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
    results_csv=RESULTS_DIR / "baseline_results.csv",
    results_dir=RESULTS_DIR,
    weights_dir=WEIGHTS_DIR,
)


def run_single_baseline(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """Train, evaluate, and save one baseline model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print("=" * 70)

    # ── Create model ──────────────────────────────────────────────────────────
    try:
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
            val_loader=val_loader,
            device=device,
            **TRAINING_CONFIG,
        )

        # ── Evaluate on test set ──────────────────────────────────────────────
        test_metrics = validate_model(
            test_loader,
            model,
            torch.nn.MSELoss(),
            device,
            print_batch_stats=True,
        )
        results["test_rmse"] = test_metrics["rmse"]
        results["test_mae"]  = test_metrics["mae"]
        results["test_r2"]   = test_metrics["r2"]

        print(f"\n{'='*70}")
        print(f"{model_name} - Final Results:")
        print(f"  Val  RMSE : {results['best_val_rmse']:.6f}  "
              f"MAE: {results['best_val_mae']:.6f}  R2: {results['best_val_r2']:.4f}  "
              f"(epoch {results['best_epoch']})")
        print(f"  Test RMSE : {results['test_rmse']:.6f}  "
              f"MAE: {results['test_mae']:.6f}  R2: {results['test_r2']:.4f}")
        print(f"  Train time: {results['train_time_s']:.1f} s")
        print(f"  Params    : {n_params / 1e6:.3f} M")
        print("=" * 70)

        # ── Save results, weights, plot ───────────────────────────────────────
        save_results_to_csv(
            results,
            OUTPUT_CONFIG["results_csv"],
            model_name=model_name,
            n_params=n_params,
        )
        save_model_weights(
            results["best_model_state"],
            model_name,
            OUTPUT_CONFIG["weights_dir"],
        )
        plot_training_history(
            results["train_history"],
            results["val_history"],
            model_name,
            OUTPUT_CONFIG["results_dir"],
        )

    except Exception as e:
        print(f"Error training model {model_name}: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main function to run all baselines."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EEG BASELINE MODELS TRAINING")
    print("=" * 70)

    device   = setup_device()
    root_dir = WINDOWS_DIR / f"{TASK}_{RELEASE}"

    # ── [1/4] Load pre-saved windowed dataset from disk ───────────────────────
    print(f"\n[1/4] Loading windowed dataset from {root_dir} ...")
    dataset = load_windowed_dataset(
        root_dir=root_dir,
        target_key="target",
        eager=False,
        excluded_subjects=EXCLUDED_SUBJECTS,
    )

    # ── [2/4] Split by subjects ───────────────────────────────────────────────
    print("\n[2/4] Splitting by subjects...")
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

    # ── [3/4] Data loaders ────────────────────────────────────────────────────
    print("\n[3/4] Creating data loaders...")
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    # ── [4/4] Train all baseline models ──────────────────────────────────────
    baseline_models = get_baseline_models()
    print(f"\n[4/4] Training {len(baseline_models)} baseline models...")

    for i, model_name in enumerate(baseline_models, 1):
        print(f"\n\nBaseline {i}/{len(baseline_models)}: {model_name}")
        run_single_baseline(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            device=device,
        )

    print("\n" + "=" * 70)
    print("ALL BASELINES COMPLETED!")
    print("=" * 70)
    print(f"\nResults saved to : {OUTPUT_CONFIG['results_csv']}")
    print(f"Model weights    : {OUTPUT_CONFIG['weights_dir']}")
    print(f"Plots            : {OUTPUT_CONFIG['results_dir']}")


if __name__ == "__main__":
    main()