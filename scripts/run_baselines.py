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


def run_baselines() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device   = setup_device()
    root_dir = WINDOWS_DIR / f"{TASK}_{RELEASE}"

    # ── Load dataset from disk ────────────────────────────────────────────────
    print(f"\nLoading windowed dataset from {root_dir} ...")
    dataset = load_windowed_dataset(
        root_dir=root_dir,
        target_key="target",
        eager=False,
        excluded_subjects=EXCLUDED_SUBJECTS,
    )

    # ── Split subjects ────────────────────────────────────────────────────────
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

    train_set, valid_set, test_set = dataset.split_by_subjects(
        train_subj, valid_subj, test_subj
    )

    print(f"  Train : {len(train_set)} windows  ({len(train_subj)} subjects)")
    print(f"  Valid : {len(valid_set)} windows  ({len(valid_subj)} subjects)")
    print(f"  Test  : {len(test_set)} windows  ({len(test_subj)} subjects)")

    # ── Data loaders — created once, reused for all models ───────────────────
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Train each baseline ───────────────────────────────────────────────────
    baselines = get_baseline_models()
    print(f"\nRunning {len(baselines)} baseline models: {baselines}\n")

    summary = []

    for i, model_name in enumerate(baselines, 1):
        print(f"\n{'#'*70}")
        print(f"  Baseline {i}/{len(baselines)}: {model_name}")
        print(f"{'#'*70}")

        try:
            model = create_model(
                model_name, n_chans=N_CHANS, n_outputs=N_OUTPUTS,
                n_times=N_TIMES, sfreq=SFREQ,
            )
            model.to(device)
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

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

            _, test_rmse = validate_model(
                test_loader, model, torch.nn.MSELoss(), device, print_batch_stats=False
            )
            results["test_rmse"] = test_rmse

            save_results_to_csv(
                results, RESULTS_DIR / "baseline_results.csv", model_name=model_name
            )
            save_model_weights(results["best_model_state"], model_name, WEIGHTS_DIR)
            plot_training_history(
                results["train_history"], results["val_history"], model_name, RESULTS_DIR
            )

            print(f"  → Best val RMSE : {results['best_val_rmse']:.6f}  "
                  f"Test RMSE : {test_rmse:.6f}")
            summary.append((model_name, results["best_val_rmse"], test_rmse, "✓"))

        except Exception as exc:
            print(f"  ✗ {model_name} failed: {exc}")
            summary.append((model_name, None, None, f"✗ {exc}"))
            continue

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Val RMSE':>10} {'Test RMSE':>10}  Status")
    print(f"  {'-'*25} {'-'*10} {'-'*10}  ------")
    for model_name, val_rmse, test_rmse, status in summary:
        val_str  = f"{val_rmse:.6f}"  if val_rmse  is not None else "—"
        test_str = f"{test_rmse:.6f}" if test_rmse is not None else "—"
        print(f"  {model_name:<25} {val_str:>10} {test_str:>10}  {status}")
    print(f"{'='*70}")
    print(f"\n✓ Results saved to {RESULTS_DIR / 'baseline_results.csv'}")


if __name__ == "__main__":
    run_baselines()