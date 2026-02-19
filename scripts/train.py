"""
scripts/train.py
----------------
Main training script. Loads the per-subject windowed dataset from disk,
splits by subjects, trains a model, and saves results.

Run prepare_data.py first if the windowed dataset doesn't exist yet.

Usage:
    python scripts/train.py
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
from src.models.registry import create_model
from src.training.trainer import train_model
from src.training.engine import validate_model
from src.utils.device import setup_device
from src.utils.io import save_results_to_csv, save_model_weights
from src.utils.plotting import plot_training_history


def train(model_name: str = "BDTCN") -> None:
    """
    Full training pipeline — loads from disk, no re-downloading.

    Parameters
    ----------
    model_name : str
        Any key from src.models.registry.MODEL_REGISTRY.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device   = setup_device()
    root_dir = WINDOWS_DIR / f"{TASK}_{RELEASE}"

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

    # ── [3/5] Data loaders ───────────────────────────────────────────────────
    print("\n[3/5] Creating data loaders...")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
    train(model_name="BDTCN")   # ← change model here