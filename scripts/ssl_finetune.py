"""
scripts/ssl_finetune.py
-----------------------
Fine-tuning script: loads a pretrained SSL encoder (EEGConformer or EEGTCNet
backbone), attaches a regression head, and trains on CCD RT prediction.

Also runs the critical ablation: the same encoder architecture trained
from scratch (supervised) for a fair comparison.

Usage:
    # Fine-tune a pretrained VICReg EEGConformer encoder
    python scripts/ssl_finetune.py --method vicreg --encoder conformer --finetune_mode full

    # Fine-tune with frozen encoder (linear probe)
    python scripts/ssl_finetune.py --method byol --encoder tcnet --finetune_mode linear

    # Run supervised-from-scratch baseline (same architecture)
    python scripts/ssl_finetune.py --method supervised --encoder conformer
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from configs.config import (
    TASK, RELEASE, WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    N_CHANS, N_TIMES, SFREQ, N_OUTPUTS,
    BATCH_SIZE, NUM_WORKERS,
    N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, MIN_DELTA,
    VALID_FRAC, TEST_FRAC, SEED, EXCLUDED_SUBJECTS,
)
from src.data.disk_dataset import load_windowed_dataset
from src.ssl.encoder import create_encoder
from src.ssl.finetune import FinetunedEEGModel
from src.training.trainer import train_model
from src.training.engine import validate_model
from src.utils.device import setup_device
from src.utils.io import save_results_to_csv, save_model_weights
from src.utils.plotting import plot_training_history


def load_pretrained_encoder(
    method: str,
    encoder_name: str,
    embed_dim: int,
    n_chans: int,
    n_times: int,
):
    """Load pretrained encoder weights from disk."""
    run_name  = f"{method}_{encoder_name}"          # e.g. "vicreg_conformer"
    ckpt_path = WEIGHTS_DIR / f"ssl_{run_name}_encoder_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No pretrained encoder found at {ckpt_path}.\n"
            f"Run: python scripts/ssl_pretrain.py "
            f"--method {method} --encoder {encoder_name}"
        )

    encoder = create_encoder(
        name=encoder_name,
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=embed_dim,
    )
    encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print(f"  ✓ Loaded pretrained encoder from {ckpt_path}")
    return encoder


def finetune(
    method: str = "vicreg",
    encoder_name: str = "conformer",
    finetune_mode: str = "full",
    release: str = RELEASE,
    task: str = TASK,
    embed_dim: int = 256,
    head_hidden: int = 128,
    n_epochs: int = N_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> None:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device   = setup_device()
    root_dir = WINDOWS_DIR / f"{task}_{release}"

    # ── [1/5] Load CCD windowed data ──────────────────────────────────────────
    print(f"\n[1/5] Loading CCD data from {root_dir}")
    dataset = load_windowed_dataset(
        root_dir=root_dir,
        target_key="target",
        eager=False,
        excluded_subjects=EXCLUDED_SUBJECTS,
    )

    # ── [2/5] Split by subjects ───────────────────────────────────────────────
    print("\n[2/5] Splitting by subjects")
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
    print(f"  Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── [3/5] Build model ─────────────────────────────────────────────────────
    print(f"\n[3/5] Building model  "
          f"(encoder={encoder_name}, method={method}, finetune_mode={finetune_mode})")

    if method == "supervised":
        # Critical ablation: same architecture, random init, trained with labels
        encoder = create_encoder(
            name=encoder_name,
            n_chans=N_CHANS,
            n_times=N_TIMES,
            embed_dim=embed_dim,
        )
        model_name = f"SSL_{encoder_name}_supervised"
    else:
        encoder = load_pretrained_encoder(
            method=method,
            encoder_name=encoder_name,
            embed_dim=embed_dim,
            n_chans=N_CHANS,
            n_times=N_TIMES,
        )
        model_name = f"SSL_{method}_{encoder_name}_{finetune_mode}"

    model = FinetunedEEGModel(
        encoder=encoder,
        n_outputs=N_OUTPUTS,
        finetune_mode=finetune_mode if method != "supervised" else "full",
        head_hidden=head_hidden,
    )
    model.to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # ── [4/5] Fine-tune ───────────────────────────────────────────────────────
    print(f"\n[4/5] Fine-tuning")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        device=device,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        print_stats=True,
    )

    # ── [5/5] Evaluate + save ─────────────────────────────────────────────────
    test_metrics = validate_model(
        test_loader, model, torch.nn.MSELoss(), device, print_batch_stats=False
    )
    results["test_rmse"] = test_metrics["rmse"]
    results["test_mae"]  = test_metrics["mae"]
    results["test_r2"]   = test_metrics["r2"]

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"  Best Val RMSE : {results['best_val_rmse']:.6f}  (epoch {results['best_epoch']})")
    print(f"  Test RMSE     : {results['test_rmse']:.6f}")
    print(f"  Test MAE      : {results['test_mae']:.6f}")
    print(f"  Test R2       : {results['test_r2']:.4f}")
    print("="*60)

    save_results_to_csv(results, RESULTS_DIR / "ssl_results.csv", model_name=model_name)
    save_model_weights(results["best_model_state"], model_name, WEIGHTS_DIR)
    plot_training_history(results["train_history"], results["val_history"],
                          model_name, RESULTS_DIR)

    print(f"\n✓ Results → {RESULTS_DIR / 'ssl_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL Fine-tuning / Ablation")
    parser.add_argument("--method",        type=str, default="vicreg",
                        choices=["vicreg", "byol", "supervised"])
    parser.add_argument("--encoder",       type=str, default="conformer",
                        choices=["conformer", "tcnet"],
                        help="Encoder backbone: 'conformer' or 'tcnet'")
    parser.add_argument("--finetune_mode", type=str, default="full",
                        choices=["full", "linear", "partial"])
    parser.add_argument("--release",       type=str, default=RELEASE)
    parser.add_argument("--task",          type=str, default=TASK)
    parser.add_argument("--embed_dim",     type=int, default=256)
    parser.add_argument("--epochs",        type=int, default=N_EPOCHS)
    parser.add_argument("--lr",            type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    finetune(
        method=args.method,
        encoder_name=args.encoder,
        finetune_mode=args.finetune_mode,
        release=args.release,
        task=args.task,
        embed_dim=args.embed_dim,
        n_epochs=args.epochs,
        lr=args.lr,
    )