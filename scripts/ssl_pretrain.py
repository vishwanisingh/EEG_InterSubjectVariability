"""
scripts/ssl_pretrain.py
-----------------------
SSL pretraining script. Trains VICReg, BYOL, or MAE on windowed EEG data
(no labels used). Saves pretrained encoder weights for fine-tuning.

Key design: pretrain on ALL available tasks (resting, movies, CCD, etc.)
for maximum unsupervised signal, then fine-tune on CCD only.

Usage:
    python scripts/ssl_pretrain.py --method vicreg --task all
    python scripts/ssl_pretrain.py --method byol   --task contrastChangeDetection
    python scripts/ssl_pretrain.py --method mae    --task all
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from configs.config import (
    DATA_DIR, WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    N_CHANS, N_TIMES, SFREQ,
    BATCH_SIZE, NUM_WORKERS,
)
from src.data.disk_dataset import load_windowed_dataset
from src.ssl.encoder import EEGEncoder
from src.ssl.vicreg import VICReg
from src.ssl.byol import BYOL
from src.ssl.mae import EEGMaskedAutoencoder
from src.ssl.augmentations import get_default_augmentation, get_strong_augmentation
from src.utils.device import setup_device


# ── Tasks to use for pretraining ─────────────────────────────────────────────
ALL_PRETRAIN_TASKS = [
    "contrastChangeDetection",
    "RestingState",
    "surroundSupp",
    "seqLearning8target",
    "seqLearning6target",
    "symbolSearch",
    # Add movie tasks if you have them:
    # "DiaryOfAWimpyKid", "DespicableMe", "ThePresent",
]


def load_pretrain_data(tasks: list, release: str, batch_size: int) -> DataLoader:
    """Load and concatenate windowed data from multiple tasks."""
    datasets = []
    for task in tasks:
        root_dir = WINDOWS_DIR / f"{task}_{release}"
        if not root_dir.exists():
            print(f"  [skip] {task} — not found at {root_dir}")
            continue
        try:
            ds = load_windowed_dataset(root_dir=root_dir, eager=False)
            datasets.append(ds)
            print(f"  ✓ {task}: {len(ds)} windows")
        except Exception as e:
            print(f"  [skip] {task}: {e}")

    if not datasets:
        raise RuntimeError("No pretraining data found. Run prepare_data.py first.")

    combined = ConcatDataset(datasets)
    print(f"\n  Total pretraining windows: {len(combined)}")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,   # important for batch norm stability
    )


def pretrain(
    method: str = "vicreg",
    tasks: list = None,
    release: str = "R5",
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    embed_dim: int = 256,
    batch_size: int = None,
    save_every: int = 10,
) -> Path:
    """
    Run SSL pretraining.

    Parameters
    ----------
    method : str
        One of 'vicreg', 'byol', 'mae'.
    tasks : list
        List of task names to pretrain on. None = ALL_PRETRAIN_TASKS.
    release : str
        Release label used to find the windowed data folders.
    n_epochs : int
        Number of pretraining epochs.
    lr : float
        Learning rate.
    weight_decay : float
    embed_dim : int
        Encoder embedding dimension.
    batch_size : int
        Batch size. Defaults to config BATCH_SIZE.
    save_every : int
        Save a checkpoint every N epochs.

    Returns
    -------
    Path to the saved pretrained encoder weights.
    """
    if tasks is None:
        tasks = ALL_PRETRAIN_TASKS
    if batch_size is None:
        batch_size = BATCH_SIZE

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = setup_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading pretraining data (tasks: {tasks})")
    loader = load_pretrain_data(tasks, release, batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building {method.upper()} model")

    encoder = EEGEncoder(
        n_chans=N_CHANS,
        n_times=N_TIMES,
        embed_dim=embed_dim,
    )

    if method == "vicreg":
        aug = get_default_augmentation()
        aug.train()
        model = VICReg(encoder, proj_dim=512, augmentation=aug)

    elif method == "byol":
        aug = get_default_augmentation()
        aug.train()
        model = BYOL(encoder, proj_dim=256, augmentation=aug)

    elif method == "mae":
        model = EEGMaskedAutoencoder(
            n_chans=N_CHANS,
            n_times=N_TIMES,
            patch_size=10,
            mask_ratio=0.75,
            encoder_d_model=embed_dim,
        )
        encoder = model.encoder   # MAE encoder is inside the MAE model

    else:
        raise ValueError(f"Unknown method: {method}. Choose vicreg / byol / mae")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training for {n_epochs} epochs")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    log_file = RESULTS_DIR / f"ssl_{method}_pretrain_log.txt"
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        bar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs} [{method}]", leave=False)
        for batch in bar:
            # batch[0] is the EEG tensor from EEGWindowDataset
            x = batch[0].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            loss, log = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # BYOL requires EMA update after each step
            if method == "byol":
                model.update_target()

            total_loss += loss.item()
            n_batches  += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        print(f"  Epoch {epoch:03d}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = WEIGHTS_DIR / f"ssl_{method}_encoder_best.pt"
            torch.save(encoder.state_dict(), best_ckpt)

        if epoch % save_every == 0:
            ckpt = WEIGHTS_DIR / f"ssl_{method}_encoder_epoch{epoch:03d}.pt"
            torch.save(encoder.state_dict(), ckpt)

    # ── Save final encoder ────────────────────────────────────────────────────
    print(f"\n[4/4] Saving pretrained encoder")
    final_path = WEIGHTS_DIR / f"ssl_{method}_encoder_final.pt"
    torch.save(encoder.state_dict(), final_path)
    print(f"  ✓ Final encoder → {final_path}")
    print(f"  ✓ Best encoder  → {best_ckpt}")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG SSL Pretraining")
    parser.add_argument("--method",     type=str,   default="vicreg",
                        choices=["vicreg", "byol", "mae"])
    parser.add_argument("--release",    type=str,   default="R5")
    parser.add_argument("--tasks",      type=str,   default="all",
                        help="Comma-separated task names, or 'all'")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--embed_dim",  type=int,   default=256)
    parser.add_argument("--batch_size", type=int,   default=None)
    args = parser.parse_args()

    tasks = None if args.tasks == "all" else args.tasks.split(",")

    pretrain(
        method=args.method,
        tasks=tasks,
        release=args.release,
        n_epochs=args.epochs,
        lr=args.lr,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
    )
