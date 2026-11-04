"""
ssl_pretrain_multirelease.py
-----------------------------
SSL pretraining on MULTIPLE RELEASES of sliding windows.

This script is optimized for datasets with sliding windows across many releases.
It loads data from multiple releases simultaneously for maximum pretraining data.

Usage:
    # Pretrain on ALL available sliding windows (recommended)
    python ssl_pretrain_multirelease.py \
        --method vicreg \
        --encoder conformer \
        --tasks contrastChangeDetection,seqLearning6target,seqLearning8target \
        --releases all \
        --epochs 300

    # Pretrain on specific releases only
    python ssl_pretrain_multirelease.py \
        --method vicreg \
        --encoder conformer \
        --tasks contrastChangeDetection,seqLearning6target,seqLearning8target \
        --releases R1,R2,R3,R6,R8,R9,R10,R11 \
        --epochs 300
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
from src.ssl.encoder import create_encoder
from src.ssl.vicreg import VICReg
from src.ssl.byol import BYOL
from src.ssl.augmentations import get_default_augmentation
from src.utils.device import setup_device


# Available releases per task (based on your data)
AVAILABLE_SLIDING = {
    "contrastChangeDetection": ["R1", "R2", "R3", "R6", "R8", "R9", "R10", "R11"],
    "seqLearning6target": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"],
    "seqLearning8target": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"],
}


def discover_available_data(tasks: list, releases: list) -> dict:
    """
    Discover which task/release combinations exist on disk.
    
    Returns
    -------
    dict mapping (task, release) -> path
    """
    found = {}
    
    for task in tasks:
        for release in releases:
            # Try sliding window path first
            sliding_path = WINDOWS_DIR / f"{task}_sliding_{release}"
            if sliding_path.exists() and (sliding_path / "subjects.txt").exists():
                found[(task, release)] = sliding_path
                continue
            
            # Fallback: some releases might not have "_sliding" suffix
            alt_path = WINDOWS_DIR / f"{task}_{release}"
            if alt_path.exists() and (alt_path / "subjects.txt").exists():
                found[(task, release)] = alt_path
    
    return found


def load_multirelease_data(
    tasks: list,
    releases: list,
    batch_size: int,
) -> DataLoader:
    """
    Load and concatenate sliding windows from multiple tasks and releases.
    
    This creates a MASSIVE pretraining dataset by combining:
    - Multiple tasks (CCD, seqLearning6, seqLearning8)
    - Multiple releases (R1-R11)
    """
    print(f"\n{'='*70}")
    print("DISCOVERING SLIDING WINDOW DATA")
    print(f"{'='*70}")
    print(f"  Tasks requested:    {len(tasks)}")
    print(f"  Releases requested: {len(releases)}")
    print()
    
    available = discover_available_data(tasks, releases)
    
    if not available:
        raise RuntimeError(
            f"No sliding window data found.\n"
            f"Searched for tasks: {tasks}\n"
            f"Searched for releases: {releases}\n"
            f"Expected paths like: {WINDOWS_DIR}/{{task}}_sliding_{{release}}/"
        )
    
    print(f"Found {len(available)} task/release combinations:")
    
    datasets = []
    total_windows = 0
    total_subjects = 0
    
    # Group by task for cleaner output
    by_task = {}
    for (task, release), path in sorted(available.items()):
        if task not in by_task:
            by_task[task] = []
        by_task[task].append((release, path))
    
    for task in sorted(by_task.keys()):
        print(f"\n  {task}:")
        for release, path in sorted(by_task[task]):
            try:
                ds = load_windowed_dataset(
                    root_dir=path,
                    eager=False,
                    target_key="target",
                )
                n_windows = len(ds)
                n_subjects = len(ds.get_subject_ids())
                
                datasets.append(ds)
                total_windows += n_windows
                total_subjects += n_subjects
                
                print(f"    {release}: {n_windows:>7,} windows  ({n_subjects:>3} subjects)")
                
            except Exception as e:
                print(f"    {release}: [ERROR] {e}")
    
    if not datasets:
        raise RuntimeError("No datasets could be loaded successfully.")
    
    print(f"\n{'='*70}")
    print(f"TOTAL PRETRAINING DATA")
    print(f"{'='*70}")
    print(f"  Datasets:  {len(datasets)}")
    print(f"  Windows:   {total_windows:,}")
    print(f"  Subjects:  {total_subjects}")
    print(f"  Batches:   {total_windows // batch_size:,}")
    print(f"{'='*70}\n")
    
    combined = ConcatDataset(datasets)
    
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )


def pretrain(
    method: str = "vicreg",
    encoder_name: str = "conformer",
    tasks: list = None,
    releases: list = None,
    n_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    embed_dim: int = 256,
    batch_size: int = None,
    save_every: int = 25,
) -> Path:
    """
    Run SSL pretraining on multi-release sliding windows.
    
    Parameters
    ----------
    method : str
        'vicreg' or 'byol'
    encoder_name : str
        'conformer' or 'tcnet'
    tasks : list
        Task names. Default: all 3 main tasks
    releases : list
        Release names. 'all' = use all available per task
    n_epochs : int
        Pretraining epochs (300+ recommended for multi-release)
    lr : float
        Learning rate
    batch_size : int
        Batch size
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    if tasks is None:
        tasks = ["contrastChangeDetection", "seqLearning6target", "seqLearning8target"]
    
    # Resolve releases
    if releases is None or releases == ['all']:
        # Use all available releases per task
        all_releases = set()
        for task in tasks:
            all_releases.update(AVAILABLE_SLIDING.get(task, []))
        releases = sorted(all_releases)
        print(f"[Auto] Using all available releases: {releases}")
    
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = setup_device()
    
    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SSL PRETRAINING - MULTI-RELEASE")
    print(f"{'='*70}")
    print(f"  Method:  {method.upper()}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Tasks:   {', '.join(tasks)}")
    print(f"{'='*70}")
    
    loader = load_multirelease_data(tasks, releases, batch_size)
    
    # ── Build encoder ─────────────────────────────────────────────────────────
    print(f"Building encoder: {encoder_name}")
    encoder = create_encoder(
        name=encoder_name,
        n_chans=N_CHANS,
        n_times=N_TIMES,
        embed_dim=embed_dim,
    )
    
    # ── Build SSL model ───────────────────────────────────────────────────────
    run_name = f"{method}_{encoder_name}_multirelease"
    
    aug = get_default_augmentation()
    aug.train()
    
    if method == "vicreg":
        model = VICReg(encoder, proj_dim=512, augmentation=aug)
    elif method == "byol":
        model = BYOL(encoder, proj_dim=256, augmentation=aug)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)\n")
    
    # ── Optimizer ─────────────────────────────────────────────────────────────
    print(f"Training for {n_epochs} epochs")
    print(f"  Learning rate: {lr:.2e}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Weight decay:  {weight_decay:.2e}\n")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    
    # ── Training loop ─────────────────────────────────────────────────────────
    log_file = RESULTS_DIR / f"ssl_{run_name}_pretrain_log.txt"
    best_loss = float("inf")
    best_ckpt = WEIGHTS_DIR / f"ssl_{run_name}_encoder_best.pt"
    
    with open(log_file, "w") as f:
        f.write("epoch,loss,lr\n")
    
    print(f"{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        bar = tqdm(
            loader,
            desc=f"Epoch {epoch:03d}/{n_epochs}",
            leave=False,
        )
        
        for batch in bar:
            x = batch[0].to(device).float()
            
            optimizer.zero_grad(set_to_none=True)
            loss, log_dict = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if method == "byol":
                model.update_target()
            
            total_loss += loss.item()
            n_batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Print every 5 epochs or on best
        is_best = avg_loss < best_loss
        if epoch % 5 == 0 or is_best:
            status = "★ BEST" if is_best else ""
            print(
                f"Epoch {epoch:03d}  "
                f"loss={avg_loss:.6f}  "
                f"lr={current_lr:.2e}  "
                f"{status}"
            )
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{current_lr:.2e}\n")
        
        # Save best
        if is_best:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), best_ckpt)
        
        # Periodic checkpoints
        if epoch % save_every == 0:
            ckpt = WEIGHTS_DIR / f"ssl_{run_name}_encoder_epoch{epoch:03d}.pt"
            torch.save(encoder.state_dict(), ckpt)
    
    # ── Save final ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PRETRAINING COMPLETE")
    print(f"{'='*70}")
    
    final_path = WEIGHTS_DIR / f"ssl_{run_name}_encoder_final.pt"
    torch.save(encoder.state_dict(), final_path)
    
    print(f"  Best loss:     {best_loss:.6f}")
    print(f"  Best weights:  {best_ckpt.name}")
    print(f"  Final weights: {final_path.name}")
    print(f"  Training log:  {log_file.name}")
    print(f"{'='*70}\n")
    
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, default="vicreg",
        choices=["vicreg", "byol"]
    )
    parser.add_argument(
        "--encoder", type=str, default="conformer",
        choices=["conformer", "tcnet"]
    )
    parser.add_argument(
        "--tasks", type=str,
        default="contrastChangeDetection,seqLearning6target,seqLearning8target",
        help="Comma-separated task names"
    )
    parser.add_argument(
        "--releases", type=str, default="all",
        help="Comma-separated releases or 'all'"
    )
    parser.add_argument(
        "--epochs", type=int, default=300
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--batch_size", type=int, default=None
    )
    parser.add_argument(
        "--embed_dim", type=int, default=256
    )
    parser.add_argument(
        "--save_every", type=int, default=25
    )
    args = parser.parse_args()
    
    tasks = args.tasks.split(",")
    releases = None if args.releases == "all" else args.releases.split(",")
    
    pretrain(
        method=args.method,
        encoder_name=args.encoder,
        tasks=tasks,
        releases=releases,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        save_every=args.save_every,
    )