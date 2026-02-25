"""High-level training orchestration with early stopping."""

import copy
from typing import Any, Dict

import time

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from .engine import train_one_epoch, validate_model


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    min_delta: float = 1e-4,
    print_stats: bool = True,
) -> Dict[str, Any]:
    """
    Train model with early stopping.

    Returns
    -------
    results : Dict[str, Any]
        - best_model_state  : state dict at best val RMSE
        - best_epoch        : epoch of best val RMSE
        - best_val_rmse     : best validation RMSE
        - best_val_mae      : val MAE  at best epoch
        - best_val_r2       : val R2   at best epoch
        - train_history     : list of (loss, rmse) per epoch
        - val_history       : list of (loss, rmse) per epoch
        - final_train_rmse  : train RMSE at last epoch
        - final_val_rmse    : val   RMSE at last epoch
        - train_time_s      : total wall-clock training time (seconds)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs - 1
    )
    loss_fn = torch.nn.MSELoss()

    best_rmse = float("inf")
    best_mae  = float("nan")
    best_r2   = float("nan")
    epochs_no_improve = 0
    best_state = None
    best_epoch = None

    train_history = []
    val_history   = []

    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        if print_stats:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{n_epochs}")
            print("=" * 60)

        train_loss, train_rmse = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch, device, print_stats,
        )

        val_metrics = validate_model(val_loader, model, loss_fn, device, print_stats)
        val_loss = val_metrics["avg_loss"]
        val_rmse = val_metrics["rmse"]
        val_mae  = val_metrics["mae"]
        val_r2   = val_metrics["r2"]

        train_history.append((train_loss, train_rmse))
        val_history.append((val_loss, val_rmse))

        if print_stats:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train -- Loss: {train_loss:.6f}  RMSE: {train_rmse:.6f}")
            print(f"  Val   -- Loss: {val_loss:.6f}  RMSE: {val_rmse:.6f}  "
                  f"MAE: {val_mae:.6f}  R2: {val_r2:.4f}")

        if val_rmse < best_rmse - min_delta:
            best_rmse  = val_rmse
            best_mae   = val_mae
            best_r2    = val_r2
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
            if print_stats:
                print(f"  >> New best val RMSE: {best_rmse:.6f}")
        else:
            epochs_no_improve += 1
            if print_stats:
                print(f"  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            if print_stats:
                print(f"\n{'='*60}")
                print(f"Early stopping at epoch {epoch}")
                print(f"Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
                print("=" * 60)
            break

    train_time_s = time.time() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_model_state": best_state,
        "best_epoch":       best_epoch,
        "best_val_rmse":    best_rmse,
        "best_val_mae":     best_mae,
        "best_val_r2":      best_r2,
        "train_history":    train_history,
        "val_history":      val_history,
        "final_train_rmse": train_history[-1][1] if train_history else None,
        "final_val_rmse":   val_history[-1][1]   if val_history   else None,
        "train_time_s":     train_time_s,
    }