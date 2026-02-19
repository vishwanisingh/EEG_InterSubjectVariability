"""High-level training orchestration with early stopping."""

import copy
from typing import Any, Dict

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
    weight_decay: float = 1e-4,
    patience: int = 5,
    min_delta: float = 1e-4,
    print_stats: bool = True,
) -> Dict[str, Any]:
    """
    Train a model with AdamW + cosine LR schedule and early stopping.

    Parameters
    ----------
    model : Module
    train_loader, val_loader : DataLoader
    device : torch.device
    n_epochs : int
        Maximum training epochs.
    lr : float
        Initial learning rate.
    weight_decay : float
    patience : int
        Stop if val RMSE does not improve for this many consecutive epochs.
    min_delta : float
        Minimum improvement to reset the patience counter.
    print_stats : bool
        Whether to print per-epoch summaries.

    Returns
    -------
    dict with keys:
        best_model_state, best_epoch, best_val_rmse,
        train_history, val_history,
        final_train_rmse, final_val_rmse
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
    loss_fn   = torch.nn.MSELoss()

    best_rmse       = float("inf")
    epochs_no_impr  = 0
    best_state      = None
    best_epoch      = None
    train_history   = []
    val_history     = []

    for epoch in range(1, n_epochs + 1):
        if print_stats:
            print(f"\n{'='*60}\nEpoch {epoch}/{n_epochs}\n{'='*60}")

        train_loss, train_rmse = train_one_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, device, print_stats
        )
        val_loss, val_rmse = validate_model(val_loader, model, loss_fn, device, print_stats)

        train_history.append((train_loss, train_rmse))
        val_history.append((val_loss, val_rmse))

        if print_stats:
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_rmse={train_rmse:.4f}"
                  f"  |  val_loss={val_loss:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < best_rmse - min_delta:
            best_rmse      = val_rmse
            best_state     = copy.deepcopy(model.state_dict())
            best_epoch     = epoch
            epochs_no_impr = 0
            if print_stats:
                print(f"  ✓ New best val RMSE: {best_rmse:.6f}")
        else:
            epochs_no_impr += 1
            if print_stats:
                print(f"  No improvement for {epochs_no_impr} epoch(s)")

        if epochs_no_impr >= patience:
            if print_stats:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_model_state": best_state,
        "best_epoch":       best_epoch,
        "best_val_rmse":    best_rmse,
        "train_history":    train_history,
        "val_history":      val_history,
        "final_train_rmse": train_history[-1][1] if train_history else None,
        "final_val_rmse":   val_history[-1][1]   if val_history   else None,
    }