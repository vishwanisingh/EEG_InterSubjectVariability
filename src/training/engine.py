"""Low-level training and validation loop functions."""

from typing import Optional, Tuple

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device,
    print_batch_stats: bool = True,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    rmse : float
    """
    model.train()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples  = 0

    bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        disable=not print_batch_stats,
        desc=f"Epoch {epoch} [train]",
    )

    for _, batch in bar:
        X, y = batch[0].to(device).float(), batch[1].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss  = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_flat  = preds.detach().view(-1)
        y_flat      = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples  += y_flat.numel()

        if print_batch_stats:
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            rmse=f"{(sum_sq_err / max(n_samples, 1)) ** 0.5:.4f}")

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader), (sum_sq_err / max(n_samples, 1)) ** 0.5


@torch.no_grad()
def validate_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate model on a dataloader.

    Returns
    -------
    avg_loss : float
    rmse : float
    """
    model.eval()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples  = 0

    bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        disable=not print_batch_stats,
        desc="[valid]",
    )

    for _, batch in bar:
        X, y    = batch[0].to(device).float(), batch[1].to(device).float()
        preds   = model(X)
        b_loss  = loss_fn(preds, y).item()
        total_loss += b_loss

        preds_flat  = preds.detach().view(-1)
        y_flat      = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples  += y_flat.numel()

        if print_batch_stats:
            bar.set_postfix(loss=f"{b_loss:.4f}",
                            rmse=f"{(sum_sq_err / max(n_samples, 1)) ** 0.5:.4f}")

    n = len(dataloader)
    return (total_loss / n if n else float("nan"),
            (sum_sq_err / max(n_samples, 1)) ** 0.5)