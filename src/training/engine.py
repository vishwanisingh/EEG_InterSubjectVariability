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
    Train model for one epoch.

    Returns
    -------
    avg_loss : float
    rmse     : float
    """
    model.train()
    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples  = 0

    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        disable=not print_batch_stats,
        desc=f"Epoch {epoch}",
    )

    for batch_idx, batch in progress_bar:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

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
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rmse": f"{running_rmse:.4f}",
            })

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    rmse     = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse


@torch.no_grad()
def validate_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model -- returns RMSE, MAE, and R2.

    Returns
    -------
    dict with keys: avg_loss, rmse, mae, r2
    """
    model.eval()
    total_loss  = 0.0
    sum_sq_err  = 0.0
    sum_abs_err = 0.0
    n_batches   = len(dataloader)
    n_samples   = 0

    all_preds   = []
    all_targets = []

    iterator = tqdm(
        enumerate(dataloader),
        total=n_batches,
        disable=not print_batch_stats,
        desc="Validation",
    )

    for batch_idx, batch in iterator:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        preds      = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        preds_flat  = preds.detach().view(-1)
        y_flat      = y.detach().view(-1)

        sum_sq_err  += torch.sum((preds_flat - y_flat) ** 2).item()
        sum_abs_err += torch.sum(torch.abs(preds_flat - y_flat)).item()
        n_samples   += y_flat.numel()

        all_preds.append(preds_flat.cpu())
        all_targets.append(y_flat.cpu())

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            iterator.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "rmse": f"{running_rmse:.4f}",
            })

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    rmse     = (sum_sq_err / max(n_samples, 1)) ** 0.5
    mae      = sum_abs_err / max(n_samples, 1)

    # R2 = 1 - SS_res / SS_tot
    all_targets_t = torch.cat(all_targets)
    all_preds_t   = torch.cat(all_preds)
    ss_res = torch.sum((all_targets_t - all_preds_t) ** 2).item()
    ss_tot = torch.sum((all_targets_t - all_targets_t.mean()) ** 2).item()
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"avg_loss": avg_loss, "rmse": rmse, "mae": mae, "r2": r2}
