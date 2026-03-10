"""Low-level training and validation loop functions."""

from typing import Dict, Optional, Tuple

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
    rmse     : float  (in normalized space if targets are z-scored)
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
        # batch may be (X, y) or (X, y, norm) — handle both
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
    Evaluate model.

    If the dataloader yields (X, y, norm) batches where norm is (B, 2)
    containing [target_mean, target_std] per sample, predictions and targets
    are inverse-transformed to the original scale before computing
    RMSE / MAE / R². The loss is always computed in normalized space.

    Returns
    -------
    dict with keys: avg_loss, rmse, mae, r2
      - avg_loss : MSE loss in normalized space
      - rmse     : RMSE in original scale (seconds) if norm provided, else normalized
      - mae      : MAE  in original scale if norm provided, else normalized
      - r2       : R²   in original scale if norm provided, else normalized
    """
    model.eval()
    total_loss  = 0.0
    n_batches   = len(dataloader)
    n_samples   = 0

    all_preds   = []
    all_targets = []
    has_norm    = False

    iterator = tqdm(
        enumerate(dataloader),
        total=n_batches,
        disable=not print_batch_stats,
        desc="Validation",
    )

    for batch_idx, batch in iterator:
        X   = batch[0].to(device).float()
        y   = batch[1].to(device).float()
        norm = batch[2].to(device).float() if len(batch) > 2 else None

        preds      = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss
        n_samples  += y.numel()

        preds_flat = preds.detach().view(-1).cpu()
        y_flat     = y.detach().view(-1).cpu()

        # Inverse-transform if norm stats are available
        if norm is not None:
            has_norm = True
            means = norm[:, 0].cpu()   # (B,)
            stds  = norm[:, 1].cpu()   # (B,)
            # preds and y are (B, 1) → view as (B,) for broadcasting
            preds_flat = preds_flat * stds + means
            y_flat     = y_flat     * stds + means

        all_preds.append(preds_flat)
        all_targets.append(y_flat)

        if print_batch_stats:
            # Show running RMSE in original scale
            p_so_far = torch.cat(all_preds)
            t_so_far = torch.cat(all_targets)
            running_rmse = ((p_so_far - t_so_far) ** 2).mean().sqrt().item()
            iterator.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "rmse": f"{running_rmse:.4f}" + ("s" if has_norm else ""),
            })

    avg_loss = total_loss / n_batches if n_batches else float("nan")

    all_targets_t = torch.cat(all_targets)
    all_preds_t   = torch.cat(all_preds)

    ss_res = ((all_targets_t - all_preds_t) ** 2).sum().item()
    ss_tot = ((all_targets_t - all_targets_t.mean()) ** 2).sum().item()

    rmse = (ss_res / max(n_samples, 1)) ** 0.5
    mae  = (all_targets_t - all_preds_t).abs().mean().item()
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"avg_loss": avg_loss, "rmse": rmse, "mae": mae, "r2": r2}