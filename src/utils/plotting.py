"""Plotting utilities for training curves and data distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(
    train_history,
    val_history,
    model_name: str,
    save_dir: str | Path = "results",
) -> None:
    """Plot and save loss + RMSE curves for a training run."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs        = range(1, len(train_history) + 1)
    train_losses, train_rmses = zip(*train_history)
    val_losses,   val_rmses   = zip(*val_history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, ylabel, title in [
        (axes[0], train_losses, val_losses, "Loss",  f"{model_name} — Loss"),
        (axes[1], train_rmses,  val_rmses,  "RMSE",  f"{model_name} — RMSE"),
    ]:
        ax.plot(epochs, train_vals, "b-", label="Train", linewidth=2)
        ax.plot(epochs, val_vals,   "r-", label="Val",   linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / f"{model_name}_training_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved → {out}")


def plot_response_time_distribution(
    metadata: pd.DataFrame,
    save_path: str | Path = "results/rt_distribution.png",
) -> None:
    """Plot histogram of reaction-time targets."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(metadata["target"].dropna(), bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Reaction Time (s)", fontsize=12)
    ax.set_ylabel("Count",             fontsize=12)
    ax.set_title("Response Time Distribution", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"RT distribution plot saved → {save_path}")