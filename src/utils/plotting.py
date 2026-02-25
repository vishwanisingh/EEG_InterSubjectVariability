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
    """
    Plot and save training history.
    
    Parameters
    ----------
    train_history : List[Tuple[float, float]]
        List of (loss, rmse) for training
    val_history : List[Tuple[float, float]]
        List of (loss, rmse) for validation
    model_name : str
        Name of the model
    save_dir : str | Path
        Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(train_history) + 1)
    train_losses, train_rmses = zip(*train_history)
    val_losses, val_rmses = zip(*val_history)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot RMSE
    axes[1].plot(epochs, train_rmses, 'b-', label='Train RMSE', linewidth=2)
    axes[1].plot(epochs, val_rmses, 'r-', label='Val RMSE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title(f'{model_name} - Training RMSE', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = save_dir / f"{model_name}_training_history.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {filename}")


def plot_response_time_distribution(
    metadata: pd.DataFrame,
    save_path: str | Path = "results/response_time_distribution.png",
) -> None:
    """
    Plot distribution of response times.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata containing target (response time) values
    save_path : str | Path
        Path to save the plot
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(metadata["target"], bins=30, color='lightblue', edgecolor='black')
    ax.set_xlabel("Response Time (s)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Response Times", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Response time distribution saved to {save_path}")