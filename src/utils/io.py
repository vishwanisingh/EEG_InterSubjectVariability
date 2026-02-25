"""File I/O utilities for results and model weights."""

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch


def save_results_to_csv(
    results: Dict[str, Any],
    csv_file: str | Path,
    model_name: str = None,
    n_params: int = None,
) -> None:
    """
    Save training results to CSV file.

    Parameters
    ----------
    results : Dict[str, Any]
        Training results dictionary from train_model() / run_single_baseline()
    csv_file : str | Path
        Path to CSV file
    model_name : str, optional
        Name of the model
    n_params : int, optional
        Total trainable parameter count
    """
    row = {
        "Model":            model_name,
        # ── primary metrics (test set) ────────────────────────────────────────
        "Test RMSE":        results.get("test_rmse",  float("nan")),
        "Test MAE":         results.get("test_mae",   float("nan")),
        "Test R2":          results.get("test_r2",    float("nan")),
        # ── validation metrics at best epoch ─────────────────────────────────
        "Val RMSE":         results.get("best_val_rmse", float("nan")),
        "Val MAE":          results.get("best_val_mae",  float("nan")),
        "Val R2":           results.get("best_val_r2",   float("nan")),
        # ── training efficiency ───────────────────────────────────────────────
        "Best Epoch":       results.get("best_epoch",    float("nan")),
        "Train Time (s)":   round(results.get("train_time_s", float("nan")), 1),
        # ── model complexity ─────────────────────────────────────────────────
        "Params (M)":       round(n_params / 1e6, 3) if n_params else float("nan"),
    }

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")


def save_model_weights(
    model_state: Dict,
    model_name: str,
    save_dir: str | Path = "weights",
) -> Path:
    """
    Save model weights to file.
    
    Parameters
    ----------
    model_state : Dict
        Model state dictionary
    model_name : str
        Name of the model
    save_dir : str | Path
        Directory to save weights
    
    Returns
    -------
    Path
        Path to saved weights file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = save_dir / f"{model_name}_weights.pt"
    torch.save(model_state, filename)
    print(f"Model weights saved to {filename}")
    
    return filename

