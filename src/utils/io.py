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
) -> None:
    """Append one row of training results to a CSV file."""
    row = {
        "Model":           model_name,
        "Best Epoch":      results["best_epoch"],
        "Best Val RMSE":   results["best_val_rmse"],
        "Final Train RMSE":results["final_train_rmse"],
        "Final Val RMSE":  results["final_val_rmse"],
        "Test RMSE":       results.get("test_rmse"),
    }
    csv_file = Path(csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        df = pd.concat([pd.read_csv(csv_file), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_file, index=False)
    print(f"Results saved → {csv_file}")


def save_model_weights(
    model_state: Dict,
    model_name: str,
    save_dir: str | Path = "weights",
) -> Path:
    """Save a model state dict to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{model_name}_weights.pt"
    torch.save(model_state, path)
    print(f"Weights saved → {path}")
    return path