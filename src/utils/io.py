"""File I/O utilities for results and model weights."""

import math
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
    # ── Test metrics ──────────────────────────────────────────────────────────
    test_rmse = results.get("test_rmse", float("nan"))
    test_mae  = results.get("test_mae",  float("nan"))
    test_r2   = results.get("test_r2",   float("nan"))
    test_nrmse = results.get("test_nrmse",
                 math.sqrt(max(1.0 - test_r2, 0.0))
                 if not math.isnan(test_r2) else float("nan"))

    # ── Val metrics — prefer seconds versions if available ────────────────────
    val_rmse = results.get("val_rmse_sec",
               results.get("best_val_rmse", float("nan")))
    val_mae  = results.get("val_mae_sec",
               results.get("best_val_mae",  float("nan")))
    val_r2   = results.get("best_val_r2", float("nan"))
    val_nrmse = results.get("val_nrmse",
                math.sqrt(max(1.0 - val_r2, 0.0))
                if not math.isnan(val_r2) else float("nan"))

    row = {
        "Model":          model_name,
        # ── test set ─────────────────────────────────────────────────────────
        "Test RMSE (s)":  test_rmse,
        "Test MAE (s)":   test_mae,
        "Test R2":        test_r2,
        "Test NRMSE":     test_nrmse,
        # ── validation ───────────────────────────────────────────────────────
        "Val RMSE (s)":   val_rmse,
        "Val MAE (s)":    val_mae,
        "Val R2":         val_r2,
        "Val NRMSE":      val_nrmse,
        # ── training ─────────────────────────────────────────────────────────
        "Best Epoch":     results.get("best_epoch",   float("nan")),
        "Train Time (s)": round(results.get("train_time_s", float("nan")), 1),
        # ── norm stats ───────────────────────────────────────────────────────
        "RT Mean Train":  results.get("rt_mean_train", float("nan")),
        "RT Std Train":   results.get("rt_std_train",  float("nan")),
        "Norm Method":    results.get("norm_method",   ""),
        # ── model ────────────────────────────────────────────────────────────
        "Params (M)":     round(n_params / 1e6, 3) if n_params else float("nan"),
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