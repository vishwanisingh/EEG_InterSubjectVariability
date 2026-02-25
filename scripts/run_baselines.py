"""Run all baseline models on the EEG dataset."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src import (
    load_eegdash_dataset,
    preprocess_dataset,
    create_windows,
    split_by_subjects,
    create_model,
    train_model,
    save_results_to_csv,
    save_model_weights,
    plot_training_history,
    plot_response_time_distribution,
    setup_device,
)
from experiments.baseline_configs import (
    DATA_CONFIG,
    PREPROCESS_CONFIG,
    WINDOW_CONFIG,
    SPLIT_CONFIG,
    DATALOADER_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    BASELINE_MODELS,
    OUTPUT_CONFIG,
)


def run_single_baseline(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Run training for a single baseline model.
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print('='*70)

    # Create model
    try:
        model = create_model(model_name, **MODEL_CONFIG)
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{model_name} architecture:")
        print(model)
        print(f"\nTrainable parameters: {n_params:,}  ({n_params/1e6:.3f} M)")
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        return

    # Train model
    try:
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            **TRAINING_CONFIG,
        )

        # Evaluate on test set (RMSE + MAE + R2)
        from src.training import validate_model
        test_metrics = validate_model(
            test_loader,
            model,
            torch.nn.MSELoss(),
            device,
            print_batch_stats=True,
        )
        results["test_rmse"] = test_metrics["rmse"]
        results["test_mae"]  = test_metrics["mae"]
        results["test_r2"]   = test_metrics["r2"]

        print(f"\n{'='*70}")
        print(f"{model_name} - Final Results:")
        print(f"  Val  RMSE : {results['best_val_rmse']:.6f}  "
              f"MAE: {results['best_val_mae']:.6f}  R2: {results['best_val_r2']:.4f}  "
              f"(epoch {results['best_epoch']})")
        print(f"  Test RMSE : {results['test_rmse']:.6f}  "
              f"MAE: {results['test_mae']:.6f}  R2: {results['test_r2']:.4f}")
        print(f"  Train time: {results['train_time_s']:.1f} s")
        print(f"  Params    : {n_params/1e6:.3f} M")
        print('='*70)

        # Save results
        save_results_to_csv(
            results,
            OUTPUT_CONFIG["results_csv"],
            model_name=model_name,
            n_params=n_params,
        )

        # Save model weights
        save_model_weights(
            results["best_model_state"],
            model_name,
            OUTPUT_CONFIG["weights_dir"],
        )

        # Plot training history
        plot_training_history(
            results["train_history"],
            results["val_history"],
            model_name,
            OUTPUT_CONFIG["results_dir"],
        )

    except Exception as e:
        print(f"Error training model {model_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run all baselines."""
    print("="*70)
    print("EEG BASELINE MODELS TRAINING")
    print("="*70)
    
    # Setup device
    device = setup_device()
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_eegdash_dataset(**DATA_CONFIG)
    print(f"Loaded {len(dataset)} recordings")
    
    # Preprocess dataset
    print("\n[2/6] Preprocessing dataset...")
    dataset = preprocess_dataset(dataset, **PREPROCESS_CONFIG)
    
    # Create windows
    print("\n[3/6] Creating windows...")
    windowed_dataset = create_windows(dataset, **WINDOW_CONFIG)
    print(f"Created {len(windowed_dataset)} windows")
    
    # Plot response time distribution
    metadata = windowed_dataset.get_metadata()
    plot_response_time_distribution(
        metadata,
        OUTPUT_CONFIG["results_dir"] / "response_time_distribution.png"
    )
    
    # Split dataset
    print("\n[4/6] Splitting dataset...")
    train_set, valid_set, test_set = split_by_subjects(
        windowed_dataset,
        **SPLIT_CONFIG,
    )
    
    # Create data loaders
    print("\n[5/6] Creating data loaders...")
    train_loader = DataLoader(
        train_set,
        batch_size=DATALOADER_CONFIG["batch_size"],
        shuffle=True,
        num_workers=DATALOADER_CONFIG["num_workers"],
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=DATALOADER_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DATALOADER_CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=DATALOADER_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DATALOADER_CONFIG["num_workers"],
    )
    
    # Train all baseline models
    print("\n[6/6] Training baseline models...")
    for i, model_name in enumerate(BASELINE_MODELS, 1):
        print(f"\n\nBaseline {i}/{len(BASELINE_MODELS)}: {model_name}")
        run_single_baseline(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            device=device,
        )
    
    print("\n" + "="*70)
    print("ALL BASELINES COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_CONFIG['results_csv']}")
    print(f"Model weights saved to: {OUTPUT_CONFIG['weights_dir']}")
    print(f"Plots saved to: {OUTPUT_CONFIG['results_dir']}")


if __name__ == "__main__":
    main()