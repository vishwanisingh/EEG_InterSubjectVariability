# EEG Regression — Repository Structure

```
eeg_project/
│
├── configs/                   # All configuration lives here
│   ├── config.py              # Paths, hyperparameters, global constants
│   ├── task_configs.py        # Per-task preprocessing settings (anchor, target, etc.)
│   └── __init__.py
│
├── src/                       # Importable source package
│   ├── data/
│   │   ├── loader.py          # load_eegdash_dataset, load_challenge_dataset, preprocess_dataset
│   │   ├── windowing.py       # create_windows
│   │   └── splitting.py       # split_by_subjects
│   │
│   ├── models/
│   │   ├── registry.py        # create_model(), MODEL_REGISTRY, get_baseline_models()
│   │   ├── eeg_unified.py     # EEGUnifiedModel (custom backbone)
│   │   ├── moe.py             # SimpleMoE, MoE, MoEConfig
│   │   ├── mlp.py             # MLP building block
│   │   └── multi_temporal_kernels.py   # MultiKernelConv1d
│   │
│   ├── training/
│   │   ├── engine.py          # train_one_epoch, validate_model
│   │   └── trainer.py         # train_model (with early stopping)
│   │
│   └── utils/
│       ├── io.py              # save_results_to_csv, save_model_weights
│       ├── plotting.py        # plot_training_history, plot_response_time_distribution
│       └── device.py          # setup_device
│
├── scripts/                   # Runnable entry points
│   ├── prepare_data.py        # Step 1: download → preprocess → window → save to disk
│   ├── train.py               # Step 2: load from disk → train one model
│   └── run_baselines.py       # Step 2 (alt): train all baseline models sequentially
│
└── data/
    └── windows/               # Saved windowed datasets (created by prepare_data.py)
```

## Workflow

```bash
# Step 1 — run ONCE (or when you change task / dataset)
python scripts/prepare_data.py

# Step 2 — run as many times as needed
python scripts/train.py          # single model
python scripts/run_baselines.py  # all baselines
```

## Switching tasks

Edit `configs/config.py`:
```python
TASK       = "surroundSupp"     # any key in configs/task_configs.py
DATASET_ID = "ds005509"
```

Then re-run `prepare_data.py` once for the new task.

## Switching models

In `scripts/train.py`, change the last line:
```python
train(model_name="EEGNet")   # any key in src/models/registry.MODEL_REGISTRY
```
