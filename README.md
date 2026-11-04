# EEG Reaction-Time Prediction

This repository prepares HBN EEG windows and trains regression models to predict
reaction time. The pipeline is organized by lifecycle stage; Python entry points
live in `scripts/`, reusable code lives in `src/`, and shell launchers live in
`bash/`.

## Pipeline Stages

### 1. Prepare data

Use the task-specific preparation entry point:

```bash
python scripts/prepare_ccd_data.py
python scripts/prepare_ss_data.py
python scripts/prepare_sl_data.py
```

For multi-release CCD preparation, use `scripts/prepare_all_releases.py`. The
release-aware shell pipelines (`bash/ccd_pipeline.sh`, `bash/ss_pipeline.sh`,
and `bash/sl_pipeline.sh`) additionally handle HPC upload, verification, and
local cleanup. They accept `--releases`, `--skip`, `--keep_local`, and
`--dry_run` options.

Run the data pipeline from the project root, for example:

```bash
bash bash/ccd_pipeline.sh --dry_run --releases R5
```

### 2. Train and evaluate

```bash
# Train one model / run a normalization experiment
python scripts/train.py --help

# Compare baseline models; defaults to subject split and Symbol Search transfer
python scripts/run_baselines.py --help

# Baselines only, no transfer evaluation
python scripts/run_baselines.py --transfer_task none
```

The Slurm launchers `bash/run_ablation.sh`, `bash/run_ablation_all.sh`, and
`bash/run_transfer.sh` wrap the training entry points. Submit them from the
project root so Slurm writes logs under the repository's `logs/` directory.
`bash/run_pipeline.sh` is the general baseline launcher. Specialized SSL work
is started with `bash/run_ssl_pretrain_multirelease.sh`.

### 3. Outputs

Prepared windows are stored under `data/windows/`; experiment tables and plots
under `results/`; model checkpoints under `weights/`; and run logs under `logs/`.

## Code Map

- `configs/`: dataset, task, and training configuration.
- `bash/`: local and Slurm shell launchers.
- `scripts/`: data preparation, training, transfer, and analysis entry points.
- `src/data/`, `src/models/`, `src/training/`, `src/utils/`: reusable pipeline
    components.
- `data/`, `results/`, `weights/`, `logs/`: generated inputs and outputs.

Task preprocessing settings are in `configs/task_config.py`. Baseline model
selection is in `scripts/run_baselines.py` and model construction is in
`src/models/registry.py`.
