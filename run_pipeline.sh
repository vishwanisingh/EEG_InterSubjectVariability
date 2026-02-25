#!/bin/bash
#SBATCH --job-name=eeg_baseline_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu              # <-- change to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com   # <-- change this

set -euo pipefail

echo "============================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "============================================================"

# ── Activate conda env ───────────────────────────────────────────────────────
# Try loading a module first (common on HPC clusters)
if command -v module &>/dev/null; then
    module load miniconda3 2>/dev/null \
        || module load anaconda3 2>/dev/null \
        || module load conda 2>/dev/null \
        || true
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate eeg

echo "Python : $(which python)"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Move to project root ─────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results weights data

# ── Step 1: Prepare data ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP 1/2 — Preparing data"
echo "============================================================"

python create_and_save_splits.py

if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed. Aborting."
    exit 1
fi

echo "Data preparation complete."

# ── Step 2: Run baselines ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP 2/2 — Running baseline models"
echo "============================================================"

python run_baselines.py

if [ $? -ne 0 ]; then
    echo "ERROR: Baseline training failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "Pipeline complete — $(date)"
echo "Results : results/baseline_results.csv"
echo "Weights : weights/"
echo "============================================================"