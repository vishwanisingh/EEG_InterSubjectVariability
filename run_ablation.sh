#!/bin/bash
#SBATCH --job-name=eeg_ablation_single
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

# Usage: sbatch run_ablation_single.sh <norm> <model>
# Example: sbatch run_ablation_single.sh robust EEGNet
# Riemannian: sbatch --gres=gpu:h200:1 run_ablation_single.sh riemannian EEGConformer

set -euo pipefail

NORM="${1:-global_zscore}"
MODEL="${2:-EEGNet}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

echo "============================================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time : $(date)"
echo "norm=$NORM  model=$MODEL"
echo "============================================================"

module load python/3.13.5
source "$SUBMIT_DIR/venv/bin/activate"

echo "Python   : $(which python)"
echo "Torch    : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA     : $(python -c 'import torch; print(torch.cuda.is_available())')"

cd "$SUBMIT_DIR"
mkdir -p logs results weights

python scripts/train.py \
    --model "$MODEL" \
    --norm  "$NORM"  \
    --train_releases R1,R2,R3,R4,R5,R6,R8,R9,R10 \
    --val_releases   R11 \
    --test_releases  R5

EXIT_CODE=$?
echo "============================================================"
[ $EXIT_CODE -eq 0 ] && echo "Done — $(date)  norm=$NORM  model=$MODEL" || echo "ERROR: $EXIT_CODE  norm=$NORM  model=$MODEL"
echo "============================================================"
exit $EXIT_CODE