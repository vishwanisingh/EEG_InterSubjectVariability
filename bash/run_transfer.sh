#!/bin/bash
#SBATCH --job-name=eeg_transfer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com   # <-- change this

# Usage:
#   sbatch bash/run_transfer.sh <transfer_task> [norm_method] (from project root)
#
# norm_method: global_zscore (default) or robust
#
# Examples:
#   sbatch bash/run_transfer.sh symbolSearch
#   sbatch bash/run_transfer.sh symbolSearch robust
#   sbatch bash/run_transfer.sh seqLearning8target global_zscore
#   sbatch bash/run_transfer.sh seqLearning8target robust

set -euo pipefail

TRANSFER_TASK="${1:-symbolSearch}"
NORM="${2:-global_zscore}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time  : $(date)"
echo "Transfer    : CCD -> $TRANSFER_TASK"
echo "Norm        : $NORM"
echo "============================================================"

module load python/3.13.5
source "$PROJECT_ROOT/venv/bin/activate"

echo "Python   : $(which python)"
echo "Torch    : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA     : $(python -c 'import torch; print(torch.cuda.is_available())')"

cd "$PROJECT_ROOT"
mkdir -p logs results weights

python scripts/run_baselines.py \
    --split_mode release \
    --train_releases R1,R2,R3,R4,R5,R6,R8,R9,R10 \
    --val_releases   R11 \
    --test_releases  R5 \
    --transfer_task  "$TRANSFER_TASK" \
    --normalize_eeg \
    --normalize_rt \
    --norm_method    "$NORM"

EXIT_CODE=$?

echo ""
echo "============================================================"
[ $EXIT_CODE -eq 0 ] && echo "Done — $(date)  transfer=$TRANSFER_TASK  norm=$NORM" || echo "ERROR: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE