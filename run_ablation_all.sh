#!/bin/bash
#SBATCH --job-name=eeg_ablation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-29
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

set -euo pipefail

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

NORM_LIST=(global_zscore global_zscore global_zscore global_zscore global_zscore
           robust        robust        robust        robust        robust
           euclidean_alignment euclidean_alignment euclidean_alignment euclidean_alignment euclidean_alignment
           temporal_baseline   temporal_baseline   temporal_baseline   temporal_baseline   temporal_baseline
           frequency_band      frequency_band      frequency_band      frequency_band      frequency_band
           riemannian          riemannian          riemannian          riemannian          riemannian)

MODEL_LIST=(EEGNet EEGTCNet EEGConformer Deep4Net ShallowFBCSPNet
            EEGNet EEGTCNet EEGConformer Deep4Net ShallowFBCSPNet
            EEGNet EEGTCNet EEGConformer Deep4Net ShallowFBCSPNet
            EEGNet EEGTCNet EEGConformer Deep4Net ShallowFBCSPNet
            EEGNet EEGTCNet EEGConformer Deep4Net ShallowFBCSPNet
            EEGNet EEGTCNet EEGConformer Deep4Net ShallowFBCSPNet)

NORM="${NORM_LIST[$ARRAY_ID]}"
MODEL="${MODEL_LIST[$ARRAY_ID]}"

echo "============================================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}  (array task $ARRAY_ID)"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time : $(date)"
echo "This run   : norm=$NORM  model=$MODEL"
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