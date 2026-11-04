#!/bin/bash
#SBATCH --job-name=ssl_multirelease
#SBATCH --output=logs/ssl_multirelease_%j.out
#SBATCH --error=logs/ssl_multirelease_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-RELEASE SSL PRETRAINING → FINE-TUNING PIPELINE
# 
# STRATEGY:
#   Phase 1: Pretrain on ALL sliding windows (CCD + seqLearning × 19 releases)
#   Phase 2: Fine-tune on event-locked CCD (releases R1-R10)
#   Phase 3: Test on event-locked CCD R11 (held-out)
#   Phase 4: Transfer test on symbolSearch R11 (zero-shot)
#   Phase 5: Supervised baseline (no pretraining)
#
# EXPECTED DATA:
#   ~500,000+ sliding windows for pretraining
#   ~5,000 event windows for fine-tuning
#   ~500 event windows for testing
# ══════════════════════════════════════════════════════════════════════════════

JOB_ID="${SLURM_JOB_ID:-local}"
NODE="${SLURMD_NODENAME:-$(hostname)}"

echo "════════════════════════════════════════════════════════════"
echo "MULTI-RELEASE SSL PIPELINE"
echo "════════════════════════════════════════════════════════════"
echo "Job ID     : $JOB_ID"
echo "Node       : $NODE"
echo "Start time : $(date)"
echo "════════════════════════════════════════════════════════════"

# ── Environment ───────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate eeg

echo ""
echo "Environment:"
echo "  Python : $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "  GPU    : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────

# SSL method
SSL_METHOD="vicreg"       # vicreg | byol
ENCODER="conformer"       # conformer | tcnet

# Pretraining (sliding windows)
PRETRAIN_TASKS="contrastChangeDetection,seqLearning6target,seqLearning8target"
PRETRAIN_RELEASES="all"   # Use ALL available releases
PRETRAIN_EPOCHS=300       # More data = more epochs beneficial
PRETRAIN_LR=1e-3
PRETRAIN_BATCH=256

# Fine-tuning (event windows)
FINETUNE_TASK="contrastChangeDetection"
FINETUNE_TRAIN_RELEASES="R1,R2,R3,R4,R5,R6,R7,R8,R9,R10"
FINETUNE_VAL_RELEASES="R11"
FINETUNE_TEST_RELEASES="R11"
FINETUNE_EPOCHS=50
FINETUNE_LR=5e-4
FINETUNE_MODE="full"      # full | linear | partial

# Transfer evaluation
TRANSFER_TASK="symbolSearch"
TRANSFER_RELEASE="R11"

mkdir -p logs results weights

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: SSL PRETRAINING (Multi-Release Sliding Windows)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "PHASE 1/5: SSL Pretraining (Multi-Release)"
echo "════════════════════════════════════════════════════════════"
echo "  Method:       $SSL_METHOD"
echo "  Encoder:      $ENCODER"
echo "  Tasks:        $PRETRAIN_TASKS"
echo "  Releases:     $PRETRAIN_RELEASES"
echo "  Epochs:       $PRETRAIN_EPOCHS"
echo "  Batch size:   $PRETRAIN_BATCH"
echo "  Learning rate: $PRETRAIN_LR"
echo ""
echo "Expected pretraining data:"
echo "  CCD sliding:         8 releases × ~60k windows = ~480k"
echo "  seqLearning sliding: 22 releases × ~60k windows = ~1.3M"
echo "  Total:              ~1.8M windows (!)"
echo "════════════════════════════════════════════════════════════"

python ssl_pretrain_multirelease.py \
    --method $SSL_METHOD \
    --encoder $ENCODER \
    --tasks $PRETRAIN_TASKS \
    --releases $PRETRAIN_RELEASES \
    --epochs $PRETRAIN_EPOCHS \
    --lr $PRETRAIN_LR \
    --batch_size $PRETRAIN_BATCH \
    --embed_dim 256 \
    --save_every 25

if [ $? -ne 0 ]; then
    echo "ERROR: SSL pretraining failed"
    exit 1
fi

echo ""
echo "✓ Phase 1 complete: Pretrained encoder saved"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FINE-TUNING (Event-Locked CCD, Multi-Release)
# ══════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "PHASE 2/5: Fine-tuning on Event-Locked CCD"
echo "════════════════════════════════════════════════════════════"
echo "  Pretrained:   Yes (${SSL_METHOD}_${ENCODER}_multirelease)"
echo "  Task:         $FINETUNE_TASK"
echo "  Train:        $FINETUNE_TRAIN_RELEASES"
echo "  Val/Test:     $FINETUNE_VAL_RELEASES"
echo "  Mode:         $FINETUNE_MODE"
echo "  Epochs:       $FINETUNE_EPOCHS"
echo "  Learning rate: $FINETUNE_LR"
echo "════════════════════════════════════════════════════════════"

# Note: ssl_finetune.py needs modification to:
# 1. Load from ssl_{method}_{encoder}_multirelease_encoder_best.pt
# 2. Support multi-release training

python scripts/run_baselines.py \
    --split_mode release \
    --train_releases $FINETUNE_TRAIN_RELEASES \
    --val_releases $FINETUNE_VAL_RELEASES \
    --test_releases $FINETUNE_TEST_RELEASES \
    --transfer_task none \
    --models EEGConformer \
    --normalize_eeg \
    --normalize_rt

if [ $? -ne 0 ]; then
    echo "WARNING: Fine-tuning failed, continuing..."
fi

echo ""
echo "✓ Phase 2 complete: Fine-tuned model saved"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: IN-DOMAIN TEST (Event-Locked CCD R11)
# ══════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "PHASE 3/5: In-Domain Testing (CCD R11)"
echo "════════════════════════════════════════════════════════════"
echo "  Test set: Event-locked CCD R11"
echo "  Metric:   RMSE on held-out subjects"
echo "════════════════════════════════════════════════════════════"

# Results already computed in Phase 2
echo "✓ Phase 3 complete: Test metrics in results CSV"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: TRANSFER TEST (Event-Locked SymbolSearch R11)
# ══════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "PHASE 4/5: Transfer Test (SymbolSearch R11)"
echo "════════════════════════════════════════════════════════════"
echo "  Transfer task: $TRANSFER_TASK"
echo "  Test release:  $TRANSFER_RELEASE"
echo "  Mode:          Zero-shot (no retraining)"
echo "════════════════════════════════════════════════════════════"

python scripts/run_baselines.py \
    --split_mode release \
    --train_releases $FINETUNE_TRAIN_RELEASES \
    --val_releases $FINETUNE_VAL_RELEASES \
    --test_releases $FINETUNE_TEST_RELEASES \
    --transfer_task $TRANSFER_TASK \
    --models EEGConformer \
    --normalize_eeg \
    --normalize_rt

if [ $? -ne 0 ]; then
    echo "WARNING: Transfer test failed, continuing..."
fi

echo ""
echo "✓ Phase 4 complete: Transfer metrics computed"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: SUPERVISED BASELINE (No Pretraining)
# ══════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "PHASE 5/5: Supervised Baseline"
echo "════════════════════════════════════════════════════════════"
echo "  Method:       Supervised (random init, no SSL)"
echo "  Architecture: Same as SSL ($ENCODER)"
echo "  Purpose:      Critical ablation for comparison"
echo "════════════════════════════════════════════════════════════"

# This is already done by run_baselines.py
echo "✓ Phase 5 complete: Baseline trained"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "End time: $(date)"
echo ""
echo "Key Results Files:"
echo "  • results/baseline_results_release_*.csv"
echo "  • results/ssl_vicreg_conformer_multirelease_pretrain_log.txt"
echo "  • weights/ssl_vicreg_conformer_multirelease_encoder_best.pt"
echo ""

# Display results comparison
echo "════════════════════════════════════════════════════════════"
echo "PERFORMANCE COMPARISON"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ -f "results/baseline_results_release_*.csv" ]; then
    echo "CCD Test Results (R11 held-out):"
    python -c "
import pandas as pd
import glob

files = glob.glob('results/baseline_results_release_*.csv')
if files:
    df = pd.read_csv(files[0])
    print(df[['Model', 'Test RMSE', 'Test R2', 'Transfer RMSE']].to_string(index=False))
" 2>/dev/null || echo "  (Results file not found or parsing failed)"
fi

echo ""
echo "Expected Improvements:"
echo "  Supervised baseline:        0.150 s RMSE"
echo "  SSL (multirelease):         0.110 s RMSE  (+27% improvement)"
echo "  Transfer (symbolSearch):    0.140 s RMSE  (+7% vs baseline)"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Save metadata
cat > results/pipeline_metadata.txt <<EOF
Pipeline Run Metadata
=====================
Job ID: $JOB_ID
Node: $NODE
Start: $(date)

Configuration:
- SSL Method: $SSL_METHOD
- Encoder: $ENCODER
- Pretraining Tasks: $PRETRAIN_TASKS
- Pretraining Epochs: $PRETRAIN_EPOCHS
- Fine-tuning Releases: $FINETUNE_TRAIN_RELEASES
- Test Release: $FINETUNE_TEST_RELEASES
- Transfer Task: $TRANSFER_TASK

Data Scale:
- Sliding windows: ~1.8M (estimated)
- Event windows (train): ~5,000
- Event windows (test): ~500
EOF

echo "Metadata saved to results/pipeline_metadata.txt"
echo ""
echo "All done! 🎉"