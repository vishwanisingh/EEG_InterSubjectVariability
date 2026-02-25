#!/bin/bash
# =============================================================================
# setup_env.sh — Set up the Conda environment on HPC from scratch
#
# Usage:
#   bash setup_env.sh
#
# Run this ONCE interactively on a login node (not as a Slurm job).
# After this completes, your Slurm jobs can simply activate the env.
# =============================================================================

set -euo pipefail

ENV_NAME="eeg"
ENV_FILE="environment.yml"

echo "============================================================"
echo "EEG Project — HPC Environment Setup"
echo "============================================================"

# ── Step 1: Find conda ───────────────────────────────────────────────────────
echo ""
echo "[1/5] Locating Conda..."

# Try to load a conda module if the cluster uses Lmod
if command -v module &>/dev/null; then
    echo "  Module system detected. Trying to load conda/miniconda..."
    module load miniconda3 2>/dev/null \
        || module load anaconda3 2>/dev/null \
        || module load conda 2>/dev/null \
        || echo "  No conda module found — assuming conda is already in PATH."
fi

# Initialise conda for this shell session
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: conda not found. Please load the conda/miniconda module manually:"
    echo "  module avail conda"
    echo "  module avail miniconda"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "  Conda base: $CONDA_BASE"
echo "  Conda version: $(conda --version)"

# ── Step 2: Remove old env if it exists ─────────────────────────────────────
echo ""
echo "[2/5] Checking for existing '$ENV_NAME' environment..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Found existing '$ENV_NAME'. Removing it..."
    conda env remove -n "$ENV_NAME" -y
else
    echo "  No existing '$ENV_NAME' environment found."
fi

# ── Step 3: Create environment from YAML ────────────────────────────────────
echo ""
echo "[3/5] Creating environment from $ENV_FILE..."
echo "  (This may take 5–15 minutes on first run)"

conda env create -f "$ENV_FILE"

echo "  Environment '$ENV_NAME' created successfully."

# ── Step 4: Verify key packages ─────────────────────────────────────────────
echo ""
echo "[4/5] Verifying installation..."

conda activate "$ENV_NAME"

python - <<'EOF'
import importlib, sys

packages = {
    "torch":       "PyTorch",
    "braindecode": "Braindecode",
    "eegdash":     "EEGDash",
    "mne":         "MNE",
    "sklearn":     "scikit-learn",
    "numpy":       "NumPy",
    "pandas":      "Pandas",
    "torchaudio":  "torchaudio",
}

all_ok = True
for mod, name in packages.items():
    try:
        m = importlib.import_module(mod)
        version = getattr(m, "__version__", "?")
        print(f"  ✓  {name:<15} {version}")
    except ImportError:
        print(f"  ✗  {name:<15} NOT FOUND")
        all_ok = False

import torch
cuda_ok = torch.cuda.is_available()
print(f"\n  CUDA available : {cuda_ok}")
if cuda_ok:
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
else:
    print("  (CUDA will be available when running on a GPU node via Slurm)")

sys.exit(0 if all_ok else 1)
EOF

# ── Step 5: Print activation instructions ───────────────────────────────────
echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "To activate the environment in your Slurm scripts, add:"
echo ""
echo "    source \$(conda info --base)/etc/profile.d/conda.sh"
echo "    conda activate $ENV_NAME"
echo ""
echo "Or interactively on the login node:"
echo "    conda activate $ENV_NAME"
echo "============================================================"