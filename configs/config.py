"""Central configuration for all EEG regression experiments."""

from pathlib import Path

# ── Task & Dataset ────────────────────────────────────────────────────────────
TASK        = "contrastChangeDetection"
RELEASE     = "R5"         # used as folder suffix: data/windows/{TASK}_{RELEASE}
DATASET_ID  = "ds005509"   # R5 → ds005509; see release map below

# R11 → ds005516 | R10 → ds005515 | R9  → ds005514 | R8 → ds005512
# R7  → ds005511 | R6  → ds005510 | R5  → ds005509 | R4 → ds005508
# R3  → ds005507 | R2  → ds005506 | R1  → ds005505

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
WINDOWS_DIR = Path("data/windows")
RESULTS_DIR = Path("results")
WEIGHTS_DIR = Path("weights")

# ── EEG signal ────────────────────────────────────────────────────────────────
SFREQ      = 100    # Hz (competition data is downsampled to 100 Hz)
N_CHANS    = 129    # EEG channels in HBN dataset
WINDOW_LEN = 2.0    # seconds per epoch window
N_TIMES    = int(WINDOW_LEN * SFREQ)   # 200 samples

# ── Model output ──────────────────────────────────────────────────────────────
N_OUTPUTS  = 1      # regression → single RT value

# ── Train / val / test split ──────────────────────────────────────────────────
VALID_FRAC = 0.1
TEST_FRAC  = 0.1
SEED       = 2025

# Subjects with known data quality issues — excluded from all splits
EXCLUDED_SUBJECTS = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
    "NDARBA381JGH",
]

# ── DataLoader ────────────────────────────────────────────────────────────────
BATCH_SIZE   = 128
NUM_WORKERS  = 0    # set >0 on Linux for faster loading

# ── Training ──────────────────────────────────────────────────────────────────
N_EPOCHS      = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5       # early-stopping patience (epochs)
MIN_DELTA     = 1e-4    # minimum val-RMSE improvement to reset patience