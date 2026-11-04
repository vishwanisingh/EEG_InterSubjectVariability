#!/bin/bash
# =============================================================================
# ss_pipeline.sh — SymbolSearch: download → event windows → upload
#
# NOTE: No sliding windows — SS has event-only windowing.
#       Raw files are shared with CCD/SL datasets (same dataset IDs).
#       If raw was already downloaded by bash/ccd_pipeline.sh, download is skipped.
#
# Output folders per release:
#   data/windows/symbolSearch_{release}/
#
# Usage
# -----
#   bash bash/ss_pipeline.sh                      # all releases
#   bash bash/ss_pipeline.sh --releases R1,R2,R3  # subset
#   bash bash/ss_pipeline.sh --skip R1,R2         # skip done releases
#   bash bash/ss_pipeline.sh --keep_local
#   bash bash/ss_pipeline.sh --dry_run
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── CONFIG ────────────────────────────────────────────────────────────────────
HPC_USER="singh.vishwa"
HPC_HOST="xfer.discovery.neu.edu"
HPC_WINDOWS="/scratch/${HPC_USER}/thesis-eeg/data/windows"
SSH_KEY="${HOME}/.ssh/id_rsa"

PYTHON="python"
PREPARE="scripts/prepare_ss_data.py"
LOG_DIR="logs/pipeline"
DONE_FILE="${LOG_DIR}/ss_completed.txt"
SIZE_TOL=1048576

ALL_RELEASES=("R1" "R2" "R3" "R4" "R5" "R6" "R7" "R8" "R9" "R10" "R11")
declare -A RELEASE_MAP=(
    [R1]=ds005505 [R2]=ds005506 [R3]=ds005507 [R4]=ds005508
    [R5]=ds005509 [R6]=ds005510 [R7]=ds005511 [R8]=ds005512
    [R9]=ds005514 [R10]=ds005515 [R11]=ds005516
)
# ─────────────────────────────────────────────────────────────────────────────

DRY_RUN=0; KEEP_LOCAL=0; RELEASES=(); SKIP_RELEASES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)    DRY_RUN=1 ;;
        --keep_local) KEEP_LOCAL=1 ;;
        --releases)   IFS=',' read -ra RELEASES <<< "$2"; shift ;;
        --skip)       IFS=',' read -ra SKIP_RELEASES <<< "$2"; shift ;;
        *) echo "[warn] Unknown arg: $1" ;;
    esac
    shift
done

[[ ${#RELEASES[@]} -eq 0 ]] && RELEASES=("${ALL_RELEASES[@]}")

if [[ ${#SKIP_RELEASES[@]} -gt 0 ]]; then
    FILTERED=()
    for r in "${RELEASES[@]}"; do
        skip=0
        for s in "${SKIP_RELEASES[@]}"; do [[ "$r" == "$s" ]] && skip=1 && break; done
        [[ $skip -eq 0 ]] && FILTERED+=("$r")
    done
    RELEASES=("${FILTERED[@]}")
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"; touch "$DONE_FILE"
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo ""; echo "── $*"; }
disk() { df -h . | awk 'NR==2 {print $4 " free"}'; }
is_done()   { grep -qx "$1" "$DONE_FILE" 2>/dev/null; }
mark_done() { echo "$1" >> "$DONE_FILE"; }

run() { [[ $DRY_RUN -eq 1 ]] && echo "  [dry] $*" || "$@"; }

delete_local() {
    local dir="$1"
    [[ -d "$dir" ]] || { log "  [skip delete] $dir not found"; return 0; }
    local size; size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    log "  Deleting $dir  ($size)"
    run rm -rf "$dir"
}

upload_verify_clean() {
    local local_dir="$1"
    local folder; folder=$(basename "$local_dir")
    local remote="${HPC_WINDOWS}/${folder}"

    [[ -d "$local_dir" ]] || { log "  [skip] $local_dir not found"; return 0; }

    log "  Uploading $folder  ($(du -sh "$local_dir" | cut -f1)) ..."
    run rsync -avzP -e "ssh -i $SSH_KEY" "$local_dir/" "${HPC_USER}@${HPC_HOST}:${remote}/" \
        || { log "  ✗ rsync failed — keeping local"; return 1; }

    [[ $DRY_RUN -eq 1 ]] && return 0

    local lc; lc=$(find "$local_dir" -name "*.npy" -o -name "*.npz" | wc -l)
    local rc; rc=$(ssh -i "$SSH_KEY" "${HPC_USER}@${HPC_HOST}" \
        "find '$remote' -name '*.npy' -o -name '*.npz' 2>/dev/null | wc -l" || echo 0)
    local ls; ls=$(du -sb "$local_dir" | cut -f1)
    local rs; rs=$(ssh -i "$SSH_KEY" "${HPC_USER}@${HPC_HOST}" \
        "du -sb '$remote' 2>/dev/null | cut -f1 || echo 0" || echo 0)
    local diff=$(( ls - rs )); diff=${diff#-}

    log "  Verify $folder — files: local=$lc HPC=$rc  size: local=${ls}B HPC=${rs}B"

    if [[ "$lc" -ne "$rc" ]]; then
        log "  ✗ File count mismatch — keeping local"; return 1
    fi
    if [[ "$diff" -gt "$SIZE_TOL" ]]; then
        log "  ✗ Size mismatch — keeping local"; return 1
    fi
    log "  ✓ Verified"

    [[ $KEEP_LOCAL -eq 0 ]] && delete_local "$local_dir" || log "  Keeping local (--keep_local)"
}

# ── SSH check ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "SS PIPELINE  —  download → event windows → HPC"
echo "  HPC      : ${HPC_USER}@${HPC_HOST}:${HPC_WINDOWS}"
echo "  Releases : ${RELEASES[*]}"
echo "  Dry run  : $DRY_RUN  |  Keep local: $KEEP_LOCAL"
echo "  Disk     : $(disk)"
echo "============================================================"

if [[ $DRY_RUN -eq 0 ]]; then
    log "Testing SSH ..."
    ssh -i "$SSH_KEY" -o ConnectTimeout=10 "${HPC_USER}@${HPC_HOST}" \
        "mkdir -p '$HPC_WINDOWS' && echo 'SSH OK'" || { log "SSH failed"; exit 1; }
fi

FAILED=()

# ── Main loop ─────────────────────────────────────────────────────────────────
for release in "${RELEASES[@]}"; do

    echo ""
    echo "============================================================"
    echo "RELEASE $release  |  $(disk)"
    echo "============================================================"

    if is_done "$release" && [[ $DRY_RUN -eq 0 ]]; then
        log "$release already completed — skipping"; continue
    fi

    DATASET_ID="${RELEASE_MAP[$release]}"
    RAW_DIR="data/${DATASET_ID}"
    WINDOW_DIR="data/windows/symbolSearch_${release}"
    LOG_FILE="${LOG_DIR}/ss_${release}.log"

    # ── 1. Download ───────────────────────────────────────────────────────────
    step "1/3  Download raw"
    if [[ -d "$RAW_DIR" && -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]]; then
        log "  [skip] $RAW_DIR already on disk"
    else
        run $PYTHON "$PREPARE" --mode download --release "$release" \
            2>&1 | tee -a "$LOG_FILE" \
            || { log "✗ Download failed"; FAILED+=("$release:download"); continue; }
    fi

    # ── 2. Event windows ──────────────────────────────────────────────────────
    step "2/3  Event-locked windows"
    if [[ -f "${WINDOW_DIR}/subjects.txt" ]]; then
        log "  [skip] already complete"
    else
        run $PYTHON "$PREPARE" --mode window --release "$release" --subject_batch_size 5 \
            2>&1 | tee -a "$LOG_FILE" \
            || { log "✗ Windowing failed"; FAILED+=("$release:window"); }
    fi

    # ── 3. Upload + verify + clean ────────────────────────────────────────────
    # NOTE: Raw is NOT deleted here — it may be shared with SL pipeline.
    # Delete raw manually once both CCD, SL, and SS pipelines are done for a release.
    step "3/3  Upload to HPC"
    upload_verify_clean "$WINDOW_DIR" || FAILED+=("$release:upload")

    mark_done "$release"
    log "Release $release DONE  |  $(disk)"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "SS PIPELINE COMPLETE  |  $(disk)"
[[ ${#FAILED[@]} -eq 0 ]] && echo "  ✓ All done" || { echo "  ✗ Failed:"; printf "    - %s\n" "${FAILED[@]}"; }
echo "============================================================"