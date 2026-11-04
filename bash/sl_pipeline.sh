#!/bin/bash
# =============================================================================
# sl_pipeline.sh — SeqLearning: event + sliding windows → upload
#
# NOTE: No download step — raw files are shared with CCD/SS datasets.
#       Run bash/ccd_pipeline.sh (or bash/ss_pipeline.sh) first to download raw,
#       OR run CCD pipeline which downloads the shared dataset.
#
# Output folders per release:
#   data/windows/seqLearning8target_{release}/
#   data/windows/seqLearning6target_{release}/
#   data/windows/seqLearning8target_sliding_{release}/  (skips R5, R7)
#   data/windows/seqLearning6target_sliding_{release}/  (skips R5, R7)
#
# Usage
# -----
#   bash bash/sl_pipeline.sh                      # all releases
#   bash bash/sl_pipeline.sh --releases R1,R2,R3  # subset
#   bash bash/sl_pipeline.sh --skip R1,R2         # skip done releases
#   bash bash/sl_pipeline.sh --keep_local
#   bash bash/sl_pipeline.sh --dry_run
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
PREPARE="scripts/prepare_sl_data.py"
LOG_DIR="logs/pipeline"
DONE_FILE="${LOG_DIR}/sl_completed.txt"
SIZE_TOL=1048576

ALL_RELEASES=("R1" "R2" "R3" "R4" "R5" "R6" "R7" "R8" "R9" "R10" "R11")
TASKS=("seqLearning8target" "seqLearning6target")
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
echo "SL PIPELINE  —  event + sliding → HPC"
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

    LOG_FILE="${LOG_DIR}/sl_${release}.log"

    # ── 1. Event windows (both tasks) ─────────────────────────────────────────
    step "1/3  Event-locked windows"
    ALL_EVENT_DONE=1
    for task in "${TASKS[@]}"; do
        [[ -f "data/windows/${task}_${release}/subjects.txt" ]] || { ALL_EVENT_DONE=0; break; }
    done

    if [[ $ALL_EVENT_DONE -eq 1 ]]; then
        log "  [skip] both tasks already complete"
    else
        run $PYTHON "$PREPARE" --mode event --release "$release" --subject_batch_size 5 \
            2>&1 | tee -a "$LOG_FILE" \
            || { log "✗ Event windowing failed"; FAILED+=("$release:event"); }
    fi

    # ── 2. Sliding windows (both tasks, R5/R7 handled inside python) ──────────
    step "2/3  Sliding windows"
    ALL_SLIDE_DONE=1
    for task in "${TASKS[@]}"; do
        [[ -f "data/windows/${task}_sliding_${release}/subjects.txt" ]] || { ALL_SLIDE_DONE=0; break; }
    done

    if [[ $ALL_SLIDE_DONE -eq 1 ]]; then
        log "  [skip] both tasks already complete"
    else
        run $PYTHON "$PREPARE" --mode sliding --release "$release" --subject_batch_size 5 \
            2>&1 | tee -a "$LOG_FILE" \
            || { log "✗ Sliding windowing failed"; FAILED+=("$release:sliding"); }
    fi

    # ── 3. Upload + verify + clean ────────────────────────────────────────────
    step "3/3  Upload to HPC"
    for task in "${TASKS[@]}"; do
        upload_verify_clean "data/windows/${task}_${release}"           || FAILED+=("$release:upload_${task}_event")
        upload_verify_clean "data/windows/${task}_sliding_${release}"   || FAILED+=("$release:upload_${task}_sliding")
    done

    mark_done "$release"
    log "Release $release DONE  |  $(disk)"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "SL PIPELINE COMPLETE  |  $(disk)"
[[ ${#FAILED[@]} -eq 0 ]] && echo "  ✓ All done" || { echo "  ✗ Failed:"; printf "    - %s\n" "${FAILED[@]}"; }
echo "============================================================"