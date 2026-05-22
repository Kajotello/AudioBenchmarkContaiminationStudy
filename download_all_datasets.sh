#!/usr/bin/env bash
set -euo pipefail

module load Miniconda3
eval "$(conda shell.bash hook)"
module load CUDA/12.8

USERNAME="$(whoami)"

export SCRATCH_BASE="/net/tscratch/people/${USERNAME}"
export ENV_PREFIX="$SCRATCH_BASE/conda/py311_env"
conda activate "$ENV_PREFIX"

# ============================================================
# Fetch-once: materialise every dataset to local WAV + JSONL.
# Run from the repo root:  bash download_all_datasets.sh
#
# Re-running is cheap: existing WAVs are skipped and, when every WAV is
# already present, only the JSONL index is rebuilt (no audio re-decode).
# ============================================================

log() { echo "[download_all] $*"; }
die() { echo "[download_all][ERROR] $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DOWNLOAD="scripts/download_dataset.py"

[[ -f "$DOWNLOAD" ]] || die "$DOWNLOAD not found — run from the repo root."

log "Repo root : $REPO_ROOT"

# --- Clotho (CLAPv2/Clotho): member=train, non-member=validation ----------
log "=== CLAPv2/Clotho ==="
python "$DOWNLOAD" \
    --dataset-id CLAPv2/Clotho \
    --caption-mode clotho \
    --split train,validation \
    --back-translate

# --- AudioCaps (OpenSound/AudioCaps): member=train, non-member=test --------
log "=== OpenSound/AudioCaps ==="
python "$DOWNLOAD" \
    --dataset-id OpenSound/AudioCaps \
    --caption-mode audiocaps \
    --split train,test \
    --back-translate

# --- AudioSet (agkphysics/AudioSet, balanced): member=train, non-member=test
log "=== agkphysics/AudioSet (balanced) ==="
python "$DOWNLOAD" \
    --dataset-id agkphysics/AudioSet \
    --caption-mode audioset \
    --config-name balanced \
    --split train,test

log "All datasets downloaded under ./data/"
