#!/usr/bin/env bash
set -euo pipefail

module load Miniconda3
eval "$(conda shell.bash hook)"
module load CUDA/12.8

USERNAME="$(whoami)"

export SCRATCH_BASE="/net/tscratch/people/${USERNAME}"
export ENV_PREFIX="/net/tscratch/people/plgwzarzecki/pw/envs/download_env/"
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

OUTPUT_DIR="/net/tscratch/people/plgwzarzecki/pw/datasets"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir|-o) OUTPUT_DIR="$2"; shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Ensure spacy model is available
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null \
  || python -m spacy download en_core_web_sm

DOWNLOAD="scripts/download_dataset.py"

[[ -f "$DOWNLOAD" ]] || die "$DOWNLOAD not found — run from the repo root."

log "Repo root  : $REPO_ROOT"
log "Output dir : $OUTPUT_DIR"

# --- Clotho (CLAPv2/Clotho): member=train, non-member=validation ----------
log "=== CLAPv2/Clotho ==="
python "$DOWNLOAD" \
    --dataset-id CLAPv2/Clotho \
    --caption-mode clotho \
    --split train,validation \
    --output-dir "$OUTPUT_DIR" \
    --back-translate

# --- AudioCaps (OpenSound/AudioCaps): member=train, non-member=test --------
log "=== OpenSound/AudioCaps ==="
python "$DOWNLOAD" \
    --dataset-id OpenSound/AudioCaps \
    --caption-mode audiocaps \
    --split train,test \
    --output-dir "$OUTPUT_DIR" \
    --back-translate

# --- AudioSet (agkphysics/AudioSet, balanced): member=train, non-member=test
log "=== agkphysics/AudioSet (balanced) ==="
python "$DOWNLOAD" \
    --dataset-id agkphysics/AudioSet \
    --caption-mode audioset \
    --config-name balanced \
    --split train,test \
    --output-dir "$OUTPUT_DIR"

# --- MMAU (TwinkStart/MMAU): benchmark QA ----
log "=== TwinkStart/MMAU ==="
python "$DOWNLOAD" \
    --dataset-id TwinkStart/MMAU \
    --caption-mode qa \
    --question-col question \
    --answer-col answer \
    --split test \
    --output-dir "$OUTPUT_DIR" \
    --no-back-translate

# --- MuchoMusic (mulab-mir/muchomusic): benchmark QA ----
log "=== mulab-mir/muchomusic ==="
python "$DOWNLOAD" \
    --dataset-id mulab-mir/muchomusic \
    --caption-mode qa \
    --question-col question \
    --answer-col answer \
    --split test \
    --output-dir "$OUTPUT_DIR" \
    --no-back-translate

# --- ClothoAQA (lmms-lab/ClothoAQA, clotho_aqa): benchmark QA ----
log "=== lmms-lab/ClothoAQA (clotho_aqa) ==="
python "$DOWNLOAD" \
    --dataset-id lmms-lab/ClothoAQA \
    --config-name clotho_aqa \
    --caption-mode qa \
    --question-col question \
    --answer-col answer \
    --split clotho_aqa_test_filtered \
    --output-dir "$OUTPUT_DIR" \
    --no-back-translate

log "All datasets downloaded under $OUTPUT_DIR"
