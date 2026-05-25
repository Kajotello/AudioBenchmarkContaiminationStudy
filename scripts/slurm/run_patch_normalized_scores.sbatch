#!/usr/bin/env bash
#SBATCH --account=plgmusicxai01-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=patch_norm
#SBATCH --output=logs/grid/patch_normalized_scores_%j.out
#SBATCH --error=logs/grid/patch_normalized_scores_%j.err

# Back-fills member_normalized_score / non_member_normalized_score onto all
# existing CoDeC run directories and updates matching wandb run summaries.
# Scans both the project-local logs/ and the scratch contam_logs/ directory.
#
# Launch with:
#   mkdir -p logs/grid
#   sbatch run_patch_normalized_scores.sbatch

set -euo pipefail

# -------------------- env --------------------------------------------------
module load Miniconda3
eval "$(conda shell.bash hook)"

USERNAME="$(whoami)"
export SCRATCH_BASE="/net/tscratch/people/${USERNAME}"
export ENV_PREFIX="/net/tscratch/people/plgwzarzecki/pw/envs/inference_env/"
conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export PYTHONNOUSERSITE=1

REPO_ROOT="/net/people/plgrid/plgwzarzecki/pw/AudioBenchmarkContaiminationStudy"
export PROJECT_ROOT="$REPO_ROOT"

SCRIPT="$REPO_ROOT/scripts/patch_normalized_scores.py"
ENTITY="nask-di"
PROJECT="audio-benchmark"

# -------------------- patch ------------------------------------------------
for LOGS_DIR in \
    "$REPO_ROOT/logs/contamination/runs" \
    "$SCRATCH_BASE/contam_logs/contamination/runs"
do
    if [ ! -d "$LOGS_DIR" ]; then
        echo "Skipping missing dir: $LOGS_DIR"
        continue
    fi
    echo
    echo "=== Patching: $LOGS_DIR ==="
    python "$SCRIPT" \
        --logs-dir "$LOGS_DIR" \
        --entity "$ENTITY" \
        --project "$PROJECT"
done
