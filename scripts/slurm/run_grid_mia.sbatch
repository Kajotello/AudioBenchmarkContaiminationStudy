#!/usr/bin/env bash
#SBATCH --account=plgmusicxai01-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=ct_mia
#SBATCH --output=logs/grid/mia_%A_%a.out
#SBATCH --error=logs/grid/mia_%A_%a.err
#SBATCH --array=0-59

# All-MIA grid: 4 models × 5 MIA methods × 3 paired datasets = 60 combos.
# Entrypoint: src/eval_mia.py (config_name=eval.yaml).
# Methods: MIA_perplexity, yeom_perplexity, min_k, min_k_pp, vl_mia_entropy.
# Datasets: clotho, audiocaps, audioset (paired member train vs non-member val/test).
# 1000 random samples per split.
#
# Launch:
#   mkdir -p logs/grid
#   sbatch run_grid_mia.sbatch
#
# Subset (e.g. AF3 only — first 15 combos):
#   sbatch --array=0-14 run_grid_mia.sbatch

set -euo pipefail

module load Miniconda3
eval "$(conda shell.bash hook)"
module load CUDA/12.8

USERNAME="$(whoami)"
export SCRATCH_BASE="/net/tscratch/people/${USERNAME}"
export ENV_PREFIX="/net/tscratch/people/plgwzarzecki/pw/envs/inference_env/"
conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

REPO_ROOT="/net/people/plgrid/plgwzarzecki/pw/AudioBenchmarkContaiminationStudy"
DATA_DIR="/net/tscratch/people/plgwzarzecki/pw/datasets"
export PROJECT_ROOT="$REPO_ROOT"
cd "$REPO_ROOT"

MAPPING="$REPO_ROOT/configs/grid_jobs_mia.txt"
LINE=$(grep -v '^#' "$MAPPING" | grep -v '^[[:space:]]*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [[ -z "$LINE" ]]; then
  echo "[mia] no mapping for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
IFS=$'\t' read -r MODEL METHOD DATA_M DATA_NM TAG <<< "$LINE"

echo "[mia] task=$SLURM_ARRAY_TASK_ID  model=$MODEL  method=$METHOD  data_member=$DATA_M  data_non_member=$DATA_NM  tag=$TAG"
echo "[mia] node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"

MAX_MEMBER=1000
MAX_NON_MEMBER=1000
echo "[mia] sample cap: member=$MAX_MEMBER  non_member=$MAX_NON_MEMBER"

sleep "$SLURM_ARRAY_TASK_ID"
cd "$DATA_DIR"

python "$REPO_ROOT/src/eval_mia.py" \
  model="$MODEL" \
  method="$METHOD" \
  data_member="$DATA_M" \
  data_non_member="$DATA_NM" \
  paths.data_dir="$DATA_DIR" \
  paths.log_dir="$SCRATCH_BASE/contam_logs/" \
  max_member_samples="$MAX_MEMBER" \
  max_non_member_samples="$MAX_NON_MEMBER" \
  tags="[mia_1k,$TAG]"

echo "[mia] done  task=$SLURM_ARRAY_TASK_ID  tag=$TAG"
