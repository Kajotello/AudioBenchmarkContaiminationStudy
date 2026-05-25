#!/usr/bin/env bash
#SBATCH --account=plgmusicxai01-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=ct_af2_big
#SBATCH --output=logs/grid/af2_big_%A_%a.out
#SBATCH --error=logs/grid/af2_big_%A_%a.err
#SBATCH --array=0-15

# AF2 1.5B + AF2 3B grid: 16 combos per configs/grid_jobs_af2_big.txt
# (per size: codec × 5 datasets + mm_detect × 3 paired datasets)
# at 1000 random samples per split.
#
# Launch:
#   mkdir -p logs/grid
#   sbatch run_grid_af2_big.sbatch

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

MAPPING="$REPO_ROOT/configs/grid_jobs_af2_big.txt"
LINE=$(grep -v '^#' "$MAPPING" | grep -v '^[[:space:]]*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [[ -z "$LINE" ]]; then
  echo "[af2_big] no mapping for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
IFS=$'\t' read -r MODEL METHOD DATA_M DATA_NM TAG <<< "$LINE"

echo "[af2_big] task=$SLURM_ARRAY_TASK_ID  model=$MODEL  method=$METHOD  data_member=$DATA_M  data_non_member=$DATA_NM  tag=$TAG"
echo "[af2_big] node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"

case "$METHOD" in
  codec)      ENTRY="src/contamination.py"   ;;
  mm_detect)  ENTRY="src/eval_mm_detect.py"  ;;
  *) echo "[af2_big] unknown method: $METHOD" >&2; exit 1 ;;
esac

MAX_MEMBER=1000
MAX_NON_MEMBER=1000
echo "[af2_big] sample cap: member=$MAX_MEMBER  non_member=$MAX_NON_MEMBER"

sleep "$SLURM_ARRAY_TASK_ID"
cd "$DATA_DIR"

python "$REPO_ROOT/$ENTRY" \
  model="$MODEL" \
  method="$METHOD" \
  data_member="$DATA_M" \
  data_non_member="$DATA_NM" \
  paths.data_dir="$DATA_DIR" \
  paths.log_dir="$SCRATCH_BASE/contam_logs/" \
  max_member_samples="$MAX_MEMBER" \
  max_non_member_samples="$MAX_NON_MEMBER" \
  tags="[af2_big_1k,$TAG]"

echo "[af2_big] done  task=$SLURM_ARRAY_TASK_ID  tag=$TAG"
