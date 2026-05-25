#!/usr/bin/env bash
#SBATCH --account=plgmusicxai01-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=ct_af3
#SBATCH --output=logs/grid/af3_%A_%a.out
#SBATCH --error=logs/grid/af3_%A_%a.err
#SBATCH --array=0-7

# AF3-only grid: 8 combos (codec × 5 datasets + mm_detect × 3 paired datasets)
# at 1000 random samples per split. Mapping: configs/grid_jobs_af3.txt.
#
# Launch:
#   mkdir -p logs/grid
#   sbatch run_grid_af3.sbatch

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

MAPPING="$REPO_ROOT/configs/grid_jobs_af3.txt"
LINE=$(grep -v '^#' "$MAPPING" | grep -v '^[[:space:]]*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [[ -z "$LINE" ]]; then
  echo "[af3] no mapping for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
IFS=$'\t' read -r MODEL METHOD DATA_M DATA_NM TAG <<< "$LINE"

echo "[af3] task=$SLURM_ARRAY_TASK_ID  model=$MODEL  method=$METHOD  data_member=$DATA_M  data_non_member=$DATA_NM  tag=$TAG"
echo "[af3] node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"

case "$METHOD" in
  codec)      ENTRY="src/contamination.py"   ;;
  mm_detect)  ENTRY="src/eval_mm_detect.py"  ;;
  *) echo "[af3] unknown method: $METHOD" >&2; exit 1 ;;
esac

MAX_MEMBER=1000
MAX_NON_MEMBER=1000
echo "[af3] sample cap: member=$MAX_MEMBER  non_member=$MAX_NON_MEMBER"

# AF3 + codec: transformers 5.9's AudioFlamingo3 processor enforces 1:1
# text:audio inside apply_chat_template, which multi-audio codec chats violate.
# mode=no_audio drops the context audio so only the target audio remains.
EXTRA_OVERRIDES=()
if [[ "$METHOD" == "codec" ]]; then
  EXTRA_OVERRIDES+=("method.mode=no_audio")
fi

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
  "${EXTRA_OVERRIDES[@]}" \
  tags="[af3_1k,$TAG]"

echo "[af3] done  task=$SLURM_ARRAY_TASK_ID  tag=$TAG"
