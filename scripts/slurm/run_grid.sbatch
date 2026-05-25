#!/usr/bin/env bash
#SBATCH --account=plgmusicxai01-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=ct_grid
#SBATCH --output=logs/grid/%A_%a.out
#SBATCH --error=logs/grid/%A_%a.err
#SBATCH --array=0-31

# Grid of (model x method x dataset) combinations.
# Mapping defined in configs/grid_jobs.txt — one record per non-comment line:
#   <model>\t<method>\t<data_member>\t<data_non_member>\t<tag>
#
# Indexing: SLURM_ARRAY_TASK_ID picks the i-th non-comment line (0-based).
#
# Launch with:
#   mkdir -p logs/grid
#   sbatch run_grid.sbatch
#
# Or a subset:
#   sbatch --array=0-9 run_grid.sbatch        # only first 10
#   sbatch --array=21-32 run_grid.sbatch      # only mm_detect jobs

set -euo pipefail

# -------------------- env (matches download_all_datasets.sh) ---------------
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

# -------------------- pick combination ------------------------------------
MAPPING="$REPO_ROOT/configs/grid_jobs.txt"
LINE=$(grep -v '^#' "$MAPPING" | grep -v '^[[:space:]]*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [[ -z "$LINE" ]]; then
  echo "[grid] no mapping for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
IFS=$'\t' read -r MODEL METHOD DATA_M DATA_NM TAG <<< "$LINE"

echo "[grid] task=$SLURM_ARRAY_TASK_ID  model=$MODEL  method=$METHOD  data_member=$DATA_M  data_non_member=$DATA_NM  tag=$TAG"
echo "[grid] node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"

# -------------------- pick entry script ------------------------------------
case "$METHOD" in
  codec)      ENTRY="src/contamination.py"   ;;
  mm_detect)  ENTRY="src/eval_mm_detect.py"  ;;
  *) echo "[grid] unknown method: $METHOD" >&2; exit 1 ;;
esac

# Caps differ by model size — AF3 is far slower, so fewer samples.
if [[ "$MODEL" == "audio_flamingo3" ]]; then
  MAX_MEMBER=100
  MAX_NON_MEMBER=100
else
  MAX_MEMBER=1000
  MAX_NON_MEMBER=1000
fi
echo "[grid] sample cap: member=$MAX_MEMBER  non_member=$MAX_NON_MEMBER"

# AF3 + codec: transformers 5.9's AudioFlamingo3 processor enforces 1:1
# text:audio inside apply_chat_template, which multi-audio codec chats violate.
# mode=no_audio drops the context audio so only the target audio remains.
EXTRA_OVERRIDES=()
if [[ "$MODEL" == "audio_flamingo3" && "$METHOD" == "codec" ]]; then
  EXTRA_OVERRIDES+=("method.mode=no_audio")
fi

# -------------------- run --------------------------------------------------
# Audio paths in JSONL are stored relative to data_dir — running from data_dir
# lets the dataset's `Path.cwd() / p` resolution branch succeed.
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
  tags="[grid,$TAG]"

echo "[grid] done  task=$SLURM_ARRAY_TASK_ID  tag=$TAG"
