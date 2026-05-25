#!/usr/bin/env bash
#SBATCH --account=plgmusicxai01-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=ct_smk_big
#SBATCH --output=logs/grid/smoke_af2_big_%j.out
#SBATCH --error=logs/grid/smoke_af2_big_%j.err

# Smoke for AF2 1.5B + AF2 3B only — 16 combos, 5 samples each, sequential.
# Verifies the lang_encoder_path override and the torch.load weights_only=False
# patch are sufficient for the larger AF2 variants.

set -uo pipefail   # NOT -e — continue past per-combo failures

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

MAPPING="$REPO_ROOT/configs/grid_jobs_af2_big.txt"
N=5

declare -a FAILED=()
declare -a OK=()
TOTAL=0

while IFS=$'\t' read -r MODEL METHOD DATA_M DATA_NM TAG; do
  [[ -z "${MODEL:-}" || "${MODEL:0:1}" == "#" ]] && continue
  TOTAL=$((TOTAL + 1))

  case "$METHOD" in
    codec)     ENTRY="src/contamination.py"  ;;
    mm_detect) ENTRY="src/eval_mm_detect.py" ;;
    *) echo "[smoke] unknown method: $METHOD"; FAILED+=("$TAG (unknown)"); continue ;;
  esac

  EXTRA_OVERRIDES=()
  if [[ "$METHOD" != "mm_detect" ]]; then
    EXTRA_OVERRIDES+=("batch_size=1")
  fi

  echo
  echo "================================================================"
  echo "[smoke] $TOTAL  model=$MODEL  method=$METHOD  dm=$DATA_M  dnm=$DATA_NM"
  echo "================================================================"

  cd "$DATA_DIR"
  if python "$REPO_ROOT/$ENTRY" \
      model="$MODEL" method="$METHOD" \
      data_member="$DATA_M" data_non_member="$DATA_NM" \
      paths.data_dir="$DATA_DIR" \
      paths.log_dir="$SCRATCH_BASE/contam_logs/" \
      max_member_samples="$N" max_non_member_samples="$N" \
      "${EXTRA_OVERRIDES[@]}" \
      tags="[smoke,$TAG]"; then
    OK+=("$TAG")
    echo "[smoke] OK  $TAG"
  else
    rc=$?
    FAILED+=("$TAG (exit $rc)")
    echo "[smoke] FAIL  $TAG  (exit $rc)"
  fi
done < "$MAPPING"

echo
echo "================================================================"
echo "[smoke] summary  total=$TOTAL  ok=${#OK[@]}  fail=${#FAILED[@]}"
echo "================================================================"
if (( ${#FAILED[@]} > 0 )); then
  echo "[smoke] failed combos:"
  for f in "${FAILED[@]}"; do echo "   - $f"; done
fi
