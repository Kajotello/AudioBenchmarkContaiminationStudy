#!/usr/bin/env bash
#SBATCH --account=plgmusicxai01-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=ct_smoke
#SBATCH --output=logs/grid/smoke_%j.out
#SBATCH --error=logs/grid/smoke_%j.err

# Single-job smoke: iterate every (model, method, dataset) combination from
# configs/grid_jobs.txt sequentially with max_samples=5. Failures are logged
# and the loop continues so we get a full pass/fail matrix in one run.
#
# Launch with:
#   mkdir -p logs/grid
#   sbatch run_grid_smoke.sbatch

set -uo pipefail   # NOT -e — we want to continue past per-combo failures

# -------------------- env --------------------------------------------------
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

MAPPING="$REPO_ROOT/configs/grid_jobs.txt"
N=5  # samples per split for smoke

# -------------------- iterate ---------------------------------------------
declare -a FAILED=()
declare -a OK=()
TOTAL=0

while IFS=$'\t' read -r MODEL METHOD DATA_M DATA_NM TAG; do
  # skip comments / blanks
  [[ -z "${MODEL:-}" || "${MODEL:0:1}" == "#" ]] && continue
  TOTAL=$((TOTAL + 1))

  case "$METHOD" in
    codec)     ENTRY="src/contamination.py"  ;;
    mm_detect) ENTRY="src/eval_mm_detect.py" ;;
    *) echo "[smoke] unknown method: $METHOD  -- skipped"; FAILED+=("$TAG (unknown method)"); continue ;;
  esac

  echo
  echo "================================================================"
  echo "[smoke] $TOTAL  model=$MODEL  method=$METHOD  dm=$DATA_M  dnm=$DATA_NM"
  echo "================================================================"

  # AF3 codec workaround: transformers 5.9's AudioFlamingo3 processor enforces
  # 1:1 text:audio inside apply_chat_template, which multi-audio codec chats
  # violate. mode=no_audio drops the context audio so only the target remains.
  EXTRA_OVERRIDES=()
  if [[ "$MODEL" == "audio_flamingo3" && "$METHOD" == "codec" ]]; then
    EXTRA_OVERRIDES+=("method.mode=no_audio")
  fi
  # batch_size only exists in contamination.yaml / eval.yaml — not eval_mm_detect.yaml.
  if [[ "$METHOD" != "mm_detect" ]]; then
    EXTRA_OVERRIDES+=("batch_size=1")
  fi

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

# -------------------- summary ----------------------------------------------
echo
echo "================================================================"
echo "[smoke] summary"
echo "================================================================"
echo "  total : $TOTAL"
echo "  ok    : ${#OK[@]}"
echo "  fail  : ${#FAILED[@]}"
if (( ${#FAILED[@]} > 0 )); then
  echo
  echo "[smoke] failed combos:"
  for f in "${FAILED[@]}"; do echo "   - $f"; done
fi
