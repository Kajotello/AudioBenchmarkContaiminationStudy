#!/usr/bin/env bash
set -euo pipefail

module load Miniconda3
eval "$(conda shell.bash hook)"
module load CUDA/12.8

USERNAME="$(whoami)"

export SCRATCH_BASE="/net/tscratch/people/${USERNAME}"
export ENV_PREFIX="$SCRATCH_BASE/conda/py311_env"

export MAMBA_ROOT_PREFIX="$SCRATCH_BASE/micromamba"
export XDG_CACHE_HOME="$SCRATCH_BASE/.cache"
export PIP_CACHE_DIR="$SCRATCH_BASE/.cache/pip"
export CONDA_PKGS_DIRS="$SCRATCH_BASE/.cache/conda/pkgs"

export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export TMPDIR="$SCRATCH_BASE/.tmp"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$MAMBA_ROOT_PREFIX" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$CONDA_PKGS_DIRS" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$TMPDIR"

conda activate "$ENV_PREFIX"

export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ENV_PREFIX/bin:${PATH}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$ENV_PREFIX}"

PROJECT_DIR="$SCRATCH_BASE/AudioBenchmarkContaiminationStudy"
cd "$PROJECT_DIR"
export PROJECT_ROOT="$PROJECT_DIR"

declare -A MAX_N=( [clotho]=100000 [audiocaps]=6630 [audioset]=10000 )


python src/eval_mm_detect.py \
      model=audio_flamingo3 method=mm_detect \
      data_member=mm_detect_clotho data_non_member=mm_detect_clotho \
      max_member_samples=${MAX_N[clotho]} max_non_member_samples=${MAX_N[clotho]} \
      tags="[full,mm_detect]"

# Full MIA battery: every method × every dataset (AF3)
for DATA in clotho audiocaps audioset; do
  N=${MAX_N[$DATA]}
  for METHOD in yeom_perplexity min_k min_k_pp vl_mia_entropy; do
    echo "=== full MIA: $DATA / $METHOD (N=$N) ==="
    python src/eval_mia.py \
        model=audio_flamingo3 method=$METHOD \
        data_member=$DATA data_non_member=$DATA \
        batch_size=16 max_member_samples=$N max_non_member_samples=$N \
        tags="[full,mia,$DATA,$METHOD]"
  done
done

# k%/top% sweeps (multirun) on every dataset
for DATA in clotho audiocaps audioset; do
  N=${MAX_N[$DATA]}
  python src/eval_mia.py -m model=audio_flamingo3 \
      method=min_k method.k_pct=10,30,50 \
      data_member=$DATA data_non_member=$DATA \
      batch_size=16 max_member_samples=$N max_non_member_samples=$N \
      tags="[full,sweep,$DATA,min_k]"
  python src/eval_mia.py -m model=audio_flamingo3 \
      method=min_k_pp method.k_pct=10,30,50 \
      data_member=$DATA data_non_member=$DATA \
      batch_size=16 max_member_samples=$N max_non_member_samples=$N \
      tags="[full,sweep,$DATA,min_k_pp]"
  python src/eval_mia.py -m model=audio_flamingo3 \
      method=vl_mia_entropy method.top_pct=10,30,50 \
      data_member=$DATA data_non_member=$DATA \
      batch_size=16 max_member_samples=$N max_non_member_samples=$N \
      tags="[full,sweep,$DATA,vl_mia]"
done

# AudioSet label-template robustness (Phase C)
declare -a TEMPLATES=( '{labels}' 'Sounds of {labels}.' 'This audio contains: {labels}.' )
for i in "${!TEMPLATES[@]}"; do
  TPL="${TEMPLATES[$i]}"
  python src/eval_mia.py model=audio_flamingo3 method=min_k_pp \
      data_member=audioset data_non_member=audioset \
      "data_member.label_template=\"$TPL\"" \
      "data_non_member.label_template=\"$TPL\"" \
      batch_size=16 max_member_samples=${MAX_N[audioset]} max_non_member_samples=${MAX_N[audioset]} \
      tags="[full,audioset_C,template_$i]"
done

# Contamination (CoDeC): both modes (AF3, Clotho)
for MODE in full no_audio; do
  python src/contamination.py \
      model=audio_flamingo3 method=codec method.mode=$MODE \
      context_pool_size=50 \
      max_member_samples=${MAX_N[clotho]} max_non_member_samples=${MAX_N[clotho]} \
      batch_size=1 tags="[full,codec,$MODE]"
done


echo "[run_full] all full runs completed OK"
