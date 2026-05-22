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
# Required by configs/paths/default.yaml (${oc.env:PROJECT_ROOT})
export PROJECT_ROOT="$PROJECT_DIR"

# W&B (optional): export WANDB_API_KEY before running, or `wandb login` in the env.
# Runs appear under project "audio-benchmark" (configs/logger/wandb.yaml).
# Local logs: $PROJECT_ROOT/logs/<task_name>/runs/<timestamp>/

# SMOKE="max_member_samples=2 max_non_member_samples=2 batch_size=2"

# # MIA eval: every method × every dataset (AF3)
# for DATA in clotho audiocaps audioset; do
#   for METHOD in yeom_perplexity min_k min_k_pp vl_mia_entropy; do
#     python src/eval_mia.py \
#         model=audio_flamingo3 method=$METHOD \
#         data_member=$DATA data_non_member=$DATA \
#         $SMOKE tags="[smoke,mia,$DATA,$METHOD]"
#   done
# done

# python src/contamination.py \
#     model=audio_flamingo3 method=codec method.mode=no_audio \
#     context_pool_size=4 \
#     max_member_samples=2 max_non_member_samples=2 batch_size=1 \
#     tags="[smoke,codec,no_audio]"

# MM-DETECT (2 samples/split; needs back-translated JSONL — see run_mm_detect_smoke.sh)
python src/eval_mm_detect.py \
    model=audio_flamingo3 method=mm_detect \
    data_member=mm_detect_clotho data_non_member=mm_detect_clotho \
    max_member_samples=2 max_non_member_samples=2 \
    tags="[smoke,mm_detect]"

# --- OPTIONAL: AF2 wrapper sanity. Run ONLY under the AF2 env
# --- (py310_af2_env); will fail under py311_env.
# python src/eval_mia.py model=audio_flamingo2 method=min_k_pp \
#     data_member=clotho data_non_member=clotho $SMOKE \
#     tags="[smoke,mia,af2]"
# python src/contamination.py model=audio_flamingo2 method=codec method.mode=full \
#     context_pool_size=4 max_member_samples=2 max_non_member_samples=2 batch_size=1 \
#     tags="[smoke,codec,af2]"

echo "[run_smoke] all smoke runs completed OK"
