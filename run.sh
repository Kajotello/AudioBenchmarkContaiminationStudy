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

python src/eval.py
