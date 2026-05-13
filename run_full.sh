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

# ── Phase A: main comparison (≈ 2 h) ─────────────────────────────────────────
for METHOD in yeom_perplexity min_k min_k_pp vl_mia_entropy; do
  python src/eval.py method=$METHOD batch_size=4 \
      max_member_samples=1000 max_non_member_samples=1000 \
      tags="[phase_A,$METHOD,default]"
done

python src/eval.py -m method=min_k     method.k_pct=10,20,30,40,50 batch_size=4 \
    max_member_samples=500 max_non_member_samples=500 \
    tags='[phase_B,min_k,sweep]'

python src/eval.py -m method=min_k_pp  method.k_pct=10,20,30,40,50 batch_size=4 \
    max_member_samples=500 max_non_member_samples=500 \
    tags='[phase_B,min_k_pp,sweep]'

python src/eval.py -m method=vl_mia_entropy method.top_pct=10,20,30,40,50 batch_size=4 \
    max_member_samples=500 max_non_member_samples=500 \
    tags='[phase_B,vl_mia,sweep]'

# ── Phase C: caption robustness for one strong method (≈ 1.5 h) ──────────────
# Pick the strongest Phase A method here; using min_k_pp as a placeholder.
for CAP in 2 3 4 5; do
  python src/eval.py method=min_k_pp batch_size=4 \
      data_member.caption_index=$CAP data_non_member.caption_index=$CAP \
      max_member_samples=500 max_non_member_samples=500 \
      tags="[phase_C,min_k_pp,caption_$CAP]"
done
