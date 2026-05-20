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

# ── AudioCaps Phase A: FULL members (balanced max 6630/6630, non-member=val+test) ──
for METHOD in yeom_perplexity min_k min_k_pp vl_mia_entropy; do
  python src/eval.py method=$METHOD \
      data_member=audiocaps data_non_member=audiocaps \
      data_non_member.split=validation+test \
      batch_size=16 max_member_samples=6630 max_non_member_samples=6630 \
      tags="[audiocaps_A_full,$METHOD]"
done

# ── AudioCaps Phase B: coarse sweep (3 points), high N (3000/3000) ──────────
# k_pct=20 already comes from Phase A at full N, so we sweep 10/30/50.
python src/eval.py -m method=min_k     method.k_pct=10,30,50 \
    data_member=audiocaps data_non_member=audiocaps \
    data_non_member.split=validation+test \
    batch_size=16 max_member_samples=3000 max_non_member_samples=3000 \
    tags='[audiocaps_B,min_k,sweep]'
python src/eval.py -m method=min_k_pp  method.k_pct=10,30,50 \
    data_member=audiocaps data_non_member=audiocaps \
    data_non_member.split=validation+test \
    batch_size=16 max_member_samples=3000 max_non_member_samples=3000 \
    tags='[audiocaps_B,min_k_pp,sweep]'
python src/eval.py -m method=vl_mia_entropy method.top_pct=10,30,50 \
    data_member=audiocaps data_non_member=audiocaps \
    data_non_member.split=validation+test \
    batch_size=16 max_member_samples=3000 max_non_member_samples=3000 \
    tags='[audiocaps_B,vl_mia,sweep]'
