#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_full.sh — full experiment battery.
# ASSUMES datasets are already on disk under ./data
# (run download_all_datasets.sh once beforehand).
#
#   AF3 (py311_env):      all MIA methods x all datasets, k%/top% sweeps,
#                         AudioSet label-template robustness, and
#                         CoDeC no_audio on Clotho.
#   AF2 (py310_af2_env):  CoDeC full + no_audio on Clotho.
#
# Why the split: AF3's processor rejects >1 audio per turn, so full-mode
# contamination (target audio + context audio) only runs on AF2.
# ============================================================

module load Miniconda3
eval "$(conda shell.bash hook)"
module load CUDA/12.8

USERNAME="$(whoami)"
export SCRATCH_BASE="/net/tscratch/people/${USERNAME}"
export ENV_PREFIX="$SCRATCH_BASE/conda/py311_env"
AF2_ENV_PREFIX="${AF2_ENV_PREFIX:-$SCRATCH_BASE/conda/py310_af2_env}"

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
# Once model weights are warm in the cache, uncomment to skip all hub
# re-validation (zero network at run time):
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export TMPDIR="$SCRATCH_BASE/.tmp"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$MAMBA_ROOT_PREFIX" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" "$TMPDIR"

conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ENV_PREFIX/bin:${PATH}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$ENV_PREFIX}"

PROJECT_DIR="$SCRATCH_BASE/AudioBenchmarkContaiminationStudy"
cd "$PROJECT_DIR"

declare -A MAX_N=( [clotho]=100000 [audiocaps]=6630 [audioset]=10000 )

# ── AF3: full MIA battery (every method x every dataset) ─────────────────────
for DATA in clotho audiocaps audioset; do
  N=${MAX_N[$DATA]}
  for METHOD in yeom_perplexity min_k min_k_pp vl_mia_entropy; do
    echo "=== [AF3] MIA: $DATA / $METHOD (N=$N) ==="
    python src/eval_mia.py \
        model=audio_flamingo3 method=$METHOD \
        data_member=$DATA data_non_member=$DATA \
        batch_size=16 max_member_samples=$N max_non_member_samples=$N \
        tags="[full,mia,$DATA,$METHOD]"
  done
done

# ── AF3: k%/top% sweeps (multirun) on every dataset ──────────────────────────
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

# ── AF3: AudioSet label-template robustness (Phase C) ────────────────────────
declare -a TEMPLATES=( '{labels}' 'Sounds of {labels}.' 'This audio contains: {labels}.' )
for i in "${!TEMPLATES[@]}"; do
  TPL="${TEMPLATES[$i]}"
  python src/eval_mia.py model=audio_flamingo3 method=min_k_pp \
      data_member=audioset data_non_member=audioset \
      "data_member.label_template=\"$TPL\"" \
      "data_non_member.label_template=\"$TPL\"" \
      batch_size=16 \
      max_member_samples=${MAX_N[audioset]} max_non_member_samples=${MAX_N[audioset]} \
      tags="[full,audioset_C,template_$i]"
done

# ── AF3: CoDeC contamination — no_audio ONLY (full needs AF2) ────────────────
echo "=== [AF3] CoDeC no_audio (Clotho) ==="
python src/contamination.py \
    model=audio_flamingo3 method=codec method.mode=no_audio \
    context_pool_size=50 \
    max_member_samples=${MAX_N[clotho]} max_non_member_samples=${MAX_N[clotho]} \
    batch_size=1 tags="[full,codec,af3,no_audio]"

# ── AF2: CoDeC contamination — full + no_audio (separate env!) ───────────────
# Switch envs inside a subshell so the change does not leak back to AF3.
if [[ -d "$AF2_ENV_PREFIX" ]]; then
  echo "=== [AF2] CoDeC full + no_audio (Clotho) — env: $AF2_ENV_PREFIX ==="
  (
    conda deactivate || true
    conda activate "$AF2_ENV_PREFIX"
    export LD_LIBRARY_PATH="$AF2_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$AF2_ENV_PREFIX/bin:${PATH}"
    for MODE in full no_audio; do
      python -m src.contamination \
          model=audio_flamingo2 method=codec method.mode=$MODE \
          context_pool_size=50 \
          max_member_samples=${MAX_N[clotho]} max_non_member_samples=${MAX_N[clotho]} \
          batch_size=1 tags="[full,codec,af2,$MODE]"
    done
  )
else
  echo "[run_full] WARNING: AF2 env not found at $AF2_ENV_PREFIX — skipping AF2 contamination."
  echo "[run_full]          Build it with: bash setup_env_af2.sh"
fi

echo "[run_full] all full runs completed OK"
