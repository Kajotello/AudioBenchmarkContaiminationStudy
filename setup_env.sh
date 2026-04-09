#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Robust HPC environment setup for AudioBenchmarkContaiminationStudy
# - derives username with whoami
# - uses /net/tscratch/people/{username}/conda/py311_env
# - redirects caches to scratch
# - exports runtime variables needed in both setup and jobs
# ============================================================

log() {
  echo "[setup_env] $*"
}

die() {
  echo "[setup_env][ERROR] $*" >&2
  exit 1
}

# ---------- identify user ----------
USERNAME="$(whoami)"
[[ -n "$USERNAME" ]] || die "whoami returned empty username"

# ---------- base paths ----------
export SCRATCH_BASE="${SCRATCH_BASE:-/net/tscratch/people/${USERNAME}}"
export ENV_PREFIX="${ENV_PREFIX:-${SCRATCH_BASE}/conda/py311_env}"
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"

# ---------- cache / package locations ----------
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$SCRATCH_BASE/micromamba}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_BASE/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH_BASE/.cache/conda/pkgs}"

export HF_HOME="${HF_HOME:-$SCRATCH_BASE/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# Optional but often useful on shared systems
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/.tmp}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

log "Resolved username       : $USERNAME"
log "SCRATCH_BASE            : $SCRATCH_BASE"
log "ENV_PREFIX              : $ENV_PREFIX"
log "PROJECT_DIR             : $PROJECT_DIR"
log "XDG_CACHE_HOME          : $XDG_CACHE_HOME"
log "PIP_CACHE_DIR           : $PIP_CACHE_DIR"
log "CONDA_PKGS_DIRS         : $CONDA_PKGS_DIRS"
log "HF_HOME                 : $HF_HOME"

# ---------- create required directories ----------
mkdir -p \
  "$(dirname "$ENV_PREFIX")" \
  "$MAMBA_ROOT_PREFIX" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$CONDA_PKGS_DIRS" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$TMPDIR"

# ---------- load conda ----------
if command -v module >/dev/null 2>&1; then
  log "Loading Miniconda3 module"
  module load Miniconda3 || die "Failed to load Miniconda3 module"
fi

command -v conda >/dev/null 2>&1 || die "conda not found"
eval "$(conda shell.bash hook)"

# ---------- accept Anaconda TOS if needed ----------
log "Accepting conda TOS if required"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ---------- create env if missing ----------
if [[ ! -d "$ENV_PREFIX" ]]; then
  log "Creating conda environment at $ENV_PREFIX"
  conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
else
  log "Conda environment already exists at $ENV_PREFIX"
fi

# ---------- activate env ----------
log "Activating environment"
conda activate "$ENV_PREFIX"

# ---------- runtime library exports ----------
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ENV_PREFIX/bin:${PATH}"

# This is sometimes needed for builds or native extensions
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$ENV_PREFIX}"

# ---------- install conda-side deps ----------
log "Installing ffmpeg and libstdcxx-ng"
conda install -p "$ENV_PREFIX" ffmpeg libstdcxx-ng -c conda-forge -y

# ---------- install Python deps ----------
log "Upgrading pip"
python -m pip install --upgrade pip

log "Installing core packages"
python -m pip install --upgrade transformers accelerate
python -m pip install soundfile librosa peft huggingface_hub
python -m pip install hydra-core hydra-colorlog lightning
python -m pip install rootutils datasets

log "Installing torchcodec for CUDA 12.8"
python -m pip install torchcodec --index-url=https://download.pytorch.org/whl/cu128

# ---------- install project requirements ----------
if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
  log "Installing requirements.txt from $PROJECT_DIR"
  cd "$PROJECT_DIR"
  python -m pip install -r requirements.txt
else
  log "No requirements.txt found in $PROJECT_DIR, skipping"
fi

# ---------- diagnostics ----------
echo "--------------------------------------------------"
echo "USERNAME                : $USERNAME"
echo "SCRATCH_BASE            : $SCRATCH_BASE"
echo "ENV_PREFIX              : $ENV_PREFIX"
echo "PROJECT_DIR             : $PROJECT_DIR"
echo "PYTHON                  : $(python --version 2>&1)"
echo "PIP                     : $(python -m pip --version)"
echo "CONDA_PREFIX            : ${CONDA_PREFIX:-unset}"
echo "LD_LIBRARY_PATH         : $LD_LIBRARY_PATH"
echo "TMPDIR                  : $TMPDIR"
echo "HF_HOME                 : $HF_HOME"
echo "HF_DATASETS_CACHE       : $HF_DATASETS_CACHE"
echo "--------------------------------------------------"

log "Checking imports"
python - <<'PY'
import importlib

packages = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "librosa",
    "soundfile",
    "lightning",
    "rootutils",
]

failed = False
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except Exception as e:
        failed = True
        print(f"[FAIL] {pkg}: {e}")

if failed:
    raise SystemExit(1)
PY

log "ffmpeg check"
ffmpeg -version | head -n 1 || true

log "Setup finished successfully"
