#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Recreates the pinned working environment from requirements.txt.
#
# Override via env vars if needed:
#   DATA_DIR        base data dir  (default: $PROJECT_DIR/data)
#   ENV_PREFIX      conda env prefix  (default: $DATA_DIR/conda/py311_env)
#   PROJECT_DIR     where requirements.txt lives (default: $PWD)
#   PYTHON_VERSION  python version (default: 3.11)
#   RECREATE        if "1", delete existing env first (default: 0)
# ============================================================

log() { echo "[setup_env] $*"; }
die() { echo "[setup_env][ERROR] $*" >&2; exit 1; }

# ── paths ─────────────────────────────────────────────────────────────────────
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
export ENV_PREFIX="${ENV_PREFIX:-${DATA_DIR}/conda/py311_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# ── cache redirects ───────────────────────────────────────────────────────────
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DIR/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_DIR/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_DIR/.cache/conda/pkgs}"
export HF_HOME="${HF_HOME:-$DATA_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$DATA_DIR/.tmp}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

log "PROJECT_DIR : $PROJECT_DIR"
log "DATA_DIR    : $DATA_DIR"
log "ENV_PREFIX  : $ENV_PREFIX"
log "PYTHON      : $PYTHON_VERSION"

mkdir -p \
  "$(dirname "$ENV_PREFIX")" \
  "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" \
  "$TMPDIR"

# ── conda bootstrap ──────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  log "Loading Miniconda3 module"
  module load Miniconda3 || die "Failed to load Miniconda3 module"
fi
command -v conda >/dev/null 2>&1 || die "conda not found"
eval "$(conda shell.bash hook)"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

# ── optional clean rebuild ───────────────────────────────────────────────────
if [[ "${RECREATE:-0}" == "1" && -d "$ENV_PREFIX" ]]; then
  log "RECREATE=1 — removing existing env at $ENV_PREFIX"
  rm -rf "$ENV_PREFIX"
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
  log "Creating conda env"
  conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
else
  log "Env exists; will install/update in place. Use RECREATE=1 to wipe."
fi

log "Activating env"
conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ENV_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$ENV_PREFIX}"

# ── conda-side native deps ───────────────────────────────────────────────────
# ffmpeg     : torchcodec/librosa audio decoding
# libstdcxx-ng : recent libstdc++ matching PyTorch wheels' ABI
log "Installing ffmpeg and libstdcxx-ng from conda-forge"
conda install -p "$ENV_PREFIX" ffmpeg libstdcxx-ng -c conda-forge -y

# ── frozen-requirements install ──────────────────────────────────────────────
[[ -f "$PROJECT_DIR/requirements.txt" ]] \
  || die "requirements.txt missing in $PROJECT_DIR"

# The frozen file pins `packaging` to a conda-forge build worker's local path
# (file:///home/conda/feedstock_root/...) which doesn't exist on this machine.
# Rewrite it to a regular PyPI lookup before install.
REQUIREMENTS_CLEAN="$TMPDIR/requirements_clean.txt"
sed -E 's|^packaging[[:space:]]*@[[:space:]]*file://[^[:space:]]*|packaging|' \
    "$PROJECT_DIR/requirements.txt" > "$REQUIREMENTS_CLEAN"

log "Upgrading pip"
python -m pip install --upgrade pip

# Three indexes:
#   PyPI (default)                  — general packages
#   download.pytorch.org/whl/cu128  — torch*==2.11.0+cu128, triton, cuda-bindings
#   pypi.nvidia.com                 — cuda-toolkit==12.8.1, nvidia-*-cu12 / nvidia-*
log "Installing pinned requirements (will take several minutes; ~3 GB of wheels)"
python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.nvidia.com \
    -r "$REQUIREMENTS_CLEAN"

# ── sanity check ─────────────────────────────────────────────────────────────
echo "--------------------------------------------------"
echo "PROJECT_DIR : $PROJECT_DIR"
echo "DATA_DIR    : $DATA_DIR"
echo "ENV_PREFIX  : $ENV_PREFIX"
echo "PYTHON      : $(python --version 2>&1)"
echo "PIP         : $(python -m pip --version)"
echo "--------------------------------------------------"

log "Verifying critical imports + torch CUDA build"
python - <<'PY'
import importlib, sys
required = [
    "torch", "torchvision", "torchaudio", "torchcodec",
    "transformers", "accelerate", "datasets",
    "librosa", "soundfile",
    "lightning", "hydra", "rootutils", "peft", "huggingface_hub",
]
failed = []
for pkg in required:
    try:
        m = importlib.import_module(pkg)
        version = getattr(m, "__version__", "?")
        print(f"[OK]   {pkg:<20} {version}")
    except Exception as e:
        failed.append(pkg)
        print(f"[FAIL] {pkg:<20} {e}")

import torch
print()
print(f"torch.__version__  : {torch.__version__}")
print(f"torch.version.cuda : {torch.version.cuda}")
if not torch.version.cuda.startswith("12.8"):
    print(f"WARNING: expected torch built for CUDA 12.8, got {torch.version.cuda}",
          file=sys.stderr)
    sys.exit(1)
if failed:
    print(f"\n{len(failed)} package(s) failed to import: {failed}", file=sys.stderr)
    sys.exit(1)
PY

log "ffmpeg check"
ffmpeg -version | head -n 1 || true

log "Setup finished. Activate with: conda activate $ENV_PREFIX"
