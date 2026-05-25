#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Creates the conda environment for src/audio_flamingo_2
# (NVIDIA/audio-flamingo, branch audio_flamingo_2).
#
# Usage: setup_env_af2.sh [-b BASE_DIR] [-h]
#   -b BASE_DIR     arbitrary base directory for the conda env and caches
#                   (overrides SCRATCH_BASE; takes precedence over env var)
#                   default: /net/tscratch/people/$USER
#
# Override remaining settings via env vars:
#   SCRATCH_BASE    base dir when -b is not given (default: /net/tscratch/people/$USER)
#   ENV_PREFIX      conda env prefix  (default: $SCRATCH_BASE/conda/py310_af2_env)
#   PROJECT_DIR     repo root with src/audio_flamingo_2/ (default: $PWD)
#   PYTHON_VERSION  python version (default: 3.10)
#   RECREATE        if "1", delete existing env first (default: 0)
#
# Python 3.10 chosen: torch==2.0.1 officially supports 3.8-3.11;
# 3.10 gives the widest wheel availability for all pinned packages.
# ============================================================

log() { echo "[setup_env_af2] $*"; }
die() { echo "[setup_env_af2][ERROR] $*" >&2; exit 1; }

# ── CLI args ──────────────────────────────────────────────────────────────────
BASE_DIR_ARG=""
while getopts ":b:h" opt; do
  case $opt in
    b) BASE_DIR_ARG="$OPTARG" ;;
    h)
      sed -n '/^# ====/,/^# ====/p' "$0"
      exit 0
      ;;
    :) die "Option -$OPTARG requires an argument" ;;
    \?) die "Unknown option: -$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))

# ── identity / paths ─────────────────────────────────────────────────────────
USERNAME="$(whoami)"
[[ -n "$USERNAME" ]] || die "whoami returned empty username"

if [[ -n "$BASE_DIR_ARG" ]]; then
  export SCRATCH_BASE="$BASE_DIR_ARG"
else
  export SCRATCH_BASE="${SCRATCH_BASE:-/net/tscratch/people/${USERNAME}}"
fi
export ENV_PREFIX="${ENV_PREFIX:-${SCRATCH_BASE}/conda/py310_af2_env}"
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

AF2_DIR="$PROJECT_DIR/src/audio_flamingo_2"
REQUIREMENTS_FILE="$AF2_DIR/requirements.txt"

# ── cache / scratch redirects (don't hammer $HOME) ───────────────────────────
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_BASE/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH_BASE/.cache/conda/pkgs}"
export HF_HOME="${HF_HOME:-$SCRATCH_BASE/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/.tmp}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

log "USERNAME    : $USERNAME"
log "SCRATCH_BASE: $SCRATCH_BASE"
log "ENV_PREFIX  : $ENV_PREFIX"
log "PROJECT_DIR : $PROJECT_DIR"
log "AF2_DIR     : $AF2_DIR"
log "PYTHON      : $PYTHON_VERSION"

[[ -d "$AF2_DIR" ]] || die "src/audio_flamingo_2 not found — run from repo root or set PROJECT_DIR"
[[ -f "$REQUIREMENTS_FILE" ]] || die "requirements.txt not found at $REQUIREMENTS_FILE"

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
  log "Creating conda env at $ENV_PREFIX"
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
# ffmpeg   : librosa / pydub / soundfile audio decoding
# libstdcxx-ng : recent libstdc++ matching PyTorch 2.0.1 ABI
log "Installing ffmpeg and libstdcxx-ng from conda-forge"
conda install -p "$ENV_PREFIX" ffmpeg libstdcxx-ng -c conda-forge -y

# ── pip install ───────────────────────────────────────────────────────────────
log "Upgrading pip"
python -m pip install --upgrade pip

# torch==2.0.1 was released against CUDA 11.7/11.8; use cu118 index.
# einops_exts is available on PyPI under the same name.
log "Installing requirements (torch==2.0.1+cu118 and friends; may take several minutes)"
python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r "$REQUIREMENTS_FILE"

# ── project root extras (unversioned, not in af2 requirements) ───────────────
log "Installing project root extras"
python -m pip install rootutils pre-commit rich pytest datasets "pyarrow>=12.0.0,<15.0.0" "fsspec<2024.3.1"

# ── sanity check ─────────────────────────────────────────────────────────────
echo "--------------------------------------------------"
echo "USERNAME    : $USERNAME"
echo "ENV_PREFIX  : $ENV_PREFIX"
echo "PYTHON      : $(python --version 2>&1)"
echo "PIP         : $(python -m pip --version)"
echo "--------------------------------------------------"

log "Verifying critical imports + torch CUDA build"
python - <<'PY'
import importlib, sys

required = [
    "torch", "torchaudio", "torchvision",
    "transformers", "tokenizers",
    "librosa", "soundfile",
    "einops", "huggingface_hub",
    "laion_clap", "nnAudio",
    "scipy", "sklearn",
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
if torch.version.cuda and not torch.version.cuda.startswith("11."):
    print(f"WARNING: expected torch built for CUDA 11.x, got {torch.version.cuda}",
          file=sys.stderr)

if failed:
    print(f"\n{len(failed)} package(s) failed to import: {failed}", file=sys.stderr)
    sys.exit(1)
PY

log "ffmpeg check"
ffmpeg -version | head -n 1 || true

log "Setup finished. Activate with: conda activate $ENV_PREFIX"
