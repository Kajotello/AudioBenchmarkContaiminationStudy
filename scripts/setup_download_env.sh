#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Creates a lightweight conda env for scripts/download_dataset.py
#
# Usage: bash setup_download_env.sh [-b <base_dir>] [-p <python_version>] [--recreate]
#
#   -b DIR   Base directory for HF caches and the conda env
#            (default: ./data, relative to repo root)
#   -p VER   Python version (default: 3.11)
#   --recreate  Delete existing env and rebuild from scratch
# ============================================================

log() { echo "[setup_download_env] $*"; }
die() { echo "[setup_download_env][ERROR] $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
BASE_DIR="${REPO_ROOT}/data"
PYTHON_VERSION="3.11"
RECREATE=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b) BASE_DIR="$(realpath -m "$2")"; shift 2 ;;
        -p) PYTHON_VERSION="$2"; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

export ENV_PREFIX="${BASE_DIR}/envs/download_env"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

log "BASE_DIR   : $BASE_DIR"
log "ENV_PREFIX : $ENV_PREFIX"
log "PYTHON     : $PYTHON_VERSION"

mkdir -p "$(dirname "$ENV_PREFIX")" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE"

if command -v module >/dev/null 2>&1; then
    log "Loading Miniconda3 module"
    module load Miniconda3 || die "Failed to load Miniconda3 module"
fi
command -v conda >/dev/null 2>&1 || die "conda not found"
eval "$(conda shell.bash hook)"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

if [[ "$RECREATE" == "1" && -d "$ENV_PREFIX" ]]; then
    log "RECREATE=1 — removing existing env"
    rm -rf "$ENV_PREFIX"
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
    log "Creating conda env"
    conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
else
    log "Env exists; installing/updating in place. Use RECREATE=1 to wipe."
fi

conda activate "$ENV_PREFIX"

log "Installing ffmpeg (needed by soundfile/librosa)"
conda install -p "$ENV_PREFIX" ffmpeg -c conda-forge -y

log "Installing pip requirements"
python -m pip install --upgrade pip
python -m pip install -r "$SCRIPT_DIR/requirements_download.txt"

log "Verifying imports"
python - <<'PY'
import importlib, sys
required = ["datasets", "soundfile", "librosa", "tqdm", "huggingface_hub", "numpy", "torchcodec"]
failed = []
for pkg in required:
    try:
        m = importlib.import_module(pkg)
        print(f"[OK]   {pkg:<20} {getattr(m, '__version__', '?')}")
    except Exception as e:
        failed.append(pkg)
        print(f"[FAIL] {pkg:<20} {e}")
if failed:
    print(f"\n{len(failed)} package(s) failed: {failed}", file=sys.stderr)
    sys.exit(1)
PY

log "Setup finished. Run with:"
log "  conda run -p $ENV_PREFIX python scripts/download_dataset.py -b $BASE_DIR"
