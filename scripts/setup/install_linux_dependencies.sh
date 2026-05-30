#!/usr/bin/env bash
set -euo pipefail

# Linux dependency bootstrap for the computational pathology research repo.
# Tested target: Ubuntu/Debian-style systems. For CUDA, install the PyTorch
# wheel matching your NVIDIA driver/CUDA runtime from https://pytorch.org first
# or edit the TORCH_INDEX_URL below.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_TORCH_CPU="${INSTALL_TORCH_CPU:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find $PYTHON_BIN. Install Python 3.10+ first." >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "Installing system packages with apt-get..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    git \
    curl \
    pkg-config \
    python3-dev \
    python3-venv \
    libopenslide0 \
    openslide-tools \
    libgl1 \
    libglib2.0-0 \
    redis-server
else
  echo "apt-get not found. Install equivalent packages manually:"
  echo "  Python headers/venv, build-essential toolchain, OpenSlide, libgl, glib, Redis."
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ "$INSTALL_TORCH_CPU" = "1" ]; then
  echo "Installing CPU PyTorch wheels from $TORCH_INDEX_URL..."
  python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

# Install runtime + development + federated + ML extras from pyproject.toml.
# Quotes are required so shells do not interpret the brackets.
python -m pip install -e ".[dev,federated,ml]"

# A few legacy/integration tests import optional platform dependencies that are
# not part of the core research runner path but are useful for full collection.
python -m pip install \
  sqlalchemy \
  hypothesis \
  pytest-cov \
  redis \
  requests \
  httpx

echo ""
echo "Dependency installation complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Recommended quick checks:"
echo "  python -m pytest tests/test_pathology_fl_privacy_regressions.py -q -o addopts=''"
echo "  python -m pytest tests/test_secure_aggregation_contract.py -q -o addopts=''"
echo "  python scripts/experiments/run_fair_weights_h_panda_feature_stress.py --help"
