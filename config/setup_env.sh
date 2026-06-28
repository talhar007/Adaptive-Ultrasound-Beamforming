#!/bin/bash
# One-time environment setup on the Makalu cluster.
# Run this ONCE from a login node:
#     bash setup_env.sh
#
# Creates ~/able_env  (home directory is NFS-shared across all compute
# nodes, so the environment is visible to batch jobs automatically).

set -euo pipefail

PYTHON=/usr/bin/python3.11
ENV_DIR="$HOME/able_env"

echo "Creating virtual environment at $ENV_DIR ..."
$PYTHON -m venv "$ENV_DIR"

source "$ENV_DIR/bin/activate"
echo "Python: $(which python)  $(python --version)"

echo "Installing PyTorch (CUDA 12.1 wheel) ..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib

echo ""
echo "All done. Test with:"
echo "  source ~/able_env/bin/activate"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
