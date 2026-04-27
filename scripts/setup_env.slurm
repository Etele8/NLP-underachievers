#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd ~/projects/NLP-underachievers
export PATH="$HOME/.local/bin:$PATH"

uv sync --no-install-package torch

source .venv/bin/activate

cd ~/projects/NLP-underachievers
source .venv/bin/activate

uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0

TORCH_SITE_PACKAGES="$(echo "$PWD"/.venv/lib/python*/site-packages)"
TORCH_LIB_DIR="$TORCH_SITE_PACKAGES/torch/lib"
NVIDIA_LIB_DIRS="$(find "$TORCH_SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"

python - <<'EOF'
import os
import torch, sys
print("python:", sys.executable)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch file:", torch.__file__)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
EOF
