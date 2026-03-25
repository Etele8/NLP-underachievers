#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd ~/projects/NLP-underachievers
export PATH="$HOME/.local/bin:$PATH"

uv sync

source .venv/bin/activate

cd ~/projects/NLP-underachievers
source .venv/bin/activate

uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0

python - <<'EOF'
import torch, sys
print("python:", sys.executable)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch file:", torch.__file__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
EOF
