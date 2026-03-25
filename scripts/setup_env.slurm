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

uv pip install --index-url https://download.pytorch.org/whl/cu126 torch

python -c "import torch; print(torch.__version__)"