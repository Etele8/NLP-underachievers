#!/bin/bash
#SBATCH --job-name=real_ner_predict
#SBATCH --output=outputs/slurm_logs/predict_%j.out
#SBATCH --error=outputs/slurm_logs/predict_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail

CONFIG=${CONFIG:-configs/mbert.yaml}
CHECKPOINT=${CHECKPOINT:?Please set CHECKPOINT=/path/to/best.pt}
SPLIT=${SPLIT:-test}

mkdir -p outputs/slurm_logs

echo "Host: $(hostname)"
echo "Python: $(command -v python || true)"

uv run python - <<'PY'
import torch
import sys
print("sys.executable:", sys.executable)
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY

uv run python -m src.predict --config "$CONFIG" --checkpoint "$CHECKPOINT" --split "$SPLIT"
