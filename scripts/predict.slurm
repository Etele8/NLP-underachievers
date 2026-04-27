#!/bin/bash
#SBATCH --job-name=real_ner_predict
#SBATCH --output=outputs/slurm_logs/predict_%j.out
#SBATCH --error=outputs/slurm_logs/predict_%j.err
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/NLP-underachievers}"
cd "$PROJECT_DIR"

CONFIG=${CONFIG:-configs/mbert.yaml}
CHECKPOINT=${CHECKPOINT:?Please set CHECKPOINT=/path/to/best.pt}
SPLIT=${SPLIT:-test}

mkdir -p outputs/slurm_logs

echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(command -v python || true)"
echo "Venv Python: $PROJECT_DIR/.venv/bin/python"

.venv/bin/python - <<'PY'
import torch
import sys
print("sys.executable:", sys.executable)
print("torch_version:", torch.__version__)
print("torch_cuda_version:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY

.venv/bin/python -m src.predict --config "$CONFIG" --checkpoint "$CHECKPOINT" --split "$SPLIT"
