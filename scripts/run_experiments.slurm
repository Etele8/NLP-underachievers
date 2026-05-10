#!/bin/bash
#SBATCH --job-name=nlp_experiments
#SBATCH --partition=acltr
#SBATCH --array=1-24%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm_logs/experiments_%A_%a.out
#SBATCH --error=outputs/slurm_logs/experiments_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/NLP-underachievers}"
cd "$PROJECT_DIR"
mkdir -p outputs/slurm_logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"
echo "Host: $(hostname)"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

TORCH_SITE_PACKAGES="$(echo "$PROJECT_DIR"/.venv/lib/python*/site-packages)"
TORCH_LIB_DIR="$TORCH_SITE_PACKAGES/torch/lib"
NVIDIA_LIB_DIRS="$(find "$TORCH_SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"

echo "Python: $PYTHON_BIN"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
"$PYTHON_BIN" run_experiments.py --experiment-index "$SLURM_ARRAY_TASK_ID"
