#!/bin/bash
#SBATCH --job-name=nlp_experiments
#SBATCH --array=1-22%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/experiments_%A_%a.out
#SBATCH --error=logs/experiments_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi

"$PYTHON_BIN" run_experiments.py --experiment-index "$SLURM_ARRAY_TASK_ID"
