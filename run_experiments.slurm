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

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: No Python interpreter found. Expected .venv/bin/python, python3, or python." >&2
    exit 1
  fi
elif [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN is set but not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" run_experiments.py --experiment-index "$SLURM_ARRAY_TASK_ID"
