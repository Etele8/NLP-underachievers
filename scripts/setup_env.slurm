#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd ~/projects/nlp-underachievers
export PATH="$HOME/.local/bin:$PATH"

uv sync

source .venv/bin/activate