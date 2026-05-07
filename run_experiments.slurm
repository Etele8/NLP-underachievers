#!/bin/bash
#SBATCH --job-name=nlp_experiments
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/experiments_%j.out
#SBATCH --error=logs/experiments_%j.err

set -euo pipefail

mkdir -p logs

python run_experiments.py
