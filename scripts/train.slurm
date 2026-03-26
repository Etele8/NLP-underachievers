#!/bin/bash
#SBATCH --job-name=nlp_train
#SBATCH --output=logs/nlp_train_%j.out
#SBATCH --error=logs/nlp_train_%j.err
#SBATCH --partition=acltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR="$HOME/projects/NLP-underachievers"
cd "$PROJECT_DIR"

mkdir -p logs outputs checkpoints

RUN_NAME="${RUN_NAME:-mbert_baseline}"
OUT_DIR="outputs/$RUN_NAME"
CKPT_DIR="checkpoints/$RUN_NAME"

mkdir -p "$OUT_DIR" "$CKPT_DIR"

echo "===== JOB INFO ====="
echo "RUN_NAME: $RUN_NAME"
echo "HOSTNAME: $(hostname)"
echo

nvidia-smi
echo

# HuggingFace cache (important for HPC)
export HF_HOME="$PROJECT_DIR/.cache/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

echo "===== TRAIN START ====="

.venv/bin/python src/train.py \
  --model_name google-bert/bert-base-multilingual-cased \
  --train_file data-baseline/en_ewt-ud-train.iob2 \
  --dev_file data-baseline/en_ewt-ud-dev.iob2 \
  --output_dir "$OUT_DIR" \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --max_length 128 \
  --seed 42 \
  --checkpoint_dir "$CKPT_DIR" \
  --resume auto

echo "===== TRAIN END ====="
