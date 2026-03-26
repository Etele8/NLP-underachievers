#!/bin/bash
#SBATCH --job-name=nlp_predict
#SBATCH --output=logs/nlp_predict_%j.out
#SBATCH --error=logs/nlp_predict_%j.err
#SBATCH --partition=acltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_DIR="$HOME/projects/NLP-underachievers"
cd "$PROJECT_DIR"

mkdir -p logs outputs

RUN_NAME="${RUN_NAME:-mbert_baseline}"
CHECKPOINT="${CHECKPOINT:-outputs/$RUN_NAME/best_google-bert-bert-base-multilingual-cased}"
TEST_FILE="${TEST_FILE:-data-baseline/en_ewt-ud-test-masked.iob2}"
PRED_OUT_DIR="${PRED_OUT_DIR:-outputs/$RUN_NAME/test_predictions}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MODEL_NAME="${MODEL_NAME:-}"

mkdir -p "$PRED_OUT_DIR"

echo "===== JOB INFO ====="
echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
echo "RUN_NAME: $RUN_NAME"
echo "CHECKPOINT: $CHECKPOINT"
echo "TEST_FILE: $TEST_FILE"
echo "PRED_OUT_DIR: $PRED_OUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_LENGTH: $MAX_LENGTH"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

echo "===== NVIDIA-SMI ====="
nvidia-smi
echo

echo "===== PYTORCH CHECK ====="
.venv/bin/python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device[0] =", torch.cuda.get_device_name(0))
PY
echo

echo "===== PREDICTION START ====="

CMD=(
  .venv/bin/python src/predict.py
  --checkpoint "$CHECKPOINT"
  --test_file "$TEST_FILE"
  --output_dir "$PRED_OUT_DIR"
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
)

if [ -n "$MODEL_NAME" ]; then
  CMD+=(--model_name "$MODEL_NAME")
fi

printf 'Running command:\n%s\n\n' "${CMD[*]}"
"${CMD[@]}"

echo "===== PREDICTION END ====="
echo "Generated files:"
find "$PRED_OUT_DIR" -maxdepth 2 -type f | sort
