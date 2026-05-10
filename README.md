# NLP Underachievers

Transformer-based token classification baseline for NER-style `.iob2` data. The repo currently contains:

- Training and evaluation for two encoder backbones:
  - `google-bert/bert-base-multilingual-cased`
  - `FacebookAI/xlm-roberta-base`
- Prediction/export for masked test files while preserving the original file structure
- Slurm jobs for cluster environment setup, training, and prediction
- A Kaggle download helper for the LINCE SpaEng dataset

## Repository Layout

```text
.
|-- data/
|   `-- raw/                         # downloaded raw data
|-- data-baseline/
|   |-- en_ewt-ud-train.iob2
|   |-- en_ewt-ud-dev.iob2
|   `-- en_ewt-ud-test-masked.iob2
|-- outputs/                        # example run outputs
|-- scripts/
|   |-- download_data.py
|   |-- setup_env.slurm
|   |-- train.slurm
|   |-- prediction.slurm
|   `-- span_f1.py
|-- src/
|   |-- data.py
|   |-- modeling.py
|   |-- predict.py
|   |-- train.py
|   `-- utils.py
|-- pyproject.toml
`-- uv.lock
```

## Python Environment

This project uses `uv` and targets Python `>=3.11`.

Install dependencies locally:

```powershell
uv sync
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Installed dependencies are defined in `pyproject.toml`, including:

- `transformers`
- `datasets`
- `seqeval`
- `accelerate`
- `evaluate`
- `scikit-learn`
- `numpy`
- `pandas`
- `tqdm`
- `pyyaml`
- `kaggle`

## Data

The training and prediction scripts currently default to the English EWT baseline files in `data-baseline/`:

- `data-baseline/en_ewt-ud-train.iob2`
- `data-baseline/en_ewt-ud-dev.iob2`
- `data-baseline/en_ewt-ud-test-masked.iob2`

The parser in `src/data.py` supports common CoNLL/IOB2-style layouts, including files with:

- `token<TAB>label`
- `index<TAB>token<TAB>label`
- comment lines starting with `#`
- sentence breaks as blank lines

### Optional Kaggle Download

If you want the SpaEng LINCE data that the helper script references, make sure the Kaggle CLI is configured, then run:

```powershell
python .\scripts\download_data.py
```

This downloads the dataset archive into `data\raw\`, extracts the target CSV files, and removes the ZIP afterwards.

## Local Training

`src/train.py` trains a token classification model, evaluates on the dev set after each epoch, writes dev predictions, and stores the best checkpoint by dev F1.

### Train with mBERT

```powershell
python .\src\train.py `
  --model_name google-bert/bert-base-multilingual-cased `
  --train_file data-baseline/en_ewt-ud-train.iob2 `
  --dev_file data-baseline/en_ewt-ud-dev.iob2 `
  --output_dir outputs\mbert_baseline_run `
  --epochs 3 `
  --batch_size 16 `
  --learning_rate 2e-5 `
  --max_length 128 `
  --seed 42 `
  --checkpoint_dir checkpoints\mbert_baseline_run `
  --resume auto
```

### Train with XLM-R

```powershell
python .\src\train.py `
  --model_name FacebookAI/xlm-roberta-base `
  --train_file data-baseline/en_ewt-ud-train.iob2 `
  --dev_file data-baseline/en_ewt-ud-dev.iob2 `
  --output_dir outputs\xlm-r_baseline_run `
  --epochs 3 `
  --batch_size 16 `
  --learning_rate 2e-5 `
  --max_length 128 `
  --seed 42 `
  --checkpoint_dir checkpoints\xlm-r_baseline_run `
  --resume auto
```

### Training Outputs

Each run writes:

- `outputs/<run>/best_<safe_model_name>/` for the best saved model and tokenizer
- `outputs/<run>/dev_predictions_<safe_model_name>.iob2` for token-level dev predictions
- `checkpoints/<run>/last.pt` if `--checkpoint_dir` is set

Important implementation details:

- The label vocabulary is built from train + dev labels.
- Evaluation uses `seqeval`.
- `--resume auto` resumes from `last.pt` when `--checkpoint_dir` is provided.
- `--model_name` is restricted to the two baseline backbones listed above.

## Local Prediction

`src/predict.py` loads a saved checkpoint and reconstructs a prediction file that keeps blank lines, comments, and token order from the input test file.

Example:

```powershell
python .\src\predict.py `
  --checkpoint outputs\mbert_baseline_run\best_google-bert-bert-base-multilingual-cased `
  --test_file data-baseline/en_ewt-ud-test-masked.iob2 `
  --output_dir outputs\mbert_baseline_run\test_predictions `
  --batch_size 16 `
  --max_length 128 `
  --model_name google-bert/bert-base-multilingual-cased
```

The output file name is generated as:

```text
predictions_<safe_model_name>.iob2
```

For the mBERT example above:

```text
outputs/mbert_baseline_run/test_predictions/predictions_google-bert-bert-base-multilingual-cased.iob2
```

## Span-Level Evaluation Helper

For additional span-level evaluation, the repo includes:

```powershell
python .\scripts\span_f1.py <gold_file> <predicted_file>
```

It reports:

- exact labeled span precision / recall / F1
- unlabeled span precision / recall / F1
- loose overlap precision / recall / F1

## Slurm Usage

The Slurm scripts assume the repo lives at:

```bash
$HOME/projects/NLP-underachievers
```

They also expect a project virtual environment at:

```bash
.venv/bin/python
```

If your cluster path is different, update `PROJECT_DIR` inside the scripts before submitting.

### 1. Environment Setup Job

This creates or syncs the `uv` environment and installs the CUDA-enabled PyTorch wheel used by the cluster job.

Submit:

```bash
sbatch scripts/setup_env.slurm
```

Useful follow-up commands:

```bash
squeue -u $USER
tail -f logs/setup_env_<jobid>.out
```

### 2. Training Job

Default submission:

```bash
sbatch scripts/train.slurm
```

This currently runs mBERT with:

- `RUN_NAME=mbert_baseline`
- output directory `outputs/$RUN_NAME`
- checkpoint directory `checkpoints/$RUN_NAME`
- `--resume auto`

Submit with a custom run name:

```bash
sbatch --export=ALL,RUN_NAME=mbert_baseline_run scripts/train.slurm
```

Submit an XLM-R training run by overriding the script command after copying or editing the model argument inside `scripts/train.slurm`.

If you want to keep the script as-is but store outputs under another name:

```bash
sbatch --export=ALL,RUN_NAME=xlm-r_baseline_run scripts/train.slurm
```

Logs:

```bash
tail -f logs/nlp_train_<jobid>.out
tail -f logs/nlp_train_<jobid>.err
```

### 3. Prediction Job

Default submission:

```bash
sbatch scripts/prediction.slurm
```

The prediction job supports runtime overrides through `sbatch --export`.

Example using the mBERT checkpoint:

```bash
sbatch --export=ALL,RUN_NAME=mbert_baseline_run,CHECKPOINT=outputs/mbert_baseline_run/best_google-bert-bert-base-multilingual-cased,PRED_OUT_DIR=outputs/mbert_baseline_run/test_predictions,MODEL_NAME=google-bert/bert-base-multilingual-cased scripts/prediction.slurm
```

Example using the XLM-R checkpoint:

```bash
sbatch --export=ALL,RUN_NAME=xlm-r_baseline_run,CHECKPOINT=outputs/xlm-r_baseline_run/best_FacebookAI-xlm-roberta-base,PRED_OUT_DIR=outputs/xlm-r_baseline_run/test_predictions,MODEL_NAME=FacebookAI/xlm-roberta-base scripts/prediction.slurm
```

You can also override the test file and batch settings:

```bash
sbatch --export=ALL,RUN_NAME=mbert_baseline_run,CHECKPOINT=outputs/mbert_baseline_run/best_google-bert-bert-base-multilingual-cased,TEST_FILE=data-baseline/en_ewt-ud-test-masked.iob2,PRED_OUT_DIR=outputs/mbert_baseline_run/test_predictions,BATCH_SIZE=16,MAX_LENGTH=128,MODEL_NAME=google-bert/bert-base-multilingual-cased scripts/prediction.slurm
```

Logs:

```bash
tail -f logs/nlp_predict_<jobid>.out
tail -f logs/nlp_predict_<jobid>.err
```

### 4. Cancel or Inspect Jobs

```bash
squeue -u $USER
scancel <jobid>
```

## Current Example Outputs in the Repo

The repository already contains sample output folders:

- `outputs/mbert_baseline_run/`
- `outputs/xlm-r_baseline_run/`
- `outputs/test_run/`

These are useful as references for expected prediction filenames and directory layout.

## Real NER Pipeline

The real Spanish-English code-switched tweet NER data lives in:

- `data/raw/ner_spaeng_train.csv`
- `data/raw/ner_spaeng_validation.csv`
- `data/raw/ner_spaeng_test.csv`

The new clean pipeline is implemented in:

- `src/data.py`
- `src/model_factory.py`
- `src/train.py`
- `src/predict.py`
- `src/evaluate.py`
- `src/utils.py`

### Train mBERT

```powershell
uv run python -m src.train --config configs/mbert.yaml
```

### Train XLM-R

```powershell
uv run python -m src.train --config configs/xlmr.yaml
```

### Train Language-Bias Variants

```powershell
uv run python -m src.train --config configs/mbert_language_bias.yaml
uv run python -m src.train --config configs/xlmr_language_bias.yaml
```

### Smoke Test

```powershell
uv run python -m src.train --config configs/mbert.yaml --smoke-test
```

You can also smoke-test XLM-R:

```powershell
uv run python -m src.train --config configs/xlmr.yaml --smoke-test
```

### Prediction

```powershell
uv run python -m src.predict --config configs/mbert.yaml --checkpoint outputs/real_ner/<run_dir>/best.pt --split test
```

If the selected split has usable gold labels, prediction also writes metrics and a classification report. The current `test` CSV appears to contain masked blank labels, so the pipeline saves predictions and skips seqeval metrics for that split.

### Entity-Type Language Bias Module

The workflow now supports an optional entity-type-specific language bias head on top of the multilingual encoder:

- token-level `lid` labels are parsed from the CSV and aligned to subwords
- the classifier can learn an entity-type-by-language bias table
- an optional soft gate combines hidden states with a language embedding before adjusting the label logits
- training exports `train_entity_language_bias.json` and `validation_entity_language_bias.json` so you can inspect which entity types skew toward which language IDs

Use the baseline configs (`configs/mbert.yaml`, `configs/xlmr.yaml`) for standard fine-tuning and the language-bias configs (`configs/mbert_language_bias.yaml`, `configs/xlmr_language_bias.yaml`) for the new architecture.

### SLURM Commands

```bash
sbatch scripts/train.slurm
sbatch --export=CONFIG=configs/xlmr.yaml scripts/train.slurm
sbatch --export=CONFIG=configs/mbert.yaml,CHECKPOINT=outputs/real_ner/<run_dir>/best.pt,SPLIT=test scripts/predict.slurm
```

### Output Files

Each training run writes a timestamped directory under `outputs/real_ner/` containing:

- `config.yaml`
- `label_maps.json`
- `metrics.jsonl`
- `last.pt`
- `best.pt`
- `best_validation_metrics.json`
- `validation_predictions_last.jsonl`
- `validation_predictions_last.csv`
- `validation_predictions_best.jsonl`
- `validation_predictions_best.csv`
- `run_summary.json`

Prediction writes split-specific exports alongside the selected checkpoint:

- `predictions_<split>.jsonl`
- `predictions_<split>.csv`
- `metrics_<split>.json`
- `classification_report_<split>.txt` when gold labels exist
