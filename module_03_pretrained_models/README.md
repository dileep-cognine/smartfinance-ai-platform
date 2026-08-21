# Module 3: Pre-Trained Models and LoRA

This module fine-tunes pre-trained transformer models for Financial PhraseBank
sentiment classification and includes a separate LoRA summarization workflow.

- `finbert` fine-tunes `ProsusAI/finbert` for three-class financial sentiment.
- `lora-classifier` applies LoRA to `facebook/opt-125m` for the same task.
- `lora-summarizer` applies LoRA to `facebook/bart-large-cnn` using reviewed
  document/summary pairs.

The classification workflows use a deterministic 70/15/15 stratified split,
evaluate accuracy and macro F1, save the best checkpoint, and record up to 20
misclassified test examples for review.

## Data

The classification commands default to the shared dataset:

```text
../data/processed/cleaned/financial_phrasebank_clean.csv
```

It must contain `Sentence` and `Sentiment` columns. The summarization workflow
requires a separate JSONL file with one object per line and both `document` and
`summary` fields. No such reviewed pair dataset is currently included, so run
the two classification workflows first.

## Run classification

Run these commands from `module_03_pretrained_models`. The first execution
downloads the Hugging Face model weights and can take several minutes,
especially on CPU.

```powershell
# Validate the module first
python -m pytest -q

# Quick combined smoke run
python -m src.main run-all --epochs 1 --no-mlflow

# Full FinBERT + LoRA comparison
python -m src.main run-all --epochs 5 --no-mlflow
```

`--no-mlflow` prevents local MLflow tracking during standalone runs. Omit it
when the MLflow tracking server is configured and you want experiment records.

Individual workflows are also available:

```powershell
python -m src.main finbert --epochs 5 --no-mlflow
python -m src.main lora-classifier --epochs 5 --no-mlflow
```

## Run summarization

After creating and reviewing a JSONL pairs file:

```powershell
python -m src.main lora-summarizer --pairs C:\path\to\filing_pairs.jsonl --epochs 3 --no-mlflow
```

## Outputs

Classification outputs are stored in the shared artifact directory:

```text
../artifacts/module_03/
├── finbert/
│   ├── checkpoints/
│   ├── model/
│   └── results.json
├── lora_classifier/
│   ├── checkpoints/
│   ├── model/
│   └── results.json
└── classification_comparison.json
```

`results.json` includes elapsed training time, parameter counts, test metrics,
per-class classification metrics, and failure candidates. Populate the measured
comparison and failure analysis in `REPORT.md` only after reviewing these files.

## Quality checks

```powershell
python -m pytest -q
python -m ruff check .
```

The tests validate parameter counting, required OPT LoRA target modules, and
the combined command's shared data/artifact defaults.
