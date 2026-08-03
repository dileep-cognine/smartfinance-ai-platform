# Module 3 Report - Pre-Trained Models

## FinBERT

`finbert` fine-tunes `ProsusAI/finbert` on the same deterministic Financial
PhraseBank split used by Module 2. Hugging Face Trainer evaluates every epoch,
uses macro-F1 for best-model selection, stops after two non-improving epochs,
and reports to MLflow. The result JSON contains per-class metrics and up to 20
misclassified candidates. Select and analyze three genuine failures, including
negation, mixed sentiment, or missing financial context, rather than choosing
examples that merely support a desired conclusion.

## LoRA

`lora-classifier` uses `facebook/opt-125m` because its attention blocks expose
the required `q_proj` and `v_proj` modules. Configuration is `r=8`,
`lora_alpha=16`, dropout 0.1. The workflow records total/trainable parameters,
training seconds, peak allocated GPU memory, and held-out metrics. Run the
corresponding full OPT fine-tune if the full-vs-LoRA comparison is required
literally; FinBERT versus OPT-LoRA alone does not isolate the tuning method.

## Filing Summarization

`lora-summarizer` expects reviewed JSONL filing-summary pairs, trains BART with
LoRA, evaluates ROUGE-1/2/L, and supports hierarchical summarization for inputs
longer than 1,024 tokens. Factual completeness still requires a human checklist
of reported amounts, dates, entities, and risk statements; ROUGE is not a
factuality metric.

## Results

Populate parameter counts, time, memory, classification improvements, ROUGE,
five sample summaries, and three failure analyses from generated result files.
No measured value should be written before the corresponding run completes.

The assessment container uses pinned CPU PyTorch on
`python:3.11.9-slim-bookworm` for portable builds. Production GPU training can
use the document's recommended CUDA image without changing the workflows.
