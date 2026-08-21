"""FinBERT full tuning and OPT LoRA classification workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..evaluation.metrics import classification_metrics, detailed_classification_metrics

LABELS = {"negative": 0, "neutral": 1, "positive": 2}


def load_splits(path: Path) -> dict[str, Dataset]:
    frame = pd.read_csv(path).dropna(subset=["Sentence", "Sentiment"])
    frame["label"] = frame["Sentiment"].str.lower().map(LABELS)
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame["label"],
        random_state=42,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder["label"],
        random_state=42,
    )
    return {
        name: Dataset.from_pandas(split[["Sentence", "label"]], preserve_index=False)
        for name, split in {
            "train": train,
            "validation": validation,
            "test": test,
        }.items()
    }


def parameter_counts(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total,
    }


def train_classifier(
    data_path: Path,
    output_dir: Path,
    model_name: str = "ProsusAI/finbert",
    use_lora: bool = False,
    epochs: int = 5,
    mlflow_enabled: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={value: key for key, value in LABELS.items()},
        label2id=LABELS,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if use_lora:
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.SEQ_CLS,
            ),
        )
    counts = parameter_counts(model)
    splits = load_splits(data_path)

    def tokenize(batch):
        return tokenizer(batch["Sentence"], truncation=True, max_length=256)

    tokenized = {name: split.map(tokenize, batched=True) for name, split in splits.items()}
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5 if not use_lora else 1e-4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to=["mlflow"] if mlflow_enabled else [],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=classification_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    prediction = trainer.predict(tokenized["test"])
    predicted = np.argmax(prediction.predictions, axis=-1).tolist()
    labels = prediction.label_ids.tolist()
    failures = []
    for row, label, predicted_label in zip(splits["test"], labels, predicted, strict=True):
        if label != predicted_label:
            failures.append(
                {
                    "sentence": row["Sentence"],
                    "actual": int(label),
                    "predicted": int(predicted_label),
                }
            )
        if len(failures) == 20:
            break
    summary = {
        "model": model_name,
        "method": "lora" if use_lora else "full",
        "training_seconds": elapsed,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        ),
        "parameters": counts,
        "test_metrics": prediction.metrics,
        "classification_report": detailed_classification_metrics(labels, predicted),
        "failure_candidates": failures,
    }
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(output_dir / "model")
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
