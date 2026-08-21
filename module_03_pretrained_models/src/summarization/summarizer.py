"""LoRA BART training and long-filing summarization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ..evaluation.metrics import rouge_metrics


def load_pairs(path: Path) -> Dataset:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all({"document", "summary"}.issubset(row) for row in rows):
        raise ValueError("JSONL must contain document and summary fields")
    return Dataset.from_list(rows)


def train_lora_summarizer(
    pairs_path: Path,
    output_dir: Path,
    model_name: str = "facebook/bart-large-cnn",
    epochs: int = 3,
    mlflow_enabled: bool = True,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        ),
    )
    dataset = load_pairs(pairs_path).train_test_split(test_size=0.2, seed=42)

    def tokenize(batch):
        inputs = tokenizer(batch["document"], max_length=1024, truncation=True)
        labels = tokenizer(text_target=batch["summary"], max_length=256, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["document", "summary"])
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["mlflow"] if mlflow_enabled else [],
        seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    prediction = trainer.predict(tokenized["test"])
    decoded_predictions = tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True)
    label_ids = prediction.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    metrics = rouge_metrics(references, decoded_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(output_dir / "model")
    results = {"rouge": metrics, "samples": list(zip(references, decoded_predictions, strict=True))}
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def summarize_long_text(
    text: str,
    model,
    tokenizer,
    max_input_tokens: int = 1024,
    max_summary_tokens: int = 180,
) -> str:
    token_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    device = next(model.parameters()).device
    partial: list[str] = []
    for start in range(0, len(token_ids), max_input_tokens):
        chunk = token_ids[start : start + max_input_tokens].unsqueeze(0).to(device)
        with torch.no_grad():
            generated = model.generate(
                chunk,
                max_new_tokens=max_summary_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        partial.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    if len(partial) == 1:
        return partial[0]
    combined = tokenizer(
        " ".join(partial),
        return_tensors="pt",
        max_length=max_input_tokens,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        generated = model.generate(**combined, max_new_tokens=max_summary_tokens, num_beams=4)
    return tokenizer.decode(generated[0], skip_special_tokens=True)
